"""Export the six raw detection branches and verify them with CPU ONNX Runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import types

from _common import ROOT, load_model, setup_runtime

OUTPUT_NAMES = ["box0", "score0", "box1", "score1", "box2", "score2"]


def six_branch_forward(self, features):
    """Raw box distances and class logits; decoding/NMS are downstream operations."""
    return tuple(value for i in range(3) for value in (self.cv2[i](features[i]), self.cv3[i](features[i])))


def prepare_model(yolo):
    setup_runtime()
    from ultralytics.nn.modules import Detect

    model = copy.deepcopy(yolo.model).cpu().float().eval()
    heads = [module for module in model.modules() if isinstance(module, Detect)]
    if len(heads) != 1 or heads[0].nl != 3 or heads[0].nc != 1:
        raise ValueError("Export requires exactly one single-class, three-scale Detect head")
    if getattr(heads[0], "end2end", False):
        raise ValueError("This export contract requires end2end=False")
    heads[0].forward = types.MethodType(six_branch_forward, heads[0])
    return model


def export_verified(yolo, output, *, imgsz=512, samples=2, seed=0, provenance=None):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    output = Path(output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite output or report: {output}")
    if output.suffix.lower() != ".onnx" or imgsz < 32 or imgsz % 32 or samples < 1:
        raise ValueError("Use an .onnx output, an input size divisible by 32, and at least one sample")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = prepare_model(yolo)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    example = torch.rand((1, 3, imgsz, imgsz), generator=generator)
    with tempfile.TemporaryDirectory(prefix="efw-export-", dir=output.parent) as temp:
        candidate = Path(temp) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model, example, str(candidate), input_names=["images"], output_names=OUTPUT_NAMES,
                opset_version=18, dynamo=False, do_constant_folding=True,
            )
        graph = onnx.load(str(candidate))
        onnx.checker.check_model(graph)
        if [x.name for x in graph.graph.output] != OUTPUT_NAMES:
            raise ValueError("Unexpected exported output names")
        if next(x.version for x in graph.opset_import if x.domain == "") != 18:
            raise ValueError("Unexpected ONNX opset")
        for key, value in {
            "efw_output_contract": "raw distances and class logits; no sigmoid, decode or NMS",
            "efw_export_device": "cpu", "efw_random_init": str(bool((provenance or {}).get("random_init"))),
        }.items():
            entry = graph.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(graph, str(candidate))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        session = ort.InferenceSession(str(candidate), sess_options=options, providers=["CPUExecutionProvider"])
        max_errors = {name: 0.0 for name in OUTPUT_NAMES}
        shapes = {}
        for index in range(samples):
            sample = example if index == 0 else torch.rand((1, 3, imgsz, imgsz), generator=generator)
            with torch.inference_mode():
                expected = model(sample)
            actual = session.run(OUTPUT_NAMES, {"images": sample.numpy()})
            for name, reference, result in zip(OUTPUT_NAMES, expected, actual):
                reference = reference.numpy()
                if not np.isfinite(reference).all() or not np.isfinite(result).all():
                    raise ValueError(f"Non-finite values in output {name}")
                np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5, err_msg=name)
                max_errors[name] = max(max_errors[name], float(np.max(np.abs(result - reference))))
                shapes[name] = list(result.shape)
        report = {
            "status": "passed", "output": str(output), "opset": 18, "input_shape": list(example.shape),
            "output_shapes": shapes, "verification": {"samples": samples, "seed": seed,
            "provider": "CPUExecutionProvider", "rtol": 1e-4, "atol": 1e-5, "max_abs_error": max_errors},
            "provenance": provenance or {}, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        }
        # Exclusive creation also protects against another process creating the destination during verification.
        with candidate.open("rb") as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--weights", type=Path)
    origin.add_argument("--random-init", action="store_true", help="Engineering verification only; not trained results")
    parser.add_argument("--config", type=Path, help="Architecture for --random-init; a checkpoint already contains its architecture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.weights is not None and args.config is not None:
        parser.error("--config applies to --random-init only; checkpoint architecture must not be silently replaced")
    if args.weights is not None and not args.weights.is_file():
        parser.error("--weights must name an existing local file; downloads are not supported")
    if args.config is not None and not args.config.is_file():
        parser.error("--config must name an existing local file")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        parser.error("Output or report already exists; choose a new path")
    setup_runtime()
    import torch
    torch.manual_seed(args.seed)
    model = load_model(weights=args.weights, config=args.config)
    config_path = (args.config or ROOT / "configs/models/efw_yolo.yaml").resolve() if args.random_init else None
    provenance = {"random_init": args.random_init, "weights": str(args.weights.resolve()) if args.weights else None,
                  "config": str(config_path) if config_path else None}
    if config_path:
        provenance["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if args.weights:
        provenance["weights_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    report = export_verified(model, args.output, imgsz=args.imgsz, samples=args.samples, seed=args.seed, provenance=provenance)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
