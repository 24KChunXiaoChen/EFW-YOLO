"""Rewrite static channel shuffles, then require CPU ORT numerical equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import shutil

from _common import setup_runtime


def optimize_verified(input_path, output_path, *, samples=3, seed=0, atol=1e-4, rtol=1e-4):
    setup_runtime()
    import numpy as np
    import onnx
    import onnxruntime as ort
    from efw_yolo.deployment.shuffle import optimize_graph

    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or output.exists() or report_path.exists():
        raise FileExistsError("Input, existing output and existing reports must never be overwritten")
    if samples < 1 or output.suffix.lower() != ".onnx":
        raise ValueError("Use an .onnx output and at least one verification sample")
    original = onnx.load(str(source))
    transformed, rewrite_report = optimize_graph(original)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    before = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    if len(before.get_inputs()) != 1:
        raise ValueError("Verification requires a single static float32 image input")
    tensor = before.get_inputs()[0]
    if tensor.type != "tensor(float)" or any(not isinstance(d, int) or d <= 0 for d in tensor.shape):
        raise ValueError("Verification requires a static float32 input shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="efw-shuffle-", dir=output.parent) as temp:
        candidate = Path(temp) / "optimized.onnx"
        onnx.save(transformed, str(candidate))
        after = ort.InferenceSession(str(candidate), sess_options=options, providers=["CPUExecutionProvider"])
        names = [o.name for o in before.get_outputs()]
        if names != [o.name for o in after.get_outputs()]:
            raise ValueError("Optimization changed the public output contract")
        rng = np.random.default_rng(seed)
        errors = {name: 0.0 for name in names}
        for _ in range(samples):
            values = rng.random(tensor.shape, dtype=np.float32)
            expected = before.run(names, {tensor.name: values})
            actual = after.run(names, {tensor.name: values})
            for name, ref, result in zip(names, expected, actual):
                if not np.isfinite(ref).all() or not np.isfinite(result).all():
                    raise ValueError(f"Non-finite values in output {name}")
                np.testing.assert_allclose(result, ref, rtol=rtol, atol=atol, err_msg=name)
                errors[name] = max(errors[name], float(np.max(np.abs(result - ref))))
        report = {
            "status": "passed", "input": str(source), "output": str(output),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "rewrite": rewrite_report, "verification": {"samples": samples, "seed": seed,
            "provider": "CPUExecutionProvider", "input_shape": tensor.shape, "rtol": rtol, "atol": atol,
            "max_abs_error": errors}, "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
        }
        with candidate.open("rb") as handle, output.open("xb") as target:
            shutil.copyfileobj(handle, target)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(optimize_verified(args.input, args.output, samples=args.samples, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
