"""Measure runnable configurations without training or borrowing manuscript metrics."""
import argparse
import copy
from pathlib import Path

from _common import ROOT, environment, setup_runtime, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', action='append', type=Path)
    parser.add_argument('--imgsz', default=512, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    setup_runtime()
    import torch
    import yaml
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(2)
    configs = args.config or sorted((ROOT / 'configs/models').glob('*.yaml'))
    results = []
    for path in configs:
        try:
            raw = yaml.safe_load(path.read_text(encoding='utf-8'))
            torch.manual_seed(0)
            model = DetectionModel(copy.deepcopy(raw), nc=1, ch=3, verbose=False).eval()
            image = torch.zeros(1, 3, args.imgsz, args.imgsz)
            with torch.inference_mode():
                prediction = model(image)[0]
            end2end = bool(model.model[-1].end2end)
            # End-to-end heads return postprocessed [batch, detections, xyxy+score+class].
            # One-to-many heads return decoded [batch, xywh+class_scores, anchors].
            valid_layout = prediction.ndim == 3 and prediction.shape[0] == image.shape[0]
            if valid_layout:
                valid_layout = prediction.shape[2] == 6 if end2end else prediction.shape[1] == 5
            if not valid_layout or not torch.isfinite(prediction).all():
                raise ValueError('Invalid single-class predictions.')
            raw_params = sum(p.numel() for p in model.parameters())
            flops = get_flops(model, imgsz=args.imgsz)
            # Fuse on a copy so measurement never mutates the tested model.
            fused = copy.deepcopy(model).fuse(verbose=False)
            record = {'config': path.name, 'sha256': sha256(path), 'status': 'passed',
                      'parameters_unfused': raw_params, 'parameters_fused': sum(p.numel() for p in fused.parameters()),
                      'gflops_at_requested_imgsz': float(flops), 'imgsz': args.imgsz,
                      'stride': model.stride.tolist(), 'prediction_shape': list(prediction.shape),
                      'end2end': end2end, 'nc': model.model[-1].nc,
                      'historical_run_identity_verified': False}
        except Exception as error:
            record = {'config': path.name, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        results.append(record)
        print(f"{path.name}: {record['status']}")
    write_json(args.output, {'environment': environment(), 'results': results, 'final_paper_result': False})
    if any(record['status'] != 'passed' for record in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
