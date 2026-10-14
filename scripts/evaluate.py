"""Evaluate a named local checkpoint with an explicit split and timing scope."""
import argparse
from pathlib import Path

from _common import bind_dataset, code_snapshot, environment, load_model, new_output_dir, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32 or args.batch <= 0:
        parser.error('imgsz must be a positive multiple of 32 and batch must be positive.')
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    import yaml
    data_config = yaml.safe_load(args.data.read_text(encoding='utf-8-sig'))
    if not data_config.get(args.split):
        parser.error(f'The dataset has no {args.split} split.')
    data_identity = bind_dataset(args.data, args.data_manifest)
    snapshot = code_snapshot()
    output = new_output_dir(args.output)
    write_json(output / 'source_snapshot.json', snapshot)
    model = load_model(weights=args.weights)
    result = model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
                       batch=args.batch, device=args.device, workers=0, plots=False,
                       project=str(output), name='validation', exist_ok=False, verbose=False)
    if code_snapshot()['sha256'] != snapshot['sha256'] or bind_dataset(args.data, args.data_manifest) != data_identity:
        raise RuntimeError('Code or data changed during evaluation; results require separate review.')
    report = {'weights': str(args.weights.resolve()), 'weights_sha256': sha256(args.weights),
              'data': str(args.data.resolve()), 'data_yaml_sha256': sha256(args.data),
              'split': args.split, 'imgsz': args.imgsz, 'batch': args.batch, 'environment': environment(),
              'data_identity': data_identity, 'code_sha256': snapshot['sha256'],
              'metrics': {key: float(value) for key, value in result.results_dict.items()},
              'speed_ms_per_image': {key: float(value) for key, value in result.speed.items()},
              'timing_scope': 'Ultralytics validation preprocess/inference/loss/postprocess; not ESP32-P4 latency',
              'final_paper_result': False}
    write_json(output / 'metrics.json', report)


if __name__ == '__main__':
    main()
