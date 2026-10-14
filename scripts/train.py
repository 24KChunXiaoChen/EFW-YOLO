"""Train one explicitly selected configuration; never start a comparison sweep or test evaluation."""
import argparse
import csv
from pathlib import Path

from _common import ROOT, bind_dataset, code_snapshot, environment, load_model, new_output_dir, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/models/efw_yolo.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=['smoke', 'experiment'], required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--scratch', action='store_true')
    initialization.add_argument('--weights', type=Path)
    parser.add_argument('--patience', type=int, required=True, help='Explicit stopping rule; 0 disables early stopping.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    for value in (args.data, args.config, args.weights):
        if value is not None and not value.resolve().is_file():
            parser.error(f'File not found: {value}')
    epochs = args.epochs if args.epochs is not None else (10 if args.phase == 'smoke' else 1000)
    batch = args.batch if args.batch is not None else (4 if args.phase == 'smoke' else 48)
    if epochs <= 0 or batch <= 0 or args.patience < 0 or args.imgsz <= 0 or args.imgsz % 32:
        parser.error('epochs/batch must be positive, patience >= 0, and imgsz a positive multiple of 32.')
    data_identity = bind_dataset(args.data, args.data_manifest)
    snapshot = code_snapshot()
    output = new_output_dir(args.output)
    write_json(output / 'source_snapshot.json', snapshot)
    config = dict(data=str(args.data.resolve()), imgsz=args.imgsz, epochs=epochs, batch=batch,
                  device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
                  optimizer='MuSGD', lr0=0.01, weight_decay=0.0005, patience=args.patience,
                  pretrained=False, single_cls=True, amp=False, resume=False, val=True, plots=False,
                  cache=False, project=str(output), name='train', exist_ok=False,
                  close_mosaic=0 if args.phase == 'smoke' else 10)
    provenance = {'phase': args.phase, 'final_paper_result': False, 'status': 'started',
                  'environment': environment(), 'config': str(args.config.resolve()),
                  'config_sha256': sha256(args.config), 'data_yaml_sha256': sha256(args.data),
                  'initialization': 'scratch' if args.scratch else str(args.weights.resolve()),
                  'weights_sha256': sha256(args.weights) if args.weights else None,
                  'data_identity': data_identity, 'code_sha256': snapshot['sha256'],
                  'train_arguments': config,
                  'note': 'Engineering settings explicitly recorded. Historical seed/initialization/stopping are unverified.'}
    write_json(output / 'run.json', provenance)
    try:
        # Seed before model construction as well as inside the trainer.
        from ultralytics.utils.torch_utils import init_seeds
        init_seeds(args.seed, deterministic=True)
        model = load_model(config=args.config)
        if args.weights:
            model.load(str(args.weights.resolve()))
        model.train(**config)
        csv_path = Path(model.trainer.save_dir) / 'results.csv'
        with csv_path.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        epochs_seen = [int(float(row['epoch'])) for row in rows]
        if not epochs_seen or epochs_seen != list(range(1, len(rows) + 1)):
            raise RuntimeError(f'Invalid epoch sequence: {epochs_seen}')
        if args.patience == 0 and len(rows) != epochs:
            raise RuntimeError(f'Expected {epochs} epochs, found {len(rows)}')
        weights = Path(model.trainer.best)
        if code_snapshot()['sha256'] != snapshot['sha256'] or bind_dataset(args.data, args.data_manifest) != data_identity:
            raise RuntimeError('Code or data changed during training; results require separate review.')
        provenance.update(status='completed', epochs_completed=len(rows),
                          best_weights=str(weights), best_sha256=sha256(weights),
                          results_csv=str(csv_path), final_validation={k: float(v) for k, v in model.trainer.metrics.items()})
        write_json(output / 'run.json', provenance)
    except Exception as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', provenance)
        raise


if __name__ == '__main__':
    main()
