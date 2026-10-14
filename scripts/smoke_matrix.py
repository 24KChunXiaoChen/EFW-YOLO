"""Run the configured model variants serially and check training and checkpoint reload."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from _common import ROOT, bind_dataset, code_snapshot, new_output_dir, sha256, write_json


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def run_logged(command, log, timeout):
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f'Exit {process.returncode}; see {log}')


def verify_training(run_path, epochs, expected_code, expected_config):
    record = json.loads(run_path.read_text(encoding='utf-8'))
    if record['status'] != 'completed' or record['epochs_completed'] != epochs:
        raise ValueError('Training did not finish the requested epochs.')
    if record['code_sha256'] != expected_code or record['config_sha256'] != expected_config:
        raise ValueError('Training source/config differs from the campaign snapshot.')
    with Path(record['results_csv']).open(encoding='utf-8') as stream:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    if [int(float(row['epoch'])) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError('Missing, duplicated, or unordered epochs in results.csv.')
    loss_fields = [key for key in rows[0] if key.startswith(('train/', 'val/')) and key.endswith('_loss')]
    if not loss_fields or not any(key.startswith('train/') for key in loss_fields) or not any(key.startswith('val/') for key in loss_fields):
        raise ValueError('Training/validation losses are missing.')
    for row in rows:
        for key, value in row.items():
            if not math.isfinite(float(value)):
                raise ValueError(f'Non-finite {key} at epoch {row["epoch"]}')
    best = Path(record['best_weights'])
    last = best.with_name('last.pt')
    if not best.is_file() or not last.is_file() or sha256(best) != record['best_sha256']:
        raise ValueError('Missing or changed checkpoint.')
    return record, {'epochs_verified': len(rows), 'finite_loss_fields': loss_fields,
                    'all_csv_values_finite': True, 'best_sha256': record['best_sha256'],
                    'last_sha256': sha256(last), 'last_epoch_losses': {key: float(rows[-1][key]) for key in loss_fields}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--models', nargs='+', help='Explicit experiment IDs; otherwise process the complete inventory.')
    parser.add_argument('--timeout-per-step', type=int, default=900)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.timeout_per_step < 1:
        parser.error('epochs, batch, and timeout must be positive.')
    data = args.data.resolve()
    identity = bind_dataset(data)
    if identity['mode'] != 'smoke':
        parser.error('Use a prepared train/val-only smoke subset.')
    matrix = json.loads((ROOT / 'configs/experiments.json').read_text(encoding='utf-8'))
    experiments = matrix['experiments']
    if args.models:
        unknown = set(args.models) - {item['id'] for item in experiments}
        if unknown:
            parser.error(f'Unknown experiment IDs: {sorted(unknown)}')
        experiments = [item for item in experiments if item['id'] in args.models]
    output = new_output_dir(args.output)
    snapshot = code_snapshot()
    write_json(output / 'source_snapshot.json', snapshot)
    report = {'started_at': now(), 'status': 'running', 'python': sys.executable,
              'code_sha256': snapshot['sha256'], 'data_identity': identity,
              'settings': {'epochs': args.epochs, 'batch': args.batch, 'imgsz': 512, 'device': args.device,
                           'seed': 0, 'initialization': 'scratch', 'optimizer': 'MuSGD', 'amp': False,
                           'patience': 0, 'workers': 0, 'split': 'val'},
              'final_paper_result': False, 'cases': []}
    for item in experiments:
        filename = item.get('target_config_filename')
        config = ROOT / 'configs/models' / filename if filename else None
        case = {'id': item['id'], 'config': filename, 'historical_run_identity_verified': False,
                'status': 'pending', 'started_at': now()}
        report['cases'].append(case)
        if config is None or not config.is_file():
            case.update(status='not_tested_missing_configuration', reason='Original comparison configuration is not identified.')
            write_json(output / 'summary.json', report)
            print(f"{item['id']}: missing configuration, not counted as passed", flush=True)
            continue
        case.update(config_sha256=sha256(config), status='running')
        case_dir = output / item['id']
        case_dir.mkdir()
        write_json(output / 'summary.json', report)
        print(f"{now()} START {item['id']}", flush=True)
        started = time.monotonic()
        try:
            if code_snapshot()['sha256'] != snapshot['sha256']:
                raise RuntimeError('Campaign code changed; freeze inputs before continuing.')
            train_command = [sys.executable, str(ROOT / 'scripts/train.py'), '--data', str(data),
                             '--config', str(config), '--output', str(case_dir / 'training'),
                             '--phase', 'smoke', '--scratch', '--patience', '0', '--epochs', str(args.epochs),
                             '--batch', str(args.batch), '--device', args.device, '--workers', '0', '--seed', '0']
            case['train_command'] = train_command
            run_logged(train_command, case_dir / 'train.log', args.timeout_per_step)
            record, checked = verify_training(case_dir / 'training/run.json', args.epochs, snapshot['sha256'], case['config_sha256'])
            val_command = [sys.executable, str(ROOT / 'scripts/evaluate.py'), '--weights', record['best_weights'],
                           '--data', str(data), '--output', str(case_dir / 'reload_validation'), '--split', 'val',
                           '--imgsz', '512', '--batch', '1', '--device', args.device]
            case['validation_command'] = val_command
            run_logged(val_command, case_dir / 'validation.log', args.timeout_per_step)
            validation = json.loads((case_dir / 'reload_validation/metrics.json').read_text(encoding='utf-8'))
            if validation['weights_sha256'] != record['best_sha256'] or validation['code_sha256'] != snapshot['sha256']:
                raise ValueError('Reload validation used different weights or code.')
            if validation['data_identity'] != identity or not all(math.isfinite(value) for value in validation['metrics'].values()):
                raise ValueError('Validation data identity differs or metrics are non-finite.')
            checked['checkpoint_reload_validation_passed'] = True
            case.update(status='passed', checks=checked,
                        run_record=str((case_dir / 'training/run.json').relative_to(output)),
                        validation_record=str((case_dir / 'reload_validation/metrics.json').relative_to(output)))
        except Exception as error:
            case.update(status='failed', error=f'{type(error).__name__}: {error}')
        case.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(output / 'summary.json', report)
        print(f"{now()} {case['status'].upper()} {item['id']} ({case['elapsed_seconds']}s)", flush=True)
    counts = {state: sum(case['status'] == state for case in report['cases'])
              for state in ('passed', 'failed', 'not_tested_missing_configuration')}
    report.update(finished_at=now(), counts=counts,
                  status='failed' if counts['failed'] else 'available_configurations_passed_with_pending' if counts['not_tested_missing_configuration'] else 'passed')
    write_json(output / 'summary.json', report)
    print(json.dumps({'status': report['status'], 'counts': counts, 'report': str(output / 'summary.json')}, ensure_ascii=False), flush=True)
    if counts['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
