"""Audit a YOLO dataset or copy a deterministic train/val subset for engineering tests."""
import argparse
import json
import shutil
from pathlib import Path

from _common import new_output_dir, sha256, write_json

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def split_files(source, split):
    folder = source / split / 'images'
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_record(image, source, split):
    label = source / split / 'labels' / (image.stem + '.txt')
    count = 0
    formats = set()
    if label.is_file():
        for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if values[0] != 0 or not all(0 <= v <= 1 for v in values[1:]):
                raise ValueError(f'Invalid single-class detection label: {label}:{number}')
            if len(values) == 5 and min(values[3:]) > 0:
                formats.add('xywh')
            elif len(values) >= 7 and len(values) % 2 == 1:
                # Ultralytics accepts polygon labels and derives detection boxes.
                # Preserve those labels rather than silently converting the source.
                formats.add('polygon')
            else:
                raise ValueError(f'Invalid box/polygon label: {label}:{number}')
            count += 1
    return {'image': image.relative_to(source).as_posix(), 'image_sha256': sha256(image),
            'label': label.relative_to(source).as_posix() if label.is_file() else None,
            'label_sha256': sha256(label) if label.is_file() else None, 'instances': count,
            'label_formats': sorted(formats)}


def prepare(source, output, mode='audit', train_count=32, val_count=8):
    import yaml
    source = Path(source).expanduser().resolve()
    original_yaml = source / 'data.yaml'
    metadata = yaml.safe_load(original_yaml.read_text(encoding='utf-8-sig'))
    if metadata.get('nc', 1) != 1 or len(metadata.get('names', [])) != 1:
        raise ValueError('This EFW-YOLO workflow expects exactly one class.')
    output = new_output_dir(output)
    records, counts = {}, {}
    for split in ('train', 'val', 'test'):
        files = split_files(source, split)
        counts[split] = len(files)
        if mode == 'smoke' and split == 'test':
            continue  # Count the original partition, but never read test pixels/labels for a smoke run.
        if mode == 'smoke':
            wanted = train_count if split == 'train' else val_count
            if wanted <= 0 or len(files) < wanted:
                raise ValueError(f'{split}: requested {wanted} images, available {len(files)}')
            files = files[:wanted]
        records[split] = []
        stems = set()
        for image in files:
            if image.stem in stems:
                raise ValueError(f'Duplicate image stem in {split}: {image.stem}')
            stems.add(image.stem)
            record = label_record(image, source, split)
            records[split].append(record)
            if mode == 'smoke':
                target = output / split / 'images' / image.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target)
                target_label = output / split / 'labels' / (image.stem + '.txt')
                target_label.parent.mkdir(parents=True, exist_ok=True)
                if record['label']:
                    shutil.copy2(source / record['label'], target_label)
                else:
                    target_label.write_text('', encoding='utf-8')
        (output / f'{split}.txt').write_text(''.join(x['image'] + '\n' for x in records[split]), encoding='utf-8')
    report = {'mode': mode, 'source': str(source), 'source_data_yaml_sha256': sha256(original_yaml),
              'original_metadata': metadata, 'original_counts': counts,
              'selected_counts': {key: len(value) for key, value in records.items()},
              'selection': 'lexicographic first N per original partition' if mode == 'smoke' else 'all original files',
              'paper_expected_counts': {'train': 1669, 'val': 477, 'test': 238},
              'counts_match_manuscript': counts == {'train': 1669, 'val': 477, 'test': 238},
              'final_paper_result': False, 'files': records}
    report['label_formats'] = {kind: sum(kind in item['label_formats'] for items in records.values() for item in items)
                               for kind in ('xywh', 'polygon')}
    write_json(output / 'manifest.json', report)
    config = {'path': str(output if mode == 'smoke' else source), 'train': 'train/images', 'val': 'val/images',
              'nc': 1, 'names': {0: 'fire'}}
    if mode == 'audit':
        config['test'] = 'test/images'
    (output / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['audit', 'smoke'], default='audit')
    parser.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--val-count', type=int, default=8)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.mode, args.train_count, args.val_count)
    print(json.dumps({key: value for key, value in result.items() if key not in {'files', 'original_metadata'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
