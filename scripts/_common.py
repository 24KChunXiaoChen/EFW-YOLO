"""Shared, location-independent setup for the paper's command-line tools."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup_runtime():
    """Prefer the frozen local runtime and isolate its settings from other projects."""
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs/.ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/.matplotlib'))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    for path in (ROOT, ROOT / 'vendor'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import ultralytics
    expected = ROOT / 'vendor/ultralytics/__init__.py'
    if Path(ultralytics.__file__).resolve() != expected.resolve():
        raise RuntimeError('A different Ultralytics runtime is already loaded; use a fresh process.')
    ultralytics.settings.update({key: False for key in
        ('sync', 'hub', 'clearml', 'comet', 'mlflow', 'neptune', 'wandb')})
    from efw_yolo import register_modules
    register_modules()


def load_model(weights=None, config=None):
    setup_runtime()
    from ultralytics import YOLO
    path = Path(weights or config or ROOT / 'configs/models/efw_yolo.yaml').expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if weights and path.suffix.lower() != '.pt':
        raise ValueError('Expected a local PyTorch .pt checkpoint.')
    return YOLO(str(path), task='detect')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def new_output_dir(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_snapshot():
    """Hash the actual runtime and entry points, independently of their source history."""
    records = []
    for folder in ('efw_yolo', 'vendor/ultralytics', 'scripts'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            records.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode('utf-8')).hexdigest()
    return {'sha256': digest, 'files': records}


def bind_dataset(data_yaml, manifest_path=None):
    """Check actual image/label bytes and partition membership against a prepared manifest."""
    import yaml
    data_yaml = Path(data_yaml).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else data_yaml.with_name('manifest.json')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Prepare/audit the dataset first or provide --data-manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = yaml.safe_load(data_yaml.read_text(encoding='utf-8-sig'))
    base = Path(config.get('path', data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    base = base.resolve()
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    verified = 0
    for split in ('train', 'val', 'test'):
        if not config.get(split):
            continue
        if split not in manifest['files']:
            raise ValueError(f'No manifest records for configured split {split}')
        split_path = Path(config[split])
        split_path = (split_path if split_path.is_absolute() else base / split_path).resolve()
        expected_folder = (base / split / 'images').resolve()
        if split_path != expected_folder:
            raise ValueError(f'Use normalized train/val/test image directories from prepare_data.py: {split_path}')
        records = manifest['files'][split]
        expected_images = set()
        for item in records:
            image = (base / item['image']).resolve()
            if image.parent != expected_folder:
                raise ValueError(f'Unexpected image location in manifest: {image}')
            expected_images.add(image)
            if not image.is_file() or sha256(image) != item['image_sha256']:
                raise ValueError(f'Image no longer matches manifest: {image}')
            label = base / split / 'labels' / (image.stem + '.txt')
            if item['label']:
                if not label.is_file() or sha256(label) != item['label_sha256']:
                    raise ValueError(f'Label no longer matches manifest: {label}')
            elif label.exists() and label.read_text(encoding='utf-8-sig').strip():
                raise ValueError(f'Previously empty/missing label now has content: {label}')
            verified += 1
        actual = {p.resolve() for p in split_path.iterdir() if p.is_file() and p.suffix.lower() in suffixes}
        if actual != expected_images:
            raise ValueError(f'Partition membership differs from manifest: {split}')
    if not config.get('train') or not config.get('val'):
        raise ValueError('A prepared dataset must include train and val.')
    return {'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
            'data_yaml_sha256': sha256(data_yaml), 'verified_images': verified,
            'mode': manifest['mode'], 'counts_match_manuscript': manifest['counts_match_manuscript']}


def environment():
    setup_runtime()
    import torch
    import ultralytics
    import platform
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'PyYAML', 'Pillow', 'opencv-python',
                 'scipy', 'matplotlib', 'polars', 'onnx', 'onnxruntime', 'onnxruntime-gpu', 'ultralytics-thop'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'ultralytics_version': ultralytics.__version__, 'runtime': str(Path(ultralytics.__file__).resolve()),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
