"""Regression tests for input identity and CLI guards that prevent unintended runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _common import bind_dataset, setup_runtime
from prepare_data import prepare


class InputIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        for split in ('train', 'val', 'test'):
            (self.source / split / 'images').mkdir(parents=True)
            (self.source / split / 'labels').mkdir()
            (self.source / split / 'images/a.jpg').write_bytes(b'identity-fixture')
            (self.source / split / 'labels/a.txt').write_text('0 0.5 0.5 0.2 0.2\n')
        (self.source / 'data.yaml').write_text('nc: 1\nnames: [fire]\n')
        self.output = self.base / 'smoke'
        prepare(self.source, self.output, mode='smoke', train_count=1, val_count=1)

    def tearDown(self):
        self.temp.cleanup()

    def test_smoke_identity_excludes_test_and_detects_label_changes(self):
        record = bind_dataset(self.output / 'data.yaml')
        self.assertEqual(record['verified_images'], 2)
        self.assertFalse((self.output / 'test').exists())
        (self.output / 'val/labels/a.txt').write_text('0 0.4 0.4 0.2 0.2\n')
        with self.assertRaisesRegex(ValueError, 'Label no longer matches'):
            bind_dataset(self.output / 'data.yaml')

    def test_extra_image_does_not_silently_change_partition(self):
        (self.output / 'train/images/extra.jpg').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'Partition membership'):
            bind_dataset(self.output / 'data.yaml')

    def test_zero_epoch_does_not_start_default_training(self):
        output = self.base / 'must-not-be-created'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
            '--data', str(self.output / 'data.yaml'), '--output', str(output),
            '--phase', 'experiment', '--scratch', '--patience', '0', '--epochs', '0'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())

    def test_evaluation_rejects_rounded_input_size(self):
        output = self.base / 'must-not-be-created'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'),
            '--data', str(self.output / 'data.yaml'), '--weights', 'absent.pt',
            '--output', str(output), '--imgsz', '513'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('positive multiple of 32', result.stderr)
        self.assertFalse(output.exists())


class LocalModelPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_runtime()

    def test_existing_yolov5u_yaml_is_not_replaced_by_alias(self):
        from ultralytics.utils.checks import check_file

        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / 'yolov5nu.yaml'
            alias = Path(directory) / 'yolov5n.yaml'
            explicit.write_text('nc: 1\n', encoding='utf-8')
            alias.write_text('nc: 80\n', encoding='utf-8')
            resolved = check_file(str(explicit), suffix='.yaml', download=False)
            self.assertEqual(Path(resolved), explicit)
            self.assertEqual(Path(resolved).read_text(encoding='utf-8'), 'nc: 1\n')

    def test_missing_yolov5u_yaml_still_falls_back_to_legacy_name(self):
        from ultralytics.utils.checks import check_file

        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / 'yolov5nu.yaml'
            alias = Path(directory) / 'yolov5n.yaml'
            alias.write_text('nc: 1\n', encoding='utf-8')
            resolved = check_file(str(requested), suffix='.yaml', download=False)
            self.assertEqual(Path(resolved), alias)
            self.assertFalse(requested.exists())


if __name__ == '__main__':
    unittest.main()
