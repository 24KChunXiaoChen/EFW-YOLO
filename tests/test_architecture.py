"""Behavioral checks for the paper refactor, not claims of metric reproduction.

Run from the project root: python -m unittest discover -s tests -p test_architecture.py -v
"""

import contextlib
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import setup_runtime

setup_runtime()

import torch
import yaml
from ultralytics.nn import tasks
from ultralytics.nn.add_modules.ESPBlock import ESPBlock as OriginalESPBlock
from ultralytics.nn.add_modules.LSKA import LSKA as OriginalLSKA
from ultralytics.nn.add_modules.ShuffleNetV2 import shufflenetv2_05 as original_shufflenet
from ultralytics.nn.tasks import DetectionModel

from efw_yolo import register_modules
from efw_yolo.modules import DSBottleneck, LKAA, PfAAM, SRSM
from efw_yolo.modules.comparators import TripletAttention


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.update(nc=1, scale="n", end2end=False, reg_max=1)
    return config


@contextlib.contextmanager
def original_parser_modules():
    """Use untouched vendor implementations for an independent reference model."""
    tasks.ESPBlock = OriginalESPBlock
    tasks.LSKA = OriginalLSKA
    tasks.shufflenetv2_05 = original_shufflenet
    try:
        yield
    finally:
        register_modules()


def build(config, seed=143):
    torch.manual_seed(seed)
    return DetectionModel(copy.deepcopy(config), ch=3, nc=1, verbose=False).eval()


def assert_tensor_tree_close(case, actual, expected):
    """Include decoded predictions and raw detection outputs, not only shapes."""
    if isinstance(actual, torch.Tensor):
        case.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        case.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            assert_tensor_tree_close(case, actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        case.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_tensor_tree_close(case, a, b)
    else:
        case.assertEqual(actual, expected)


class ArchitectureCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))
        source = ROOT / "vendor/ultralytics/cfg/models/New/yolo26_New.yaml"
        with original_parser_modules():
            cls.original = build(load_config(source))
        cls.cleaned = build(load_config(ROOT / "configs/models/efw_yolo.yaml"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_seeded_initialization_and_state_dict_compatibility(self):
        original_state = self.original.state_dict()
        cleaned_state = self.cleaned.state_dict()
        self.assertEqual(list(original_state), list(cleaned_state))
        for key in original_state:
            with self.subTest(key=key):
                torch.testing.assert_close(cleaned_state[key], original_state[key], rtol=0, atol=0)
        # The source state can be loaded without missing or unexpected keys.
        self.cleaned.load_state_dict(original_state, strict=True)

    def test_full_512_eval_output_matches_source(self):
        generator = torch.Generator().manual_seed(991)
        image = torch.randn(1, 3, 512, 512, generator=generator)
        with torch.inference_mode():
            original_output = self.original(image)
            cleaned_output = self.cleaned(image)
        assert_tensor_tree_close(self, cleaned_output, original_output)
        self.assertEqual(tuple(cleaned_output[0].shape), (1, 5, 5376))

    def test_source_architecture_facts_and_head(self):
        self.assertEqual(sum(p.numel() for p in self.cleaned.parameters()),
                         sum(p.numel() for p in self.original.parameters()))
        backbone = self.cleaned.model[0]
        self.assertEqual(backbone.stage_repeats, [8, 4, 4])
        self.assertEqual(backbone.width_list, [48, 48, 96, 192])
        self.assertEqual(len(backbone.features), 16)
        srsm = [module for module in self.cleaned.modules() if isinstance(module, SRSM)]
        self.assertEqual(len(srsm), 3)
        for module in srsm:
            self.assertEqual(module.c, 32)
            self.assertEqual(module.cv2.conv.out_channels, 64)
            self.assertEqual(len(module.m), 1)
            self.assertFalse(module.m[0].add)
            self.assertEqual(module.m[0].cv1.pointwise.out_channels, 32)
        torch.testing.assert_close(self.cleaned.stride, torch.tensor([8.0, 16.0, 32.0]))
        head = self.cleaned.model[-1]
        self.assertEqual(head.nc, 1)
        self.assertEqual(head.reg_max, 1)
        self.assertFalse(head.end2end)
        self.assertEqual(head.f, [17, 20, 23])

    def test_m2_refactor_matches_existing_source(self):
        source = ROOT / "vendor/ultralytics/cfg/models/New/yolo26_neck精修版目前更加轻量和高效的版本.yaml"
        with original_parser_modules():
            original = build(load_config(source), seed=81)
        cleaned = build(load_config(ROOT / "configs/models/m2_mfmn.yaml"), seed=81)
        image = torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(19))
        with torch.inference_mode():
            assert_tensor_tree_close(self, cleaned(image), original(image))

    def test_all_candidates_have_valid_single_class_outputs(self):
        image = torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(919))
        for name in ("baseline", "m1_shufflenet", "m2_mfmn", "efw_yolo", "attention_lkdsc", "attention_none", "attention_triplet"):
            with self.subTest(config=name):
                model = build(load_config(ROOT / f"configs/models/{name}.yaml"), seed=83)
                with torch.inference_mode():
                    prediction = model(image)[0]
                self.assertEqual(tuple(prediction.shape), (1, 5, 336))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual(model.model[-1].nc, 1)
                self.assertFalse(model.model[-1].end2end)
                torch.testing.assert_close(model.stride, torch.tensor([8.0, 16.0, 32.0]))

    def test_no_attention_refactor_matches_existing_source(self):
        source = ROOT / "vendor/ultralytics/cfg/models/New/yolo26_无注意力机制版本.yaml"
        with original_parser_modules():
            original = build(load_config(source), seed=91)
        cleaned = build(load_config(ROOT / "configs/models/attention_none.yaml"), seed=91)
        image = torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(21))
        with torch.inference_mode():
            assert_tensor_tree_close(self, cleaned(image), original(image))


class AttentionAndGradientTests(unittest.TestCase):
    def test_pfaam_matches_paper_equation_and_has_no_parameters(self):
        attention = PfAAM(8, 8)
        x = torch.linspace(-2, 3, 2 * 8 * 5 * 7).reshape(2, 8, 5, 7).requires_grad_()
        actual = attention(x)
        expected = x * torch.sigmoid(x.mean((2, 3), keepdim=True) * x.mean(1, keepdim=True))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(sum(p.numel() for p in attention.parameters()), 0)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_lkaa_and_srsm_backpropagate_finite_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        try:
            for module in (LKAA(32), SRSM(64, 64), DSBottleneck(32, 32), TripletAttention()):
                with self.subTest(module=type(module).__name__):
                    channels = 64 if isinstance(module, SRSM) else 32
                    x = torch.randn(2, channels, 8, 8, requires_grad=True)
                    module(x).square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        finally:
            torch.set_num_threads(previous_threads)

    def test_unsupported_srsm_repeats_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "n=1"):
            SRSM(64, 64, n=2)


if __name__ == "__main__":
    unittest.main()
