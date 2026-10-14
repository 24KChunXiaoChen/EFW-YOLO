"""Check source-equivalent square math and the C2PSAMLLA runtime repairs."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import setup_runtime
setup_runtime()

import torch
import yaml
from ultralytics.nn.add_modules.MLLA import C2PSAMLLA as OriginalC2PSAMLLA
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from efw_yolo.modules.mlla import C2PSAMLLA, MLLAttention


class C2PSAMLLATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_supported_original_square_outputs_and_gradients_are_exact(self):
        for dtype in (torch.float32, torch.float64):
            for training in (False, True):
                with self.subTest(dtype=dtype, training=training):
                    torch.manual_seed(743)
                    original = OriginalC2PSAMLLA(256, 256).to(dtype=dtype).train(training)
                    torch.manual_seed(743)
                    cleaned = C2PSAMLLA(256, 256).to(dtype=dtype).train(training)
                    self.assertEqual(list(original.state_dict()), list(cleaned.state_dict()))
                    for name, value in original.state_dict().items():
                        torch.testing.assert_close(cleaned.state_dict()[name], value, rtol=0, atol=0)
                    cleaned.load_state_dict(original.state_dict(), strict=True)
                    x_original = torch.randn(2, 256, 16, 16, dtype=dtype, requires_grad=True)
                    x_cleaned = x_original.detach().clone().requires_grad_()
                    expected, actual = original(x_original), cleaned(x_cleaned)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    weight = torch.randn_like(expected)
                    (expected.square().mean() + (expected * weight).mean()).backward()
                    (actual.square().mean() + (actual * weight).mean()).backward()
                    torch.testing.assert_close(x_cleaned.grad, x_original.grad, rtol=0, atol=0)
                    original_params = dict(original.named_parameters())
                    for name, parameter in cleaned.named_parameters():
                        self.assertIsNotNone(parameter.grad)
                        torch.testing.assert_close(parameter.grad, original_params[name].grad, rtol=0, atol=0)
                    for name, value in original.state_dict().items():
                        torch.testing.assert_close(cleaned.state_dict()[name], value, rtol=0, atol=0)

    def test_nano_square_and_rectangular_outputs_and_gradients_are_finite(self):
        for height, width in ((16, 16), (16, 12), (12, 16), (8, 24)):
            with self.subTest(height=height, width=width):
                module = C2PSAMLLA(128, 128).train()
                self.assertEqual(module.m[0].attn.num_heads, 1)
                x = torch.randn(2, 128, height, width, requires_grad=True)
                result = module(x)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_heads_and_feature_dimensions_fail_early(self):
        for dim, heads in ((32, 0), (32, 3), (31, 1)):
            with self.subTest(dim=dim, heads=heads), self.assertRaises(ValueError):
                MLLAttention(dim, num_heads=heads)

    def test_registered_configuration_supports_square_rectangular_and_loss_backward(self):
        cfg = yaml.safe_load((ROOT / "configs/models/attention_mlla.yaml").read_text(encoding="utf-8"))
        model = DetectionModel(copy.deepcopy(cfg), verbose=False).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 386447)
        self.assertEqual(model.model[-1].nc, 1)
        self.assertFalse(model.model[-1].end2end)
        torch.testing.assert_close(model.stride, torch.tensor([8.0, 16.0, 32.0]))
        with torch.inference_mode():
            for height, width in ((512, 512), (512, 384), (384, 512)):
                with self.subTest(height=height, width=width):
                    prediction = model(torch.randn(1, 3, height, width))[0]
                    anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
                    self.assertEqual(list(prediction.shape), [1, 5, anchors])
                    self.assertTrue(torch.isfinite(prediction).all())
        model.train()
        model.args = SimpleNamespace(**copy.deepcopy(DEFAULT_CFG_DICT))
        batch = {"img": torch.randn(2, 3, 128, 96), "batch_idx": torch.tensor([0, 1]),
                 "cls": torch.zeros(2, 1),
                 "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.25], [0.4, 0.4, 0.3, 0.2]])}
        loss, _ = model.loss(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
