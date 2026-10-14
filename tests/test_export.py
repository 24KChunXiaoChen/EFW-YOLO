"""Real CPU exports and inference; random models do not validate paper accuracy."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import load_model, setup_runtime
from export_onnx import OUTPUT_NAMES, export_verified
from optimize_shuffle import optimize_verified


class DeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_runtime()
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(7)
        cls.temp = tempfile.TemporaryDirectory(prefix="efw-deployment-test-")
        cls.output = Path(cls.temp.name) / "random.onnx"
        cls.report = export_verified(load_model(), cls.output, imgsz=64, samples=2,
                                     provenance={"random_init": True})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_six_branches_match_torch(self):
        self.assertEqual(list(self.report["output_shapes"]), OUTPUT_NAMES)
        for name in OUTPUT_NAMES:
            self.assertEqual(self.report["output_shapes"][name][1], 4 if name.startswith("box") else 1)
        self.assertEqual(self.report["opset"], 18)
        self.assertEqual(self.report["verification"]["provider"], "CPUExecutionProvider")

    def test_shuffle_numerical_equivalence(self):
        result = optimize_verified(self.output, Path(self.temp.name) / "optimized.onnx", samples=3)
        self.assertGreater(result["rewrite"]["recognized"], 0)
        self.assertEqual(result["status"], "passed")

    def test_existing_output_is_preserved(self):
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            optimize_verified(self.output, self.output)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
