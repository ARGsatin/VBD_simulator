import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PackageAndConfigTests(unittest.TestCase):
    def test_package_imports_version(self):
        import hydrogel_vbd

        self.assertIsInstance(hydrogel_vbd.__version__, str)
        self.assertTrue(hydrogel_vbd.__version__)

    def test_config_templates_have_required_keys(self):
        expected = {
            "material_hydrogel.json": {"density", "young_modulus_min", "young_modulus_max", "poisson_ratio", "damping"},
            "printer_dlp.json": {"layer_height", "exposure_time", "dt", "num_layers"},
            "electrode_config.json": {"electrode_ids", "voltage_min", "voltage_max", "regularization"},
            "solver_vbd.json": {"substeps", "iterations", "damping", "backend"},
        }

        for filename, keys in expected.items():
            path = ROOT / "configs" / filename
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertLessEqual(keys, set(data), f"{filename} missing {keys - set(data)}")


if __name__ == "__main__":
    unittest.main()
