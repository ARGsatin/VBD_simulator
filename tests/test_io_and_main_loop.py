import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class IoAndMainLoopTests(unittest.TestCase):
    def test_npz_roundtrip_and_report_writer(self):
        from hydrogel_vbd.io.npz_state import load_layer_state, save_layer_state
        from hydrogel_vbd.io.report_writer import write_metrics_csv
        from hydrogel_vbd.state import FieldCommand, LayerResult

        root = ROOT / "outputs" / "test_tmp_io"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            result = LayerResult(
                layer_id=2,
                x_sim=np.array([[0.0, 0.0, 1.0]]),
                v_sim=np.array([[0.0, 0.0, 0.2]]),
                error_metrics={"rms_error": 0.1, "max_error": 0.2},
                field_command_next=FieldCommand(voltage=np.array([1.5]), electrode_ids=["top"]),
                max_deformation=0.2,
                rms_error=0.1,
                success=True,
            )

            state_path = save_layer_state(root / "layer_0002.npz", result)
            loaded = load_layer_state(state_path)

            self.assertEqual(loaded["layer_id"], 2)
            np.testing.assert_allclose(loaded["x_sim"], result.x_sim)
            np.testing.assert_allclose(loaded["voltage"], np.array([1.5]))

            csv_path = write_metrics_csv(root / "metrics.csv", [result])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
                self.assertEqual(len(header), len(set(header)))
                handle.seek(0)
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["layer_id"], "2")
            self.assertEqual(rows[0]["success"], "True")
            self.assertEqual(float(rows[0]["rms_error"]), 0.1)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_gcode_exporter_inserts_field_commands(self):
        from hydrogel_vbd.io.gcode_exporter import insert_field_commands
        from hydrogel_vbd.state import FieldCommand

        source = ";LAYER: 0\nG1 Z0.000\n;LAYER: 1\nG1 Z0.050\n"
        commands = {
            1: FieldCommand(voltage=np.array([2.0, -1.0]), duration=0.8, electrode_ids=["left", "right"])
        }

        output = insert_field_commands(source, commands)

        self.assertIn(";E_FIELD: ELECTRODE=left, VOLTAGE=2.000000", output)
        self.assertIn(";E_FIELD: ELECTRODE=right, VOLTAGE=-1.000000", output)
        self.assertIn(";E_FIELD: OFF", output)

    def test_demo_loop_creates_outputs(self):
        from hydrogel_vbd.main_loop import run_demo

        output_dir = ROOT / "outputs" / "test_demo"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        try:
            results = run_demo(layers=3, output=output_dir)

            self.assertEqual(len(results), 3)
            self.assertTrue((output_dir / "states" / "layer_0000.npz").exists())
            self.assertTrue((output_dir / "vtk" / "layer_0000.vtu").exists())
            self.assertTrue((output_dir / "reports" / "error_metrics.csv").exists())
            self.assertTrue((output_dir / "gcode" / "compensated_print.gcode").exists())

            commands = json.loads((output_dir / "simulation_field_commands.json").read_text(encoding="utf-8"))
            self.assertEqual(len(commands["layers"]), 3)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
