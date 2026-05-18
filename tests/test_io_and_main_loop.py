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
        from hydrogel_vbd.core.state import FieldCommand, LayerResult

        root = ROOT / "outputs" / "test_tmp_io"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            result = LayerResult(
                layer_id=2,
                x_sim=np.array([[0.0, 0.0, 1.0]]),
                v_sim=np.array([[0.0, 0.0, 0.2]]),
                error_metrics={
                    "rms_error": 0.1,
                    "max_error": 0.2,
                    "field_control_effective": "scalar_pid",
                },
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
            self.assertEqual(
                loaded["error_metrics"]["field_control_effective"],
                "scalar_pid",
            )

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
        from hydrogel_vbd.core.state import FieldCommand

        source = ";LAYER: 0\nG1 Z0.000\n;LAYER: 1\nG1 Z0.050\n"
        commands = {
            1: FieldCommand(voltage=np.array([2.0, -1.0]), duration=0.8, electrode_ids=["left", "right"])
        }

        output = insert_field_commands(source, commands)

        self.assertIn(";E_FIELD: ELECTRODE=left, VOLTAGE=2.000000", output)
        self.assertIn(";E_FIELD: ELECTRODE=right, VOLTAGE=-1.000000", output)
        self.assertIn(";E_FIELD: OFF", output)

    def test_demo_loop_creates_outputs(self):
        from hydrogel_vbd.core.main_loop import run_demo

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

    def test_rms_guard_selects_candidate_with_one_percent_tolerance(self):
        from hydrogel_vbd.core.main_loop import _select_rms_guarded_result
        from hydrogel_vbd.core.state import FieldCommand, LayerResult

        def result(name: str, rms: float) -> LayerResult:
            return LayerResult(
                layer_id=0,
                x_sim=np.zeros((1, 3)),
                v_sim=np.zeros((1, 3)),
                error_metrics={"field_control_effective": name},
                field_command_next=FieldCommand(voltage=np.array([0.0])),
                max_deformation=rms,
                rms_error=rms,
                success=True,
            )

        baseline = result("scalar_pid", 1.0)
        passing_candidate = result("bottom_z", 1.009)
        selected, passed = _select_rms_guarded_result(
            baseline, passing_candidate, tolerance=0.01
        )
        self.assertTrue(passed)
        self.assertIs(selected, passing_candidate)

        failing_candidate = result("bottom_z", 1.011)
        selected, passed = _select_rms_guarded_result(
            baseline, failing_candidate, tolerance=0.01
        )
        self.assertFalse(passed)
        self.assertIs(selected, baseline)

    def test_demo_loop_bottom_z_guarded_never_exceeds_scalar_pid_rms_guard(self):
        from hydrogel_vbd.core.main_loop import run_demo

        baseline_dir = ROOT / "outputs" / "test_demo_scalar_pid"
        guarded_dir = ROOT / "outputs" / "test_demo_bottom_z_guarded"
        for path in (baseline_dir, guarded_dir):
            if path.exists():
                shutil.rmtree(path)

        try:
            baseline = run_demo(
                layers=2,
                output=baseline_dir,
                field_control_mode="scalar_pid",
            )
            guarded = run_demo(
                layers=2,
                output=guarded_dir,
                field_control_mode="bottom_z_guarded",
            )

            self.assertEqual(len(guarded), len(baseline))
            for baseline_result, guarded_result in zip(baseline, guarded):
                self.assertLessEqual(
                    guarded_result.rms_error,
                    baseline_result.rms_error * 1.01 + 1e-12,
                )
                self.assertIn(
                    guarded_result.error_metrics["field_control_effective"],
                    {"scalar_pid", "bottom_z"},
                )
                self.assertIn("rms_guard_passed", guarded_result.error_metrics)

            with (guarded_dir / "reports" / "error_metrics.csv").open(newline="", encoding="utf-8") as handle:
                fieldnames = csv.DictReader(handle).fieldnames
            for name in [
                "field_control_requested",
                "field_control_effective",
                "rms_guard_passed",
                "rms_guard_baseline",
                "rms_guard_candidate",
                "bottom_z_mean_error",
                "bottom_z_max_error",
                "bottom_z_E_z",
            ]:
                self.assertIn(name, fieldnames)
        finally:
            for path in (baseline_dir, guarded_dir):
                if path.exists():
                    shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
