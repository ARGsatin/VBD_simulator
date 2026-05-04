from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrogel_vbd.state import LayerResult


def save_layer_state(path: str | Path, result: LayerResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        layer_id=np.array(result.layer_id, dtype=int),
        x_sim=result.x_sim,
        v_sim=result.v_sim,
        voltage=result.field_command_next.voltage,
        max_deformation=np.array(result.max_deformation, dtype=float),
        rms_error=np.array(result.rms_error, dtype=float),
        success=np.array(result.success, dtype=bool),
        metric_keys=np.array(list(result.error_metrics.keys())),
        metric_values=np.array(list(result.error_metrics.values()), dtype=float),
    )
    return output


def load_layer_state(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        metric_keys = [str(item) for item in data["metric_keys"]]
        metric_values = [float(item) for item in data["metric_values"]]
        return {
            "layer_id": int(data["layer_id"]),
            "x_sim": data["x_sim"].copy(),
            "v_sim": data["v_sim"].copy(),
            "voltage": data["voltage"].copy(),
            "max_deformation": float(data["max_deformation"]),
            "rms_error": float(data["rms_error"]),
            "success": bool(data["success"]),
            "error_metrics": dict(zip(metric_keys, metric_values)),
        }
