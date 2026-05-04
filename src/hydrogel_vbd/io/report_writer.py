from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from hydrogel_vbd.state import LayerResult


def write_metrics_csv(path: str | Path, results: Iterable[LayerResult]) -> Path:
    rows = list(results)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_names = {"layer_id", "success", "max_deformation", "rms_error"}
    metric_names = sorted({name for result in rows for name in result.error_metrics if name not in fixed_names})
    fieldnames = ["layer_id", "success", "max_deformation", "rms_error", *metric_names]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            row = {
                "layer_id": result.layer_id,
                "success": result.success,
                "max_deformation": result.max_deformation,
                "rms_error": result.rms_error,
            }
            row.update(result.error_metrics)
            writer.writerow(row)
    return output
