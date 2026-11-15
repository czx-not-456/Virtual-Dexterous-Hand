from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


class CSVRunLogger:
    def __init__(self, path: Path, joint_names: list[str], extra_columns: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.joint_names = joint_names
        self.extra_columns = list(extra_columns or [])
        self.writer.writerow([
            "frame", "timestamp", "intent", "pinch_distance_m", "wrap_score",
            "pipeline_latency_ms", "optimizer_ms", "objective", "joint_rmse_to_base_rad",
            "velocity_rms_rad_s", *self.extra_columns, *[f"q_{j}" for j in joint_names]
        ])

    def write(self, row: Iterable) -> None:
        self.writer.writerow(list(row))

    def close(self) -> None:
        self.file.close()
