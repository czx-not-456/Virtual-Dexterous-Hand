from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time

from .driver import GloveSample


@dataclass(slots=True)
class DatasetRow:
    trial_id: int
    frame: int
    gesture: str
    timestamp_s: float
    channels: dict[str, float]
    palm_rpy: tuple[float, float, float]


class CSVGloveDatasetDriver:
    """Replay a reproducible synthetic glove CSV as if it were a live glove.

    Expected columns:
      trial_id, frame, gesture, timestamp_s,
      palm_roll, palm_pitch, palm_yaw, <all glove channels>

    The driver intentionally exposes the same ``sample`` interface as the live/mock
    driver, so the rest of the pipeline does not need to know whether input came
    from a real sensor or an offline synthetic dataset.
    """

    def __init__(
        self,
        path: str | Path,
        channels: list[str],
        *,
        sampling_hz: float = 60.0,
        trial_id: int | None = None,
        loop: bool = True,
    ) -> None:
        self.path = Path(path)
        self.channels = list(channels)
        self.sampling_hz = float(sampling_hz)
        self.dt = 1.0 / self.sampling_hz
        self.loop = bool(loop)
        self._rows_all = self._read_rows(self.path)
        if trial_id is None:
            self.rows = list(self._rows_all)
        else:
            self.rows = [r for r in self._rows_all if r.trial_id == int(trial_id)]
            if not self.rows:
                available = sorted({r.trial_id for r in self._rows_all})
                raise ValueError(f"dataset trial_id={trial_id} 不存在；可用 trial_id={available}")
        if not self.rows:
            raise ValueError(f"synthetic dataset is empty: {self.path}")
        self.index = 0

    def _read_rows(self, path: Path) -> list[DatasetRow]:
        if not path.is_file():
            raise FileNotFoundError(f"模拟数据集不存在：{path}")
        rows: list[DatasetRow] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"dataset has no header: {path}")
            missing = [c for c in self.channels if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"dataset 缺少手套通道：{missing}")
            for raw in reader:
                channels = {c: float(raw[c]) for c in self.channels}
                rows.append(
                    DatasetRow(
                        trial_id=int(raw.get("trial_id", 0) or 0),
                        frame=int(raw.get("frame", len(rows)) or len(rows)),
                        gesture=str(raw.get("gesture", "UNKNOWN") or "UNKNOWN"),
                        timestamp_s=float(raw.get("timestamp_s", 0.0) or 0.0),
                        channels=channels,
                        palm_rpy=(
                            float(raw.get("palm_roll", 0.0) or 0.0),
                            float(raw.get("palm_pitch", 0.0) or 0.0),
                            float(raw.get("palm_yaw", 0.0) or 0.0),
                        ),
                    )
                )
        return rows

    @property
    def available_trials(self) -> list[int]:
        return sorted({r.trial_id for r in self._rows_all})

    @property
    def current_gesture(self) -> str:
        if not self.rows:
            return "UNKNOWN"
        i = min(self.index, len(self.rows) - 1)
        return self.rows[i].gesture

    def calibration_samples(self) -> list[GloveSample]:
        """Use the replay trial itself for per-trial min/max calibration.

        Every generated trial contains OPEN, NEUTRAL, PINCH and WRAP/FIST-like
        regions, therefore it spans the expected flexion range. This models the
        project's offline calibration stage without requiring real hardware.
        """
        return [GloveSample(r.timestamp_s, r.channels, r.palm_rpy) for r in self.rows]

    def reset(self) -> None:
        self.index = 0

    def sample(self) -> GloveSample:
        if self.index >= len(self.rows):
            if not self.loop:
                raise StopIteration("synthetic dataset replay finished")
            self.index = 0
        row = self.rows[self.index]
        self.index += 1
        # Use wall-clock timestamp for pipeline latency logging, while the synthetic
        # dataset keeps its own timestamp_s column for reproducibility/debugging.
        return GloveSample(time.perf_counter(), row.channels, row.palm_rpy)
