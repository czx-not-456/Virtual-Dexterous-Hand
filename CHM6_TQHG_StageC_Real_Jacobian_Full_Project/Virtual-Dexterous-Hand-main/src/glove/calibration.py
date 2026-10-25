from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .driver import GloveSample


@dataclass
class AdaptiveCalibrator:
    channels: list[str]
    epsilon: float = 1.0
    min_values: dict[str, float] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.min_values:
            self.min_values = {c: float("inf") for c in self.channels}
        if not self.max_values:
            self.max_values = {c: float("-inf") for c in self.channels}

    def update(self, sample: GloveSample) -> None:
        for c in self.channels:
            x = float(sample.channels[c])
            self.min_values[c] = min(self.min_values[c], x)
            self.max_values[c] = max(self.max_values[c], x)

    def fit(self, samples: list[GloveSample]) -> None:
        for s in samples:
            self.update(s)
        self._validate()

    def _validate(self) -> None:
        bad = [c for c in self.channels if self.max_values[c] - self.min_values[c] <= self.epsilon]
        if bad:
            raise ValueError(f"Calibration span too small for channels: {bad}")

    def normalize(self, sample: GloveSample) -> dict[str, float]:
        self._validate()
        out: dict[str, float] = {}
        for c in self.channels:
            mn = self.min_values[c]
            mx = self.max_values[c]
            out[c] = float(np.clip((float(sample.channels[c]) - mn) / (mx - mn), 0.0, 1.0))
        return out
