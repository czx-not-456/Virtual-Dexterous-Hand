from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MovingAverageFilter:
    channels: list[str]
    window: int = 5
    _buffers: dict[str, deque] = field(init=False)

    def __post_init__(self) -> None:
        self._buffers = {c: deque(maxlen=self.window) for c in self.channels}

    def apply(self, values: dict[str, float]) -> dict[str, float]:
        out = {}
        for c in self.channels:
            self._buffers[c].append(float(values[c]))
            out[c] = float(np.mean(self._buffers[c]))
        return out


@dataclass
class LowPassVectorFilter:
    alpha: float = 0.45
    previous: np.ndarray | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.previous is None:
            self.previous = x.copy()
        else:
            self.previous = self.alpha * x + (1.0 - self.alpha) * self.previous
        return self.previous.copy()


def rate_limit(current: np.ndarray, previous: np.ndarray | None, max_velocity: np.ndarray, dt: float) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    if previous is None:
        return current.copy()
    limit = np.asarray(max_velocity, dtype=float) * float(dt)
    delta = np.clip(current - previous, -limit, limit)
    return previous + delta
