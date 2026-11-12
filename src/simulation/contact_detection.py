"""Calibratable force-backed contact evidence and debounced controller state.

The filtered estimate is for *control only*: an exit grace period is not
physical contact and MUST NOT be used as proof of free-body grasp success.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math


@dataclass(frozen=True)
class ContactDetectionConfig:
    mode: str = 'force_and_temporal'
    enter_force_n: float = 0.01
    exit_force_n: float = 0.003
    enter_frames: int = 2
    exit_frames: int = 3

    def __post_init__(self) -> None:
        if self.mode not in ('geometry_only', 'force_only', 'temporal_only', 'force_and_temporal'):
            raise ValueError('Unknown contact detector mode')
        if (not math.isfinite(self.enter_force_n) or not math.isfinite(self.exit_force_n)
                or self.enter_force_n <= 0 or self.exit_force_n <= 0
                or self.exit_force_n > self.enter_force_n):
            raise ValueError('Require 0 < exit_force_n <= enter_force_n')
        if self.enter_frames < 1 or self.exit_frames < 1:
            raise ValueError('Contact dwell frames must be >= 1')


class ForceContactDebouncer:
    """Per-pad/finger Schmitt trigger with enter/exit dwell; pure-Python testable."""
    def __init__(self, cfg: ContactDetectionConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._active: dict[str, bool] = {}
        self._on: dict[str, int] = {}
        self._off: dict[str, int] = {}

    def update(self, normal_forces_n: Mapping[str, float]) -> dict[str, bool]:
        keys = set(normal_forces_n) | set(self._active)
        result = {}
        for key in keys:
            value = float(normal_forces_n.get(key, 0.0))
            if not math.isfinite(value) or value < 0:
                value = 0.0
            old = self._active.get(key, False)
            if old:
                # Keep the controller engaged across a *short* force dropout,
                # but not as a reported physical-contact success.
                off = self._off.get(key, 0) + 1 if value < self.cfg.exit_force_n else 0
                new = off < self.cfg.exit_frames
                self._off[key] = off if new else 0
                self._on[key] = 0
            else:
                on = self._on.get(key, 0) + 1 if value >= self.cfg.enter_force_n else 0
                new = on >= self.cfg.enter_frames
                self._on[key] = 0 if new else on
                self._off[key] = 0
            self._active[key] = new
            result[key] = new
        return result
