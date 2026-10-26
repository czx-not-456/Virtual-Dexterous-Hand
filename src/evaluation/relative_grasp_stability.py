"""Post-release relative-pose holding metric (no grasp-force or slip claims).

A valid window comprises uninterrupted *free-body* task contact.  We use the
object pose relative to the palm, not the world/initial object pose. A rolling
window allows a brief settling motion without declaring permanent failure.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class RelativeGraspHold:
    sampling_hz: float = 60.0
    hold_window_s: float = 1.0
    max_translation_m: float = 0.001
    max_rotation_deg: float = 1.0
    _positions: deque = field(default_factory=deque, init=False, repr=False)
    _rotations: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sampling_hz <= 0 or self.hold_window_s <= 0:
            raise ValueError('sampling_hz and hold_window_s must be positive')
        if self.max_translation_m < 0 or self.max_rotation_deg < 0:
            raise ValueError('pose thresholds must be nonnegative')
        # N+1 samples are required to cover at least N * dt seconds.
        self.window_frames = int(math.ceil(self.hold_window_s * self.sampling_hz - 1e-9)) + 1
        self._positions = deque(maxlen=self.window_frames)
        self._rotations = deque(maxlen=self.window_frames)

    def reset(self) -> None:
        self._positions.clear()
        self._rotations.clear()

    def update(self, *, contact: bool, free_object: bool,
               palm_pos: np.ndarray | None, palm_rot: np.ndarray | None,
               object_pos: np.ndarray | None, object_rot: np.ndarray | None) -> dict:
        available = all(x is not None for x in (palm_pos, palm_rot, object_pos, object_rot))
        if not (contact and free_object and available):
            self.reset()
            return self._status(False, free_object, available, 0.0, 0.0)
        p_p = np.asarray(palm_pos, dtype=float).reshape(3)
        R_p = np.asarray(palm_rot, dtype=float).reshape(3, 3)
        p_o = np.asarray(object_pos, dtype=float).reshape(3)
        R_o = np.asarray(object_rot, dtype=float).reshape(3, 3)
        if not all(np.all(np.isfinite(x)) for x in (p_p, R_p, p_o, R_o)):
            self.reset()
            return self._status(False, free_object, False, 0.0, 0.0)
        self._positions.append(R_p.T @ (p_o - p_p))
        self._rotations.append(R_p.T @ R_o)
        positions = np.stack(tuple(self._positions))
        rotations = np.stack(tuple(self._rotations))
        # Maximum pairwise span over the whole window (not a scalar norm range).
        d = positions[:, None, :] - positions[None, :, :]
        translation_span = float(np.sqrt(np.max(np.sum(d*d, axis=-1))))
        traces = np.einsum('nij,mij->nm', rotations, rotations)
        angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        rotation_span = float(np.degrees(np.max(angles)))
        window_ready = len(self._positions) >= self.window_frames
        stable = bool(window_ready and translation_span <= self.max_translation_m
                      and rotation_span <= self.max_rotation_deg)
        return self._status(stable, free_object, True, translation_span, rotation_span)

    def _status(self, stable: bool, free_object: bool, available: bool,
                translation: float, rotation: float) -> dict:
        return {
            'hold_stable_proxy': bool(stable),
            'hold_object_free': bool(free_object),
            'hold_pose_available': bool(available),
            'hold_window_samples': len(self._positions),
            'hold_window_required_samples': self.window_frames,
            'hold_relative_translation_span_m': float(translation),
            'hold_relative_rotation_span_deg': float(rotation),
        }
