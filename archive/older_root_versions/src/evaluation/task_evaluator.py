from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import numpy as np


@dataclass
class TaskEpisodeEvaluator:
    """Aggregate per-frame simulation metrics into a reproducible task summary.

    Contact acquisition and stable contact are deliberately reported as separate
    success proxies. Neither metric claims full pick-and-place success because the
    current hand base is fixed and no lift/transport phase is evaluated.
    """

    task_name: str
    sampling_hz: float
    min_success_streak_frames: int = 12
    total_frames: int = 0
    active_frames: int = 0
    contact_frames: int = 0
    stable_frames: int = 0
    contact_success_streak: int = 0
    stable_success_streak: int = 0
    max_contact_success_streak: int = 0
    max_stable_success_streak: int = 0
    first_active_frame: int | None = None
    first_contact_frame: int | None = None
    first_contact_success_frame: int | None = None
    first_stable_success_frame: int | None = None
    contact_success: bool = False
    stable_success: bool = False
    _latency_ms: list[float] = field(default_factory=list)
    _optimizer_ms: list[float] = field(default_factory=list)
    _rmse_rad: list[float] = field(default_factory=list)
    _velocity_rad_s: list[float] = field(default_factory=list)
    _tip_ratio: list[float] = field(default_factory=list)
    _orientation_drift_deg: list[float] = field(default_factory=list)
    _object_displacement_m: list[float] = field(default_factory=list)
    _servo_mean_error_m: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        frame: int,
        task_active: bool,
        contact: dict[str, Any],
        latency_ms: float,
        optimizer_ms: float,
        joint_rmse_rad: float,
        velocity_rms_rad_s: float,
        servo_mean_error_m: float = 0.0,
    ) -> dict[str, int | bool]:
        self.total_frames += 1
        self._latency_ms.append(float(latency_ms))
        self._optimizer_ms.append(float(optimizer_ms))
        self._rmse_rad.append(float(joint_rmse_rad))
        self._velocity_rad_s.append(float(velocity_rms_rad_s))

        if not task_active:
            self.contact_success_streak = 0
            self.stable_success_streak = 0
            return {
                "task_active": False,
                "task_contact_proxy": False,
                "contact_success_proxy": bool(self.contact_success),
                "stable_success_proxy": bool(self.stable_success),
                "contact_success_streak_frames": 0,
                "stable_success_streak_frames": 0,
            }

        self.active_frames += 1
        if self.first_active_frame is None:
            self.first_active_frame = int(frame)

        task_proxy = bool(contact.get("task_contact_proxy", contact.get("stable_contact_proxy", False)))
        stable = bool(contact.get("stable_contact_proxy", False))
        if task_proxy:
            self.contact_frames += 1
            if self.first_contact_frame is None:
                self.first_contact_frame = int(frame)
        if stable:
            self.stable_frames += 1

        if task_proxy:
            self.contact_success_streak += 1
            self.max_contact_success_streak = max(
                self.max_contact_success_streak, self.contact_success_streak
            )
        else:
            self.contact_success_streak = 0
        if stable:
            self.stable_success_streak += 1
            self.max_stable_success_streak = max(
                self.max_stable_success_streak, self.stable_success_streak
            )
        else:
            self.stable_success_streak = 0

        if (
            not self.contact_success
            and self.contact_success_streak >= self.min_success_streak_frames
        ):
            self.contact_success = True
            self.first_contact_success_frame = int(frame)
        if (
            not self.stable_success
            and self.stable_success_streak >= self.min_success_streak_frames
        ):
            self.stable_success = True
            self.first_stable_success_frame = int(frame)

        self._tip_ratio.append(float(contact.get("tip_contact_ratio", 0.0) or 0.0))
        self._orientation_drift_deg.append(float(contact.get("orientation_drift_deg", 0.0) or 0.0))
        self._object_displacement_m.append(float(contact.get("object_displacement_m", 0.0) or 0.0))
        if servo_mean_error_m > 0:
            self._servo_mean_error_m.append(float(servo_mean_error_m))

        return {
            "task_active": True,
            "task_contact_proxy": task_proxy,
            "contact_success_proxy": bool(self.contact_success),
            "stable_success_proxy": bool(self.stable_success),
            "contact_success_streak_frames": int(self.contact_success_streak),
            "stable_success_streak_frames": int(self.stable_success_streak),
        }

    @staticmethod
    def _avg(values: list[float]) -> float:
        return float(mean(values)) if values else 0.0

    @staticmethod
    def _p95(values: list[float]) -> float:
        return float(np.percentile(values, 95)) if values else 0.0

    def _frames_to_ms(self, start: int | None, end: int | None) -> float | None:
        if start is None or end is None:
            return None
        return float((end - start) / self.sampling_hz * 1000.0)

    def summary(self) -> dict[str, Any]:
        active = max(self.active_frames, 1)
        return {
            "task": self.task_name,
            "contact_success_proxy": bool(self.contact_success),
            "stable_success_proxy": bool(self.stable_success),
            "contact_success_definition": (
                f"task_contact_proxy consecutive >= {self.min_success_streak_frames} active frames; "
                "may include configured approach-fixture frames"
            ),
            "stable_success_definition": (
                f"stable_contact_proxy consecutive >= {self.min_success_streak_frames} active frames; "
                "fixed-base simulation, not full lift/transport success"
            ),
            "total_frames": int(self.total_frames),
            "active_frames": int(self.active_frames),
            "contact_frames": int(self.contact_frames),
            "stable_frames": int(self.stable_frames),
            "contact_frame_ratio": float(self.contact_frames / active) if self.active_frames else 0.0,
            "stable_frame_ratio": float(self.stable_frames / active) if self.active_frames else 0.0,
            "max_contact_success_streak_frames": int(self.max_contact_success_streak),
            "max_stable_success_streak_frames": int(self.max_stable_success_streak),
            "first_contact_latency_ms": self._frames_to_ms(self.first_active_frame, self.first_contact_frame),
            "first_contact_success_latency_ms": self._frames_to_ms(
                self.first_active_frame, self.first_contact_success_frame
            ),
            "first_stable_success_latency_ms": self._frames_to_ms(
                self.first_active_frame, self.first_stable_success_frame
            ),
            "mean_pipeline_latency_ms": self._avg(self._latency_ms),
            "p95_pipeline_latency_ms": self._p95(self._latency_ms),
            "mean_optimizer_ms": self._avg(self._optimizer_ms),
            "mean_joint_rmse_rad": self._avg(self._rmse_rad),
            "mean_velocity_rms_rad_s": self._avg(self._velocity_rad_s),
            "mean_tip_contact_ratio": self._avg(self._tip_ratio),
            "max_orientation_drift_deg": max(self._orientation_drift_deg, default=0.0),
            "max_object_displacement_m": max(self._object_displacement_m, default=0.0),
            "mean_servo_error_m": self._avg(self._servo_mean_error_m),
        }
