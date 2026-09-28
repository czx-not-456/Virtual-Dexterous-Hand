from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class HumanHandFrame:
    """Source-independent hand frame consumed by the retargeting pipeline."""

    timestamp: float
    joint_order: tuple[str, ...]
    joint_angles_rad: np.ndarray
    palm_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source: str = "unknown"
    robot_joint_targets_rad: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_angles_rad = np.asarray(self.joint_angles_rad, dtype=float)
        if self.joint_angles_rad.shape != (len(self.joint_order),):
            raise ValueError(
                f"HumanHandFrame expected {len(self.joint_order)} joint angles, "
                f"got {self.joint_angles_rad.shape}"
            )
        if self.robot_joint_targets_rad is not None:
            self.robot_joint_targets_rad = np.asarray(self.robot_joint_targets_rad, dtype=float)

    @property
    def joint_angles(self) -> dict[str, float]:
        return dict(zip(self.joint_order, self.joint_angles_rad.tolist()))
