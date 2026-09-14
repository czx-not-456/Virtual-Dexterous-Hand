from __future__ import annotations

import numpy as np


def joint_rmse(q: np.ndarray, q_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(q) - np.asarray(q_ref)) ** 2)))


def fingertip_rmse(robot_tips: dict[str, np.ndarray], target_tips: dict[str, np.ndarray], scale: float = 1.0) -> float:
    errs = []
    for f in target_tips:
        errs.append(np.sum((robot_tips[f] - target_tips[f] * scale) ** 2))
    return float(np.sqrt(np.mean(errs)))


def velocity_rms(q: np.ndarray, q_prev: np.ndarray | None, dt: float) -> float:
    if q_prev is None:
        return 0.0
    v = (np.asarray(q) - np.asarray(q_prev)) / dt
    return float(np.sqrt(np.mean(v * v)))
