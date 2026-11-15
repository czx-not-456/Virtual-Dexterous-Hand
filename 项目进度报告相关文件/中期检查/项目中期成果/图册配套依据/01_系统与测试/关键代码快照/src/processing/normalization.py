from __future__ import annotations

import numpy as np


def normalized_to_angle(value: float, angle_min: float, angle_max: float) -> float:
    return float(angle_min + np.clip(value, 0.0, 1.0) * (angle_max - angle_min))
