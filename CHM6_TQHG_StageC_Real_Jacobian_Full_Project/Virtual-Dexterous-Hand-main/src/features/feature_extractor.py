from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.hand_model.human_hand import HumanHandModel


@dataclass(slots=True)
class HandFeatures:
    joint_angles: np.ndarray
    joint_normalized: np.ndarray
    wrap_score: float
    pinch_distance_m: float
    fingertip_positions: dict[str, np.ndarray]


class FeatureExtractor:
    def __init__(self, human_model: HumanHandModel) -> None:
        self.human = human_model
        self.idx = {name: i for i, name in enumerate(human_model.joint_order)}

    def extract(self, q_h: np.ndarray) -> HandFeatures:
        qn = self.human.normalized_from_angles(q_h)
        non_thumb = [i for name, i in self.idx.items() if not name.startswith("thumb_")]
        wrap_score = float(np.mean(qn[non_thumb]))
        tips = self.human.fingertips(q_h)
        pinch_distance = float(np.linalg.norm(tips["thumb"] - tips["index"]))
        return HandFeatures(q_h.copy(), qn, wrap_score, pinch_distance, tips)
