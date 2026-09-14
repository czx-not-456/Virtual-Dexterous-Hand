from __future__ import annotations

import numpy as np

from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel


class TopologyMapper:
    """基于关节拓扑对应与活动范围比例的基础映射 q_base。"""

    def __init__(self, human: HumanHandModel, robot: RobotHandModel) -> None:
        if human.joint_order != robot.joint_order:
            raise ValueError("Current prototype requires the same topological joint names for human and robot models")
        self.human = human
        self.robot = robot

    def map(self, q_h: np.ndarray) -> np.ndarray:
        u = self.human.normalized_from_angles(q_h)
        return self.robot.ranges[:, 0] + u * (self.robot.ranges[:, 1] - self.robot.ranges[:, 0])
