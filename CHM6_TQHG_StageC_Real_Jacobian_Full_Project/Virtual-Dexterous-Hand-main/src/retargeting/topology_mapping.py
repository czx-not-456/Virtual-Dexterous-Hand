from __future__ import annotations
import numpy as np
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel

class TopologyMapper:
    """Map normalized 15-channel human semantics into CH-M6 native joints."""
    def __init__(self, human: HumanHandModel, robot: RobotHandModel) -> None:
        self.human = human
        self.robot = robot

    def map(self, q_h: np.ndarray) -> np.ndarray:
        return self.robot.map_human_normalized(self.human.normalized_from_angles(q_h), self.human.joint_order)