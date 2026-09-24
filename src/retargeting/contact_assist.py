from __future__ import annotations
import math
import numpy as np
from src.hand_model.robot_hand import RobotHandModel

class TaskContactAssist:
    """Config-driven fallback closure controller for CH-M6."""
    def __init__(self, robot: RobotHandModel, cfg: dict | None, dt: float) -> None:
        self.robot, self.cfg, self.dt = robot, dict(cfg or {}), float(dt)
        self.enabled = bool(self.cfg.get("enabled", False))
        self.release_rate = math.radians(float(self.cfg.get("release_rate_deg_s", 220.0)))
        self.idx = {name: i for i, name in enumerate(robot.joint_order)}
        self.offset = np.zeros(len(robot.joint_order), dtype=float)
    def reset(self) -> None: self.offset[:] = 0.0
    def _decay(self) -> None:
        step = self.release_rate * self.dt
        self.offset = np.sign(self.offset) * np.maximum(np.abs(self.offset)-step, 0.0)
    def apply(self, q: np.ndarray, *, task_name: str, intent_value: str, contact: dict) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        task = self.cfg.get(task_name)
        if not self.enabled or not isinstance(task, dict) or str(task.get("active_intent")) != str(intent_value):
            self._decay(); return np.clip(q+self.offset, self.robot.ranges[:,0], self.robot.ranges[:,1])
        step = math.radians(float(task.get("close_rate_deg_s", 80.0))) * self.dt
        limits = dict(task.get("max_extra_deg", {}))
        active = ("thumb", "index") if task_name == "pinch" else tuple(self.robot.finger_joints)
        for finger in active:
            tip = bool(contact.get(f"{finger}_tip_contact", False))
            any_contact = bool(contact.get(f"{finger}_contact", False))
            if tip or any_contact: continue
            for joint in self.robot.joints_for_finger(finger):
                if joint in limits:
                    i=self.idx[joint]; self.offset[i]=min(self.offset[i]+step, math.radians(float(limits[joint])))
        return np.clip(q+self.offset, self.robot.ranges[:,0], self.robot.ranges[:,1])