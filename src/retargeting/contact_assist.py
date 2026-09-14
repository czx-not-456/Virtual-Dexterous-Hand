from __future__ import annotations

import math
import numpy as np

from src.hand_model.robot_hand import RobotHandModel


class TaskContactAssist:
    """任务级接触闭环辅助器。

    设计目标：
    - 不替代上游 intent-driven retargeting；
    - 仅在 PINCH / WRAP 对应任务和意图期间，逐帧增加少量屈曲；
    - 某一手指与任务物体接触后停止继续增加该手指的额外闭合量；
    - 离开目标意图后平滑释放额外闭合量。

    这属于 v0.3.2 的工程共享控制层，主要用来把“动作像抓握”推进到
    “真实发生手-物接触”。真实数据手套到位后仍需重新调参。
    """

    FINGER_JOINTS = {
        "thumb": ("thumb_mcp", "thumb_pip", "thumb_dip"),
        "index": ("index_mcp", "index_pip", "index_dip"),
        "middle": ("middle_mcp", "middle_pip", "middle_dip"),
        "ring": ("ring_mcp", "ring_pip", "ring_dip"),
        "little": ("little_mcp", "little_pip", "little_dip"),
    }

    CONTACT_KEYS = {
        "thumb": "thumb_contact",
        "index": "index_contact",
        "middle": "middle_contact",
        "ring": "ring_contact",
        "little": "little_contact",
    }

    PINCH_CONTACT_KEYS = {
        "thumb": "thumb_tip_contact",
        "index": "index_tip_contact",
    }

    def __init__(self, robot: RobotHandModel, cfg: dict | None, dt: float) -> None:
        self.robot = robot
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.dt = float(dt)
        self.release_rate = math.radians(float(self.cfg.get("release_rate_deg_s", 220.0)))
        self.idx = {name: i for i, name in enumerate(robot.joint_order)}
        self.offset = np.zeros(len(robot.joint_order), dtype=float)

    def reset(self) -> None:
        self.offset[:] = 0.0

    def _decay(self) -> None:
        step = self.release_rate * self.dt
        self.offset = np.sign(self.offset) * np.maximum(np.abs(self.offset) - step, 0.0)

    def _task_cfg(self, task_name: str) -> dict | None:
        cfg = self.cfg.get(task_name)
        return dict(cfg) if isinstance(cfg, dict) else None

    def _finger_is_in_contact(self, task_name: str, finger: str, contact: dict) -> bool:
        # PINCH 要求拇指/食指指腹真正接触；若当前 MuJoCo 尚未形成 pad 接触，
        # 一般手指接触仍作为安全停止条件，避免持续挤压模型。
        if task_name == "pinch" and finger in self.PINCH_CONTACT_KEYS:
            if bool(contact.get(self.PINCH_CONTACT_KEYS[finger], False)):
                return True
        return bool(contact.get(self.CONTACT_KEYS[finger], False))

    def apply(
        self,
        q: np.ndarray,
        *,
        task_name: str,
        intent_value: str,
        contact: dict,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not self.enabled:
            return q.copy()

        task_cfg = self._task_cfg(task_name)
        if task_cfg is None or str(task_cfg.get("active_intent", "")) != str(intent_value):
            self._decay()
            return np.clip(q + self.offset, self.robot.ranges[:, 0], self.robot.ranges[:, 1])

        close_step = math.radians(float(task_cfg.get("close_rate_deg_s", 80.0))) * self.dt
        max_extra = dict(task_cfg.get("max_extra_deg", {}))

        active_fingers = ("thumb", "index") if task_name == "pinch" else (
            "thumb", "index", "middle", "ring", "little"
        )

        for finger in active_fingers:
            if self._finger_is_in_contact(task_name, finger, contact):
                continue
            for joint in self.FINGER_JOINTS[finger]:
                if joint not in max_extra:
                    continue
                i = self.idx[joint]
                limit = math.radians(float(max_extra[joint]))
                self.offset[i] = min(self.offset[i] + close_step, limit)

        return np.clip(q + self.offset, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
