from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.hand_model.robot_hand import RobotHandModel


class TaskSpaceSimulator(Protocol):
    def joint_positions(self) -> np.ndarray: ...
    def task_contact_target(self, finger: str, task_name: str, clearance_m: float, vertical_margin_m: float) -> np.ndarray: ...
    def fingertip_position(self, finger: str) -> np.ndarray: ...
    def fingertip_jacobian(self, finger: str, joint_names: tuple[str, ...]) -> np.ndarray: ...


@dataclass(slots=True)
class ServoDiagnostics:
    active: bool = False
    thumb_error_m: float = 0.0
    index_error_m: float = 0.0
    mean_error_m: float = 0.0
    max_error_m: float = 0.0
    corrected_digits: int = 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "task_servo_active": bool(self.active),
            "servo_thumb_error_m": float(self.thumb_error_m),
            "servo_index_error_m": float(self.index_error_m),
            "servo_mean_error_m": float(self.mean_error_m),
            "servo_max_error_m": float(self.max_error_m),
            "servo_corrected_digits": int(self.corrected_digits),
        }


class ObjectAwareTaskServo:
    """基于当前物体位姿与 MuJoCo 指尖 Jacobian 的轻量任务空间伺服。

    v0.3.4 的目的不是替换上游 retargeting，而是在 PINCH/WRAP 进入任务阶段后，
    将“固定角度闭合”升级为“指尖朝物体表面收敛”。

    - 拇指保留 3 DoF DLS 修正；
    - 非拇指按单腱欠驱动方向做 1 DoF 有效修正；
    - 已与物体发生接触的手指保持当前姿态，避免继续穿透/挤压；
    - 每帧修正受 max_joint_step_deg 限制，最终仍受关节限位保护。
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

    TIP_CONTACT_KEYS = {
        "thumb": "thumb_tip_contact",
        "index": "index_tip_contact",
        "middle": "middle_tip_contact",
        "ring": "ring_tip_contact",
        "little": "little_tip_contact",
    }

    def __init__(self, robot: RobotHandModel, cfg: dict | None) -> None:
        self.robot = robot
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.gain = float(self.cfg.get("gain", 0.70))
        self.damping = float(self.cfg.get("damping", 0.020))
        self.blend = float(self.cfg.get("blend", 0.80))
        self.max_joint_step = math.radians(float(self.cfg.get("max_joint_step_deg", 4.0)))
        self.stop_error_m = float(self.cfg.get("stop_error_m", 0.0045))
        self.idx = {name: i for i, name in enumerate(robot.joint_order)}

    def reset(self) -> None:
        # 预留状态接口；当前 servo 无积分状态。
        return None

    def _task_cfg(self, task_name: str) -> dict | None:
        cfg = self.cfg.get(task_name)
        return dict(cfg) if isinstance(cfg, dict) else None

    def _active_fingers(self, task_name: str) -> tuple[str, ...]:
        if task_name == "pinch":
            return ("thumb", "index")
        if task_name in {"wrap", "sphere"}:
            return ("thumb", "index", "middle", "ring", "little")
        return ()

    def _finger_contacted(self, finger: str, contact: dict) -> bool:
        # 指尖接触优先；任一该手指接触也作为安全停止条件。
        return bool(contact.get(self.TIP_CONTACT_KEYS[finger], False)) or bool(
            contact.get(self.CONTACT_KEYS[finger], False)
        )

    def _thumb_delta(self, J: np.ndarray, error: np.ndarray) -> np.ndarray:
        """3x3 DLS：dq = J^T (J J^T + lambda^2 I)^-1 e。"""
        J = np.asarray(J, dtype=float)
        error = np.asarray(error, dtype=float)
        A = J @ J.T + (self.damping**2) * np.eye(3)
        try:
            dq = J.T @ np.linalg.solve(A, self.gain * error)
        except np.linalg.LinAlgError:
            dq = J.T @ np.linalg.pinv(A) @ (self.gain * error)
        return np.clip(dq, -self.max_joint_step, self.max_joint_step)

    def _underactuated_delta(self, finger: str, J: np.ndarray, error: np.ndarray) -> np.ndarray:
        """沿单腱协同方向进行 1D Jacobian 修正。"""
        c_cfg = self.robot.coupling[finger]
        direction = np.array(
            [1.0, float(c_cfg["pip_over_mcp"]), float(c_cfg["dip_over_mcp"])],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        J_eff = np.asarray(J, dtype=float) @ direction
        denom = float(J_eff @ J_eff + self.damping**2)
        du = float(self.gain * (J_eff @ np.asarray(error, dtype=float)) / denom)
        du = float(np.clip(du, -self.max_joint_step, self.max_joint_step))
        return direction * du

    def apply(
        self,
        q_target: np.ndarray,
        *,
        simulator: TaskSpaceSimulator,
        task_name: str,
        intent_value: str,
        contact: dict,
    ) -> tuple[np.ndarray, ServoDiagnostics]:
        q_target = np.asarray(q_target, dtype=float)
        diag = ServoDiagnostics()

        if not self.enabled:
            return q_target.copy(), diag

        task_cfg = self._task_cfg(task_name)
        if task_cfg is None:
            return q_target.copy(), diag

        active_intent = str(task_cfg.get("active_intent", ""))
        if str(intent_value) != active_intent:
            return q_target.copy(), diag

        clearance = float(task_cfg.get("clearance_m", 0.0070))
        vertical_margin = float(task_cfg.get("vertical_margin_m", 0.010))
        q_actual = np.asarray(simulator.joint_positions(), dtype=float)
        q_cmd = q_target.copy()
        errors: list[float] = []
        diag.active = True

        for finger in self._active_fingers(task_name):
            joints = self.FINGER_JOINTS[finger]
            indices = np.array([self.idx[j] for j in joints], dtype=int)

            current = np.asarray(simulator.fingertip_position(finger), dtype=float)
            target = np.asarray(
                simulator.task_contact_target(
                    finger,
                    task_name,
                    clearance_m=clearance,
                    vertical_margin_m=vertical_margin,
                ),
                dtype=float,
            )
            error = target - current
            err = float(np.linalg.norm(error))
            errors.append(err)

            if finger == "thumb":
                diag.thumb_error_m = err
            elif finger == "index":
                diag.index_error_m = err

            if self._finger_contacted(finger, contact):
                # 接触后保持实际姿态，避免上游 target 把手指重新拉离物体。
                q_cmd[indices] = q_actual[indices]
                continue

            if err <= self.stop_error_m:
                q_cmd[indices] = self.blend * q_actual[indices] + (1.0 - self.blend) * q_target[indices]
                continue

            J = simulator.fingertip_jacobian(finger, joints)
            if finger == "thumb":
                dq = self._thumb_delta(J, error)
            else:
                dq = self._underactuated_delta(finger, J, error)

            desired = q_actual[indices] + dq
            q_cmd[indices] = self.blend * desired + (1.0 - self.blend) * q_target[indices]
            diag.corrected_digits += 1

        q_cmd = np.clip(q_cmd, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
        if errors:
            diag.mean_error_m = float(np.mean(errors))
            diag.max_error_m = float(np.max(errors))
        return q_cmd, diag
