from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.features.intent_recognition import Intent
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from .constraints import coupling_error, fingertip_collision_penalty, vector_shape_error
from .topology_mapping import TopologyMapper

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


@dataclass(slots=True)
class RetargetResult:
    q_target: np.ndarray
    q_base: np.ndarray
    objective: float
    success: bool
    iterations: int
    components: dict[str, float]


class IntentDrivenRetargeter:
    """文献驱动的轻量化混合映射器。

    v0.3 在 v0.2 的“末端位置 + 拓扑 + 平滑 + 协同”基础上加入：
    1) 关键指尖相对向量/形状项；
    2) 简化自碰撞安全项；
    3) 仍由 PINCH/WRAP 意图动态调权。

    该实现吸收了 Huang et al. (ICIRA 2023) 与后续 tele-grasping 工作中的
    混合映射思想，但没有复制论文的具体机器人尺寸、向量集合和权重。
    """

    def __init__(self, human: HumanHandModel, robot: RobotHandModel, optimizer_cfg: dict) -> None:
        self.human = human
        self.robot = robot
        self.topology = TopologyMapper(human, robot)
        self.cfg = optimizer_cfg
        self.weights = optimizer_cfg["weights"]
        self.vector_pairs = optimizer_cfg.get("vector_pairs", [])
        self.collision_cfg = optimizer_cfg.get("collision", {})
        h_lengths = [sum(human.kin.finger_lengths[f]) for f in human.kin.finger_lengths]
        r_lengths = [sum(robot.kin.finger_lengths[f]) for f in robot.kin.finger_lengths]
        self.task_scale = float(np.mean(r_lengths) / np.mean(h_lengths))

    def _active_fingers(self, intent: Intent) -> tuple[str, ...]:
        if intent == Intent.PINCH:
            return ("thumb", "index")
        return ("thumb", "index", "middle", "ring", "little")

    def _position_error(self, q_r: np.ndarray, human_tips: dict[str, np.ndarray], intent: Intent) -> float:
        rt = self.robot.fingertips(q_r)
        fingers = self._active_fingers(intent)
        e = 0.0
        for f in fingers:
            target = human_tips[f] * self.task_scale
            e += float(np.sum((rt[f] - target) ** 2))
        return e / len(fingers)

    def objective_components(
        self,
        q: np.ndarray,
        q_base: np.ndarray,
        q_prev: np.ndarray,
        human_tips: dict[str, np.ndarray],
        intent: Intent,
    ) -> dict[str, float]:
        rt = self.robot.fingertips(q)
        e_pos = self._position_error(q, human_tips, intent)
        e_vector = vector_shape_error(rt, human_tips, self.task_scale, self.vector_pairs)
        e_topo = float(np.mean((q - q_base) ** 2))
        e_smooth = float(np.mean((q - q_prev) ** 2))
        e_coupling = coupling_error(q, self.robot)
        e_collision = fingertip_collision_penalty(
            q,
            self.robot,
            intent.value,
            adjacent_tip_min_m=float(self.collision_cfg.get("adjacent_tip_min_m", 0.010)),
            non_target_thumb_tip_min_m=float(self.collision_cfg.get("non_target_thumb_tip_min_m", 0.014)),
            penalty_gain=float(self.collision_cfg.get("penalty_gain", 1.0)),
        )
        return {
            "pos": e_pos,
            "vector": e_vector,
            "topo": e_topo,
            "smooth": e_smooth,
            "coupling": e_coupling,
            "collision": e_collision,
        }

    def solve(
        self,
        q_h: np.ndarray,
        human_tips: dict[str, np.ndarray],
        intent: Intent,
        q_prev: np.ndarray | None = None,
        dynamic: bool = True,
    ) -> RetargetResult:
        q_base = self.topology.map(q_h)
        if q_prev is None:
            q_prev = q_base.copy()
        mode = intent.value if dynamic else Intent.NEUTRAL.value
        w = self.weights[mode]
        objective_intent = intent if dynamic else Intent.NEUTRAL

        def objective(q: np.ndarray) -> float:
            c = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
            return float(sum(float(w.get(name, 0.0)) * value for name, value in c.items()))

        if minimize is None:
            q = np.clip(q_base, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
            comp = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
            return RetargetResult(q, q_base, float(objective(q)), False, 0, comp)

        result = minimize(
            objective,
            x0=np.clip(q_prev, self.robot.ranges[:, 0], self.robot.ranges[:, 1]),
            method="SLSQP",
            bounds=[tuple(x) for x in self.robot.ranges],
            options={"maxiter": int(self.cfg["maxiter"]), "ftol": float(self.cfg["ftol"]), "disp": False},
        )
        q = np.clip(np.asarray(result.x, dtype=float), self.robot.ranges[:, 0], self.robot.ranges[:, 1])
        comp = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
        return RetargetResult(
            q,
            q_base,
            float(result.fun),
            bool(result.success),
            int(getattr(result, "nit", 0)),
            comp,
        )
