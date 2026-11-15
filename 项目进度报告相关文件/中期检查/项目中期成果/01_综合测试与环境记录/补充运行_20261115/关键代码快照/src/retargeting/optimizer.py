from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.features.intent_recognition import Intent
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from .constraints import coupling_error, fingertip_collision_penalty, vector_shape_error
from .topology_mapping import TopologyMapper
from .latent_synergy import LatentSynergyMapper
from .adaptive_synergy import TaskAdaptiveSynergyMapper
from .reachability_synergy import ReachabilityGatedSynergyMapper, ReachabilityDiagnostics
from .credible_contact_synergy import CredibleContactDiagnostics
from .task_effect_spectrum import TaskEffectDiagnostics

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

    def __init__(self, human: HumanHandModel, robot: RobotHandModel, optimizer_cfg: dict,
                 latent_mapper: LatentSynergyMapper | None = None,
                 adaptive_mapper: TaskAdaptiveSynergyMapper | None = None,
                 adaptive_source: str = "sensor") -> None:
        self.human = human
        self.robot = robot
        self.latent_mapper = latent_mapper
        self.adaptive_mapper = adaptive_mapper
        self.adaptive_source = adaptive_source
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
        q_base: np.ndarray | None = None,
        sensors_deg: np.ndarray | None = None,
        controls_deg: np.ndarray | None = None,
    ) -> RetargetResult:
        if q_base is None:
            q_base = self.topology.map(q_h)
        else:
            q_base = np.asarray(q_base, dtype=float)
            if q_base.shape != (len(self.robot.joint_order),):
                raise ValueError(
                    f"external base mapping expected {len(self.robot.joint_order)} targets, "
                    f"got {q_base.shape}"
                )
            q_base = np.clip(q_base, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
        adaptive = self.adaptive_mapper
        control_target = None
        if adaptive is not None:
            if sensors_deg is not None or controls_deg is not None:
                control_target = adaptive.controls_from_observation(
                    sensors_deg, controls_deg, source=self.adaptive_source)
            else:
                # Direct calls without glove metadata can still be benchmarked.
                control_target = adaptive.encode_joints(q_base)
            q_base = adaptive.decode_controls(control_target)
        elif self.latent_mapper is not None:
            q_base = self.latent_mapper.project_joints(q_base)
        if q_prev is None:
            q_prev = q_base.copy()
        mode = intent.value if dynamic else Intent.NEUTRAL.value
        w = self.weights[mode]
        objective_intent = intent if dynamic else Intent.NEUTRAL

        def objective(q: np.ndarray) -> float:
            c = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
            return float(sum(float(w.get(name, 0.0)) * value for name, value in c.items()))

        if minimize is None and adaptive is None:
            q = np.clip(q_base, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
            comp = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
            return RetargetResult(q, q_base, float(objective(q)), False, 0, comp)

        if adaptive is not None:
            opt_kwargs = {'ftol': float(self.cfg['ftol'])}
            if isinstance(adaptive, ReachabilityGatedSynergyMapper):
                opt_kwargs['task_targets_m'] = {
                    f: human_tips[f] * self.task_scale
                    for f in adaptive.FINGERS[objective_intent]
                }
            q, result, diag = adaptive.optimize(
                objective, q_prev, objective_intent, control_target, **opt_kwargs)
            comp = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
            comp.update({'adaptive_k': float(diag.dimension),
                         'adaptive_projection_error_m': diag.projection_error_m,
                         'adaptive_residual_norm': diag.residual_norm,
                         'adaptive_feasible': float(diag.feasible)})
            if isinstance(diag, ReachabilityDiagnostics):
                comp.update({'reach_variables': float(diag.optimized_variables),
                             'reach_residual_axes': float(diag.active_residuals),
                             'reach_stage': float(diag.stage),
                             'reach_task_linear_error_m': diag.task_linear_error_m,
                             'reach_unreachable_error_m': diag.task_full_approx_error_m,
                             'reach_fallback_used': float(diag.fallback_full_reference),
                             'reach_command_rank': float(diag.command_subspace_rank)})
            if isinstance(diag, CredibleContactDiagnostics):
                comp.update({
                    'trust_risk_bound_m': diag.conditional_risk_bound_m,
                    'trust_budget_m': diag.risk_budget_m,
                    'trust_feasible': float(diag.risk_feasible),
                    'trust_contact_phase': {'UNOBSERVED': -1., 'FREE': 0.,
                        'APPROACH': 1., 'SECURE': 2., 'LOSS_RISK': 3.}.get(diag.contact_phase, -2.),
                    'trust_cov_trace': diag.sensor_uncertainty_trace,
                    'trust_nonlinear_tip_error_m': diag.nonlinear_tip_error_m,
                    'trust_ref_tip_error_m': diag.nonlinear_reference_error_m,
                    'trust_guard': float(diag.nonlinear_guard_used),
                    'trust_hold': float(diag.uncertainty_hold_used),
                    'trust_policy_enforced': float(diag.risk_policy_enforced),
                })
            if isinstance(diag, TaskEffectDiagnostics):
                comp.update({
                    'spectrum_modes': float(diag.command_subspace_rank),
                    'spectrum_omission_m': diag.spectral_omission_m,
                    'spectrum_risk_m': diag.spectral_ellipsoid_risk_m,
                    'spectrum_energy': diag.selected_energy_fraction,
                    'spectrum_regret_m': diag.baseline_regret_m,
                })
            return RetargetResult(q, q_base, float(objective(q)),
                                  bool(diag.feasible and (result is None or result.success)),
                                  int(getattr(result, 'nit', 0)), comp)
        if self.latent_mapper is None:
            # Original 11-joint optimization remains available as a baseline.
            result = minimize(
                objective,
                x0=np.clip(q_prev, self.robot.ranges[:, 0], self.robot.ranges[:, 1]),
                method="SLSQP",
                bounds=[tuple(x) for x in self.robot.ranges],
                options={"maxiter": int(self.cfg["maxiter"]), "ftol": float(self.cfg["ftol"]), "disp": False},
            )
            q = np.clip(np.asarray(result.x, dtype=float), self.robot.ranges[:, 0], self.robot.ranges[:, 1])
        else:
            # Optimize k latent coordinates, *never* 11 independent angles.
            # Every objective evaluation expands through the fixed 6->11
            # coupling model, so contact/task optimization cannot break it.
            latent = self.latent_mapper
            result = minimize(
                lambda z: objective(latent.decode_joints(z)),
                x0=np.clip(latent.encode_joints(q_prev),
                           latent.z_bounds[:, 0], latent.z_bounds[:, 1]),
                method="SLSQP",
                bounds=[tuple(x) for x in latent.z_bounds],
                options={"maxiter": int(self.cfg["maxiter"]), "ftol": float(self.cfg["ftol"]), "disp": False},
            )
            q = latent.decode_joints(result.x)
        comp = self.objective_components(q, q_base, q_prev, human_tips, objective_intent)
        return RetargetResult(
            q,
            q_base,
            float(result.fun),
            bool(result.success),
            int(getattr(result, "nit", 0)),
            comp,
        )
