from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

from src.common import ROOT, load_yaml
from src.evaluation.logger import CSVRunLogger
from src.evaluation.metrics import joint_rmse, velocity_rms
from src.evaluation.task_evaluator import TaskEpisodeEvaluator
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import IntentRecognizer
from src.glove.adapter import CapturedGloveInput, GloveAdapter, SimGloveAdapter
from src.glove.calibration import AdaptiveCalibrator
from src.glove.dataset_driver import CSVGloveDatasetDriver
from src.glove.mock_driver import MockGloveDriver
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.processing.filters import MovingAverageFilter, LowPassVectorFilter, rate_limit
from src.retargeting.optimizer import IntentDrivenRetargeter, RetargetResult
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.adaptive_synergy import TaskAdaptiveSynergyMapper
from src.retargeting.reachability_synergy import ReachabilityGatedSynergyMapper
from src.retargeting.credible_contact_synergy import CredibleContactSynergyMapper
from src.retargeting.task_effect_spectrum import TaskEffectSpectrumMapper
from src.retargeting.tqhg_online import TQHGOnlineMapper  # TQHG_ONLINE_INTEGRATED_v1
from src.retargeting.tqhg_v2 import TQHGv2OnlineMapper
from src.retargeting.tqhg_stagec import (StageCOnlineMapper, STAGEC_METHODS,
                                         parse_fault_channels, fault_mask_for_frame)
from src.retargeting.contact_assist import TaskContactAssist
from src.retargeting.task_space_servo import ObjectAwareTaskServo, ServoDiagnostics


CONTACT_COLUMNS = [
    "contact_count",
    "inactive_contact_candidates",
    "hand_normal_force_n",
    "thumb_tip_normal_force_n",
    "index_tip_normal_force_n",
    "max_contact_penetration_m",
    "raw_thumb_tip_contact",
    "raw_index_tip_contact",
    "raw_pinch_contact_proxy",
    "raw_wrap_contact_proxy",
    "filtered_task_contact_proxy",
    "environment_contact_count",
    "contact_sections",
    "thumb_contact",
    "index_contact",
    "middle_contact",
    "ring_contact",
    "little_contact",
    "finger_contact",
    "palm_contact",
    "thumb_tip_contact",
    "index_tip_contact",
    "middle_tip_contact",
    "ring_tip_contact",
    "little_tip_contact",
    "tip_contact_ratio",
    "contact_streak_frames",
    "orientation_drift_deg",
    "stable_orientation",
    "pinch_contact_proxy",
    "wrap_contact_proxy",
    "task_contact_proxy",
    "stable_contact_proxy",
    "object_displacement_m",
    "object_pos_x_m", "object_pos_y_m", "object_pos_z_m",
    "object_reset_this_step",
    "raw_thumb_contact", "raw_index_contact", "raw_middle_contact",
    "raw_ring_contact", "raw_little_contact",
    "thumb_normal_force_n", "index_normal_force_n", "middle_normal_force_n",
    "ring_normal_force_n", "little_normal_force_n",
    *[f"{finger}_force_on_object_{axis}_n" for finger in ("thumb", "index", "middle", "ring", "little") for axis in "xyz"],
    "hold_stable_proxy",
    "hold_object_free",
    "hold_pose_available",
    "hold_window_samples",
    "hold_window_required_samples",
    "hold_relative_translation_span_m",
    "hold_relative_rotation_span_deg",
]

SERVO_COLUMNS = [
    "task_servo_active",
    "servo_thumb_error_m",
    "servo_index_error_m",
    "servo_mean_error_m",
    "servo_max_error_m",
    "servo_corrected_digits",
    "servo_thumb_recovery_active", "servo_wrap_coordination_active",
    "servo_thumb_loss_event",
]

ADAPTIVE_COLUMNS = [
    "adaptive_k", "adaptive_projection_error_m", "adaptive_residual_norm", "adaptive_feasible",
    "reach_variables", "reach_residual_axes", "reach_stage",
    "reach_task_linear_error_m", "reach_unreachable_error_m", "reach_fallback_used", "reach_command_rank",
    "trust_risk_bound_m", "trust_budget_m", "trust_feasible", "trust_contact_phase",
    "trust_cov_trace", "trust_nonlinear_tip_error_m", "trust_ref_tip_error_m", "trust_guard", "trust_hold", "trust_policy_enforced",
    "spectrum_modes", "spectrum_omission_m", "spectrum_risk_m", "spectrum_energy",
    "spectrum_regret_m", "spectrum_post_guard", "spectrum_post_tip_m", "spectrum_post_baseline_m", "spectrum_speed_valid",
    "spectrum_guard_reason", "spectrum_object_error_m", "spectrum_object_baseline_m",
    "spectrum_selected_k", "spectrum_accepted_k", "spectrum_output_rank",
    "spectrum_fallback", "spectrum_fallback_reason", "spectrum_selected_indices",
    "spectrum_final_guard_enabled", "spectrum_final_guard_regret_m",
    "spectrum_candidate_attempts", "spectrum_reject_spectral_count",
    "spectrum_reject_fidelity_count", "spectrum_reject_nonlinear_count",
    "spectrum_candidate_min_fk_error_m", "spectrum_candidate_min_regret_m",
    "spectrum_candidate_max_fidelity_excess_deg", "spectrum_selection_status",
]

TRACE_JOINTS = (
    "th_cmc_joint", "th_mcp_joint", "th_ip_joint",
    "ff_mcp_joint", "ff_dip_joint",
)
TRACE_STAGES = ("opt", "servo", "filtered", "command")
SERVO_TRACE_COLUMNS = [
    f"trace_{stage}_{joint}" for stage in TRACE_STAGES for joint in TRACE_JOINTS
]

ONLINE_COLUMNS = ["mapping_w_task", "mapping_w_null", "mapping_visible_rank", "mapping_box_clipped"]
V2_ONLINE_COLUMNS = [
    "mapping_sensor_reliability", "mapping_invalid_channels", "mapping_gate_severity",
    "mapping_expert_delta_norm", "mapping_task_delta_norm", "mapping_null_delta_norm",
    "mapping_post_clip_task_delta_norm", "mapping_gate_fallback",
]
STAGEC_COLUMNS = [
    "stagec_fault_active", "stagec_fault_count",
    "mapping_stagec_projection_leakage", "mapping_stagec_expert_a_norm",
    "mapping_stagec_expert_b_norm", "mapping_stagec_task_output_shift",
    "mapping_stagec_null_output_shift",
]
STAGEC_DIAG_KEYS = [
    "stagec_projection_leakage", "stagec_expert_a_norm", "stagec_expert_b_norm",
    "stagec_task_output_shift", "stagec_null_output_shift",
]
V2_GATE_KEYS = [
    "sensor_reliability", "invalid_channels", "gate_severity", "expert_delta_norm",
    "task_delta_norm", "null_delta_norm", "post_clip_task_delta_norm", "gate_fallback",
]

EVALUATION_COLUMNS = [
    "task_active",
    "contact_success_proxy",
    "stable_success_proxy",
    "legacy_stable_success_proxy",
    "contact_success_streak_frames",
    "stable_success_streak_frames",
    "hold_stable_streak_frames",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Virtual dexterous hand v0.4 - synthetic dataset + benchmarkable MuJoCo pipeline")
    p.add_argument("--sim", choices=["auto", "none", "mujoco"], default="auto")
    p.add_argument("--render", action="store_true")
    p.add_argument("--contact-detector",
                   choices=["geometry_only", "force_only", "temporal_only", "force_and_temporal"],
                   default=None,
                   help="MuJoCo接触检测器消融：几何/力/纯时序/组合；只改变控制反馈，物理成功代理统一采用力证据")
    p.add_argument(
        "--steps",
        type=int,
        default=900,
        help="运行帧数；设为 0 表示持续运行，直到关闭 Viewer 或按 Ctrl+C",
    )
    p.add_argument("--realtime", action="store_true", help="按配置采样频率 sleep；默认尽快运行")
    p.add_argument(
        "--camera",
        choices=["overview", "closeup", "side", "free"],
        default=None,
        help="MuJoCo 自由相机初始视角预设",
    )
    p.add_argument(
        "--task",
        default=None,
        help="测试物体名称；见 configs/simulation.yaml -> task_objects，例如 wrap/pinch/sphere/card/bottle/box",
    )
    p.add_argument(
        "--input",
        choices=["sim", "glove", "mock", "dataset"],
        default="sim",
        help="输入源；mock/dataset 是兼容旧命令的 sim 别名",
    )
    p.add_argument("--sim-source", choices=["mock", "dataset"], default="dataset")
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "synthetic_glove_v1.csv",
        help="CSV 模拟数据集路径",
    )
    p.add_argument("--dataset-trial", type=int, default=0, help="使用数据集中的某个 trial_id")
    p.add_argument("--glove-samples", type=Path, default=None, help="真实手套 captured_samples (.xlsx/.csv)")
    p.add_argument("--base-mapping", type=Path, default=None, help="老师提供的 base_mapping.json")
    p.add_argument("--glove-manifest", type=Path, default=None, help="真实手套 manifest.json")
    p.add_argument("--latent-model", type=Path, default=None,
                   help="可选：15/6→k→11 低维协同模型 JSON（仅真实手套输入）")
    p.add_argument("--adaptive-synergy", action="store_true",
                   help="启用方案B：任务自适应协同维度与受限互补残差（须同时提供新版--latent-model）")
    p.add_argument("--task-effect-spectrum", action="store_true",
                   help="方案E：基于任务效应谱的按需选维和最终输出误差复核")
    p.add_argument("--spectrum-config", type=Path,
                   default=Path("configs/task_effect_spectrum.yaml"))
    p.add_argument("--credible-contact-synergy", action="store_true",
                   help="方案D：输入可信度-任务可达性协同与观测接触门控稀疏补偿")
    p.add_argument("--credible-config", type=Path,
                   default=Path("configs/credible_contact_synergy.yaml"),
                   help="方案D不确定性与接触约束配置")
    p.add_argument("--reachability-synergy", action="store_true",
                   help="启用方案C：任务雅可比协同基+真实低维求解+按需稀疏残差")
    p.add_argument("--reachability-config", type=Path, default=Path("configs/reachability_synergy.yaml"),
                   help="方案C协同可达性阈值与稀疏补偿参数")
    p.add_argument("--adaptive-config", type=Path, default=Path("configs/adaptive_synergy.yaml"),
                   help="方案B阈值、残差幅值、优化权重")
    p.add_argument("--latent-source", choices=["sensor", "control", "blend"], default="sensor",
                   help="低维输入路径；推荐 sensor，blend 使用两路编码器")
    p.add_argument("--seed", type=int, default=7, help="procedural mock 模式随机种子")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--run-name", default=None, help="固定输出文件名前缀，便于批量实验")
    p.add_argument("--no-auto-reset-object", action="store_true",
                   help="Disable only workspace-triggered task object reset; retain reproducible task-onset reset")
    p.add_argument("--no-inactive-object-reset", action="store_true",
                   help="Do not teleport the object home during inactive frames; task-onset reset remains")
    p.add_argument("--wrap-servo-policy",
                   choices=list(ObjectAwareTaskServo.WRAP_POLICIES), default="original",
                   help="WRAP-only four-way ablation: original, low_gain, "
                        "thumb_recovery, coordinated; requires legacy/uncoupled servo")
    p.add_argument("--coupled-servo-ablation", choices=["original", "prior", "hold", "both"],
                   default="original", help="诊断用：仅改变TEQS六维耦合伺服；其余路径不变")
    p.add_argument("--servo-trace", action="store_true",
                   help="在CSV记录优化器/伺服器/滤波/最终指令的拇指食指关节角（弧度）")
    p.add_argument("--mapping-method", choices=["legacy", "ridge6", "pca4", "fixed-blend", "single-gate", "tqhg", "tqhg-v2", *STAGEC_METHODS],
                   default="legacy", help="对照用15->6映射。legacy维持原控制流程")
    p.add_argument("--tqhg-calibration", type=Path, default=Path("configs/tqhg_online_v1.json"),
                   help="独立历史训练和校准形成的6维编码器与残差数据")
    p.add_argument("--tqhg-v2-calibration", type=Path, default=Path("configs/tqhg_online_v2.json"),
                   help="新增v2鲁棒4维专家及条件化双权重校准JSON")
    p.add_argument("--stagec-robust6-model", type=Path,
                   default=Path("configs/stageB_robust_ridge6.json"),
                   help="Stage B training-matched robust 30->6 expert")
    p.add_argument("--stagec-gate-model", type=Path,
                   default=Path("configs/stageC_task_gate.json"),
                   help="CAL-only paired expert errors used with real task Jacobian")
    p.add_argument("--stagec-fault-channels", default="",
                   help="Explicit missing sensors, ONE-BASED comma-separated e.g. 1,4,7")
    p.add_argument("--stagec-fault-start", type=int, default=120,
                   help="Missing-sensor first frame inclusive")
    p.add_argument("--stagec-fault-stop", type=int, default=240,
                   help="First recovered frame exclusive")
    return p


def _empty_contact_metrics() -> dict[str, float | int | bool]:
    return {
        "contact_count": 0,
        "inactive_contact_candidates": 0,
        "hand_normal_force_n": 0.0,
        "thumb_tip_normal_force_n": 0.0,
        "index_tip_normal_force_n": 0.0,
        "max_contact_penetration_m": 0.0,
        "raw_thumb_tip_contact": False,
        "raw_index_tip_contact": False,
        "raw_pinch_contact_proxy": False,
        "raw_wrap_contact_proxy": False,
        "filtered_task_contact_proxy": False,
        "environment_contact_count": 0,
        "contact_sections": 0,
        "thumb_contact": False,
        "index_contact": False,
        "middle_contact": False,
        "ring_contact": False,
        "little_contact": False,
        "finger_contact": False,
        "palm_contact": False,
        "thumb_tip_contact": False,
        "index_tip_contact": False,
        "middle_tip_contact": False,
        "ring_tip_contact": False,
        "little_tip_contact": False,
        "tip_contact_ratio": 0.0,
        "contact_streak_frames": 0,
        "orientation_drift_deg": 0.0,
        "stable_orientation": True,
        "pinch_contact_proxy": False,
        "wrap_contact_proxy": False,
        "task_contact_proxy": False,
        "stable_contact_proxy": False,
        "object_displacement_m": 0.0,
        "object_pos_x_m": float("nan"),
        "object_pos_y_m": float("nan"),
        "object_pos_z_m": float("nan"),
        "object_reset_this_step": False,
        **{f"raw_{f}_contact": False for f in ("thumb", "index", "middle", "ring", "little")},
        **{f"{f}_normal_force_n": 0.0 for f in ("thumb", "index", "middle", "ring", "little")},
        **{f"{f}_force_on_object_{axis}_n": 0.0 for f in ("thumb", "index", "middle", "ring", "little") for axis in "xyz"},
        "hold_stable_proxy": False,
        "hold_object_free": False,
        "hold_pose_available": False,
        "hold_window_samples": 0,
        "hold_window_required_samples": 0,
        "hold_relative_translation_span_m": 0.0,
        "hold_relative_rotation_span_deg": 0.0,
    }


def _make_driver(args: argparse.Namespace, glove_cfg: dict):
    channels = list(glove_cfg["channels"])
    hz = float(glove_cfg["sampling_hz"])
    source = args.input if args.input in {"mock", "dataset"} else args.sim_source
    if source == "dataset":
        driver = CSVGloveDatasetDriver(
            args.dataset,
            channels,
            sampling_hz=hz,
            trial_id=args.dataset_trial,
            loop=True,
        )
        print(
            f"[INFO] 输入=synthetic dataset: {args.dataset} "
            f"trial={args.dataset_trial} available={driver.available_trials}"
        )
        return driver
    driver = MockGloveDriver(
        channels=channels,
        sampling_hz=hz,
        raw_min=float(glove_cfg["raw_min_default"]),
        raw_max=float(glove_cfg["raw_max_default"]),
        seed=int(args.seed),
    )
    print(f"[INFO] 输入=procedural mock seed={args.seed}")
    return driver


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    # PowerShell 5.1 decodes redirected native output using the active code page.
    # Emit UTF-8 consistently so captured Chinese paths/messages remain readable.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    glove_cfg = load_yaml("configs/glove.yaml")
    algo_cfg = load_yaml("configs/algorithm.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    channels = glove_cfg["channels"]
    hz = float(glove_cfg["sampling_hz"])
    dt = 1.0 / hz

    task_name = args.task or str(sim_cfg.get("default_task", "wrap"))
    task_catalog = dict(sim_cfg.get("task_objects", {}))
    if task_name not in task_catalog:
        raise SystemExit(f"未知 task={task_name!r}；可用：{', '.join(sorted(task_catalog))}")
    task_spec = dict(task_catalog[task_name])
    task_mode = str(task_spec.get("interaction", task_name))
    active_intent = str(task_spec.get("active_intent", "PINCH" if task_mode == "pinch" else "WRAP"))

    human = HumanHandModel()
    robot = RobotHandModel()
    input_mode = "glove" if args.input == "glove" else "sim"
    sim_source = args.input if args.input in {"mock", "dataset"} else args.sim_source
    glove_input = None
    latent_mapper = None
    if args.latent_model is not None:
        if input_mode != "glove":
            raise SystemExit("--latent-model 仅适用于 --input glove")
        latent_mapper = LatentSynergyMapper(robot, _project_path(args.latent_model))
        print(f"[INFO] 已加载低维协同模型: k={latent_mapper.k}, source={args.latent_source}")
    if sum((args.adaptive_synergy, args.reachability_synergy,
            args.credible_contact_synergy, args.task_effect_spectrum)) > 1:
        raise SystemExit("方案B/C/D的开关互斥")
    if (args.adaptive_synergy or args.reachability_synergy or
        args.credible_contact_synergy or args.task_effect_spectrum) and latent_mapper is None:
        raise SystemExit("自适应协同需要 --latent-model，且模型须包含完整6维协同基")
    if args.task_effect_spectrum:
        adaptive_mapper = TaskEffectSpectrumMapper(
            latent_mapper, load_yaml(str(args.spectrum_config)))
        print("[INFO] 方案E：任务效应谱动态方向、模型误差输出护栏")
    elif args.credible_contact_synergy:
        adaptive_mapper = CredibleContactSynergyMapper(
            latent_mapper, load_yaml(str(args.credible_config)))
        print(f"[INFO] 方案D：可信度门控、实际低维优化和接触状态反馈")
    elif args.reachability_synergy:
        adaptive_mapper = ReachabilityGatedSynergyMapper(
            latent_mapper, load_yaml(str(args.reachability_config)))
        print(f"[INFO] 方案C：k=[{adaptive_mapper.k_min},{adaptive_mapper.k_max}]，按需释放稀疏残差")
    elif args.adaptive_synergy:
        adaptive_mapper = TaskAdaptiveSynergyMapper(
            latent_mapper, load_yaml(str(args.adaptive_config)))
        print(f"[INFO] 方案B：k=[{adaptive_mapper.k_min},{adaptive_mapper.k_max}]，恒优化6个变量")
    else:
        adaptive_mapper = None
    online_mapper = None
    try:
        stagec_fault_channels = parse_fault_channels(args.stagec_fault_channels)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.stagec_fault_stop <= args.stagec_fault_start or args.stagec_fault_start < 0:
        raise SystemExit("Stage C fault window requires 0 <= start < stop")
    if stagec_fault_channels and args.mapping_method not in STAGEC_METHODS:
        raise SystemExit("--stagec-fault-channels applies only to Stage C methods")
    if args.mapping_method != "legacy":
        if input_mode != "glove" or args.task_effect_spectrum or args.adaptive_synergy or args.reachability_synergy or args.credible_contact_synergy:
            raise SystemExit("--mapping-method 非legacy只支持 --input glove，不能同时启用方案B/C/D/E")
        if latent_mapper is not None:
            raise SystemExit("在线映射使用独立 --tqhg-calibration 模型；请去掉 --latent-model")
        if args.mapping_method in STAGEC_METHODS:
            online_mapper = StageCOnlineMapper(
                robot, _project_path(args.tqhg_v2_calibration),
                _project_path(args.stagec_robust6_model),
                _project_path(args.stagec_gate_model))
            print(f"[INFO] Stage C robust6 vs robust4: {args.mapping_method}; real task J when --sim mujoco")
            print(f"[INFO] Explicit missing sensors (1-based)={stagec_fault_channels}, "
                  f"frames=[{args.stagec_fault_start}, {args.stagec_fault_stop})")
        elif args.mapping_method == "tqhg-v2":
            online_mapper = TQHGv2OnlineMapper(robot, _project_path(args.tqhg_v2_calibration))
            print(f"[INFO] TQHG-v2 conditional robust online mapping={args.tqhg_v2_calibration}")
        else:
            online_mapper = TQHGOnlineMapper(robot, _project_path(args.tqhg_calibration))
            print(f"[INFO] 在线统一6维映射={args.mapping_method}；模型={args.tqhg_calibration}")
    driver = None
    sim_adapter = None
    if input_mode == "glove":
        real_cfg = dict(glove_cfg["real"])
        samples_path = _project_path(args.glove_samples or real_cfg["captured_samples"])
        mapping_path = _project_path(args.base_mapping or real_cfg["base_mapping"])
        manifest_path = _project_path(args.glove_manifest or real_cfg["manifest"])
        glove_adapter = GloveAdapter(
            human,
            robot,
            base_mapping_path=mapping_path,
            manifest_path=manifest_path,
            output_bindings=real_cfg["output_bindings"],
            latent_mapper=latent_mapper if online_mapper is None else None,
            latent_source=args.latent_source,
        )
        glove_input = CapturedGloveInput(samples_path, glove_adapter, loop=True)
        print(
            f"[INFO] 输入=glove captures: {glove_input.path} samples={len(glove_input.samples)} "
            f"mapping={glove_adapter.base_mapping.path}"
        )
    else:
        driver = _make_driver(args, glove_cfg)
        calibrator = AdaptiveCalibrator(channels, epsilon=float(glove_cfg["calibration_epsilon"]))
        calibrator.fit(driver.calibration_samples())
        input_filter = MovingAverageFilter(channels, int(glove_cfg["moving_average_window"]))
        sim_adapter = SimGloveAdapter(human, calibrator, input_filter)

    extractor = FeatureExtractor(human)
    intent_cfg = algo_cfg["intent"]
    recognizer = IntentRecognizer(
        float(intent_cfg["pinch_enter_m"]),
        float(intent_cfg["pinch_exit_m"]),
        float(intent_cfg["wrap_enter"]),
        float(intent_cfg["wrap_exit"]),
        int(intent_cfg["dwell_frames"]),
        int(intent_cfg.get("sequence_window_frames", 1)),
    )
    retargeter = IntentDrivenRetargeter(
        human, robot, algo_cfg["optimizer"], latent_mapper=latent_mapper,
        adaptive_mapper=adaptive_mapper, adaptive_source=args.latent_source)
    contact_assist = TaskContactAssist(robot, algo_cfg.get("contact_assist", {}), dt)
    if args.wrap_servo_policy != "original":
        if task_mode != "wrap":
            raise SystemExit("--wrap-servo-policy 非 original 仅适用于 --task wrap")
        if (args.sim != "mujoco" or not bool(algo_cfg.get("task_space_servo", {}).get("enabled", False))):
            raise SystemExit("WRAP 控制消融需要启用 MuJoCo 任务伺服")
        if isinstance(adaptive_mapper, TaskEffectSpectrumMapper) or online_mapper is not None:
            raise SystemExit("WRAP 控制消融尚不支持六维耦合伺服：请使用 --mapping-method legacy")
    task_servo = ObjectAwareTaskServo(
        robot, algo_cfg.get("task_space_servo", {}),
        coupled_ablation=args.coupled_servo_ablation,
        wrap_policy=args.wrap_servo_policy)
    if args.coupled_servo_ablation != "original" and not (args.task_effect_spectrum or online_mapper is not None):
        raise SystemExit("--coupled-servo-ablation 仅适用于 --task-effect-spectrum")
    if args.task_effect_spectrum or online_mapper is not None:
        print(f"[INFO] Coupled servo ablation={args.coupled_servo_ablation}; "
              f"trace={'ON' if args.servo_trace else 'OFF'}")

    out_cfg = algo_cfg["output_filter"]
    lp = LowPassVectorFilter(float(out_cfg["alpha"]))
    vmax = np.full(len(robot.joint_order), math.radians(float(out_cfg["max_velocity_deg_s"])))

    simulator = None
    use_mujoco = args.sim == "mujoco"
    if args.sim == "auto":
        try:
            import mujoco  # noqa: F401
            use_mujoco = True
        except ImportError:
            use_mujoco = False
            print("[INFO] 未检测到 MuJoCo，自动使用纯算法模式。")

    if use_mujoco:
        from src.simulation.mujoco_env import MujocoSimulator

        camera = args.camera or str(sim_cfg["default_camera"])
        simulator = MujocoSimulator(
            robot,
            model_path=str(sim_cfg["model_path"]),
            render=args.render,
            camera=camera,
            task_object_body=str(sim_cfg["task_object_body"]),
            task_name=task_name,
            task_spec=task_spec,
            auto_reset_object=bool(sim_cfg["auto_reset_object"]) and not args.no_auto_reset_object,
            object_workspace_radius_m=float(sim_cfg["object_workspace_radius_m"]),
            object_min_z_m=float(sim_cfg["object_min_z_m"]),
            stability_orientation_deg=float(sim_cfg.get("stability_orientation_deg", 3.0)),
            hold_window_s=float(sim_cfg.get("evaluation", {}).get("hold_window_s", 1.0)),
            hold_max_translation_m=float(sim_cfg.get("evaluation", {}).get("hold_max_translation_m", 0.001)),
            hold_max_rotation_deg=float(sim_cfg.get("evaluation", {}).get("hold_max_rotation_deg", 1.0)),
            camera_presets=dict(sim_cfg.get("camera_presets", {})),
            scene_config=sim_cfg,
            contact_detection={**sim_cfg.get("contact_detection", {}),
                               **({"mode": args.contact_detector} if args.contact_detector else {})},
            control_dt_s=dt,
        )
        if args.render:
            print(
                f"[INFO] MuJoCo Viewer 已启动；camera={camera}；task={task_name} "
                f"mode={task_mode}；任务空间伺服={'ON' if task_servo.enabled else 'OFF'}"
            )
        print(
            f"[INFO] Physics sync: control_dt={dt*1000:.2f}ms; "
            f"mujoco_dt={simulator.physics_dt_s*1000:.2f}ms; "
            f"avg_substeps={dt/simulator.physics_dt_s:.2f}"
        )

    print(
        f"[INFO] Robot architecture: nominal DoF={robot.nominal_dof}, "
        f"effective actuators={robot.effective_actuators}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{stamp}_{task_name}"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{run_name}.csv"
    summary_path = output_dir / f"{run_name}_summary.json"
    logger = CSVRunLogger(
        log_path,
        robot.joint_order,
        extra_columns=(CONTACT_COLUMNS + SERVO_COLUMNS + EVALUATION_COLUMNS + ADAPTIVE_COLUMNS
                       + (SERVO_TRACE_COLUMNS if args.servo_trace else []) + ONLINE_COLUMNS + V2_ONLINE_COLUMNS + STAGEC_COLUMNS),
    )

    eval_cfg = dict(sim_cfg.get("evaluation", {}))
    evaluator = TaskEpisodeEvaluator(
        task_name=task_name,
        sampling_hz=hz,
        min_success_streak_frames=int(eval_cfg.get("min_success_streak_frames", 12)),
    )

    q_prev_opt = None
    q_prev_out = None
    q_prev_command = None  # physical q_actual may lag the feasible commanded manifold
    last_intent = None
    last_contact = _empty_contact_metrics()
    approach_fixture_released = False
    frame = 0

    try:
        while args.steps <= 0 or frame < args.steps:
            if simulator is not None and not simulator.is_running():
                print("[INFO] MuJoCo Viewer 已关闭，结束运行。")
                break

            pipeline_t0 = time.perf_counter()
            if input_mode == "glove":
                assert glove_input is not None
                hand_frame = glove_input.sample()
            else:
                assert driver is not None and sim_adapter is not None
                hand_frame = sim_adapter.adapt(driver.sample())
            q_h = hand_frame.joint_angles_rad
            features = extractor.extract(q_h)
            intent = recognizer.update(features)
            task_active = intent.value == active_intent

            if simulator is not None:
                rising = simulator.set_task_active(
                    task_active,
                    hold_when_inactive=(bool(sim_cfg.get("hold_task_object_when_inactive", True))
                                        and not args.no_inactive_object_reset),
                )
                if rising:
                    if isinstance(online_mapper, TQHGv2OnlineMapper):
                        online_mapper.reset()
                    last_contact = _empty_contact_metrics()
                    approach_fixture_released = False
                    contact_assist.reset()
                    task_servo.reset()

            if isinstance(adaptive_mapper, CredibleContactSynergyMapper):
                # Previous-frame real simulator telemetry; no fabricated contact
                # feedback in --sim none, and no future-frame leakage.
                adaptive_mapper.update_contact(
                    last_contact if simulator is not None else None,
                    task_active=task_active, observed=(simulator is not None))

            if isinstance(adaptive_mapper, TaskEffectSpectrumMapper):
                adaptive_mapper.set_command_anchor(q_prev_command, vmax, dt)

            opt_t0 = time.perf_counter()
            stagec_fault_active = False
            stagec_fault_count = 0
            if online_mapper is not None:
                if isinstance(online_mapper, StageCOnlineMapper):
                    valid = fault_mask_for_frame(
                        frame, channels=stagec_fault_channels,
                        start=args.stagec_fault_start, stop=args.stagec_fault_stop)
                    stagec_fault_count = int(np.count_nonzero(~valid))
                    stagec_fault_active = bool(stagec_fault_count)
                    # ONLY the mapper sees invalid channels. Human glove
                    # intent/extractor remain clean to isolate mapping faults.
                    control = online_mapper.predict(
                        hand_frame.metadata['raw_sensors_deg'], method=args.mapping_method,
                        simulator=simulator, task_mode=task_mode,
                        sensor_valid_mask=valid)
                else:
                    control = online_mapper.predict(
                        hand_frame.metadata['raw_sensors_deg'], method=args.mapping_method,
                        simulator=simulator, task_mode=task_mode)
                q_mapped = online_mapper.decode_controls(control)
                # Direct online mapping comparison: SAME downstream coupled Hold
                # for Ridge6/PCA4/fixed blend/single gate/TQHG; no hidden NLP.
                result = RetargetResult(q_mapped, q_mapped.copy(), 0.0, True, 0, {})
            else:
                result = retargeter.solve(
                    q_h,
                    features.fingertip_positions,
                    intent,
                    q_prev_opt,
                    dynamic=True,
                    q_base=hand_frame.robot_joint_targets_rad,
                    sensors_deg=hand_frame.metadata.get('raw_sensors_deg') if adaptive_mapper else None,
                    controls_deg=(np.array([
                        hand_frame.metadata['base_mapping_outputs_deg'][name]
                        for name in glove_adapter.base_mapping.output_names]) if adaptive_mapper else None),
                )
            optimizer_ms = (time.perf_counter() - opt_t0) * 1000.0
            q_prev_opt = result.q_target.copy()

            q_target = result.q_target
            q_trace_opt = q_target.copy() if args.servo_trace else None
            servo_diag = ServoDiagnostics()
            if simulator is not None:
                if task_servo.enabled:
                    if isinstance(adaptive_mapper, TaskEffectSpectrumMapper) or online_mapper is not None:
                        q_target, servo_diag = task_servo.apply_coupled(
                            q_target, simulator=simulator, task_name=task_name,
                            task_mode=task_mode, intent_value=intent.value,
                            contact=last_contact, mapper=(online_mapper if online_mapper is not None else adaptive_mapper))
                    else:
                        q_target, servo_diag = task_servo.apply(
                            q_target, simulator=simulator, task_name=task_name,
                            task_mode=task_mode, intent_value=intent.value,
                            contact=last_contact)
                else:
                    assist_mode = "wrap" if task_mode in {"wrap", "sphere"} else task_mode
                    q_target = contact_assist.apply(
                        q_target,
                        task_name=assist_mode,
                        intent_value=intent.value,
                        contact=last_contact,
                    )

            q_trace_servo = q_target.copy() if args.servo_trace else None

            # Object-space goal must be derived from CURRENT MuJoCo telemetry;
            # never fabricate it during --sim none. Clear it when servo inactive.
            if isinstance(adaptive_mapper, TaskEffectSpectrumMapper):
                if simulator is not None and servo_diag.active:
                    task_cfg = task_servo._task_cfg(task_name, task_mode) or {}
                    clearance = float(task_cfg.get('target_offset_m',
                                      task_cfg.get('clearance_m', 0.007)))
                    margin = float(task_cfg.get('vertical_margin_m', 0.01))
                    fingers = task_servo._active_fingers(task_mode)
                    targets = {
                        finger: simulator.task_contact_target(
                            finger, task_mode, clearance_m=clearance,
                            vertical_margin_m=margin)
                        for finger in fingers
                    }
                    # Do not use an unavailable MuJoCo FK callback or silently
                    # mix world-frame goals with local-hand kinematics.
                    adaptive_mapper.set_object_task_goal(None, targets)
                else:
                    adaptive_mapper.set_object_task_goal(None, None)

            # Scheme E already uses a joint-wise, six-control speed box and
            # checks the actual final command. An additional global LPF can
            # erase the selected task-mode command; compare this choice as
            # a separate ablation rather than claiming identical filtering.
            q_filtered = (q_target if online_mapper is not None else q_target if isinstance(adaptive_mapper, TaskEffectSpectrumMapper)
                          and not adaptive_mapper.config.get('apply_lowpass', False)
                          else lp.apply(q_target))
            q_trace_filtered = q_filtered.copy() if args.servo_trace else None
            if online_mapper is not None:
                q_out = online_mapper.project_and_rate_limit(q_filtered, q_prev_command, vmax, dt)
                q_prev_command = q_out.copy()
            elif adaptive_mapper is not None:
                # Preserve the complete six-command manifold, including residuals.
                # Rate-limit against prior *commands*, not delayed physical feedback.
                q_out = adaptive_mapper.project_and_rate_limit(q_filtered, q_prev_command, vmax, dt)
                q_prev_command = q_out.copy()
                if isinstance(adaptive_mapper, TaskEffectSpectrumMapper):
                    final_diag = adaptive_mapper.last_diag
                    if final_diag is not None:
                        result.components.update({
                            'spectrum_post_guard': float(final_diag.final_guard_used),
                            'spectrum_post_tip_m': final_diag.final_tip_error_m,
                            'spectrum_post_baseline_m': final_diag.final_reference_error_m,
                            'spectrum_speed_valid': float(final_diag.rate_box_valid),
                            'spectrum_guard_reason': float(final_diag.final_guard_reason),
                            'spectrum_object_error_m': final_diag.final_object_task_error_m,
                            'spectrum_object_baseline_m': final_diag.final_object_baseline_m,
                            'spectrum_selected_k': float(final_diag.requested_mode_count),
                            'spectrum_accepted_k': float(final_diag.accepted_mode_count),
                            'spectrum_output_rank': float(final_diag.output_control_rank),
                            'spectrum_fallback': float(final_diag.fallback_full_reference),
                            'spectrum_fallback_reason': final_diag.fallback_reason,
                            'spectrum_selected_indices': ','.join(map(str, final_diag.selected_mode_indices)),
                            'spectrum_final_guard_enabled': float(final_diag.final_guard_enabled),
                            'spectrum_final_guard_regret_m': final_diag.final_guard_regret_m,
                            'spectrum_candidate_attempts': float(final_diag.candidate_attempts),
                            'spectrum_reject_spectral_count': float(final_diag.reject_spectral_count),
                            'spectrum_reject_fidelity_count': float(final_diag.reject_fidelity_count),
                            'spectrum_reject_nonlinear_count': float(final_diag.reject_nonlinear_count),
                            'spectrum_candidate_min_fk_error_m': final_diag.candidate_min_fk_error_m,
                            'spectrum_candidate_min_regret_m': final_diag.candidate_min_regret_m,
                            'spectrum_candidate_max_fidelity_excess_deg': final_diag.candidate_max_fidelity_excess_deg,
                            'spectrum_selection_status': final_diag.selection_status,
                        })
            elif latent_mapper is None:
                q_out = rate_limit(q_filtered, q_prev_out, vmax, dt)
            else:
                # Fixed-PCA baseline: enforce its k-dimensional manifold.
                q_out = latent_mapper.project_and_rate_limit(q_filtered, q_prev_command, vmax, dt)
                q_prev_command = q_out.copy()
            if simulator is not None:
                # PINCH approach fixture: keep the free calibration block upright
                # until BOTH fingertip pads have contacted it.  A single early
                # contact can otherwise knock the tall block down and make the
                # second fingertip chase a moving target.  As soon as bilateral
                # fingertip contact was observed on the previous frame, release
                # the object; success must then persist under free-body physics.
                bilateral_tip_contact = bool(
                    last_contact.get("thumb_tip_contact", False)
                    and last_contact.get("index_tip_contact", False)
                )
                if task_name == "pinch" and bilateral_tip_contact:
                    approach_fixture_released = True
                hold_pinch_object = bool(
                    task_mode == "pinch"
                    and task_active
                    and not approach_fixture_released
                )
                q_actual = simulator.step(q_out, hold_object=hold_pinch_object)
                contact = simulator.contact_metrics()
                if hold_pinch_object:
                    # Fixture-assisted frames prove physical contact acquisition,
                    # but they must not count as free-body stability.
                    contact["stable_contact_proxy"] = False
                last_contact = contact
            else:
                q_actual = q_out
                contact = _empty_contact_metrics()

            latency_ms = (time.perf_counter() - pipeline_t0) * 1000.0
            rmse = joint_rmse(q_actual, result.q_base)
            vel = velocity_rms(q_actual, q_prev_out, dt)
            q_prev_out = q_actual.copy()
            servo_dict = servo_diag.as_dict()
            eval_state = evaluator.update(
                frame=frame,
                task_active=task_active,
                contact=contact,
                latency_ms=latency_ms,
                optimizer_ms=optimizer_ms,
                joint_rmse_rad=rmse,
                velocity_rms_rad_s=vel,
                servo_mean_error_m=float(servo_dict["servo_mean_error_m"]),
            )

            trace_values = []
            if args.servo_trace:
                positions = [robot.joint_order.index(n) for n in TRACE_JOINTS]
                for q in (q_trace_opt, q_trace_servo, q_trace_filtered, q_out):
                    trace_values.extend(float(q[i]) for i in positions)
            logger.write(
                [
                    frame,
                    hand_frame.timestamp,
                    intent.value,
                    features.pinch_distance_m,
                    features.wrap_score,
                    latency_ms,
                    optimizer_ms,
                    result.objective,
                    rmse,
                    vel,
                    *[contact[name] for name in CONTACT_COLUMNS],
                    *[servo_dict[name] for name in SERVO_COLUMNS],
                    *[eval_state[name] for name in EVALUATION_COLUMNS],
                    *[result.components.get(name, float('nan')) for name in ADAPTIVE_COLUMNS],
                    *trace_values,
                    *([online_mapper.last_gate.get("w_task", float("nan")),
                        online_mapper.last_gate.get("w_null", float("nan")),
                        online_mapper.last_gate.get("visible_rank", float("nan")),
                        float(online_mapper.last_gate.get("box_clipped", False))]
                      if online_mapper is not None else [float("nan")] * len(ONLINE_COLUMNS)),
                    *([online_mapper.last_gate.get(key, float("nan")) for key in V2_GATE_KEYS]
                      if online_mapper is not None else [float("nan")] * len(V2_ONLINE_COLUMNS)),
                    *([int(stagec_fault_active), stagec_fault_count] +
                      [online_mapper.last_gate.get(key, float("nan")) for key in STAGEC_DIAG_KEYS]
                      if isinstance(online_mapper, StageCOnlineMapper)
                      else [float("nan")] * len(STAGEC_COLUMNS)),
                    *q_actual.tolist(),
                ]
            )

            # Print intent transitions plus periodic active-task diagnostics.
            # The thumb/index pad flags are essential for PINCH debugging: raw
            # contact_count alone cannot tell a phalanx hit from a true pad hit.
            report_active = bool(task_active and frame % 30 == 0)
            if intent != last_intent or report_active:
                extra = ""
                if simulator is not None:
                    extra = (
                        f" contacts={contact['contact_count']} sections={contact['contact_sections']} "
                        f"tip_ratio={float(contact['tip_contact_ratio']):.2f}"
                        f" Ttip={int(bool(contact['thumb_tip_contact']))}"
                        f" Itip={int(bool(contact['index_tip_contact']))}"
                        f" Terr={float(servo_dict['servo_thumb_error_m'])*1000:.1f}mm"
                        f" Ierr={float(servo_dict['servo_index_error_m'])*1000:.1f}mm"
                    )
                print(
                    f"frame={frame:04d} intent={intent.value:<7} "
                    f"pinch={features.pinch_distance_m:.4f}m wrap={features.wrap_score:.3f} "
                    f"opt={optimizer_ms:.2f}ms{extra}"
                )
                last_intent = intent

            if args.realtime:
                time.sleep(max(0.0, dt - (time.perf_counter() - pipeline_t0)))

            frame += 1

    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，结束运行。")
    finally:
        logger.close()
        if simulator is not None:
            simulator.close()

        summary = evaluator.summary()
        required_success_metric = str(task_spec.get("required_success_metric", "contact" if task_mode == "pinch" else "stable"))
        task_metric_pass = True if required_success_metric == "none" else bool(summary["contact_success_proxy" if required_success_metric == "contact" else "stable_success_proxy"])
        summary.update(
            {
                "version": "ch-m6-adapter-v1",
                "input_source": input_mode,
                "sim_source": sim_source if input_mode == "sim" else None,
                "dataset": str(args.dataset) if input_mode == "sim" and sim_source == "dataset" else None,
                "dataset_trial": int(args.dataset_trial) if input_mode == "sim" and sim_source == "dataset" else None,
                "mock_seed": int(args.seed) if input_mode == "sim" and sim_source == "mock" else None,
                "glove_samples": str(glove_input.path) if glove_input is not None else None,
                "base_mapping": str(glove_input.adapter.base_mapping.path) if glove_input is not None else None,
                "simulation": "mujoco" if simulator is not None else "none",
                "robot_model": "CH-M6_L",
                "robot_dof": robot.nominal_dof,
                "robot_actuators": robot.effective_actuators,
                "task_mode": task_mode,
                "active_intent": active_intent,
                "required_success_metric": required_success_metric,
                "coupled_servo_ablation": args.coupled_servo_ablation,
                "wrap_servo_policy": args.wrap_servo_policy,
                "auto_reset_object_effective": bool(sim_cfg["auto_reset_object"]) and not args.no_auto_reset_object,
                "inactive_object_reset_effective": (bool(sim_cfg.get("hold_task_object_when_inactive", True))
                                                     and not args.no_inactive_object_reset),
                "requested_steps": int(args.steps),
                "actual_processed_frames": int(frame),
                "wrap_servo_policy_config": (task_servo._wrap_params(
                    task_servo._task_cfg(task_name, task_mode) or {})
                    if args.wrap_servo_policy != "original" else None),
                "mapping_method": args.mapping_method,
                "tqhg_calibration": str(args.tqhg_calibration) if online_mapper is not None and args.mapping_method not in ("tqhg-v2", *STAGEC_METHODS) else None,
                "tqhg_v2_calibration": str(args.tqhg_v2_calibration) if args.mapping_method in ("tqhg-v2", *STAGEC_METHODS) else None,
                "mapping_v2_conditional_gate": bool(args.mapping_method == "tqhg-v2"),
                "stagec_real_jacobian_gate": bool(args.mapping_method in STAGEC_METHODS and simulator is not None),
                "stagec_fair_robust_experts": bool(args.mapping_method in STAGEC_METHODS),
                "stagec_fault_channels_1based": list(stagec_fault_channels) if args.mapping_method in STAGEC_METHODS else [],
                "stagec_fault_window_start_inclusive": args.stagec_fault_start if args.mapping_method in STAGEC_METHODS else None,
                "stagec_fault_window_stop_exclusive": args.stagec_fault_stop if args.mapping_method in STAGEC_METHODS else None,
                "stagec_fault_applied_to": "mapping sensor validity ONLY; glove intent uses clean replay" if args.mapping_method in STAGEC_METHODS else None,
                "stagec_gate_model": str(args.stagec_gate_model) if args.mapping_method in STAGEC_METHODS else None,
                "stagec_robust6_model": str(args.stagec_robust6_model) if args.mapping_method in STAGEC_METHODS else None,
                "online_comparison_bypasses_legacy_optimizer": bool(online_mapper is not None),
                "stability_metric_version": "v2_free_body_relative_palm_rolling_1s",
                "stability_window_s": float(eval_cfg.get("hold_window_s", 1.0)),
                "stability_max_translation_m": float(eval_cfg.get("hold_max_translation_m", 0.001)),
                "stability_max_rotation_deg": float(eval_cfg.get("hold_max_rotation_deg", 1.0)),
                "contact_detector": (simulator._contact_filter.cfg.mode if simulator is not None else None),
                "contact_detector_config": (vars(simulator._contact_filter.cfg) if simulator is not None else None),
                "stability_uses_contact_force": bool(simulator is not None),
                "stability_uses_lift": False,
                "stability_uses_contact_force_or_lift": bool(simulator is not None),
                "servo_trace_enabled": bool(args.servo_trace),
                "task_metric_pass": task_metric_pass,
                "log_csv": str(log_path),
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] CSV:     {log_path}")
        print(f"[DONE] Summary: {summary_path}")
        if simulator is not None:
            print(
                f"[RESULT] task={task_name} "
                f"contact_success={summary['contact_success_proxy']} "
                f"stable_success={summary['stable_success_proxy']} "
                f"contact_ratio={summary['contact_frame_ratio']:.1%} "
                f"stable_ratio={summary['stable_frame_ratio']:.1%} "
                f"p95_latency={summary['p95_pipeline_latency_ms']:.2f}ms"
            )


if __name__ == "__main__":
    main()
