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
from src.glove.calibration import AdaptiveCalibrator
from src.glove.dataset_driver import CSVGloveDatasetDriver
from src.glove.mock_driver import MockGloveDriver
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.processing.filters import MovingAverageFilter, LowPassVectorFilter, rate_limit
from src.retargeting.optimizer import IntentDrivenRetargeter
from src.retargeting.contact_assist import TaskContactAssist
from src.retargeting.task_space_servo import ObjectAwareTaskServo, ServoDiagnostics


CONTACT_COLUMNS = [
    "contact_count",
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
]

SERVO_COLUMNS = [
    "task_servo_active",
    "servo_thumb_error_m",
    "servo_index_error_m",
    "servo_mean_error_m",
    "servo_max_error_m",
    "servo_corrected_digits",
]

EVALUATION_COLUMNS = [
    "task_active",
    "contact_success_proxy",
    "stable_success_proxy",
    "contact_success_streak_frames",
    "stable_success_streak_frames",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Virtual dexterous hand v0.4 - synthetic dataset + benchmarkable MuJoCo pipeline")
    p.add_argument("--sim", choices=["auto", "none", "mujoco"], default="auto")
    p.add_argument("--render", action="store_true")
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
    p.add_argument("--input", choices=["mock", "dataset"], default="dataset", help="模拟输入来源")
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "synthetic_glove_v1.csv",
        help="CSV 模拟数据集路径",
    )
    p.add_argument("--dataset-trial", type=int, default=0, help="使用数据集中的某个 trial_id")
    p.add_argument("--seed", type=int, default=7, help="procedural mock 模式随机种子")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--run-name", default=None, help="固定输出文件名前缀，便于批量实验")
    return p


def _empty_contact_metrics() -> dict[str, float | int | bool]:
    return {
        "contact_count": 0,
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
    }


def _make_driver(args: argparse.Namespace, glove_cfg: dict):
    channels = list(glove_cfg["channels"])
    hz = float(glove_cfg["sampling_hz"])
    if args.input == "dataset":
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

    driver = _make_driver(args, glove_cfg)
    calibrator = AdaptiveCalibrator(channels, epsilon=float(glove_cfg["calibration_epsilon"]))
    calibrator.fit(driver.calibration_samples())
    input_filter = MovingAverageFilter(channels, int(glove_cfg["moving_average_window"]))

    human = HumanHandModel()
    robot = RobotHandModel()
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
    retargeter = IntentDrivenRetargeter(human, robot, algo_cfg["optimizer"])
    contact_assist = TaskContactAssist(robot, algo_cfg.get("contact_assist", {}), dt)
    task_servo = ObjectAwareTaskServo(robot, algo_cfg.get("task_space_servo", {}))

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
            auto_reset_object=bool(sim_cfg["auto_reset_object"]),
            object_workspace_radius_m=float(sim_cfg["object_workspace_radius_m"]),
            object_min_z_m=float(sim_cfg["object_min_z_m"]),
            stability_orientation_deg=float(sim_cfg.get("stability_orientation_deg", 3.0)),
            camera_presets=dict(sim_cfg.get("camera_presets", {})),
            scene_config=sim_cfg,
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
        extra_columns=CONTACT_COLUMNS + SERVO_COLUMNS + EVALUATION_COLUMNS,
    )

    eval_cfg = dict(sim_cfg.get("evaluation", {}))
    evaluator = TaskEpisodeEvaluator(
        task_name=task_name,
        sampling_hz=hz,
        min_success_streak_frames=int(eval_cfg.get("min_success_streak_frames", 12)),
    )

    q_prev_opt = None
    q_prev_out = None
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
            sample = driver.sample()
            norm = calibrator.normalize(sample)
            norm = input_filter.apply(norm)
            q_h = human.angles_from_normalized(norm)
            features = extractor.extract(q_h)
            intent = recognizer.update(features)
            task_active = intent.value == active_intent

            if simulator is not None:
                rising = simulator.set_task_active(
                    task_active,
                    hold_when_inactive=bool(sim_cfg.get("hold_task_object_when_inactive", True)),
                )
                if rising:
                    last_contact = _empty_contact_metrics()
                    approach_fixture_released = False
                    contact_assist.reset()
                    task_servo.reset()

            opt_t0 = time.perf_counter()
            result = retargeter.solve(
                q_h,
                features.fingertip_positions,
                intent,
                q_prev_opt,
                dynamic=True,
            )
            optimizer_ms = (time.perf_counter() - opt_t0) * 1000.0
            q_prev_opt = result.q_target.copy()

            q_target = result.q_target
            servo_diag = ServoDiagnostics()
            if simulator is not None:
                if task_servo.enabled:
                    q_target, servo_diag = task_servo.apply(
                        q_target,
                        simulator=simulator,
                        task_name=task_name,
                        task_mode=task_mode,
                        intent_value=intent.value,
                        contact=last_contact,
                    )
                else:
                    assist_mode = "wrap" if task_mode in {"wrap", "sphere"} else task_mode
                    q_target = contact_assist.apply(
                        q_target,
                        task_name=assist_mode,
                        intent_value=intent.value,
                        contact=last_contact,
                    )

            q_filtered = lp.apply(q_target)
            q_out = rate_limit(q_filtered, q_prev_out, vmax, dt)
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

            logger.write(
                [
                    frame,
                    sample.timestamp,
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
                "input_source": args.input,
                "dataset": str(args.dataset) if args.input == "dataset" else None,
                "dataset_trial": int(args.dataset_trial) if args.input == "dataset" else None,
                "mock_seed": int(args.seed) if args.input == "mock" else None,
                "simulation": "mujoco" if simulator is not None else "none",
                "robot_model": "CH-M6_L",
                "robot_dof": robot.nominal_dof,
                "robot_actuators": robot.effective_actuators,
                "task_mode": task_mode,
                "active_intent": active_intent,
                "required_success_metric": required_success_metric,
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
