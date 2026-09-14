from __future__ import annotations

import argparse
from datetime import datetime
import math
import time
import numpy as np

from src.common import ROOT, load_yaml
from src.evaluation.logger import CSVRunLogger
from src.evaluation.metrics import joint_rmse, velocity_rms
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import IntentRecognizer
from src.glove.calibration import AdaptiveCalibrator
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Virtual dexterous hand v0.3.4 - object-aware task-space servo prototype")
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
        help="MuJoCo 自由相机初始视角预设；启动后仍可鼠标拖动；不填写时读取 configs/simulation.yaml",
    )
    p.add_argument(
        "--task",
        choices=["wrap", "pinch", "sphere"],
        default=None,
        help="测试物体：wrap=圆柱包络，pinch=薄块捏合，sphere=球体形状适应",
    )
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
        "stable_contact_proxy": False,
        "object_displacement_m": 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    glove_cfg = load_yaml("configs/glove.yaml")
    algo_cfg = load_yaml("configs/algorithm.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    channels = glove_cfg["channels"]
    hz = float(glove_cfg["sampling_hz"])
    dt = 1.0 / hz

    driver = MockGloveDriver(
        channels=channels,
        sampling_hz=hz,
        raw_min=float(glove_cfg["raw_min_default"]),
        raw_max=float(glove_cfg["raw_max_default"]),
    )
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

    task_name = args.task or str(sim_cfg.get("default_task", "wrap"))
    if use_mujoco:
        from src.simulation.mujoco_env import MujocoSimulator

        camera = args.camera or str(sim_cfg["default_camera"])
        task_spec = dict(sim_cfg.get("task_objects", {}).get(task_name, {}))
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
        )
        if args.render:
            print(f"[INFO] MuJoCo v0.3.4 Viewer 已启动；自由相机初始预设={camera}；任务={task_name}；任务空间伺服=ON")

    print(
        f"[INFO] Robot architecture: nominal DoF={robot.nominal_dof}, "
        f"effective actuators={robot.effective_actuators}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "outputs" / f"run_{stamp}.csv"
    logger = CSVRunLogger(
        log_path,
        robot.joint_order,
        extra_columns=CONTACT_COLUMNS + SERVO_COLUMNS,
    )

    q_prev_opt = None
    q_prev_out = None
    last_intent = None
    last_contact = _empty_contact_metrics()
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

            if simulator is not None:
                active_intent = "PINCH" if task_name == "pinch" else "WRAP"
                task_active = intent.value == active_intent
                rising = simulator.set_task_active(
                    task_active,
                    hold_when_inactive=bool(sim_cfg.get("hold_task_object_when_inactive", True)),
                )
                if rising:
                    last_contact = _empty_contact_metrics()
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
                        intent_value=intent.value,
                        contact=last_contact,
                    )
                else:
                    q_target = contact_assist.apply(
                        q_target,
                        task_name=task_name,
                        intent_value=intent.value,
                        contact=last_contact,
                    )

            q_filtered = lp.apply(q_target)
            q_out = rate_limit(q_filtered, q_prev_out, vmax, dt)
            if simulator is not None:
                q_actual = simulator.step(q_out)
                contact = simulator.contact_metrics()
                last_contact = contact
            else:
                q_actual = q_out
                contact = _empty_contact_metrics()

            latency_ms = (time.perf_counter() - pipeline_t0) * 1000.0
            rmse = joint_rmse(q_actual, result.q_base)
            vel = velocity_rms(q_actual, q_prev_out, dt)
            q_prev_out = q_actual.copy()

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
                    *[servo_diag.as_dict()[name] for name in SERVO_COLUMNS],
                    *q_actual.tolist(),
                ]
            )

            if intent != last_intent:
                extra = ""
                if simulator is not None:
                    extra = (
                        f" contacts={contact['contact_count']} sections={contact['contact_sections']} "
                        f"tip_ratio={float(contact['tip_contact_ratio']):.2f}"
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

    print(f"[DONE] 输出日志：{log_path}")


if __name__ == "__main__":
    main()
