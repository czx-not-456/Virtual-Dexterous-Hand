"""Paired CH-M6 MuJoCo trials on the delivered held-out glove captures.

The delivery contains personalized predictions for six validation frames, but
does not contain the model checkpoint. Each task therefore replays its open and
active held-out capture for a configurable number of control frames.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.evaluation.task_evaluator import TaskEpisodeEvaluator
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import Intent
from src.glove.adapter import GloveAdapter
from src.glove.captured_samples import load_captured_samples, resolve_delivery_path
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.processing.filters import LowPassVectorFilter, rate_limit
from src.retargeting.contact_assist import TaskContactAssist
from src.retargeting.optimizer import IntentDrivenRetargeter
from src.retargeting.task_space_servo import ObjectAwareTaskServo, ServoDiagnostics
from src.simulation.mujoco_env import MujocoSimulator


TASK_TARGETS = {
    "pinch": (80, 23, 43, 0, 0, 0),
    "wrap": (70, 30, 80, 80, 80, 80),
}
OPEN_TARGET = (0, 0, 0, 0, 0, 0)
FRAME_COLUMNS = (
    "task", "mapping", "frame", "phase", "sample_index", "pose",
    "active", "fingertip_distance_error_m", "human_tip_tracking_error_m", "contact", "stable_contact",
    "object_displacement_m", "joint_tracking_rmse_rad", "pipeline_latency_ms",
    "optimizer_ms", "thumb_tip_contact", "index_tip_contact",
)


def _validation_cases(samples_path: Path, manifest_path: Path):
    samples = load_captured_samples(samples_path)
    manifest_file = resolve_delivery_path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    validation = manifest.get("adaptation_result", {}).get("pose_validation", [])
    held_out = {(s.pose, s.pose_frame_index): s for s in samples}
    cases = {}
    for row in validation:
        sample = held_out.get((row["pose"], 3))
        if sample is None or sample.targets_deg is None:
            raise ValueError(f"missing held-out capture for {row['pose']!r}")
        target = np.asarray(row["target_deg"], dtype=float)
        prediction = np.asarray(row["final_prediction_deg"], dtype=float)
        if target.shape != (6,) or prediction.shape != (6,) or not np.allclose(sample.targets_deg, target):
            raise ValueError(f"manifest/capture validation mismatch for {row['pose']!r}")
        cases[tuple(target.tolist())] = (sample, prediction)
    required = [OPEN_TARGET, *TASK_TARGETS.values()]
    if any(tuple(map(float, target)) not in cases for target in required):
        raise ValueError("manifest lacks an open, index-pinch, or full-wrap validation pose")
    return cases, manifest_file


def _joint_target(outputs_deg, output_names, bindings, robot):
    values = dict(zip(output_names, np.asarray(outputs_deg, dtype=float)))
    target = np.radians([values[bindings[name]] for name in robot.joint_order])
    return np.clip(target, robot.ranges[:, 0], robot.ranges[:, 1])


def _tip_error(robot, actual, human_tips, scale, intent):
    fingers = ("thumb", "index") if intent == Intent.PINCH else ("thumb", "index", "middle", "ring", "little")
    tips = robot.fingertips(actual)
    return float(np.mean([np.linalg.norm(tips[name] - scale * human_tips[name]) for name in fingers]))


def _object_tip_error(simulator, task, algo_cfg):
    cfg = algo_cfg.get("task_space_servo", {}).get(task, {})
    clearance = float(cfg.get("target_offset_m", cfg.get("clearance_m", 0.007)))
    margin = float(cfg.get("vertical_margin_m", 0.01))
    fingers = ("thumb", "index") if task == "pinch" else ("thumb", "index", "middle", "ring", "little")
    return float(np.mean([
        np.linalg.norm(
            simulator.fingertip_position(name)
            - simulator.task_contact_target(name, task, clearance_m=clearance, vertical_margin_m=margin)
        ) for name in fingers
    ]))


def run_trial(*, task, mapping, open_case, active_case, adapter, robot, human,
              algo_cfg, sim_cfg, hz, steps, warmup_frames):
    spec = dict(sim_cfg["task_objects"][task])
    intent = Intent.PINCH if task == "pinch" else Intent.WRAP
    dt = 1.0 / hz
    simulator = MujocoSimulator(
        robot, model_path=str(sim_cfg["model_path"]), render=False,
        task_object_body=str(sim_cfg["task_object_body"]), task_name=task,
        task_spec=spec, auto_reset_object=bool(sim_cfg["auto_reset_object"]),
        object_workspace_radius_m=float(sim_cfg["object_workspace_radius_m"]),
        object_min_z_m=float(sim_cfg["object_min_z_m"]),
        stability_orientation_deg=float(sim_cfg.get("stability_orientation_deg", 3.0)),
        scene_config=sim_cfg, control_dt_s=dt,
    )
    extractor = FeatureExtractor(human)
    solver = IntentDrivenRetargeter(human, robot, algo_cfg["optimizer"])
    assist = TaskContactAssist(robot, algo_cfg.get("contact_assist", {}), dt)
    servo = ObjectAwareTaskServo(robot, algo_cfg.get("task_space_servo", {}))
    lowpass = LowPassVectorFilter(float(algo_cfg["output_filter"]["alpha"]))
    velocity = np.full(len(robot.joint_order), math.radians(float(algo_cfg["output_filter"]["max_velocity_deg_s"])))
    evaluation = TaskEpisodeEvaluator(
        task, hz, int(sim_cfg.get("evaluation", {}).get("min_success_streak_frames", 12))
    )
    previous_opt = previous_out = None
    last_contact = {}
    released = False
    rows = []
    try:
        for frame_index in range(steps):
            active = frame_index >= warmup_frames
            sample, personalized_deg = active_case if active else open_case
            start = time.perf_counter()
            hand_frame = adapter.adapt(sample, timestamp=frame_index * dt)
            if mapping == "personalized":
                target = _joint_target(personalized_deg, adapter.base_mapping.output_names,
                                       adapter.output_bindings, robot)
                hand_frame = replace(hand_frame, robot_joint_targets_rad=target)
            features = extractor.extract(hand_frame.joint_angles_rad)
            simulator.set_task_active(active, hold_when_inactive=bool(sim_cfg.get("hold_task_object_when_inactive", True)))
            if frame_index == warmup_frames:
                assist.reset()
                servo.reset()
                last_contact = {}
            solve_start = time.perf_counter()
            result = solver.solve(
                hand_frame.joint_angles_rad, features.fingertip_positions,
                intent if active else Intent.NEUTRAL, previous_opt,
                dynamic=True, q_base=hand_frame.robot_joint_targets_rad,
            )
            optimizer_ms = (time.perf_counter() - solve_start) * 1000.0
            previous_opt = result.q_target.copy()
            target = result.q_target
            diagnostics = ServoDiagnostics()
            if servo.enabled:
                target, diagnostics = servo.apply(
                    target, simulator=simulator, task_name=task, task_mode=task,
                    intent_value=intent.value if active else Intent.NEUTRAL.value,
                    contact=last_contact,
                )
            else:
                target = assist.apply(
                    target, task_name=task,
                    intent_value=intent.value if active else Intent.NEUTRAL.value,
                    contact=last_contact,
                )
            output = rate_limit(lowpass.apply(target), previous_out, velocity, dt)
            if task == "pinch" and last_contact.get("thumb_tip_contact") and last_contact.get("index_tip_contact"):
                released = True
            held = task == "pinch" and active and not released
            actual = simulator.step(output, hold_object=held)
            contact = simulator.contact_metrics()
            if held:
                contact["stable_contact_proxy"] = False
            last_contact = contact
            latency_ms = (time.perf_counter() - start) * 1000.0
            rmse_rad = float(np.sqrt(np.mean((actual - result.q_base) ** 2)))
            velocity_rms = 0.0 if previous_out is None else float(np.sqrt(np.mean(((actual - previous_out) / dt) ** 2)))
            previous_out = actual.copy()
            evaluation.update(
                frame=frame_index, task_active=active, contact=contact,
                latency_ms=latency_ms, optimizer_ms=optimizer_ms,
                joint_rmse_rad=rmse_rad, velocity_rms_rad_s=velocity_rms,
                servo_mean_error_m=float(diagnostics.as_dict()["servo_mean_error_m"]),
            )
            rows.append({
                "task": task, "mapping": mapping, "frame": frame_index,
                "phase": "active" if active else "warmup", "sample_index": sample.sample_index,
                "pose": sample.pose, "active": active,
                "fingertip_distance_error_m": _object_tip_error(simulator, task, algo_cfg),
                "human_tip_tracking_error_m": _tip_error(robot, actual, features.fingertip_positions, solver.task_scale, intent),
                "contact": bool(contact.get("task_contact_proxy", False)),
                "stable_contact": bool(contact.get("stable_contact_proxy", False)),
                "object_displacement_m": float(contact.get("object_displacement_m", 0.0)),
                "joint_tracking_rmse_rad": rmse_rad,
                "pipeline_latency_ms": latency_ms, "optimizer_ms": optimizer_ms,
                "thumb_tip_contact": bool(contact.get("thumb_tip_contact", False)),
                "index_tip_contact": bool(contact.get("index_tip_contact", False)),
            })
    finally:
        simulator.close()
    summary = evaluation.summary()
    active_rows = [row for row in rows if row["active"]]
    errors = [row["fingertip_distance_error_m"] for row in active_rows]
    tracking = [row["joint_tracking_rmse_rad"] for row in active_rows]
    sample, personalized_deg = active_case
    base_deg = np.asarray(list(adapter.base_mapping.predict_deg(sample.sensors_deg).values()))
    prediction = base_deg if mapping == "base" else personalized_deg
    return rows, {
        "task": task, "mapping": mapping, "frames": steps, "active_frames": len(active_rows),
        "input_sample_index": sample.sample_index, "input_pose": sample.pose,
        "fingertip_distance_error_mean_m": float(np.mean(errors)),
        "human_tip_tracking_error_mean_m": float(np.mean([row["human_tip_tracking_error_m"] for row in active_rows])),
        "contact_success_rate": summary["contact_frame_ratio"],
        "stable_contact_ratio": summary["stable_frame_ratio"],
        "contact_success_proxy": summary["contact_success_proxy"],
        "stable_success_proxy": summary["stable_success_proxy"],
        "max_object_displacement_m": summary["max_object_displacement_m"],
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking)))),
        "mapping_rmse_deg": float(np.sqrt(np.mean((prediction - sample.targets_deg) ** 2))),
        "mean_pipeline_latency_ms": summary["mean_pipeline_latency_ms"],
        "p95_pipeline_latency_ms": summary["p95_pipeline_latency_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "数据手套数据" / "captured_samples.xlsx")
    parser.add_argument("--base-mapping", type=Path, default=ROOT / "数据手套数据" / "base_mapping.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "数据手套数据" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "generated" / "glove_tasks")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    if args.steps <= 0 or not 0 <= args.warmup_frames < args.steps:
        parser.error("require steps > 0 and 0 <= warmup-frames < steps")
    cases, manifest_path = _validation_cases(args.samples, args.manifest)
    glove_cfg = load_yaml("configs/glove.yaml")
    algo_cfg = load_yaml("configs/algorithm.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    human, robot = HumanHandModel(), RobotHandModel()
    adapter = GloveAdapter(
        human, robot, base_mapping_path=args.base_mapping, manifest_path=manifest_path,
        output_bindings=glove_cfg["real"]["output_bindings"],
    )
    all_rows, summaries = [], []
    for task, target in TASK_TARGETS.items():
        for mapping in ("base", "personalized"):
            print(f"[RUN] {task} / {mapping}", flush=True)
            rows, summary = run_trial(
                task=task, mapping=mapping, open_case=cases[tuple(map(float, OPEN_TARGET))],
                active_case=cases[tuple(map(float, target))], adapter=adapter,
                robot=robot, human=human, algo_cfg=algo_cfg, sim_cfg=sim_cfg,
                hz=float(glove_cfg["sampling_hz"]), steps=args.steps,
                warmup_frames=args.warmup_frames,
            )
            all_rows.extend(rows)
            summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    fields = list(summaries[0])
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    result = {
        "protocol": "held_out_fourth_frame_static_pose_replay",
        "note": "Personalized outputs are manifest predictions for held-out captures; no checkpoint or live personalized inference was delivered. Repeated frames are not independent measurements. Success is a simulation contact proxy, not physical grasp success.",
        "samples": str(resolve_delivery_path(args.samples)),
        "base_mapping": str(adapter.base_mapping.path), "manifest": str(manifest_path),
        "steps_per_trial": args.steps, "warmup_frames": args.warmup_frames,
        "sampling_hz": float(glove_cfg["sampling_hz"]),
        "trials": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {args.output_dir / 'summary.csv'}")
    print(f"[DONE] {args.output_dir / 'summary.json'}")
    print(f"[DONE] {args.output_dir / 'frames.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
