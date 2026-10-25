"""Replay teacher calibration pose clips using recorded within-pose timestamps.

This evaluates the existing base mapping. Pose boundaries are discontinuities;
no personalized checkpoint is loaded and no free-form motion is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
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
from src.glove.captured_samples import load_captured_samples
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.processing.filters import LowPassVectorFilter, rate_limit
from src.retargeting.contact_assist import TaskContactAssist
from src.retargeting.optimizer import IntentDrivenRetargeter
from src.retargeting.task_space_servo import ObjectAwareTaskServo, ServoDiagnostics
from src.simulation.mujoco_env import MujocoSimulator
from src.simulation.teacher_chm6_control import TeacherCHM6ControlAdapter


def aggregate(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))} if len(a) else None


def pose_clip(samples, name):
    clip = [s for s in samples if s.pose == name]
    if not clip or any(b.elapsed_ms <= a.elapsed_ms for a, b in zip(clip, clip[1:])):
        raise ValueError(f"missing or nonmonotonic pose clip: {name}")
    return clip


def evaluate_mapping(samples, adapter, control_adapter, robot):
    base_errors, tip_errors, smoothness = [], [], []
    previous = {}
    by_pose = {}
    for sample in samples:
        started = time.perf_counter()
        base = np.asarray(list(adapter.base_mapping.predict_deg(sample.sensors_deg).values()), dtype=float)
        mapping_ms = (time.perf_counter() - started) * 1000
        target = np.asarray(sample.targets_deg, dtype=float)
        if target.shape != (6,):
            raise ValueError("six target degrees required")
        base_errors.extend((base - target).tolist())
        pred_q = control_adapter.expand_degrees(base)
        target_q = control_adapter.expand_degrees(target)
        pred_tips, target_tips = robot.fingertips(pred_q), robot.fingertips(target_q)
        tip_errors.extend(float(np.linalg.norm(pred_tips[f] - target_tips[f])) for f in pred_tips)
        if sample.pose in previous:
            earlier, prev = previous[sample.pose]
            dt = (sample.elapsed_ms - earlier) / 1000
            smoothness.append(float(np.sqrt(np.mean(((base - prev) / dt) ** 2))))
        previous[sample.pose] = (sample.elapsed_ms, base)
        by_pose.setdefault(sample.pose, {"n": 0, "mapping_ms": []})
        by_pose[sample.pose]["n"] += 1
        by_pose[sample.pose]["mapping_ms"].append(mapping_ms)
    return {
        "frames": len(samples), "base_angle_rmse_deg": float(np.sqrt(np.mean(np.square(base_errors)))),
        "base_tip_error_mean_mm": float(np.mean(tip_errors) * 1000),
        "base_output_speed_rms_deg_s": float(np.sqrt(np.mean(np.square(smoothness)))),
        "mapping_latency_ms": aggregate([v for pose in by_pose.values() for v in pose["mapping_ms"]]),
        "personalized": "blocked: no verified forward implementation; checkpoint not deserialized",
    }


def run_task(task, active_pose, samples, adapter, robot, human, algo, sim_cfg):
    sequence = pose_clip(samples, "张开") + pose_clip(samples, active_pose)
    steps = [(b.elapsed_ms - a.elapsed_ms) / 1000 for a, b in zip(sequence, sequence[1:])
             if a.pose == b.pose]
    median_dt = statistics.median(steps)
    hz = 1 / median_dt
    intent = Intent.PINCH if task == "pinch" else Intent.WRAP
    sim = MujocoSimulator(
        robot, model_path=str(sim_cfg["model_path"]), task_object_body=str(sim_cfg["task_object_body"]),
        task_name=task, task_spec=dict(sim_cfg["task_objects"][task]),
        auto_reset_object=bool(sim_cfg["auto_reset_object"]),
        object_workspace_radius_m=float(sim_cfg["object_workspace_radius_m"]),
        object_min_z_m=float(sim_cfg["object_min_z_m"]),
        stability_orientation_deg=float(sim_cfg["stability_orientation_deg"]),
        scene_config=sim_cfg, control_dt_s=median_dt,
    )
    solver = IntentDrivenRetargeter(human, robot, algo["optimizer"])
    extractor = FeatureExtractor(human)
    servo = ObjectAwareTaskServo(robot, algo.get("task_space_servo", {}))
    assist = TaskContactAssist(robot, algo.get("contact_assist", {}), median_dt)
    lowpass = LowPassVectorFilter(float(algo["output_filter"]["alpha"]))
    vmax = np.full(11, math.radians(float(algo["output_filter"]["max_velocity_deg_s"])))
    evaluator = TaskEpisodeEvaluator(task, hz, int(sim_cfg["evaluation"]["min_success_streak_frames"]))
    rows = []
    previous_opt = previous_out = None
    last_contact = {}
    released = False
    try:
        for frame, sample in enumerate(sequence):
            active = sample.pose == active_pose
            clip_first = frame == 0 or sample.pose != sequence[frame - 1].pose
            dt = median_dt if clip_first else (sample.elapsed_ms - sequence[frame - 1].elapsed_ms) / 1000
            sim.control_dt_s = dt
            started = time.perf_counter()
            hand = adapter.adapt(sample, timestamp=sample.elapsed_ms / 1000)
            features = extractor.extract(hand.joint_angles_rad)
            processing_ms = (time.perf_counter() - started) * 1000
            sim.set_task_active(active, hold_when_inactive=bool(sim_cfg.get("hold_task_object_when_inactive", True)))
            if clip_first and active:
                servo.reset()
                assist.reset()
                last_contact = {}
            began = time.perf_counter()
            solved = solver.solve(hand.joint_angles_rad, features.fingertip_positions,
                                  intent if active else Intent.NEUTRAL, previous_opt,
                                  dynamic=True, q_base=hand.robot_joint_targets_rad)
            optimizer_ms = (time.perf_counter() - began) * 1000
            previous_opt = solved.q_target.copy()
            target = solved.q_target
            diagnostics = ServoDiagnostics()
            if servo.enabled:
                target, diagnostics = servo.apply(target, simulator=sim, task_name=task,
                                                 task_mode=task, intent_value=intent.value if active else "NEUTRAL",
                                                 contact=last_contact)
            else:
                target = assist.apply(target, task_name=task,
                                      intent_value=intent.value if active else "NEUTRAL", contact=last_contact)
            output = rate_limit(lowpass.apply(target), previous_out, vmax, dt)
            if task == "pinch" and last_contact.get("thumb_tip_contact") and last_contact.get("index_tip_contact"):
                released = True
            fixture_held = task == "pinch" and active and not released
            began = time.perf_counter()
            actual = sim.step(output, hold_object=fixture_held)
            simulation_ms = (time.perf_counter() - began) * 1000
            contact = sim.contact_metrics()
            if fixture_held:
                contact["stable_contact_proxy"] = False
            last_contact = contact
            elapsed_ms = (time.perf_counter() - started) * 1000
            evaluator.update(frame=frame, task_active=active, contact=contact, latency_ms=elapsed_ms,
                             optimizer_ms=optimizer_ms,
                             joint_rmse_rad=float(np.sqrt(np.mean((actual - solved.q_base) ** 2))),
                             velocity_rms_rad_s=0 if previous_out is None else
                             float(np.sqrt(np.mean(((actual - previous_out) / dt) ** 2))),
                             servo_mean_error_m=float(diagnostics.as_dict()["servo_mean_error_m"]))
            previous_out = actual.copy()
            rows.append({"frame": frame, "pose": sample.pose, "sample_index": sample.sample_index,
                         "source_elapsed_ms": sample.elapsed_ms, "clip_boundary": clip_first,
                         "dt_s": dt, "active": active, "fixture_held": fixture_held,
                         "contact": bool(contact.get("task_contact_proxy")),
                         "stable": bool(contact.get("stable_contact_proxy")),
                         "processing_ms": processing_ms, "optimizer_ms": optimizer_ms,
                         "simulation_ms": simulation_ms, "pipeline_ms": elapsed_ms})
    finally:
        sim.close()
    summary = evaluator.summary()
    active_rows = [row for row in rows if row["active"]]
    return rows, {
        "task": task, "source": "teacher_calibration_pose_clips", "frames": len(rows),
        "active_frames": len(active_rows), "median_source_interval_ms": median_dt * 1000,
        "processing_latency_ms": aggregate([row["processing_ms"] for row in rows]),
        "optimizer_latency_ms": aggregate([row["optimizer_ms"] for row in rows]),
        "simulation_latency_ms": aggregate([row["simulation_ms"] for row in rows]),
        "pipeline_latency_ms": aggregate([row["pipeline_ms"] for row in rows]),
        "fixture_contact_frames": sum(row["contact"] and row["fixture_held"] for row in active_rows),
        "free_contact_frames": sum(row["contact"] and not row["fixture_held"] for row in active_rows),
        "free_stable_frames": sum(row["stable"] and not row["fixture_held"] for row in active_rows),
        "contact_success_proxy": summary["contact_success_proxy"],
        "stable_success_proxy": summary["stable_success_proxy"],
        "limitation": "Two separate calibration pose clips; the transition was not captured and source elapsed_ms resets at boundary.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "数据手套数据/records/chm6_calibrations/20260917_102405_G15RB010901_right")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
    delivery = ROOT / "数据手套数据"
    samples = load_captured_samples(delivery / manifest["data"]["captured_samples"])
    human, robot = HumanHandModel(), RobotHandModel()
    adapter = GloveAdapter(human, robot,
                           base_mapping_path=delivery / manifest["base_mapping"]["path"],
                           manifest_path=args.snapshot / "manifest.json",
                           output_bindings=load_yaml("configs/glove.yaml")["real"]["output_bindings"])
    control = TeacherCHM6ControlAdapter(robot, delivery / "mod/joint_mapping.json")
    mapping = evaluate_mapping(samples, adapter, control, robot)
    algo = load_yaml("configs/algorithm.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task, pose in (("pinch", "对食指"), ("wrap", "握拳")):
        print(f"[RUN] {task}: teacher calibration clips, base mapping", flush=True)
        rows, summary = run_task(task, pose, samples, adapter, robot, human, algo, sim_cfg)
        tasks.append(summary)
        with (args.output_dir / f"{task}_frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {"snapshot": args.snapshot.name, "model_inference": "blocked_unverified_forward",
              "mapping": mapping, "tasks": tasks}
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "frames", "active_frames", "median_source_interval_ms",
                                                   "fixture_contact_frames", "free_contact_frames", "free_stable_frames",
                                                   "contact_success_proxy", "stable_success_proxy", "processing_mean_ms",
                                                   "optimizer_mean_ms", "simulation_mean_ms", "pipeline_mean_ms", "pipeline_p95_ms"])
        writer.writeheader()
        for row in tasks:
            writer.writerow({"task": row["task"], "frames": row["frames"], "active_frames": row["active_frames"],
                             "median_source_interval_ms": row["median_source_interval_ms"],
                             "fixture_contact_frames": row["fixture_contact_frames"],
                             "free_contact_frames": row["free_contact_frames"], "free_stable_frames": row["free_stable_frames"],
                             "contact_success_proxy": row["contact_success_proxy"],
                             "stable_success_proxy": row["stable_success_proxy"],
                             "processing_mean_ms": row["processing_latency_ms"]["mean"],
                             "optimizer_mean_ms": row["optimizer_latency_ms"]["mean"],
                             "simulation_mean_ms": row["simulation_latency_ms"]["mean"],
                             "pipeline_mean_ms": row["pipeline_latency_ms"]["mean"],
                             "pipeline_p95_ms": row["pipeline_latency_ms"]["p95"]})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
