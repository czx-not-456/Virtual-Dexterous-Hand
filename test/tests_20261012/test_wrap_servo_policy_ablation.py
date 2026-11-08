"""Regression tests for the opt-in WRAP four-way controller ablation."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.task_space_servo import ObjectAwareTaskServo
from src.main import build_parser
from experiments.benchmark_wrap_servo_policy import summarize_run, aggregate


class FakeSimulator:
    def __init__(self, robot):
        self.robot = robot
        self.q = np.zeros(len(robot.joint_order), dtype=float)
        self.targets = {f: np.array([0.002, 0.001, 0.], dtype=float) for f in robot.finger_joints}

    def joint_positions(self):
        return self.q.copy()

    def fingertip_position(self, finger):
        return np.zeros(3, dtype=float)

    def task_contact_target(self, finger, task_name, clearance_m, vertical_margin_m):
        return self.targets[finger].copy()

    def fingertip_jacobian(self, finger, joint_names):
        J = np.zeros((3, len(joint_names)), dtype=float)
        J[0, :] = 0.02
        J[1, :] = 0.01
        return J


def contact(thumb=False, index=False):
    result = {}
    for finger in ("thumb", "index", "middle", "ring", "little"):
        hit = (finger == "thumb" and thumb) or (finger == "index" and index)
        result[f"raw_{finger}_contact"] = hit
        result[f"{finger}_contact"] = hit
        result[f"{finger}_tip_contact"] = hit
    return result


def mk(policy):
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["task_space_servo"]
    return robot, ObjectAwareTaskServo(robot, cfg, wrap_policy=policy)


def apply(servo, robot, sim, contact_info, q=None, task="wrap"):
    return servo.apply(np.zeros(len(robot.joint_order)) if q is None else q,
                       simulator=sim, task_name=task,
                       intent_value="WRAP" if task == "wrap" else "PINCH",
                       contact=contact_info, task_mode=task)


def digit(q, robot, finger):
    return q[[robot.joint_order.index(j) for j in robot.joints_for_finger(finger)]]


def test_original_default_unchanged_and_pinch_unaffected():
    robot, normal = mk("original")
    _, baseline = mk("original")
    _, altered = mk("coordinated")
    sim = FakeSimulator(robot)
    for c in (contact(), contact(True, True), contact(False, True)):
        q1, _ = apply(normal, robot, sim, c)
        q2, _ = apply(baseline, robot, sim, c)
        assert np.array_equal(q1, q2)
    q1, _ = normal.apply(np.zeros(len(robot.joint_order)), simulator=sim,
                         task_name="pinch", intent_value="PINCH", contact=contact())
    q2, _ = altered.apply(np.zeros(len(robot.joint_order)), simulator=sim,
                          task_name="pinch", intent_value="PINCH", contact=contact())
    assert np.array_equal(q1, q2)


def test_low_gain_reduces_uncontacted_digit_step():
    robot, original = mk("original")
    _, low = mk("low_gain")
    sim = FakeSimulator(robot)
    qold, _ = apply(original, robot, sim, contact())
    qlow, _ = apply(low, robot, sim, contact())
    assert 0 < np.linalg.norm(qlow) < np.linalg.norm(qold)
    assert np.all(np.isfinite(qlow))


def test_recovery_requires_previous_physical_thumb_and_other_contact():
    robot, servo = mk("thumb_recovery")
    sim = FakeSimulator(robot)
    q, d = apply(servo, robot, sim, contact(False, True))
    assert not d.thumb_recovery_active and not d.thumb_loss_event
    _, d = apply(servo, robot, sim, contact(True, True))
    assert not d.thumb_recovery_active
    q, d = apply(servo, robot, sim, contact(False, True))
    assert d.thumb_recovery_active and d.thumb_loss_event
    assert np.max(np.abs(digit(q, robot, "thumb") - digit(sim.q, robot, "thumb"))) <= np.deg2rad(.8) + 1e-12
    q, d = apply(servo, robot, sim, contact(False, True))
    assert d.thumb_recovery_active and not d.thumb_loss_event
    servo.reset()
    _, d = apply(servo, robot, sim, contact(False, True))
    assert not d.thumb_recovery_active


def test_recovery_target_does_not_chase_large_object_motion():
    robot, servo = mk("thumb_recovery")
    sim = FakeSimulator(robot)
    apply(servo, robot, sim, contact(True, True))
    anchor = servo._thumb_last_contact_target.copy()
    sim.targets["thumb"] += np.array([.06, 0., 0.])
    for i in range(8):
        _, d = apply(servo, robot, sim, contact(False, True))
        assert d.thumb_recovery_active
        assert np.linalg.norm(servo._thumb_recovery_target - anchor) <= .003 + 1e-9


def test_coordinated_limits_noncontacting_fingers_and_recovers_thumb():
    robot, recovery = mk("thumb_recovery")
    _, coordinated = mk("coordinated")
    sim = FakeSimulator(robot)
    qnom = np.full(len(robot.joint_order), 0.35)
    apply(recovery, robot, sim, contact(True, True), qnom)
    apply(coordinated, robot, sim, contact(True, True), qnom)
    a, da = apply(recovery, robot, sim, contact(False, True), qnom)
    b, db = apply(coordinated, robot, sim, contact(False, True), qnom)
    assert da.thumb_recovery_active and not da.wrap_coordination_active
    assert db.thumb_recovery_active and db.wrap_coordination_active
    assert np.allclose(digit(b, robot, "index"), digit(sim.q, robot, "index"))
    assert np.max(np.abs(digit(b, robot, "middle") - digit(sim.q, robot, "middle"))) <= np.deg2rad(.2) + 1e-12
    assert np.linalg.norm(digit(b, robot, "middle")) < np.linalg.norm(digit(a, robot, "middle"))
    assert np.allclose(digit(b, robot, "thumb"), digit(a, robot, "thumb"))


def test_recovery_has_finite_timeout():
    robot, servo = mk("thumb_recovery")
    sim = FakeSimulator(robot)
    apply(servo, robot, sim, contact(True, True))
    states = [apply(servo, robot, sim, contact(False, True))[1].thumb_recovery_active
              for _ in range(20)]
    assert states[:18] == [True] * 18
    assert states[18:] == [False, False]
    apply(servo, robot, sim, contact(True, True))
    assert apply(servo, robot, sim, contact(False, True))[1].thumb_loss_event


def test_cli_guards_and_diagnostic_fields():
    parser = build_parser()
    args = parser.parse_args(["--sim", "mujoco", "--task", "wrap", "--wrap-servo-policy", "coordinated"])
    assert args.wrap_servo_policy == "coordinated"
    assert not args.no_auto_reset_object
    assert not args.no_inactive_object_reset
    with pytest.raises(SystemExit):
        parser.parse_args(["--wrap-servo-policy", "unknown"])
    robot, servo = mk("coordinated")
    diag = apply(servo, robot, FakeSimulator(robot), contact())[1]
    assert "servo_thumb_recovery_active" in diag.as_dict()


def test_report_summary_is_grounded_in_written_runs(tmp_path):
    csv_file = tmp_path / "wrap_trial0_coordinated.csv"
    summary_file = tmp_path / "wrap_trial0_coordinated_summary.json"
    summary_file.write_text(json.dumps({
        "wrap_servo_policy": "coordinated", "dataset_trial": 0, "simulation": "mujoco",
        "task_mode": "wrap", "contact_frame_ratio": .5, "stable_frame_ratio": .2,
        "stable_success_proxy": False, "ever_hold_stable_proxy": True,
        "max_stable_success_streak_frames": 13,
        "hold_max_relative_translation_span_m": .009,
        "hold_max_relative_rotation_span_deg": 1.5,
    }), encoding="utf-8")
    rows = [
        {"intent": "WRAP", "task_active": "True", "frame": i,
         "raw_thumb_contact": str(on), "filtered_task_contact_proxy": str(on),
         "object_displacement_m": ".005", "servo_mean_error_m": ".002",
         "servo_thumb_recovery_active": str(not on),
         "servo_wrap_coordination_active": str(not on), "object_reset_this_step": "False"}
        for i, on in enumerate((True, True, False, False, True))
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = summarize_run(csv_file, summary_file, trial=0, policy="coordinated")
    assert result["thumb_loss_events"] == 1
    assert result["control_switches"] == 2
    assert result["max_object_displacement_mm"] == 5.
    assert result["recovery_frames"] == 2
    with pytest.raises(ValueError):
        summarize_run(csv_file, summary_file, trial=1, policy="coordinated")
    output = aggregate([result], [0], ["coordinated"])
    assert output[0]["final_stable_count"] == 0
