import numpy as np

from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.task_space_servo import ObjectAwareTaskServo


class FakeSimulator:
    def __init__(self, robot: RobotHandModel):
        self.robot = robot
        self.q = np.zeros(len(robot.joint_order), dtype=float)

    def joint_positions(self):
        return self.q.copy()

    def fingertip_position(self, finger: str):
        # 所有指尖从原点开始，便于验证 correction 是否非零。
        return np.zeros(3, dtype=float)

    def task_contact_target(self, finger, task_name, clearance_m, vertical_margin_m):
        if finger == "thumb":
            return np.array([0.010, 0.000, 0.000], dtype=float)
        return np.array([0.000, -0.010, 0.000], dtype=float)

    def fingertip_jacobian(self, finger: str, joint_names):
        # thumb: 3 DoF 正交；非拇指：让协同方向对 Y 有作用。
        if finger == "thumb":
            return np.eye(3, dtype=float)
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.010, -0.008, -0.006],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )


def _contact():
    d = {}
    for f in ("thumb", "index", "middle", "ring", "little"):
        d[f"{f}_contact"] = False
        d[f"{f}_tip_contact"] = False
    return d


def test_task_space_servo_inactive_intent_does_not_modify_target():
    robot = RobotHandModel()
    servo = ObjectAwareTaskServo(robot, load_yaml("configs/algorithm.yaml")["task_space_servo"])
    sim = FakeSimulator(robot)
    q = np.zeros(len(robot.joint_order), dtype=float)
    out, diag = servo.apply(q, simulator=sim, task_name="pinch", intent_value="NEUTRAL", contact=_contact())
    assert np.allclose(out, q)
    assert not diag.active


def test_task_space_servo_pinch_corrects_thumb_and_index_only():
    robot = RobotHandModel()
    servo = ObjectAwareTaskServo(robot, load_yaml("configs/algorithm.yaml")["task_space_servo"])
    sim = FakeSimulator(robot)
    q = np.zeros(len(robot.joint_order), dtype=float)
    out, diag = servo.apply(q, simulator=sim, task_name="pinch", intent_value="PINCH", contact=_contact())
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    assert diag.active
    assert diag.corrected_digits == 2
    assert np.linalg.norm(out[[idx["thumb_mcp"], idx["thumb_pip"], idx["thumb_dip"]]]) > 0
    assert np.linalg.norm(out[[idx["index_mcp"], idx["index_pip"], idx["index_dip"]]]) > 0
    assert np.allclose(out[[idx["middle_mcp"], idx["middle_pip"], idx["middle_dip"]]], 0)


def test_task_space_servo_holds_digit_after_contact():
    robot = RobotHandModel()
    servo = ObjectAwareTaskServo(robot, load_yaml("configs/algorithm.yaml")["task_space_servo"])
    sim = FakeSimulator(robot)
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    sim.q[idx["thumb_mcp"]] = -0.15
    sim.q[idx["thumb_pip"]] = 0.20
    sim.q[idx["thumb_dip"]] = 0.10
    q_target = np.zeros(len(robot.joint_order), dtype=float)
    contact = _contact()
    contact["thumb_tip_contact"] = True
    out, _ = servo.apply(
        q_target,
        simulator=sim,
        task_name="pinch",
        intent_value="PINCH",
        contact=contact,
    )
    thumb_idx = [idx["thumb_mcp"], idx["thumb_pip"], idx["thumb_dip"]]
    assert np.allclose(out[thumb_idx], sim.q[thumb_idx])


def test_pinch_non_tip_contact_does_not_freeze_digit():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["task_space_servo"]
    servo = ObjectAwareTaskServo(robot, cfg)
    sim = FakeSimulator(robot)
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _contact()
    contact["thumb_contact"] = True
    contact["thumb_tip_contact"] = False
    out, diag = servo.apply(
        q, simulator=sim, task_name="pinch", intent_value="PINCH", contact=contact
    )
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    thumb_idx = [idx["thumb_mcp"], idx["thumb_pip"], idx["thumb_dip"]]
    assert diag.active
    assert np.linalg.norm(out[thumb_idx]) > 0


def test_pinch_tip_contact_still_freezes_digit():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["task_space_servo"]
    servo = ObjectAwareTaskServo(robot, cfg)
    sim = FakeSimulator(robot)
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    sim.q[idx["thumb_mcp"]] = -0.12
    sim.q[idx["thumb_pip"]] = 0.15
    sim.q[idx["thumb_dip"]] = 0.08
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _contact()
    contact["thumb_tip_contact"] = True
    out, _ = servo.apply(
        q, simulator=sim, task_name="pinch", intent_value="PINCH", contact=contact
    )
    thumb_idx = [idx["thumb_mcp"], idx["thumb_pip"], idx["thumb_dip"]]
    assert np.allclose(out[thumb_idx], sim.q[thumb_idx])


def test_task_specific_config_precedes_interaction_mode_fallback():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["task_space_servo"]
    servo = ObjectAwareTaskServo(robot, cfg)
    assert servo._task_cfg("card", "pinch")["target_offset_m"] == -0.002
    assert servo._task_cfg("unknown_pinch_object", "pinch")["target_offset_m"] == 0.006


def test_never_contact_policy_keeps_card_servo_active():
    robot = RobotHandModel()
    servo = ObjectAwareTaskServo(robot, load_yaml("configs/algorithm.yaml")["task_space_servo"])
    sim = FakeSimulator(robot)
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _contact()
    contact["thumb_tip_contact"] = True
    out, diag = servo.apply(
        q,
        simulator=sim,
        task_name="card",
        task_mode="pinch",
        intent_value="PINCH",
        contact=contact,
    )
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    thumb_idx = [idx["thumb_mcp"], idx["thumb_pip"], idx["thumb_dip"]]
    assert diag.active
    assert np.linalg.norm(out[thumb_idx]) > 0
