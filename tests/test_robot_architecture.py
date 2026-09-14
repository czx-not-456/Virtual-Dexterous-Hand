import numpy as np

from src.hand_model.robot_hand import RobotHandModel


def test_robot_exposes_15dof_7actuator_architecture():
    robot = RobotHandModel()
    assert robot.nominal_dof == 15
    assert robot.effective_actuators == 7
    assert len(robot.actuator_order) == 7
    assert len(robot.joint_order) == 15


def test_joint_targets_project_to_seven_actuators():
    robot = RobotHandModel()
    q = robot.ranges[:, 0] + 0.5 * (robot.ranges[:, 1] - robot.ranges[:, 0])
    u = robot.actuator_targets_from_joint_targets(q)
    r = robot.actuator_ctrl_ranges()
    assert u.shape == (7,)
    assert r.shape == (7, 2)
    assert np.all(u >= r[:, 0] - 1e-9)
    assert np.all(u <= r[:, 1] + 1e-9)


def test_thumb_anatomical_aliases_are_explicit():
    robot = RobotHandModel()
    assert robot.anatomical_aliases["thumb_mcp"] == "thumb_cmc"
    assert robot.anatomical_aliases["thumb_pip"] == "thumb_mcp"
    assert robot.anatomical_aliases["thumb_dip"] == "thumb_ip"


def test_thumb_positive_flexion_changes_tip_toward_palm_side():
    robot = RobotHandModel()
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    q0 = np.zeros(len(robot.joint_order), dtype=float)
    q1 = q0.copy()
    q1[idx["thumb_mcp"]] = 0.40
    q1[idx["thumb_pip"]] = 0.30
    q1[idx["thumb_dip"]] = 0.20
    p0 = robot.fingertips(q0)["thumb"]
    p1 = robot.fingertips(q1)["thumb"]
    assert np.linalg.norm(p1 - p0) > 0.01
