import numpy as np

from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.contact_assist import TaskContactAssist
from src.common import load_yaml


def _empty_contact():
    return {
        "thumb_contact": False,
        "index_contact": False,
        "middle_contact": False,
        "ring_contact": False,
        "little_contact": False,
        "thumb_tip_contact": False,
        "index_tip_contact": False,
    }


def test_pinch_contact_assist_closes_thumb_and_index_only():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["contact_assist"]
    assist = TaskContactAssist(robot, cfg, 1 / 60)
    q = np.zeros(len(robot.joint_order), dtype=float)
    out = assist.apply(q, task_name="pinch", intent_value="PINCH", contact=_empty_contact())
    idx = {n: i for i, n in enumerate(robot.joint_order)}
    assert out[idx["thumb_pip"]] > 0
    assert out[idx["index_pip"]] > 0
    assert out[idx["middle_pip"]] == 0


def test_pinch_contact_stops_contacted_thumb_but_keeps_index_closing():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["contact_assist"]
    assist = TaskContactAssist(robot, cfg, 1 / 60)
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _empty_contact()
    contact["thumb_tip_contact"] = True
    out = assist.apply(q, task_name="pinch", intent_value="PINCH", contact=contact)
    idx = {n: i for i, n in enumerate(robot.joint_order)}
    assert out[idx["thumb_pip"]] == 0
    assert out[idx["index_pip"]] > 0


def test_wrap_assist_stops_each_digit_independently():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["contact_assist"]
    assist = TaskContactAssist(robot, cfg, 1 / 60)
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _empty_contact()
    contact["index_contact"] = True
    out = assist.apply(q, task_name="wrap", intent_value="WRAP", contact=contact)
    idx = {n: i for i, n in enumerate(robot.joint_order)}
    assert out[idx["index_pip"]] == 0
    assert out[idx["middle_pip"]] > 0
    assert out[idx["thumb_pip"]] > 0


def test_assist_releases_after_intent_ends():
    robot = RobotHandModel()
    cfg = load_yaml("configs/algorithm.yaml")["contact_assist"]
    assist = TaskContactAssist(robot, cfg, 1 / 60)
    q = np.zeros(len(robot.joint_order), dtype=float)
    contact = _empty_contact()
    first = assist.apply(q, task_name="pinch", intent_value="PINCH", contact=contact)
    second = assist.apply(q, task_name="pinch", intent_value="NEUTRAL", contact=contact)
    assert np.linalg.norm(second) < np.linalg.norm(first)
