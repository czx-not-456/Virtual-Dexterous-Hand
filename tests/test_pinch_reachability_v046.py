import math
import numpy as np

from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel


def _point_to_box_distance(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    outside = np.maximum(np.abs(point - center) - half, 0.0)
    return float(np.linalg.norm(outside))


def test_index_pad_can_geometrically_intersect_pinch_box():
    """Regression for the v0.4.5 7-8 mm plateau.

    The index is effectively one-tendon driven. A representative reachable
    precision-pinch pose is near MCP/PIP/DIP = 70/95/70 deg. The center of its
    8 mm fingertip sphere must come within 8 mm of the box; otherwise no amount
    of extra run time can ever produce index_tip_contact=True.
    """
    robot = RobotHandModel()
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]
    center = np.asarray(cfg["pos"], dtype=float)
    half = np.asarray(cfg["size"], dtype=float)

    idx = {name: i for i, name in enumerate(robot.joint_order)}
    q = np.zeros(len(robot.joint_order), dtype=float)
    q[idx["index_mcp"]] = math.radians(70.0)
    q[idx["index_pip"]] = math.radians(95.0)
    q[idx["index_dip"]] = math.radians(70.0)

    palm_world = np.array([0.0, 0.0, 0.240], dtype=float)
    tip = robot.fingertips(q)["index"] + palm_world
    distance = _point_to_box_distance(tip, center, half)

    assert distance < 0.0080, (tip, distance)


def test_pinch_servo_target_matches_reachable_index_pose():
    robot = RobotHandModel()
    sim = load_yaml("configs/simulation.yaml")["mujoco"]
    algo = load_yaml("configs/algorithm.yaml")["task_space_servo"]["pinch"]
    center = np.asarray(sim["task_objects"]["pinch"]["pos"], dtype=float)
    half = np.asarray(sim["task_objects"]["pinch"]["size"], dtype=float)
    c = float(algo["target_offset_m"])
    margin = float(algo["vertical_margin_m"])

    # Index approaches the +Y face. Initial tip is above the object, so the
    # target generator clips the center Z to the upper face minus margin.
    target = np.array([
        -0.019,
        center[1] + half[1] + c,
        center[2] + half[2] - margin,
    ])

    idx = {name: i for i, name in enumerate(robot.joint_order)}
    q = np.zeros(len(robot.joint_order), dtype=float)
    q[idx["index_mcp"]] = math.radians(70.0)
    q[idx["index_pip"]] = math.radians(95.0)
    q[idx["index_dip"]] = math.radians(70.0)
    tip = robot.fingertips(q)["index"] + np.array([0.0, 0.0, 0.240])

    assert float(np.linalg.norm(tip - target)) < 0.0020, (tip, target)
