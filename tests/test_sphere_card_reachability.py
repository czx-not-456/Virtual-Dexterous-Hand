import numpy as np

from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.mujoco_env import MujocoSimulator


def test_card_has_thin_edge_and_task_specific_servo_tuning():
    sim = load_yaml("configs/simulation.yaml")["mujoco"]
    card = sim["task_objects"]["card"]
    algo = load_yaml("configs/algorithm.yaml")["task_space_servo"]
    hx, hy, hz = map(float, card["size"])
    _, _, z = map(float, card["pos"])

    assert card["interaction"] == "pinch"
    assert 2 * hx == 0.006
    assert 2 * hy == 0.030
    assert abs((z - hz) - 0.070) < 1e-9
    assert "card" in algo
    assert algo["card"]["stop_on_contact"] == "never"


def test_sphere_reachable_target_follows_underactuated_tangent():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    spec = cfg["task_objects"]["sphere"]
    robot = RobotHandModel()
    simulator = MujocoSimulator(
        robot,
        model_path=cfg["model_path"],
        task_object_body=cfg["task_object_body"],
        task_name="sphere",
        task_spec=spec,
    )
    try:
        simulator.mujoco.mj_forward(simulator.model, simulator.data)
        center, rotation = simulator.task_object_pose()
        current = simulator.fingertip_position("index")
        clearance = float(
            load_yaml("configs/algorithm.yaml")["task_space_servo"]["sphere"]["clearance_m"]
        )
        target = simulator.task_contact_target(
            "index", "sphere", clearance_m=clearance, vertical_margin_m=0.01
        )
        radius = float(spec["size"][0]) + clearance
        assert abs(np.linalg.norm(rotation.T @ (target - center)) - radius) < 1e-8

        joints = ("index_mcp", "index_pip", "index_dip")
        jacobian = simulator.fingertip_jacobian("index", joints)
        coupling = robot.coupling["index"]
        tendon = np.array(
            [1.0, coupling["pip_over_mcp"], coupling["dip_over_mcp"]], dtype=float
        )
        tangent = jacobian @ tendon
        local_current = rotation.T @ (current - center)
        local_tangent = rotation.T @ tangent
        closest = local_current - (
            np.dot(local_current, local_tangent) / np.dot(local_tangent, local_tangent)
        ) * local_tangent
        expected = center + rotation @ (closest / np.linalg.norm(closest) * radius)
        # When the tangent misses the sphere, the target is the surface point
        # nearest to that reachable line rather than the radial nearest point.
        assert np.allclose(target, expected)
    finally:
        simulator.close()
