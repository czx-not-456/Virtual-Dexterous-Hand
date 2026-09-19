import numpy as np
from src.common import load_yaml
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import Intent
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.optimizer import IntentDrivenRetargeter


def test_retargeting_respects_joint_limits():
    human = HumanHandModel()
    robot = RobotHandModel()
    solver = IntentDrivenRetargeter(human, robot, load_yaml("configs/algorithm.yaml")["optimizer"])
    qh = human.ranges[:, 0] + 0.7 * (human.ranges[:, 1] - human.ranges[:, 0])
    f = FeatureExtractor(human).extract(qh)
    result = solver.solve(qh, f.fingertip_positions, Intent.WRAP)
    assert np.all(result.q_target >= robot.ranges[:, 0] - 1e-9)
    assert np.all(result.q_target <= robot.ranges[:, 1] + 1e-9)


def test_dynamic_pinch_does_not_worsen_pinch_task_error():
    human = HumanHandModel()
    robot = RobotHandModel()
    solver = IntentDrivenRetargeter(human, robot, load_yaml("configs/algorithm.yaml")["optimizer"])
    qh = human.ranges[:, 0] + 0.45 * (human.ranges[:, 1] - human.ranges[:, 0])
    f = FeatureExtractor(human).extract(qh)
    static = solver.solve(qh, f.fingertip_positions, Intent.PINCH, dynamic=False)
    dynamic = solver.solve(qh, f.fingertip_positions, Intent.PINCH, dynamic=True)
    e0 = solver._position_error(static.q_target, f.fingertip_positions, Intent.PINCH)
    e1 = solver._position_error(dynamic.q_target, f.fingertip_positions, Intent.PINCH)
    assert e1 <= e0 + 1e-8


def test_dynamic_wrap_reduces_coupling_penalty():
    from src.retargeting.constraints import coupling_error, vector_shape_error
    human = HumanHandModel()
    robot = RobotHandModel()
    solver = IntentDrivenRetargeter(human, robot, load_yaml("configs/algorithm.yaml")["optimizer"])
    qh = human.ranges[:, 0] + 0.88 * (human.ranges[:, 1] - human.ranges[:, 0])
    f = FeatureExtractor(human).extract(qh)
    static = solver.solve(qh, f.fingertip_positions, Intent.WRAP, dynamic=False)
    dynamic = solver.solve(qh, f.fingertip_positions, Intent.WRAP, dynamic=True)
    assert coupling_error(dynamic.q_target, robot) <= coupling_error(static.q_target, robot) + 1e-8
    static_vec = vector_shape_error(
        robot.fingertips(static.q_target), f.fingertip_positions,
        solver.task_scale, solver.vector_pairs,
    )
    dynamic_vec = vector_shape_error(
        robot.fingertips(dynamic.q_target), f.fingertip_positions,
        solver.task_scale, solver.vector_pairs,
    )
    # Known multi-objective tradeoff is about 8.47%; prevent silent worsening.
    assert dynamic_vec <= static_vec * 1.10
