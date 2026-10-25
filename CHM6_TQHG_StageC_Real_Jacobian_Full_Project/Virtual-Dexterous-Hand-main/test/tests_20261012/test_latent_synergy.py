"""Low-dimensional CH-M6 mapping tests; no hardware interpretation implied."""
import numpy as np
import pytest

from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper, fit_latent_model
from src.glove.adapter import BaseMapping
from src.features.intent_recognition import Intent
from src.hand_model.human_hand import HumanHandModel
from src.retargeting.optimizer import IntentDrivenRetargeter
from src.common import load_yaml


MODEL_PATH = "configs/latent_chm6_k4.json"


def test_fit_and_shared_encoders_shapes():
    rng = np.random.default_rng(77)
    s = rng.uniform(0, 90, size=(40, 15))
    u = rng.uniform(0, 35, size=(40, 6))
    p = fit_latent_model(s, u, k=4)
    assert np.asarray(p["control_pca_basis"]).shape == (6, 4)
    assert np.asarray(p["sensor_to_latent_weights"]).shape == (15, 4)
    assert np.allclose(np.asarray(p["control_pca_basis"]).T @
                       np.asarray(p["control_pca_basis"]), np.eye(4), atol=1e-8)


def test_projected_mapping_enforces_all_five_couplings():
    robot = RobotHandModel()
    mapper = LatentSynergyMapper(robot, MODEL_PATH)
    s = np.linspace(5, 50, 15)
    u = np.array([70., 20., 45., 25., 20., 15.])
    for source in ("sensor", "control", "blend"):
        q = mapper.from_observation(s, u, source=source)
        assert q.shape == (11,)
        assert np.all(q >= robot.ranges[:, 0]) and np.all(q <= robot.ranges[:, 1])
        for driver, child, gain in mapper.teacher.couplings:
            i, j = mapper.joint_index[driver], mapper.joint_index[child]
            assert q[j] == pytest.approx(np.clip(q[i] * gain, *robot.ranges[j]), abs=1e-8)
        projected = mapper.project_joints(q)
        assert np.allclose(q, projected, atol=1e-6)


def test_legacy_base_mapping_and_latent_optional():
    mapper = LatentSynergyMapper(RobotHandModel(), MODEL_PATH)
    raw = BaseMapping("数据手套数据/base_mapping(1).json")
    sensors = np.arange(1., 16.)
    controls = raw.predict_deg(sensors)
    assert len(controls) == 6
    assert mapper.from_observation(sensors, np.asarray(list(controls.values()))).shape == (11,)


def test_latent_optimization_stays_coupled():
    human = HumanHandModel()
    robot = RobotHandModel()
    mapper = LatentSynergyMapper(robot, MODEL_PATH)
    opt = IntentDrivenRetargeter(human, robot, load_yaml("configs/algorithm.yaml")["optimizer"],
                                 latent_mapper=mapper)
    q_h = human.angles_from_normalized({name: 0.45 for name in human.joint_order})
    q_base = mapper.from_observation(np.linspace(5., 60., 15), source="sensor")
    result = opt.solve(q_h, human.fingertips(q_h), Intent.NEUTRAL, q_base=q_base)
    assert np.linalg.norm(result.q_target - mapper.project_joints(result.q_target)) < 1e-5


def test_velocity_limiter():
    robot = RobotHandModel()
    mapper = LatentSynergyMapper(robot, MODEL_PATH)
    prev = mapper.decode_joints(np.zeros(4))
    desired = mapper.decode_joints(np.array([1., 1., 1., 1.]))
    max_speed = np.full(11, 0.4)
    out = mapper.project_and_rate_limit(desired, prev, max_speed, 1/60)
    assert np.all(np.abs(out - prev) <= max_speed / 60 + 1e-9)
    assert np.linalg.norm(out - mapper.project_joints(out)) < 1e-5
