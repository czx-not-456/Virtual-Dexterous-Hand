from pathlib import Path
import numpy as np
import pytest
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.tqhg_online import TQHGOnlineMapper
from src.retargeting.task_weighted_hybrid import task_projector

CAL = Path("configs/tqhg_online_v1.json")

@pytest.fixture(scope='module')
def obj():
    if not CAL.exists():
        pytest.skip("Run python scripts/train_tqhg_online.py first")
    return TQHGOnlineMapper(RobotHandModel(), CAL)


def test_all_five_mapping_methods_remain_six_coupled(obj):
    s = obj.sensor_min + 0.3 * obj.sensor_span
    for method in obj.METHODS:
        u = obj.predict(s, method=method, simulator=None, task_mode='pinch')
        assert u.shape == (6,) and np.isfinite(u).all()
        q = obj.decode_controls(u)
        assert q.shape == (11,) and np.isfinite(q).all()
        assert np.allclose(q, obj.C @ u, atol=1e-10)
        assert np.all(q >= obj.robot.ranges[:, 0] - 1e-8)
        assert np.all(q <= obj.robot.ranges[:, 1] + 1e-8)


def test_coupled_rate_limit(obj):
    a = obj.decode_controls(obj.lower + .03*(obj.upper-obj.lower))
    b = obj.decode_controls(obj.upper - .03*(obj.upper-obj.lower))
    vmax = np.full(11, np.deg2rad(180.0))
    q = obj.project_and_rate_limit(b, a, vmax, 1.0 / 60)
    assert np.max(np.abs(q-a) - vmax / 60) <= 1e-9
    assert np.max(np.abs(obj.decode_controls(obj.encode_joints(q)) - q)) <= 1e-7


def test_task_projector_complement_is_annihilated():
    rng = np.random.default_rng(12)
    J = rng.standard_normal((3,6))
    P = task_projector(J)
    assert np.allclose(J @ (np.eye(6) - P), 0, atol=1e-9)


def test_online_jacobian_is_six_columns(obj):
    class Stub:
        def fingertip_jacobian(self, finger, joint_names):
            m = np.zeros((3, len(joint_names)))
            m[0, :] = np.arange(1, len(joint_names)+1)
            return m
    assert obj._jacobian(Stub(), 'pinch').shape == (6, 6)
    assert obj._jacobian(Stub(), 'wrap').shape == (15, 6)
    assert obj._jacobian(Stub(), 'neutral').shape == (1, 6)


def test_nonfinite_rejected(obj):
    bad = obj.sensor_min.copy(); bad[0] = np.nan
    with pytest.raises(ValueError): obj.predict(bad, method='tqhg')
