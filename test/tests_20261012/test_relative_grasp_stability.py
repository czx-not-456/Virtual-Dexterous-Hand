"""Regression tests against transient contacts, fixture assistance and pose drift."""
import math
import numpy as np

from src.evaluation.relative_grasp_stability import RelativeGraspHold
from src.evaluation.task_evaluator import TaskEpisodeEvaluator


def Rz(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0.0],
                     [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def feed(metric, *, contact=True, free=True, offset=0.0, angle=0.0,
         palm_offset=0.0, palm_angle=0.0):
    palm_rot = Rz(palm_angle)
    palm_pos = np.array([palm_offset, 0.0, 0.0])
    return metric.update(contact=contact, free_object=free,
                         palm_pos=palm_pos, palm_rot=palm_rot,
                         object_pos=palm_pos + palm_rot @ np.array([0.1 + offset, 0.0, 0.0]),
                         object_rot=palm_rot @ Rz(angle))


def test_stable_after_one_second_not_twelve_frames():
    metric = RelativeGraspHold(sampling_hz=60, hold_window_s=1.0)
    for _ in range(60):
        last = feed(metric)
        assert not last['hold_stable_proxy']
    last = feed(metric)
    assert last['hold_stable_proxy']
    assert last['hold_window_required_samples'] == 61


def test_contact_fixture_cannot_build_hold_window():
    m = RelativeGraspHold(sampling_hz=60)
    for _ in range(90):
        assert not feed(m, free=False)['hold_stable_proxy']
    assert feed(m, free=True)['hold_window_samples'] == 1
    for _ in range(60):
        status = feed(m)
    assert status['hold_stable_proxy']


def test_settle_then_stable_window_is_possible():
    m = RelativeGraspHold()
    for i in range(120):
        feed(m, offset=i * 0.00004, angle=i * 0.04)
    assert not feed(m, offset=0.0048, angle=4.8)['hold_stable_proxy']
    for _ in range(60):
        last = feed(m, offset=0.0048, angle=4.8)
    assert last['hold_stable_proxy']
    assert last['hold_relative_translation_span_m'] < 1e-8


def test_large_translation_or_rotation_invalidates_current_stability():
    m = RelativeGraspHold()
    for _ in range(61):
        ok = feed(m)
    assert ok['hold_stable_proxy']
    bad_translation = feed(m, offset=0.002)
    assert not bad_translation['hold_stable_proxy']
    assert bad_translation['hold_relative_translation_span_m'] >= 0.002 - 1e-12
    m.reset()
    for _ in range(61):
        ok = feed(m)
    assert ok['hold_stable_proxy']
    bad_rotation = feed(m, angle=2)
    assert not bad_rotation['hold_stable_proxy']
    assert bad_rotation['hold_relative_rotation_span_deg'] > 1.9


def test_missing_pose_and_lost_contact_reset_immediately():
    m = RelativeGraspHold()
    for _ in range(61):
        assert isinstance(feed(m)['hold_stable_proxy'], bool)
    assert not feed(m, contact=False)['hold_stable_proxy']
    assert feed(m)['hold_window_samples'] == 1
    missing = m.update(contact=True, free_object=True,
                       palm_pos=None, palm_rot=np.eye(3),
                       object_pos=np.zeros(3), object_rot=np.eye(3))
    assert not missing['hold_stable_proxy']
    assert not missing['hold_pose_available']


def test_relative_pose_unchanged_by_palm_whole_hand_motion():
    m = RelativeGraspHold()
    for i in range(61):
        result = feed(m, palm_offset=i * 0.005, palm_angle=i * 1.5)
    assert result['hold_stable_proxy']
    assert result['hold_relative_translation_span_m'] < 1e-8
    assert result['hold_relative_rotation_span_deg'] < 0.0001


def test_evaluator_current_vs_ever_stable_and_legacy():
    ev = TaskEpisodeEvaluator('pinch', sampling_hz=60, min_success_streak_frames=3)
    def frame(i, hold, legacy, active=True):
        return ev.update(frame=i, task_active=active,
                         contact={'task_contact_proxy': True, 'hold_object_free': True,
                                  'hold_stable_proxy': hold, 'stable_contact_proxy': legacy},
                         latency_ms=1, optimizer_ms=1, joint_rmse_rad=0,
                         velocity_rms_rad_s=0)
    for i in range(3):
        s = frame(i, hold=True, legacy=True)
    assert s['stable_success_proxy'] is True
    assert s['legacy_stable_success_proxy'] is True
    s = frame(3, hold=False, legacy=False)
    assert s['stable_success_proxy'] is False
    assert s['legacy_stable_success_proxy'] is True
    frame(4, hold=False, legacy=False, active=False)
    summary = ev.summary()
    assert summary['stable_success_proxy'] is False
    assert summary['ever_hold_stable_proxy'] is True
    assert summary['legacy_stable_success_proxy'] is True


def test_evaluator_missing_telemetry_is_never_new_stable():
    ev = TaskEpisodeEvaluator('pinch', 60, min_success_streak_frames=2)
    for i in range(5):
        ev.update(frame=i, task_active=True,
                  contact={'task_contact_proxy': True, 'stable_contact_proxy': True},
                  latency_ms=1, optimizer_ms=1,
                  joint_rmse_rad=0, velocity_rms_rad_s=0)
    assert ev.summary()['stable_success_proxy'] is False
    assert ev.summary()['legacy_stable_success_proxy'] is True
