"""Noise/dropout regression tests for the force-backed contact control state."""
import numpy as np
import pytest
from src.simulation.contact_detection import ContactDetectionConfig, ForceContactDebouncer
from experiments.benchmark_contact_detection import predict, synth_episode, run


def test_contact_requires_force_and_consecutive_frames():
    d=ForceContactDebouncer(ContactDetectionConfig(enter_force_n=.02,exit_force_n=.005,enter_frames=2,exit_frames=3))
    assert not d.update({'thumb':.019})['thumb']
    assert not d.update({'thumb':.03})['thumb']
    assert d.update({'thumb':.03})['thumb']
    assert d.update({'thumb':0.0})['thumb']
    assert d.update({'thumb':0.0})['thumb']
    assert not d.update({'thumb':0.0})['thumb']


def test_force_threshold_hysteresis_and_reset():
    d=ForceContactDebouncer(ContactDetectionConfig(enter_force_n=.02,exit_force_n=.01,enter_frames=1,exit_frames=2))
    assert d.update({'index':.03})['index']
    assert d.update({'index':.014})['index']
    assert d.update({})['index']
    assert not d.update({})['index']
    assert not d.update({'index':.013})['index']
    d.reset()
    assert not d.update({'index':0.0})['index']


def test_invalid_thresholds_rejected():
    for kw in ({'enter_force_n':-.01},{'exit_force_n':0},{'enter_force_n':.01,'exit_force_n':.02},
               {'enter_frames':0},{'exit_frames':0}):
        with pytest.raises(ValueError): ContactDetectionConfig(**kw)


def test_synthetic_ablation_reproducible_and_no_trivial_regression(tmp_path):
    a=run(tmp_path/'a',episodes=20,seed=20261108,bootstraps=150)
    b=run(tmp_path/'b',episodes=20,seed=20261108,bootstraps=150)
    assert a==b
    assert a['summary']['force_and_temporal']['f1'] > a['summary']['geometry_only']['f1']
    assert a['summary']['force_and_temporal']['state_switches'] < a['summary']['force_only']['state_switches']


def test_no_force_never_becomes_physical_contact():
    cfg=ContactDetectionConfig(enter_frames=2,exit_frames=3)
    f=np.zeros(16)
    candidates=np.ones(16,dtype=bool)
    out=predict(f,candidates,cfg)
    assert out['geometry_only'].all()
    assert not out['force_and_temporal'].any()


def test_mujoco_metrics_adapter_rejects_inactive_and_does_not_credit_filtered_success():
    """Fake MuJoCo contact structs, exercising the *actual simulator* adapter."""
    from types import SimpleNamespace
    from src.hand_model.robot_hand import RobotHandModel
    from src.evaluation.relative_grasp_stability import RelativeGraspHold
    from src.simulation.mujoco_env import MujocoSimulator

    obj = object.__new__(MujocoSimulator)
    obj.robot = RobotHandModel()
    obj._object_geom_ids = {0}
    obj._object_body_id = None
    obj._object_qpos_adr = None
    obj._object_qpos0 = None
    obj._palm_body_id = 0
    obj._task_active = True
    obj._object_held_by_fixture = False
    obj._object_reset_this_step = False
    obj._contact_streak_frames = 0
    obj.stability_orientation_deg = 5.0
    obj.task_mode = 'pinch'
    obj._hold_tracker = RelativeGraspHold()
    obj._contact_filter = ForceContactDebouncer(ContactDetectionConfig())
    obj.object_orientation_drift_deg = lambda:0.0
    obj.object_displacement_m = lambda:0.0
    geom_names = {1:'thumb_pad', 2:'index_pad'}
    body_names = {1:'th_distal_link',2:'ff_distal_link'}
    class FakeMujoco:
        mjtObj = SimpleNamespace(mjOBJ_GEOM='geom',mjOBJ_BODY='body')
        @staticmethod
        def mj_id2name(model, kind, idx):
            return (geom_names if kind=='geom' else body_names).get(idx, 'object')
        @staticmethod
        def mj_contactForce(model, data, idx, out):
            out[0]=data.forces[idx]
    obj.mujoco=FakeMujoco
    obj.model=SimpleNamespace(geom_bodyid=np.array([0,1,2]))
    obj.data=SimpleNamespace(ncon=0,contact=[],forces=[],
                              xpos=np.zeros((1,3)),xmat=np.eye(3).reshape(1,9))

    def frame(force, efc=0):
        obj.data.contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.001,efc_address=efc,
                                          frame=np.eye(3).reshape(-1)),
                          SimpleNamespace(geom1=0,geom2=2,dist=-.001,efc_address=efc,
                                          frame=np.eye(3).reshape(-1))]
        obj.data.ncon=2
        obj.data.forces=[force, force]
        return obj.contact_metrics()

    # Solver-inactive candidates should never count as physical contact.
    ghost=frame(.5,-1)
    assert ghost['inactive_contact_candidates']==2
    assert ghost['contact_count']==0
    assert not ghost['raw_thumb_contact']
    assert not ghost['task_contact_proxy']
    first=frame(.02)
    assert first['task_contact_proxy']
    assert first['raw_thumb_contact'] and first['raw_index_contact']
    assert first['thumb_normal_force_n'] > 0
    # Object is geom1, so its applied force is opposite to the contact normal.
    assert first['thumb_force_on_object_x_n'] == pytest.approx(-.02)
    assert first['index_force_on_object_x_n'] == pytest.approx(-.02)
    assert not first['filtered_task_contact_proxy']
    second=frame(.02)
    assert second['filtered_task_contact_proxy']
    assert second['thumb_tip_contact'] and second['index_tip_contact']
    dropout=frame(0.0)
    assert dropout['filtered_task_contact_proxy']  # controller grace period
    assert not dropout['task_contact_proxy']  # no fake physics success!
    assert not dropout['hold_stable_proxy']
    frame(0.0)
    assert not frame(0.0)['filtered_task_contact_proxy']
    # The legacy geometry-only ablation can still flag inactive candidate
    # contacts for controller feedback, but NEVER for scoring physical success.
    obj._contact_filter = ForceContactDebouncer(ContactDetectionConfig(mode='geometry_only'))
    ghost_control = frame(0.0, -1)
    assert ghost_control['filtered_task_contact_proxy']
    assert not ghost_control['task_contact_proxy']
    obj._contact_filter = ForceContactDebouncer(ContactDetectionConfig(mode='force_only'))
    no_force = frame(0.0, -1)
    assert not no_force['filtered_task_contact_proxy']


def test_all_four_detector_modes_are_individually_selectable():
    cfgs = [ContactDetectionConfig(mode=m) for m in (
        'geometry_only','force_only','temporal_only','force_and_temporal')]
    assert len({c.mode for c in cfgs}) == 4
    assert all(c.enter_frames == 2 for c in cfgs)
