import numpy as np
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.topology_mapping import TopologyMapper

def test_robot_exposes_ch_m6_11dof_11actuator_architecture():
    r=RobotHandModel(); assert r.nominal_dof==11; assert r.effective_actuators==11
    assert len(r.joint_order)==len(r.actuator_order)==11
    assert r.source_model_path=="CH-M6/CH-M6_L.xml"

def test_joint_targets_map_one_to_one_to_ch_m6_actuators():
    r=RobotHandModel(); q=r.ranges.mean(axis=1); u=r.actuator_targets_from_joint_targets(q)
    assert u.shape==(11,); assert np.allclose(u,q); assert np.all(u>=r.actuator_ctrl_ranges()[:,0])

def test_standard_human_pose_maps_to_ch_m6_native_joints():
    h=HumanHandModel(); r=RobotHandModel(); q=TopologyMapper(h,r).map(h.ranges.mean(axis=1))
    assert q.shape==(11,); assert np.allclose(q,r.ranges.mean(axis=1),atol=2e-6)
    assert set(r.human_mapping["ff_dip_joint"]["sources"])=={"index_pip","index_dip"}

def test_ch_m6_fk_moves_all_fingertips():
    r=RobotHandModel(); a=r.fingertips(r.ranges[:,0]); b=r.fingertips(r.ranges.mean(axis=1))
    assert all(np.linalg.norm(b[f]-a[f])>0.01 for f in r.finger_joints)