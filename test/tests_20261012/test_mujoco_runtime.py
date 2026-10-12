import pytest
from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.mujoco_env import MujocoSimulator
mujoco=pytest.importorskip("mujoco")

def test_ch_m6_runtime_loads_assets_scene_and_named_entities():
    c=load_yaml("configs/simulation.yaml")["mujoco"]; r=RobotHandModel(); s=MujocoSimulator(r,model_path=c["model_path"],task_object_body=c["task_object_body"],task_name="neutral",task_spec=c["task_objects"]["neutral"],scene_config=c)
    try:
        assert s.model.nu==11; assert s.model.nq==18; assert s.model.nv==17
        for n in r.joint_order: assert mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_JOINT,n)>=0
        for f in r.finger_joints: assert mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_SITE,r.tip_site_name(f))>=0
        assert mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_BODY,c["task_object_body"])>=0
    finally: s.close()