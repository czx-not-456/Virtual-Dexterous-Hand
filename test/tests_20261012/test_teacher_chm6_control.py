import numpy as np
import pytest

from src.common import ROOT
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.teacher_chm6_control import TeacherCHM6ControlAdapter


def test_teacher_six_controls_expand_without_changing_robot_architecture():
    robot = RobotHandModel()
    adapter = TeacherCHM6ControlAdapter(robot, ROOT / "数据手套数据/mod/joint_mapping.json")
    target = adapter.expand_degrees(np.array([70, 20, 40, 45, 30, 25], dtype=float))
    index = {name: i for i, name in enumerate(robot.joint_order)}
    assert target.shape == (11,)
    assert robot.effective_actuators == 11
    assert target[index["th_ip_joint"]] == pytest.approx(
        target[index["th_mcp_joint"]] * 1.5483870968
    )
    assert target[index["ff_dip_joint"]] == pytest.approx(
        target[index["ff_mcp_joint"]] * 0.9617834395
    )
    assert np.all(target >= robot.ranges[:, 0])
    assert np.all(target <= robot.ranges[:, 1])


def test_teacher_adapter_rejects_wrong_control_shape():
    adapter = TeacherCHM6ControlAdapter(
        RobotHandModel(), ROOT / "数据手套数据/mod/joint_mapping.json"
    )
    with pytest.raises(ValueError, match="six finite"):
        adapter.expand_degrees(np.zeros(11))
