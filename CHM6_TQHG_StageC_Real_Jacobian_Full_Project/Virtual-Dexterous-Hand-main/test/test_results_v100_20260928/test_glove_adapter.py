import math
from pathlib import Path

import numpy as np

from src.common import ROOT, load_yaml
from src.glove.adapter import BaseMapping, GloveAdapter
from src.glove.captured_samples import load_captured_samples, resolve_delivery_path
from src.hand_model.frame import HumanHandFrame
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel


DATA = ROOT / "数据手套数据"


def test_teacher_captured_xlsx_is_loaded_without_optional_excel_dependency():
    samples = load_captured_samples(DATA / "captured_samples.xlsx")
    assert len(samples) == 24
    assert {sample.pose for sample in samples} == {"张开", "对食指", "对中指", "三指捏", "半握", "握拳"}
    assert all(sample.sensors_deg.shape == (15,) for sample in samples)
    assert all(sample.targets_deg is not None and sample.targets_deg.shape == (6,) for sample in samples)


def test_delivered_base_mapping_expression_is_used():
    mapping = BaseMapping(DATA / "base_mapping.json")
    sensors = np.arange(1.0, 16.0)
    prediction = mapping.predict_deg(sensors)
    assert math.isclose(prediction["thumb_rotation"], 2.430257 * 4.0 + 45.149950)
    assert math.isclose(prediction["thumb_bend"], 1.147454 * 2.0 - 0.460135 * 3.0 + 13.189736)


def test_glove_adapter_produces_human_frame_and_native_ch_m6_base_target():
    cfg = load_yaml("configs/glove.yaml")["real"]
    human = HumanHandModel()
    robot = RobotHandModel()
    adapter = GloveAdapter(
        human,
        robot,
        base_mapping_path=DATA / "base_mapping.json",
        manifest_path=DATA / "manifest.json",
        output_bindings=cfg["output_bindings"],
    )
    frame = adapter.adapt(load_captured_samples(DATA / "captured_samples.xlsx")[4], timestamp=1.25)
    assert isinstance(frame, HumanHandFrame)
    assert frame.source == "glove"
    assert frame.timestamp == 1.25
    assert frame.joint_angles_rad.shape == (15,)
    assert frame.robot_joint_targets_rad.shape == (11,)
    assert np.all(frame.robot_joint_targets_rad >= robot.ranges[:, 0])
    assert np.all(frame.robot_joint_targets_rad <= robot.ranges[:, 1])
    assert np.isclose(frame.robot_joint_targets_rad[3], frame.robot_joint_targets_rad[4])
    assert resolve_delivery_path(DATA / "base_mapping.json").name == "base_mapping(1).json"
