from src.common import load_yaml


def test_v042_combines_wrap_and_pinch_contact_tuning():
    cfg = load_yaml("configs/algorithm.yaml")
    servo = cfg["task_space_servo"]

    # Shared tuning used by the validated WRAP setup.
    assert servo["enabled"] is True
    assert servo["gain"] == 0.88
    assert servo["damping"] == 0.018
    assert servo["max_joint_step_deg"] == 6.5
    assert servo["stop_error_m"] == 0.0008
    assert servo["wrap"]["clearance_m"] == 0.0055

    # PINCH-specific fingertip-contact fix (v0.4.6 uses non-penetrating +7 mm center offset).
    pinch = servo["pinch"]
    assert pinch["stop_on_contact"] == "tip"
    assert pinch["target_offset_m"] == 0.0070
    assert pinch["command_blend"] == 0.90
    assert pinch["posture_prior_blend"] == 1.00
