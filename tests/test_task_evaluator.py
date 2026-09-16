from src.evaluation.task_evaluator import TaskEpisodeEvaluator


def test_task_evaluator_requires_consecutive_stable_frames():
    ev = TaskEpisodeEvaluator("pinch", sampling_hz=60, min_success_streak_frames=3)
    for frame, stable in enumerate([False, True, True, False, True, True, True]):
        contact = {
            "task_contact_proxy": stable,
            "stable_contact_proxy": stable,
            "tip_contact_ratio": 0.4 if stable else 0.0,
            "orientation_drift_deg": 1.0,
            "object_displacement_m": 0.001,
        }
        state = ev.update(
            frame=frame,
            task_active=True,
            contact=contact,
            latency_ms=5.0,
            optimizer_ms=2.0,
            joint_rmse_rad=0.1,
            velocity_rms_rad_s=0.2,
        )
    assert state["task_success"] is True
    summary = ev.summary()
    assert summary["success_proxy"] is True
    assert summary["max_success_streak_frames"] == 3
    assert summary["first_success_latency_ms"] == 100.0


def test_task_evaluator_ignores_inactive_frames_for_contact_ratio():
    ev = TaskEpisodeEvaluator("wrap", sampling_hz=50, min_success_streak_frames=2)
    for frame in range(3):
        ev.update(
            frame=frame,
            task_active=False,
            contact={},
            latency_ms=1,
            optimizer_ms=1,
            joint_rmse_rad=0,
            velocity_rms_rad_s=0,
        )
    ev.update(
        frame=3,
        task_active=True,
        contact={"task_contact_proxy": True, "stable_contact_proxy": False},
        latency_ms=1,
        optimizer_ms=1,
        joint_rmse_rad=0,
        velocity_rms_rad_s=0,
    )
    assert ev.summary()["contact_frame_ratio"] == 1.0


def test_pinch_contact_mode_uses_bilateral_contact_for_success_but_keeps_stability_metric():
    ev = TaskEpisodeEvaluator(
        "pinch", sampling_hz=60, min_success_streak_frames=3, success_mode="contact"
    )
    for frame in range(3):
        state = ev.update(
            frame=frame,
            task_active=True,
            contact={
                "task_contact_proxy": True,
                "stable_contact_proxy": False,
                "tip_contact_ratio": 0.4,
                "orientation_drift_deg": 6.0,
                "object_displacement_m": 0.002,
            },
            latency_ms=1,
            optimizer_ms=1,
            joint_rmse_rad=0,
            velocity_rms_rad_s=0,
        )
    assert state["task_success"] is True
    summary = ev.summary()
    assert summary["success_proxy"] is True
    assert summary["contact_frames"] == 3
    assert summary["stable_frames"] == 0
    assert summary["success_mode"] == "contact"
