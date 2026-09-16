from pathlib import Path


def test_mujoco_step_uses_control_time_accumulator():
    src = Path("src/simulation/mujoco_env.py").read_text(encoding="utf-8")
    assert "_physics_time_accumulator_s" in src
    assert "while (" in src
    assert "self.mujoco.mj_step(self.model, self.data)" in src


def test_main_passes_control_period_to_mujoco():
    src = Path("src/main.py").read_text(encoding="utf-8")
    assert "control_dt_s=dt" in src
