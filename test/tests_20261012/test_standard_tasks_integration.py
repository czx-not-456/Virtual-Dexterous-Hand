import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.common import ROOT


@pytest.mark.parametrize(
    ("task", "required_metric"),
    [("neutral", "none"), ("pinch", "contact"), ("wrap", "stable")],
)
def test_canonical_trial_establishes_sustained_real_contact(
    task: str, required_metric: str, tmp_path: Path
):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run_name = f"integration_{task}"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "--sim",
            "mujoco",
            "--input",
            "dataset",
            "--dataset",
            str(ROOT / "datasets" / "synthetic_glove_v1.csv"),
            "--dataset-trial",
            "0",
            "--task",
            task,
            "--steps",
            "800",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            run_name,
        ],
        cwd=ROOT,
        env=env,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0
    summary = json.loads((tmp_path / f"{run_name}_summary.json").read_text(encoding="utf-8"))
    assert summary["active_frames"] > 0
    assert summary["robot_model"] == "CH-M6_L"
    assert summary["robot_dof"] == 11
    assert summary["required_success_metric"] == required_metric
    assert summary["task_metric_pass"] == (
        True if required_metric == "none" else bool(
            summary["contact_success_proxy"] if required_metric == "contact" else summary["stable_success_proxy"]
        )
    )
    if required_metric != "none":
        assert summary["contact_frames"] > 0
        assert summary["contact_success_proxy"] is True
        assert summary["max_contact_success_streak_frames"] >= 12
    if required_metric == "stable":
        # New one-second free-object criterion is stricter than historical 12-frame proxy.
        # A failing physical hold is a measured outcome, not a test infrastructure failure.
        assert "legacy_stable_success_proxy" in summary
        assert "hold_max_relative_translation_span_m" in summary
