import csv
from pathlib import Path
from types import SimpleNamespace

from experiments import benchmark_synthetic


def _write_dataset(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "frame"])
        writer.writeheader()
        writer.writerows(
            [
                {"trial_id": 0, "frame": 0},
                {"trial_id": 0, "frame": 1},
                {"trial_id": 1, "frame": 0},
            ]
        )


def test_trial_lengths_reads_complete_selected_trials(tmp_path):
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset)
    assert benchmark_synthetic._trial_lengths(dataset) == {0: 2, 1: 1}


def test_run_one_records_subcommand_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        benchmark_synthetic.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7),
    )
    result = benchmark_synthetic._run_one(
        "pinch", 0, 800, tmp_path / "data.csv", tmp_path, "mujoco"
    )
    assert result["run_ok"] is False
    assert result["returncode"] == 7
    assert result["error"] == "subcommand failed"


def test_run_one_rejects_missing_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        benchmark_synthetic.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    result = benchmark_synthetic._run_one(
        "pinch", 0, 800, tmp_path / "data.csv", tmp_path, "mujoco"
    )
    assert result["run_ok"] is False
    assert result["returncode"] == 0
    assert result["error"] == "missing summary"
