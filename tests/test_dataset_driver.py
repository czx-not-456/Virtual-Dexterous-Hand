import csv
from pathlib import Path

from src.glove.dataset_driver import CSVGloveDatasetDriver


def test_dataset_driver_replays_selected_trial(tmp_path: Path):
    channels = ["a", "b"]
    path = tmp_path / "mock.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["trial_id", "frame", "gesture", "timestamp_s", "palm_roll", "palm_pitch", "palm_yaw", *channels],
        )
        w.writeheader()
        w.writerow({"trial_id": 0, "frame": 0, "gesture": "OPEN", "timestamp_s": 0, "palm_roll": 0, "palm_pitch": 0, "palm_yaw": 0, "a": 1, "b": 2})
        w.writerow({"trial_id": 1, "frame": 0, "gesture": "PINCH", "timestamp_s": 0, "palm_roll": 0, "palm_pitch": 0, "palm_yaw": 0, "a": 3, "b": 4})
        w.writerow({"trial_id": 1, "frame": 1, "gesture": "WRAP", "timestamp_s": 0.1, "palm_roll": 0, "palm_pitch": 0, "palm_yaw": 0, "a": 5, "b": 6})
    d = CSVGloveDatasetDriver(path, channels, trial_id=1, loop=True)
    assert d.available_trials == [0, 1]
    assert d.sample().channels["a"] == 3
    assert d.sample().channels["a"] == 5
    assert d.sample().channels["a"] == 3
    assert len(d.calibration_samples()) == 2
