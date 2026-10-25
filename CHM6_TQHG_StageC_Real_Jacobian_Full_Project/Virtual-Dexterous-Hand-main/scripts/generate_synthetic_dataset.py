from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.glove.mock_driver import MockGloveDriver


def gesture_for_frame(frame: int) -> str:
    # 800-frame cycle: give precision pinch enough time to converge, while
    # keeping a comparably long WRAP phase for repeatable benchmark episodes.
    x = frame % 800
    if x < 80:
        return "OPEN"
    if x < 160:
        return "NEUTRAL"
    if x < 400:
        return "PINCH"
    if x < 480:
        return "NEUTRAL"
    if x < 720:
        return "WRAP"
    return "OPEN"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate reproducible synthetic glove trials")
    p.add_argument("--output", type=Path, default=ROOT / "datasets" / "synthetic_glove_v1.csv")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--frames", type=int, default=800, help="frames per trial; 800 contains one full gesture cycle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-raw", type=float, default=10.0)
    args = p.parse_args()

    cfg = load_yaml("configs/glove.yaml")
    channels = list(cfg["channels"])
    hz = float(cfg["sampling_hz"])
    raw_min = float(cfg["raw_min_default"])
    raw_max = float(cfg["raw_max_default"])
    span = raw_max - raw_min
    rng = np.random.default_rng(args.seed)
    template = MockGloveDriver(channels, sampling_hz=hz, raw_min=raw_min, raw_max=raw_max, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial_id", "frame", "gesture", "timestamp_s",
        "palm_roll", "palm_pitch", "palm_yaw", *channels,
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in range(args.trials):
            # Model different users / glove placements using channel-wise gain and bias.
            user_gain = {c: float(rng.normal(1.0, 0.035)) for c in channels}
            user_bias = {c: float(rng.normal(0.0, 0.012)) for c in channels}
            phase_shift = float(rng.uniform(0.0, 2.0 * math.pi))
            for frame in range(args.frames):
                gesture = gesture_for_frame(frame)
                phase = frame / hz * 2.0 * math.pi + phase_shift
                n = template._gesture_vector(gesture, phase)
                raw_channels: dict[str, float] = {}
                for c in channels:
                    n_user = float(np.clip(n[c] * user_gain[c] + user_bias[c], 0.0, 1.0))
                    raw_channels[c] = raw_min + n_user * span + float(rng.normal(0.0, args.noise_raw))
                t = frame / hz
                row = {
                    "trial_id": trial,
                    "frame": frame,
                    "gesture": gesture,
                    "timestamp_s": t,
                    "palm_roll": 0.04 * math.sin(t + phase_shift),
                    "palm_pitch": 0.03 * math.sin(0.7 * t + 0.3 * phase_shift),
                    "palm_yaw": 0.02 * math.cos(0.5 * t + 0.2 * phase_shift),
                    **raw_channels,
                }
                writer.writerow(row)

    print(f"[DONE] synthetic dataset: {args.output}")
    print(f"       trials={args.trials}, frames_per_trial={args.frames}, rows={args.trials * args.frames}")


if __name__ == "__main__":
    main()
