"""Train a shared 15/6 -> k latent CH-M6 mapping from paired glove captures.

Run from project root, for example:
python scripts/train_latent_synergy.py --samples '数据手套数据/records/chm6_calibrations/20260917_102405_G15RB010901_right/captured_samples.csv' --out configs/latent_chm6_k4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.glove.captured_samples import load_captured_samples
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG, fit_latent_model


def _estimate(data: dict, s: np.ndarray) -> np.ndarray:
    x = np.clip((s - np.array(data["sensor_min_deg"])) /
                np.array(data["sensor_span_deg"]), 0., 1.)
    z = x @ np.array(data["sensor_to_latent_weights"]) + np.array(data["sensor_to_latent_bias"])
    un = np.clip(np.array(data["control_mean_normalized"]) +
                 z @ np.array(data["control_pca_basis"]).T, 0., 1.)
    return un * CONTROL_LIMITS_DEG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.05)
    args = parser.parse_args()
    samples = [r for r in load_captured_samples(args.samples) if r.targets_deg is not None]
    sensors = np.asarray([r.sensors_deg for r in samples])
    controls = np.asarray([r.targets_deg for r in samples])
    poses = np.asarray([r.pose for r in samples])
    # Last ~20% of each pose group is a *diagnostic* temporal holdout;
    # not evidence of generalization to new gestures or new gloves.
    train_mask = np.ones(len(samples), dtype=bool)
    for pose in sorted(set(poses)):
        indices = np.flatnonzero(poses == pose)
        val_n = max(1, int(len(indices) * 0.2))
        train_mask[indices[-val_n:]] = False
    diagnostic = fit_latent_model(sensors[train_mask], controls[train_mask],
                                  k=args.k, ridge=args.ridge)
    pred = _estimate(diagnostic, sensors[~train_mask])
    val_mae = np.abs(pred - controls[~train_mask]).mean(axis=0)
    trained = fit_latent_model(sensors, controls, k=args.k, ridge=args.ridge)
    trained["source_file"] = str(args.samples)
    labels_rounded = np.round(controls, 6)
    trained["distinct_target_vectors"] = int(len(np.unique(labels_rounded, axis=0)))
    trained["centered_target_rank"] = int(np.linalg.matrix_rank(
        controls / CONTROL_LIMITS_DEG - np.mean(controls / CONTROL_LIMITS_DEG, axis=0), tol=1e-9))
    trained["label_warning"] = (
        "Six predefined target labels in this dataset, NOT independent measured continuous robot commands. "
        "The last PCA axis is unobserved as a variable target and its use is exploratory."
    )
    trained["diagnostic_temporal_holdout_mae_deg"] = val_mae.tolist()
    trained["diagnostic_note"] = "Same-pose temporal holdout; test unseen poses and real closed-loop motions separately."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trained, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    explained = sum(trained["explained_variance_ratio"][:args.k])
    print(f"saved: {args.out} | N={len(samples)}, k={args.k}, PCA explained={explained:.3%}")
    print(f"same-pose temporal holdout MAE (6 axes, degrees): {np.round(val_mae, 2).tolist()}")
    print(f"unique 6D labels={trained['distinct_target_vectors']}, rank={trained['centered_target_rank']}")
    print("CAUTION: training labels are six target poses, not measurements of actual robot joint motion.")


if __name__ == "__main__":
    main()
