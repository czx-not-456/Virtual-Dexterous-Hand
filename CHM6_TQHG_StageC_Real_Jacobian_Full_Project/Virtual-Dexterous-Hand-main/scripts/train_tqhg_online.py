"""Train solely on historical CH-M6 LEFT-hand samples. Never read test file."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retargeting.latent_synergy import fit_latent_model, CONTROL_LIMITS_DEG

HISTORY = ROOT / "数据手套数据/records/chm6_calibrations/20260917_102742_G12LC040704_left/captured_samples.csv"
OUTPUT = ROOT / "configs/tqhg_online_v1.json"


def train(history=HISTORY, output=OUTPUT):
    history, output = Path(history), Path(output)
    with history.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    s = np.asarray([[float(r[f"sensor_{i}"]) for i in range(1, 16)] for r in rows])
    y = np.asarray([[float(r[f"target_{i}"]) for i in range(1, 7)] for r in rows])
    poses = [r["pose"] for r in rows]
    tr, ca = [], []
    for pose in dict.fromkeys(poses):
        ix = [i for i, p in enumerate(poses) if p == pose]
        n = max(2, min(len(ix) - 2, int(np.floor(len(ix) * 0.66))))
        tr.extend(ix[:n]); ca.extend(ix[n:])
    tr, ca = np.asarray(tr, dtype=int), np.asarray(ca, dtype=int)
    model = fit_latent_model(s[tr], y[tr], k=4, ridge=0.05)
    sn = np.clip((s[ca] - np.asarray(model["sensor_min_deg"])) /
                 np.asarray(model["sensor_span_deg"]), 0, 1)
    ridge = np.clip(sn @ np.asarray(model["sensor_to_controls_weights"]) +
                    np.asarray(model["sensor_to_controls_bias"]), 0, 1)
    pca = np.clip(np.asarray(model["control_mean_normalized"]) +
                  (sn @ np.asarray(model["sensor_to_latent_weights"]) +
                   np.asarray(model["sensor_to_latent_bias"])) @
                  np.asarray(model["control_pca_basis"]).T, 0, 1)
    yt = y[ca] / CONTROL_LIMITS_DEG
    result = {
        "schema": "tqhg_online_calibration_v1",
        "model": model,
        "calibration_errors_ridge": (ridge - yt).tolist(),
        "calibration_errors_pca": (pca - yt).tolist(),
        "provenance": {
            "historical_training_file": str(history.relative_to(ROOT)),
            "training_rows": len(tr), "calibration_rows": len(ca),
            "split": "chronological 66% within each of six historical gestures",
            "not_in_training_or_calibration": "captured_samples(1).xlsx and CHM6_pinch_hold_diagnostic_only.xlsx",
            "teacher_labels_only": True,
            "note": "Single-device six-pose calibration; not independent human joint-angle truth.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] TQHG calibration saved: {output}")
    print(f"[INFO] Train={len(tr)}, calibration={len(ca)}, historical total={len(rows)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    a = parser.parse_args()
    train(a.history, a.output)
