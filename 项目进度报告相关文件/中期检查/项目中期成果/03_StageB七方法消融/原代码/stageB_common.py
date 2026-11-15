"""Stage B shared experimental definitions. Not a production online mapper.

Historical training/calibration split is exactly that used by train_tqhg_v2.
Evaluation uses the unchanged 24-record external test workbook and paired masks.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG
from src.retargeting.tqhg_v2 import masked_features

CONDITIONS = (("clean", 0), ("mask_one", 1), ("mask_three", 3), ("mask_five", 5))
METHODS = ("ridge6_original", "pca4_original", "ridge6_robust",
           "pca4_robust", "fixed_50_50", "calibrated_scalar_blend", "tqhg_v2_original")


def historical_split(history: Path, sensor_min, sensor_span):
    with Path(history).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    s = np.asarray([[float(r[f"sensor_{i}"]) for i in range(1, 16)] for r in rows], float)
    y = np.asarray([[float(r[f"target_{i}"]) for i in range(1, 7)] for r in rows], float) / CONTROL_LIMITS_DEG
    poses = [r["pose"] for r in rows]
    tr, ca = [], []
    for pose in dict.fromkeys(poses):
        ids = [i for i, p in enumerate(poses) if p == pose]
        cutoff = max(2, min(len(ids)-2, int(np.floor(len(ids)*0.66))))
        tr.extend(ids[:cutoff]);ca.extend(ids[cutoff:])
    tr, ca = np.asarray(tr, dtype=int), np.asarray(ca, dtype=int)
    sn = np.clip((s - np.asarray(sensor_min, float)) / np.asarray(sensor_span, float), 0, 1)
    if sn.shape != (190, 15) or not np.all(np.isfinite(y)) or len(tr) != 124 or len(ca) != 66:
        raise ValueError("Expected original 190 historical rows with train=124/cal=66; refusing silent split change")
    return sn, y, tr, ca


def paired_cases(sensors, rng, trials):
    """Same order/mask RNG contract as Stage A, with no test-label use."""
    if trials < 1:
        raise ValueError("trials must be >=1")
    for condition, n_missing in CONDITIONS:
        for trial in range(trials if n_missing else 1):
            for i in range(len(sensors)):
                mask = np.ones(15, bool)
                if n_missing:
                    mask[rng.choice(15, size=n_missing, replace=False)] = False
                yield condition, n_missing, trial, i, mask


def robust_predictions(features, old_cfg, v2_cfg, robust6_cfg):
    """Same augmented 30-D features for robust 6-D and robust 4-D experts.

    Old models see median-imputed 15D values and cannot see validity mask;
    they are kept as historical baselines, not training-matched comparators.
    """
    x = np.asarray(features, float)
    p = old_cfg["model"]
    mu = np.asarray(p["control_mean_normalized"],float)
    basis = np.asarray(p["control_pca_basis"],float)
    e = v2_cfg["robust_expert"]
    ridge_old = x[:15] @ np.asarray(p["sensor_to_controls_weights"],float) + np.asarray(p["sensor_to_controls_bias"],float)
    pca_old = mu + (x[:15] @ np.asarray(p["sensor_to_latent_weights"],float) + np.asarray(p["sensor_to_latent_bias"],float)) @ basis.T
    ridge_new = x @ np.asarray(robust6_cfg["feature_to_controls_weights"],float) + np.asarray(robust6_cfg["feature_to_controls_bias"],float)
    pca_new = mu + (x @ np.asarray(e["feature_to_latent_weights"],float) + np.asarray(e["feature_to_latent_bias"],float)) @ basis.T
    return {"ridge6_original":np.clip(ridge_old,0,1),
            "pca4_original":np.clip(pca_old,0,1),
            "ridge6_robust":np.clip(ridge_new,0,1),
            "pca4_robust":np.clip(pca_new,0,1)}


def scalar_calibration_weight(err_a, err_b, regularizer=0.03):
    """MSE-fitted mixture on calibration only; never uses external test labels."""
    a,b = np.asarray(err_a,float),np.asarray(err_b,float)
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=6:
        raise ValueError("Expected two matching [N,6] calibration residual arrays")
    d = b-a
    curvature = float(np.mean(np.sum(d*d,axis=1)))
    linear = float(np.mean(np.sum(a*d,axis=1)))
    lam = max(1.e-9, float(regularizer)*max(curvature,1.e-6))
    return float(np.clip((-linear + .5*lam) / (curvature+lam), 0.,1.))


def predict_blends(x, model_old, model_v2, model_robust6, condition):
    d = robust_predictions(x,model_old,model_v2,model_robust6)
    a,b = d["ridge6_robust"],d["pca4_robust"]
    w = float(model_robust6["calibrated_scalar_weights"][condition])
    d["fixed_50_50"] = np.clip(.5*a+.5*b,0,1)
    d["calibrated_scalar_blend"] = np.clip((1-w)*a+w*b,0,1)
    return d
