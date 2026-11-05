"""Stage A: leakage-aware, held-out expert complementarity audit for TQHG-v2.

No training occurs and no model weights are altered. Uses the SAME held-out
24 glove records, explicit faulty masks, and seed scheme as the original
scripts/evaluate_tqhg_v2_faults.py. Distinguishes *sample oracle* (diagnostic
lower bound using ground-truth labels) from a deployable predictor.

Without MuJoCo, J=0, P=0 and N=I; this audits whole-control gating, NOT
independent task/null-space contributions. Small fixed-pose test; do not use
sample-level win rates as independent-trial hypothesis tests.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG
from src.retargeting.tqhg_v2 import TQHGv2OnlineMapper, masked_features

CONDITIONS = (("clean", 0), ("mask_one", 1), ("mask_three", 3), ("mask_five", 5))


def classify(a: float, b: float, tie_epsilon: float = 1e-10) -> str:
    """Return per-record winner; a and b are per-record normalized MAE."""
    if a + tie_epsilon < b:
        return "ridge6"
    if b + tie_epsilon < a:
        return "robust4"
    return "tie"


def pearson_safe(a, b):
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(cases):
    """Diagnostics; wins are paired observations, not independent subjects."""
    if not cases:
        raise ValueError("Empty case group")
    x = np.asarray([[r["ridge6_mae"], r["robust4_mae"],
                     r["tqhg_v2_mae"], r["oracle_mae"], r["w_task"],
                     r["w_null"], r["sensor_reliability"]] for r in cases], float)
    maer, maeb, maem, oracle, task, null, rel = x.mean(axis=0)
    best = min(maer, maeb)
    out = {
        "cases": len(cases),
        "unique_poses": len(set(r["pose"] for r in cases)),
        "unique_test_frames": len(set(r["sample_index"] for r in cases)),
        "ridge6_mae": float(maer),
        "robust4_mae": float(maeb),
        "tqhg_v2_mae": float(maem),
        "best_single_expert_mae": float(best),
        "oracle_mae": float(oracle),
        "oracle_gain_over_best": float(best - oracle),
        "oracle_gain_over_best_pct": float((best-oracle)/best*100) if best else 0.,
        "tqhg_regret_vs_oracle": float(maem - oracle),
        "tqhg_minus_best_single": float(maem-best),
        "ridge6_win_pct": 100. * sum(r["winner"] == "ridge6" for r in cases)/len(cases),
        "robust4_win_pct": 100. * sum(r["winner"] == "robust4" for r in cases)/len(cases),
        "tie_pct": 100. * sum(r["winner"] == "tie" for r in cases)/len(cases),
        "abs_error_correlation": pearson_safe(x[:, 0], x[:, 1]),
        "w_task_mean": float(task),
        "w_null_mean": float(null),
        "sensor_reliability_mean": float(rel),
    }
    return out


def _write_csv(path, rows, cols):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(x, digits=6):
    return "N/A" if x is None else f"{x:.{digits}f}"


def evaluate(test_path: Path, model_path: Path, outdir: Path,
             *, trials: int = 20, seed: int = 20261025,
             tie_epsilon: float = 1e-10):
    if trials < 1:
        raise ValueError("--trials must be >= 1")
    if tie_epsilon < 0:
        raise ValueError("--tie-epsilon must be nonnegative")
    samples = load_captured_samples(test_path)
    if any(s.targets_deg is None for s in samples):
        raise ValueError("All independent test samples must contain six teacher targets")
    sensors = np.asarray([s.sensors_deg for s in samples], float)
    targets = np.asarray([s.targets_deg for s in samples], float) / CONTROL_LIMITS_DEG
    if not np.isfinite(sensors).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite teacher labels or sensor inputs")

    model = TQHGv2OnlineMapper(RobotHandModel(), model_path)
    W, b = model.W_ridge, model.b_ridge
    rng = np.random.default_rng(seed)
    all_rows = []

    for condition, missing_count in CONDITIONS:
        for trial in range(trials if missing_count else 1):
            for s, sample, truth in zip(sensors, samples, targets):
                mask = np.ones(15, dtype=bool)
                if missing_count:
                    mask[rng.choice(15, size=missing_count, replace=False)] = False
                # Match evaluate_tqhg_v2_faults.py: explicit mask and 15-channel
                # median imputation; both experts receive identical damaged input.
                model.reset()
                sn, observed, _, _ = model.quality.observe(s, mask)
                features = masked_features(sn, observed, model.medians)
                ridge = np.clip(features[:15] @ W + b, model.lower, model.upper)
                robust = np.clip(
                    model.mu + (features @ model.robust_W + model.robust_b)
                    @ model.robust_basis.T, model.lower, model.upper
                )
                model.reset()
                mixed = model.predict(s, method="tqhg-v2",
                                      sensor_valid_mask=mask, simulator=None)
                gate = model.last_gate
                ea, eb, em = (float(np.mean(np.abs(u-truth)))
                              for u in (ridge, robust, mixed))
                all_rows.append({
                    "condition": condition,
                    "missing_count": missing_count,
                    "trial": trial,
                    "sample_index": int(sample.sample_index),
                    "pose": sample.pose,
                    "missing_channels_1based": ";".join(str(j+1) for j in np.flatnonzero(~mask)),
                    "ridge6_mae": ea,
                    "robust4_mae": eb,
                    "tqhg_v2_mae": em,
                    "oracle_mae": min(ea, eb),
                    "winner": classify(ea, eb, tie_epsilon),
                    "tqhg_regret": em-min(ea, eb),
                    "expert_abs_error_gap": abs(ea-eb),
                    "expert_prediction_l2": float(np.linalg.norm(ridge-robust)),
                    "w_task": float(gate["w_task"]),
                    "w_null": float(gate["w_null"]),
                    "visible_rank": int(gate["visible_rank"]),
                    "sensor_reliability": float(gate["sensor_reliability"]),
                })
    outdir.mkdir(parents=True, exist_ok=True)
    conditions = []
    poses = []
    for name, count in CONDITIONS:
        group = [r for r in all_rows if r["condition"] == name]
        conditions.append({"condition": name, "missing_count": count, **summarize(group)})
        bypose = defaultdict(list)
        for r in group:
            bypose[r["pose"]].append(r)
        for pose, subset in bypose.items():
            poses.append({"condition": name, "pose": pose, **summarize(subset)})

    case_cols = list(all_rows[0])
    summary_cols = list(conditions[0])
    pose_cols = list(poses[0])
    _write_csv(outdir / "paired_cases.csv", all_rows, case_cols)
    _write_csv(outdir / "summary_by_condition.csv", conditions, summary_cols)
    _write_csv(outdir / "summary_by_pose.csv", poses, pose_cols)
    metadata = {
        "test_file": str(test_path),
        "model_file": str(model_path),
        "test_count": len(samples),
        "poses": sorted(set(s.pose for s in samples)),
        "trials_per_corruption": trials,
        "random_seed": seed,
        "tie_epsilon": tie_epsilon,
        "total_evaluations": len(all_rows),
        "test_label": "software 6D teacher output, normalized; NOT physical ground truth",
        "oracle": "per-case minimum of expert 6D MAE using labels; NOT deployable",
        "task_projector": "no MuJoCo; P=0, N=I, w_task=0; not a task/null dual-gate test",
        "independence_warning": "24 fixed-pose frames; repeated masks NOT independent human trials",
        "no_retraining": True,
        "by_condition": conditions,
    }
    (outdir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# TQHG-v2 Stage A | Expert complementarity diagnostic",
        "",
        f"- Held-out records: **{len(samples)}**, across **{len(metadata['poses'])} fixed poses**.",
        f"- Missing-mask replications: **{trials}** per corruption condition; seed **{seed}**.",
        "- No training or calibration updates; all methods receive paired fault masks.",
        "- Errors: normalized six-channel *software teacher-label* MAE, not robot ground truth.",
        "- No MuJoCo: **P=0, N=I**. Thus `w_null` is a whole-control gate, **not evidence of task/null separation**.",
        "- Oracle uses the test labels to pick the better expert per record. It is an **unattainable diagnostic lower bound**, not a deployable algorithm.",
        "- Repeated masks on 24 fixed-pose records must not be treated as statistically independent human movements.",
        "",
        "## Per-condition paired results",
        "",
        "| Fault condition | Ridge6 MAE | Robust4 MAE | TQHG-v2 MAE | Oracle MAE | Ridge win % | Robust win % | Oracle opportunity % | TQHG vs best single | Error corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in conditions:
        lines.append(
            f"| {x['condition']} | {_fmt(x['ridge6_mae'])} | {_fmt(x['robust4_mae'])} | {_fmt(x['tqhg_v2_mae'])} | "
            f"{_fmt(x['oracle_mae'])} | {_fmt(x['ridge6_win_pct'],1)} | {_fmt(x['robust4_win_pct'],1)} | "
            f"{_fmt(x['oracle_gain_over_best_pct'],2)} | {_fmt(x['tqhg_minus_best_single'])} | {_fmt(x['abs_error_correlation'],3)} |"
        )
    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- `oracle_gain_over_best_pct` measures how much an impossible per-case chooser could improve on the better globally chosen expert. A small number indicates limited exploitable complementarity on THIS test.",
        "- `tqhg_minus_best_single > 0`: TQHG is worse than the best expert selected per condition; `<= 0`: equal/better. This does not establish causal gate value without controlled ablations.",
        "- `ridge6_win_pct` and `robust4_win_pct` are proportions of paired **record-mask cases**, not subject-level rates.",
        "- `abs_error_correlation` is the Pearson correlation of casewise error magnitudes; a positive correlation may largely reflect shared case difficulty and alone does not disprove complementarity.",
        "- Inspect `summary_by_pose.csv` for which gestures favor each expert; do not tune weights on the held-out file.",
        "",
        "## Next experiment (NOT done here)",
        "",
        "Train a mask-augmented Ridge6 on the 124 historical training samples, calibrate only on 66 historical calibration samples, and compare against robust4 on NEW held-out movement sequences. Use actual CH-M6 task Jacobians and α/β ablations only afterward.",
    ])
    (outdir / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    print(f"INDEPENDENT TEST: {len(samples)} records, {len(metadata['poses'])} poses, {trials} masks per fault condition, seed={seed}")
    print("MAE: normalized 6D software-teacher labels; NO MuJoCo task projector; P=0, N=I.")
    print("condition       ridge6   robust4   tqhg-v2   oracle   A_win%   B_win%   oracle_gain%   tqhg-minus-best")
    for x in conditions:
        print(f"{x['condition']:<15} {x['ridge6_mae']:.6f}  {x['robust4_mae']:.6f}  {x['tqhg_v2_mae']:.6f}  "
              f"{x['oracle_mae']:.6f}  {x['ridge6_win_pct']:6.2f}   {x['robust4_win_pct']:6.2f}  "
              f"{x['oracle_gain_over_best_pct']:7.2f}       {x['tqhg_minus_best_single']:+.6f}")
    print(f"[DONE] Outputs: {outdir.resolve()}")
    print("[WARNING] Oracle sees test labels; repeated masks are not independent human trials.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "configs/tqhg_online_v2.json")
    parser.add_argument("--test", type=Path, default=ROOT / "数据手套数据/captured_samples(1).xlsx")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--tie-epsilon", type=float, default=1e-10)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/stageA_complementarity")
    args = parser.parse_args()
    evaluate(args.test, args.model, args.outdir, trials=args.trials,
             seed=args.seed, tie_epsilon=args.tie_epsilon)


if __name__ == "__main__":
    main()
