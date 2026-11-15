"""Focused Scheme-E ablation suite.

Separates three questions:
1) Dynamic task-effect mode selection vs fixed k=4/5/6 (same E mapper).
2) Analytic bounded ray step vs optional small QP refinement (same dynamic selector).
3) Final nonlinear guard on/off (same dynamic selector and analytic step).

The original 11-DoF and fixed-PCA4 rows are contextual baselines, not perfectly
matched optimization-space comparisons. Repeats measure run-to-run timing
variation; they are NOT additional independent posture/subject observations.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper

CONFIGS = {
    "original_direct": None,
    "fixed_PCA4": None,
    "E_dynamic": "configs/task_effect_spectrum.yaml",
    "E_dynamic_strict": "configs/task_effect_spectrum_dynamic_strict.yaml",
    "E_dynamic_relaxed": "configs/task_effect_spectrum_dynamic_relaxed.yaml",
    "E_fixed4": "configs/task_effect_spectrum_fixed4.yaml",
    "E_fixed5": "configs/task_effect_spectrum_fixed5.yaml",
    "E_fixed6": "configs/task_effect_spectrum_fixed6.yaml",
    "E_dynamic_no_guard": "configs/task_effect_spectrum_no_final_guard.yaml",
    "E_dynamic_qp": "configs/task_effect_spectrum_qp.yaml",
}
# QP is an optional solver-only diagnostic. It is excluded from the default
# suite because SLSQP can have high per-frame variance on some inputs.
DEFAULT_MODES = [m for m in CONFIGS if m != "E_dynamic_qp"]
PAIRED = [
    ("dynamic_minus_fixed4", "E_dynamic", "E_fixed4"),
    ("dynamic_minus_strict_threshold", "E_dynamic", "E_dynamic_strict"),
    ("dynamic_minus_relaxed_threshold", "E_dynamic", "E_dynamic_relaxed"),
    ("dynamic_minus_fixed5", "E_dynamic", "E_fixed5"),
    ("dynamic_minus_fixed6", "E_dynamic", "E_fixed6"),
    ("dynamic_minus_PCA4", "E_dynamic", "fixed_PCA4"),
    ("dynamic_minus_direct11", "E_dynamic", "original_direct"),
    ("QP_minus_analytic", "E_dynamic_qp", "E_dynamic"),
    ("no_guard_minus_guard", "E_dynamic_no_guard", "E_dynamic"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_new_output(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if overwrite:
            shutil.rmtree(path)
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_name(f"{path.name}_{stamp}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_for(mode: str, python: str, sample: Path, out: Path,
                steps: int, source: str, run_name: str) -> list[str]:
    cmd = [python, "-m", "src.main", "--input", "glove", "--sim", "none",
           "--steps", str(steps), "--glove-samples", str(sample),
           "--output-dir", str(out), "--run-name", run_name]
    config = CONFIGS[mode]
    if mode == "fixed_PCA4":
        cmd += ["--latent-model", "configs/latent_chm6_adaptive.json",
                "--latent-source", source]
    elif config is not None:
        cmd += ["--latent-model", "configs/latent_chm6_adaptive.json",
                "--latent-source", source,
                "--task-effect-spectrum", "--spectrum-config", config]
    return cmd


def read_run(path: Path, steps: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing run CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != steps:
        raise ValueError(f"{path.name}: expected {steps} rows, found {len(rows)}")
    frames = [int(r["frame"]) for r in rows]
    if frames != list(range(steps)):
        raise ValueError(f"{path.name}: frames do not align to 0..{steps-1}")
    return rows


def finite_values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def score_run(mode: str, repeat: int, rows: list[dict[str, str]], captures: list,
              mapper: LatentSynergyMapper, elapsed_s: float, csv_path: Path,
              log_path: Path, deadline_ms: float = 1000.0 / 60.0
              ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    joint_names = mapper.robot.joint_order
    active_indices = [mapper.teacher.index[name] for name in mapper.teacher.active_order]
    q = np.asarray([[float(row[f"q_{name}"]) for name in joint_names] for row in rows], dtype=float)
    labels = np.stack([sample.targets_deg for sample in captures]).astype(float)
    controls = np.rad2deg(q[:, active_indices])
    if controls.shape != labels.shape or not np.isfinite(controls).all():
        raise ValueError(f"{mode}: output/label shape or finite-value check failed")
    frame_mae = np.mean(np.abs(controls - labels), axis=1)
    opt = finite_values(rows, "optimizer_ms")
    pipeline = finite_values(rows, "pipeline_latency_ms")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")
    elapsed_ms = np.asarray([float(getattr(x, "elapsed_ms", 0.0)) for x in captures], dtype=float)
    # Captured sequences are ordered time-series holds. Compute within-sequence
    # command jitter/velocity/acceleration without treating pose transitions as jitter.
    seq = np.asarray([int(getattr(x, "pose_sequence_index", -1)) for x in captures])
    dt_s = np.diff(elapsed_ms) / 1000.0
    dq = np.diff(controls, axis=0)
    valid_dt = dt_s > 1e-6
    step_rms = np.full(len(rows), np.nan)
    velocity_rms = np.full(len(rows), np.nan)
    for i in range(1, len(rows)):
        if seq[i] == seq[i-1] and valid_dt[i-1]:
            step_rms[i] = float(np.sqrt(np.mean(dq[i-1] ** 2)))
            velocity_rms[i] = float(np.sqrt(np.mean((dq[i-1] / dt_s[i-1]) ** 2)))
    same_seq_pairs = np.isfinite(step_rms)
    sequence_step_values = step_rms[same_seq_pairs]
    sequence_velocity_values = velocity_rms[same_seq_pairs]
    deadline_miss_rate = float(np.mean(pipeline > deadline_ms))
    deadline_miss_count = int(np.sum(pipeline > deadline_ms))
    if len(opt) != len(rows) or len(pipeline) != len(rows) or np.any(opt < 0) or np.any(pipeline < 0):
        raise ValueError(f"{mode}: missing or invalid timing values")

    # Coupling and joint-limit compliance are evaluated on actual emitted q.
    coupling_err = max(
        float(np.max(np.abs(q[:, mapper.teacher.index[child]] -
                            gain * q[:, mapper.teacher.index[driver]])))
        for driver, child, gain in mapper.teacher.couplings
    )
    joint_ranges = mapper.robot.ranges
    limit_violations = int(np.sum((q < joint_ranges[:, 0] - 1e-8) |
                                  (q > joint_ranges[:, 1] + 1e-8)))
    row0 = rows[0]
    is_e = mode.startswith("E_")
    metric = {
        "repeat": repeat, "policy": mode, "frame_count": len(rows),
        "teacher_label_mae_deg": float(np.mean(frame_mae)),
        "teacher_label_rmse_deg": float(np.sqrt(np.mean((controls-labels)**2))),
        "optimizer_median_ms": float(np.median(opt)),
        "optimizer_p95_ms": float(np.percentile(opt, 95)),
        "pipeline_median_ms": float(np.median(pipeline)),
        "pipeline_p95_ms": float(np.percentile(pipeline, 95)),
        "pipeline_p99_ms": float(np.percentile(pipeline, 99)),
        "deadline_ms": float(deadline_ms),
        "deadline_miss_count": deadline_miss_count,
        "deadline_miss_rate": deadline_miss_rate,
        "deadline_miss_max_overrun_ms": float(max(0.0, np.max(pipeline) - deadline_ms)),
        "within_sequence_command_step_rms_deg": float(np.nanmean(sequence_step_values)) if len(sequence_step_values) else None,
        "within_sequence_command_velocity_rms_deg_s": float(np.nanmean(sequence_velocity_values)) if len(sequence_velocity_values) else None,
        "within_sequence_pair_count": int(np.sum(same_seq_pairs)),
        "run_wall_s": float(elapsed_s),
        "coupling_error_max_rad": coupling_err,
        "joint_limit_violations": limit_violations,
        "csv_path": str(csv_path), "log_path": str(log_path),
    }
    mode_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for i, (row, sample, err) in enumerate(zip(rows, captures, frame_mae)):
        frame_rows.append({"repeat": repeat, "policy": mode, "frame": i,
                           "pose": str(sample.pose), "sample_index": int(sample.sample_index),
                           "pose_sequence_index": int(getattr(sample, "pose_sequence_index", -1)),
                           "pose_frame_index": int(getattr(sample, "pose_frame_index", -1)),
                           "elapsed_ms": float(getattr(sample, "elapsed_ms", 0.0)),
                           "teacher_mae_deg": float(err),
                           "command_step_rms_deg": None if not np.isfinite(step_rms[i]) else float(step_rms[i]),
                           "command_velocity_rms_deg_s": None if not np.isfinite(velocity_rms[i]) else float(velocity_rms[i]),
                           "optimizer_ms": float(opt[i]), "pipeline_latency_ms": float(pipeline[i]),
                           "deadline_miss": int(pipeline[i] > deadline_ms)})
        if is_e:
            def number(key: str, default=float("nan")):
                try:
                    return float(row.get(key, default))
                except (TypeError, ValueError):
                    return default
            mode_rows.append({
                "repeat": repeat, "policy": mode, "frame": i, "pose": str(sample.pose),
                "requested_k": number("spectrum_selected_k"),
                "accepted_k": number("spectrum_accepted_k"),
                "output_rank": number("spectrum_output_rank"),
                "fallback": number("spectrum_fallback"),
                "fallback_reason": row.get("spectrum_fallback_reason", ""),
                "selected_indices": row.get("spectrum_selected_indices", ""),
                "spectral_omission_m": number("spectrum_omission_m"),
                "selected_energy_fraction": number("spectrum_energy"),
                "final_guard_enabled": number("spectrum_final_guard_enabled"),
                "final_guard_used": number("spectrum_post_guard"),
                "final_guard_regret_m": number("spectrum_final_guard_regret_m"),
                "final_fk_error_m": number("spectrum_post_tip_m"),
                "reference_fk_error_m": number("spectrum_post_baseline_m"),
                "speed_box_valid": number("spectrum_speed_valid"),
                "candidate_attempts": number("spectrum_candidate_attempts"),
                "reject_spectral_count": number("spectrum_reject_spectral_count"),
                "reject_fidelity_count": number("spectrum_reject_fidelity_count"),
                "reject_nonlinear_count": number("spectrum_reject_nonlinear_count"),
                "candidate_min_fk_error_m": number("spectrum_candidate_min_fk_error_m"),
                "candidate_min_regret_m": number("spectrum_candidate_min_regret_m"),
                "candidate_max_fidelity_excess_deg": number("spectrum_candidate_max_fidelity_excess_deg"),
                "selection_status": row.get("spectrum_selection_status", ""),
            })

    if is_e:
        req = finite_values(rows, "spectrum_selected_k")
        acc = finite_values(rows, "spectrum_accepted_k")
        rank = finite_values(rows, "spectrum_output_rank")
        fallback = finite_values(rows, "spectrum_fallback")
        guard = finite_values(rows, "spectrum_post_guard")
        omission = finite_values(rows, "spectrum_omission_m")
        energy = finite_values(rows, "spectrum_energy")
        regret = finite_values(rows, "spectrum_final_guard_regret_m")
        accepted_nonzero = acc[acc > 0]
        metric.update({
            "requested_k_mean": float(np.mean(req)) if len(req) else None,
            "accepted_k_mean_including_fallback_zero": float(np.mean(acc)) if len(acc) else None,
            "accepted_k_mean_when_candidate_accepted": float(np.mean(accepted_nonzero)) if len(accepted_nonzero) else None,
            "output_rank_mean": float(np.mean(rank)) if len(rank) else None,
            "requested_k_lt6_fraction": float(np.mean(req < 6)) if len(req) else None,
            "actual_output_low_rank_fraction": float(np.mean(rank < 6)) if len(rank) else None,
            "accepted_fallback_rate": float(np.mean(fallback)) if len(fallback) else None,
            "final_guard_trigger_rate": float(np.mean(guard)) if len(guard) else None,
            "spectral_omission_mean_mm": float(np.mean(omission) * 1000) if len(omission) else None,
            "selected_energy_fraction_mean": float(np.mean(energy)) if len(energy) else None,
            "final_guard_regret_p95_mm": float(np.percentile(regret, 95) * 1000) if len(regret) else None,
            "requested_k_histogram": {str(k): int(np.sum(req == k)) for k in range(1, 7)},
            "accepted_k_histogram": {str(k): int(np.sum(acc == k)) for k in range(0, 7)},
            "fallback_reasons": dict(Counter(r.get("spectrum_fallback_reason", "") for r in rows)),
            "reject_spectral_total": int(np.nansum(finite_values(rows, "spectrum_reject_spectral_count"))),
            "reject_fidelity_total": int(np.nansum(finite_values(rows, "spectrum_reject_fidelity_count"))),
            "reject_nonlinear_total": int(np.nansum(finite_values(rows, "spectrum_reject_nonlinear_count"))),
            "candidate_attempts_total": int(np.nansum(finite_values(rows, "spectrum_candidate_attempts"))),
        })
    return metric, frame_rows, mode_rows


def pose_bootstrap(deltas_by_pose: dict[str, list[float]], seed: int, n_boot: int) -> list[float] | None:
    poses = sorted(deltas_by_pose)
    if len(poses) < 2:
        return None
    means = np.asarray([np.mean(deltas_by_pose[p]) for p in poses], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(means[rng.integers(0, len(means), size=(n_boot, len(means)))], axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    sample_path = args.samples.resolve()
    captures = load_captured_samples(sample_path)
    if args.steps < 1 or args.steps > len(captures):
        raise ValueError(f"--steps must be 1..{len(captures)}; samples are never silently repeated")
    captures = captures[:args.steps]
    if any(x.targets_deg is None for x in captures):
        raise ValueError("all selected samples need teacher labels")
    if len({x.pose for x in captures}) < 2:
        raise ValueError("need at least two pose clusters for paired sensitivity analysis")

    modes = args.modes.split(",") if args.modes else DEFAULT_MODES
    if len(modes) != len(set(modes)) or any(m not in CONFIGS for m in modes):
        raise ValueError(f"invalid --modes; choose from {list(CONFIGS)}")
    if args.repeats < 1 or args.bootstrap < 100:
        raise ValueError("repeats must be >=1 and bootstrap >=100")

    out = ensure_new_output(args.out, args.overwrite)
    run_root = out / "runs"
    run_root.mkdir()
    mapper = LatentSynergyMapper(RobotHandModel(), ROOT / "configs/latent_chm6_adaptive.json")
    active_indices = [mapper.teacher.index[name] for name in mapper.teacher.active_order]
    joint_names = mapper.robot.joint_order
    data_hash = sha256(sample_path)
    config_hashes = {m: (sha256(ROOT / CONFIGS[m]) if CONFIGS[m] else None) for m in modes}
    manifest: dict[str, Any] = {
        "schema": "scheme_e_focused_ablation_v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_path": str(sample_path), "sample_sha256": data_hash,
        "sample_count": len(captures), "pose_clusters": sorted({str(x.pose) for x in captures}),
        "repeats": args.repeats, "seed": args.seed, "source": args.source,
        "deadline_ms": float(args.deadline_ms),
        "modes": modes, "config_sha256": config_hashes,
        "scope": "offline software teacher-label replay only; not measured robot tracking or physical grasp success",
        "deadline_scope": "60Hz frame budget on recorded software pipeline latency; not a hard real-time OS guarantee",
        "commands": [], "complete": False,
        "notes": [
            "Repeats repeat the same captured samples and are computational repeats, not new independent observations.",
            "Pose-cluster bootstrap is exploratory only; six pose groups do not support population-level significance.",
            "Direct 11-DoF and fixed-PCA baselines have different control manifolds; timing comparisons are descriptive.",
            "Dynamic-vs-fixed-k E variants share the same E mapper and feasibility/rate checks; only the spectral dimension rule differs.",
            "Analytic-vs-QP compares the same E selector with a different within-mode refinement solver.",
        ]
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_run_metrics: list[dict[str, Any]] = []
    all_frame_metrics: list[dict[str, Any]] = []
    all_mode_metrics: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for repeat in range(1, args.repeats + 1):
        repeat_dir = run_root / f"repeat_{repeat:02d}"
        repeat_dir.mkdir()
        order = list(modes)
        random.Random(args.seed + repeat * 7919).shuffle(order)
        manifest.setdefault("run_order", {})[str(repeat)] = order
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for mode in order:
            run_name = f"{mode}"
            command = command_for(mode, sys.executable, sample_path, repeat_dir,
                                  args.steps, args.source, run_name)
            log_path = repeat_dir / f"{mode}.log"
            cmd_record = {"repeat": repeat, "mode": mode, "argv": command,
                          "exit_code": None, "wall_s": None, "log": str(log_path)}
            manifest["commands"].append(cmd_record)
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            print(f"[RUN] repeat={repeat}/{args.repeats} mode={mode}", flush=True)
            start = time.perf_counter()
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log:
                    proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          check=False, env=env, timeout=args.process_timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                cmd_record["exit_code"] = "timeout"
                cmd_record["wall_s"] = elapsed
                cmd_record["error"] = f"process exceeded {args.process_timeout}s"
                (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(f"{mode} repeat {repeat} exceeded {args.process_timeout}s; see {log_path}")
            elapsed = time.perf_counter() - start
            cmd_record["exit_code"] = proc.returncode
            cmd_record["wall_s"] = elapsed
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"{mode} repeat {repeat} failed ({proc.returncode}); see {log_path}")
            csv_path = repeat_dir / f"{mode}.csv"
            rows = read_run(csv_path, args.steps)
            metrics, frame_metrics, mode_metrics = score_run(
                mode, repeat, rows, captures, mapper, elapsed, csv_path, log_path,
                deadline_ms=args.deadline_ms)
            all_run_metrics.append(metrics)
            all_frame_metrics.extend(frame_metrics)
            all_mode_metrics.extend(mode_metrics)
            print(f"[OK] {mode}: MAE={metrics['teacher_label_mae_deg']:.3f} deg; "
                  f"optimizer median={metrics['optimizer_median_ms']:.2f} ms; "
                  f"wall={elapsed:.2f}s", flush=True)

    # Aggregate repeats by method. The same data are repeated for timing only;
    # report frame count and pose count once, never inflate sample size.
    aggregate: list[dict[str, Any]] = []
    for mode in modes:
        group = [r for r in all_run_metrics if r["policy"] == mode]
        keys = ["teacher_label_mae_deg", "teacher_label_rmse_deg", "optimizer_median_ms",
                "optimizer_p95_ms", "pipeline_median_ms", "pipeline_p95_ms", "pipeline_p99_ms",
                "deadline_miss_rate", "deadline_miss_max_overrun_ms",
                "within_sequence_command_step_rms_deg", "within_sequence_command_velocity_rms_deg_s",
                "run_wall_s", "coupling_error_max_rad", "joint_limit_violations"]
        row: dict[str, Any] = {"policy": mode, "n_repeated_runs": len(group), "frame_count": args.steps,
                               "pose_cluster_count": len({str(x.pose) for x in captures})}
        for key in keys:
            values = np.asarray([float(x[key]) for x in group], dtype=float)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_sd_between_runs"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if mode.startswith("E_"):
            for key in ["requested_k_mean", "accepted_k_mean_including_fallback_zero",
                        "accepted_k_mean_when_candidate_accepted", "output_rank_mean",
                        "requested_k_lt6_fraction", "actual_output_low_rank_fraction",
                        "accepted_fallback_rate", "final_guard_trigger_rate",
                        "spectral_omission_mean_mm", "selected_energy_fraction_mean", "final_guard_regret_p95_mm",
                        "reject_spectral_total", "reject_fidelity_total", "reject_nonlinear_total",
                        "candidate_attempts_total"]:
                vals = [float(x[key]) for x in group if x.get(key) is not None]
                row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            row["requested_k_histogram_total"] = dict(Counter(
                k for x in group for k, n in (x.get("requested_k_histogram") or {}).items()
                for _ in range(int(n))))
            row["accepted_k_histogram_total"] = dict(Counter(
                k for x in group for k, n in (x.get("accepted_k_histogram") or {}).items()
                for _ in range(int(n))))
        aggregate.append(row)

    # Average repeated computations per sample first, then do pose-cluster paired
    # analysis. This avoids treating repeated identical samples as independent.
    by_mode_frame: dict[tuple[str, int], list[float]] = defaultdict(list)
    pose_by_frame = {x.sample_index: str(x.pose) for x in captures}
    for row in all_frame_metrics:
        by_mode_frame[(row["policy"], row["frame"])].append(float(row["teacher_mae_deg"]))
    mean_frame = {key: float(np.mean(values)) for key, values in by_mode_frame.items()}
    pose_deltas_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    for comparison, candidate, baseline in PAIRED:
        if candidate not in modes or baseline not in modes:
            continue
        frame_deltas: dict[int, float] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for frame, sample in enumerate(captures):
            key_a, key_b = (candidate, frame), (baseline, frame)
            if key_a not in mean_frame or key_b not in mean_frame:
                raise ValueError(f"unpaired frame {frame} for {comparison}")
            delta = mean_frame[key_a] - mean_frame[key_b]
            frame_deltas[frame] = delta
            grouped[str(sample.pose)].append(delta)
        pose_means = {pose: float(np.mean(values)) for pose, values in grouped.items()}
        interval = pose_bootstrap(grouped, args.seed + len(paired_summaries), args.bootstrap)
        run_a = [x for x in all_run_metrics if x["policy"] == candidate]
        run_b = [x for x in all_run_metrics if x["policy"] == baseline]
        summary = {
            "comparison": comparison, "candidate": candidate, "baseline": baseline,
            "candidate_teacher_mae_deg": float(np.mean([x["teacher_label_mae_deg"] for x in run_a])),
            "baseline_teacher_mae_deg": float(np.mean([x["teacher_label_mae_deg"] for x in run_b])),
            "mean_paired_frame_delta_deg": float(np.mean(list(frame_deltas.values()))),
            "mean_pose_paired_delta_deg": float(np.mean(list(pose_means.values()))),
            "pose_cluster_count": len(pose_means), "paired_frame_count": args.steps,
            "pose_cluster_bootstrap_95_interval_deg": interval,
            "candidate_optimizer_median_ms": float(np.mean([x["optimizer_median_ms"] for x in run_a])),
            "baseline_optimizer_median_ms": float(np.mean([x["optimizer_median_ms"] for x in run_b])),
            "candidate_pipeline_p95_ms": float(np.mean([x["pipeline_p95_ms"] for x in run_a])),
            "baseline_pipeline_p95_ms": float(np.mean([x["pipeline_p95_ms"] for x in run_b])),
            "interpretation": "negative delta favors candidate on preset teacher labels; interval exploratory only",
        }
        paired_summaries.append(summary)
        for pose, delta in pose_means.items():
            pose_deltas_rows.append({"comparison": comparison, "candidate": candidate,
                                     "baseline": baseline, "pose": pose,
                                     "mean_candidate_minus_baseline_deg": delta,
                                     "frame_count": len(grouped[pose])})

    write_csv(out / "run_summary.csv", all_run_metrics)
    write_csv(out / "aggregate.csv", aggregate)
    write_csv(out / "per_frame_metrics.csv", all_frame_metrics)
    write_csv(out / "mode_selection.csv", all_mode_metrics)
    write_csv(out / "paired_pose_deltas.csv", pose_deltas_rows)
    write_csv(out / "paired_comparisons.csv", paired_summaries)

    summary = {
        "schema": "scheme_e_focused_ablation_results_v2",
        "sample_sha256": data_hash, "sample_count": args.steps,
        "pose_clusters": sorted({str(x.pose) for x in captures}),
        "repeats": args.repeats, "modes": modes,
        "aggregate": aggregate, "paired_comparisons": paired_summaries,
        "interpretation_limits": manifest["notes"],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# 方案 E 深化消融报告", "",
        f"- 数据：`{sample_path.name}`；SHA256：`{data_hash}`。",
        f"- 帧数：{args.steps}；姿态簇：{len({x.pose for x in captures})}；计算重复：{args.repeats}。",
        "- 指标为相对预设软件教师标签的离线控制角度误差，不是真实机器人关节真值或抓握成功率。",
        "- 重复运行仅用于计时波动估计；相同帧/姿态不会因重复运行而增加独立样本量。",
        "- Direct 11DoF 与 E 的六维耦合控制空间不同，作为背景基线而非严格同约束竞赛。", "",
        "## 运行汇总", "",
        "| 方法 | MAE° | 优化器中位 ms | Pipeline P95 ms | 60Hz miss | 请求 k | 接受时 k | 输出秩 | 实际低秩比例 | fallback 率 | guard 率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in aggregate:
        def fmt(key, digits=3):
            v = r.get(key)
            return "—" if v is None else f"{float(v):.{digits}f}"
        report.append("| {policy} | {mae} | {opt} | {p95} | {deadline} | {req} | {acc} | {rank} | {lowrank} | {fallback} | {guard} |".format(
            policy=r["policy"], mae=fmt("teacher_label_mae_deg_mean"),
            opt=fmt("optimizer_median_ms_mean", 2), p95=fmt("pipeline_p95_ms_mean", 2),
            deadline=fmt("deadline_miss_rate_mean", 3),
            req=fmt("requested_k_mean_mean"),
            acc=fmt("accepted_k_mean_when_candidate_accepted_mean"),
            rank=fmt("output_rank_mean_mean"),
            lowrank=fmt("actual_output_low_rank_fraction_mean", 3),
            fallback=fmt("accepted_fallback_rate_mean"), guard=fmt("final_guard_trigger_rate_mean")))
    # Explicitly report the gap between requested spectral dimension and
    # the dimension actually emitted after fidelity/nonlinear checks/fallback.
    dynamic_rows = [x for x in all_mode_metrics if x.get("policy") == "E_dynamic"]
    if dynamic_rows:
        def hist(key: str) -> Counter:
            return Counter(str(int(float(x[key]))) for x in dynamic_rows
                           if x.get(key) is not None and np.isfinite(float(x[key])))
        requested_hist = hist("requested_k")
        accepted_hist = hist("accepted_k")
        output_hist = hist("output_rank")
        lowrank = sum(1 for x in dynamic_rows if float(x.get("output_rank", 6)) < 6)
        fallback_n = sum(1 for x in dynamic_rows if float(x.get("fallback", 0)) >= 0.5)
        n_dynamic = len(dynamic_rows)
        report += ["", "## 动态选维行为诊断", "",
                   f"- E_dynamic 逐帧记录数：{n_dynamic}；请求 k<6：{sum(v for k, v in requested_hist.items() if int(k) < 6)}/{n_dynamic}（{sum(v for k, v in requested_hist.items() if int(k) < 6)/n_dynamic:.1%}）。",
                   f"- 最终实际输出秩<6：{lowrank}/{n_dynamic}（{lowrank/n_dynamic:.1%}）；fallback：{fallback_n}/{n_dynamic}（{fallback_n/n_dynamic:.1%}）。",
                   "| 维数 | requested_k | accepted_k | output_rank |",
                   "|---:|---:|---:|---:|"]
        for k in range(0, 7):
            vals = [str(k), str(requested_hist.get(str(k), 0)),
                    str(accepted_hist.get(str(k), 0)), str(output_hist.get(str(k), 0))]
            if any(int(v) for v in vals[1:]):
                report.append("| " + " | ".join(vals) + " |")
        reject_spectral = sum(int(float(x.get("reject_spectral_count", 0) or 0)) for x in dynamic_rows)
        reject_fidelity = sum(int(float(x.get("reject_fidelity_count", 0) or 0)) for x in dynamic_rows)
        reject_nonlinear = sum(int(float(x.get("reject_nonlinear_count", 0) or 0)) for x in dynamic_rows)
        attempts = sum(int(float(x.get("candidate_attempts", 0) or 0)) for x in dynamic_rows)
        report += ["", "### 候选拒绝原因（次数可重叠）", "",
                   f"- 候选尝试：{attempts}；谱预算拒绝：{reject_spectral}；编码器保真/速度盒拒绝：{reject_fidelity}；非线性 FK/regret 拒绝：{reject_nonlinear}。",
                   "- 拒绝次数按候选尝试统计，同一候选可能同时违反多个条件，不能把三类次数相加后当作互斥样本量。",
                   "- `selection_status` 区分请求维数直接通过、升维后通过、回退到完整参考命令。", "",
                   "解释：requested_k 是选择器提出的维数；accepted_k 是同时通过谱预算、编码器保真度、非线性 FK/regret 与命令速度/关节盒约束的维数，回退时为 0；output_rank 是最终实际输出秩。三者不能互相替代。", ""]

    report += ["", "## 配对对比（候选 − 基线）", "",
               "负值表示候选方法的离线教师标签 MAE 较低；区间按姿态簇重采样，仅作探索性敏感度分析。", "",
               "| 对比 | MAE 差° | 姿态簇差值° | 95% 姿态簇区间° | 优化器中位 ms（候选/基线） | Pipeline P95 ms（候选/基线） |",
               "|---|---:|---:|---|---:|---:|"]
    for p in paired_summaries:
        ci = p["pose_cluster_bootstrap_95_interval_deg"]
        ci_text = "不可估计" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        report.append(f"| {p['comparison']} | {p['mean_paired_frame_delta_deg']:+.3f} | "
                      f"{p['mean_pose_paired_delta_deg']:+.3f} | {ci_text} | "
                      f"{p['candidate_optimizer_median_ms']:.2f}/{p['baseline_optimizer_median_ms']:.2f} | "
                      f"{p['candidate_pipeline_p95_ms']:.2f}/{p['baseline_pipeline_p95_ms']:.2f} |")
    report += ["", "## 解释重点", "",
               "1. `E_dynamic` vs `E_fixed4/5/6` 是本实验最关键的受控比较：它检验任务效应谱选维是否优于相同 E 求解链路下的固定维度。",
               "2. `E_dynamic_strict/relaxed` 只改变谱截断预算，用于观察维数压缩、fallback 与误差之间的敏感性；不能在同一 24 帧上挑选最优阈值后宣称泛化。",
               "3. `E_dynamic_qp` vs `E_dynamic` 是可选诊断，只改变子空间内求解器（解析有界射线步进 vs 小型 SLSQP QP）；QP 不在本次默认重复套件中，不能据此写正式比较结论。",
               "4. `E_dynamic_no_guard` vs `E_dynamic` 检查最终非线性输出守卫的代价与收益；无守卫变体仅用于离线诊断，不作为推荐部署配置。",
               "5. 记录 `requested_k`、`accepted_k`、`output_rank` 三个独立量。旧字段 `spectrum_modes` 是输出秩，不应再当作请求选维数量。",
               "6. 若动态 k 经常等于 6，或 fallback 高，则当前证据不支持动态选维具有实质收益；应考虑固定维度或重新设计谱预算。",
               "7. `E_fixed6` 是真正的固定 k=6 E 消融；历史 `E_full6_feasible` 强制回退到参考控制，不等同于固定六维优化，此处不再用它作为主比较。", "",
               "## 输出文件", "",
               "- `run_summary.csv`：每次运行的汇总指标。",
               "- `aggregate.csv`：按方法聚合的重复运行结果。",
               "- `per_frame_metrics.csv`：逐帧标签误差与耗时。",
               "- `mode_selection.csv`：请求维度、接受维度、输出秩、fallback、候选拒绝原因、FK regret 与谱残差逐帧记录。",
               "- `per_frame_metrics.csv`：包含 60Hz deadline miss 与同一姿态序列内部的命令步长/速度抖动指标。",
               "- `paired_pose_deltas.csv` / `paired_comparisons.csv`：姿态配对差异。",
               "- `manifest.json`：数据/配置哈希、命令、日志与运行顺序。", "",
               "## 数据与研究边界", "",
               "- 目前只有 24 帧、6 个预设姿态；姿态簇 bootstrap 不是显著性证明。",
               "- 真实泛化需要按受试者/采集日期/连续序列隔离的独立测试集。",
               "- E 的谱最优性仅限代码中声明的局部线性模型与特定正交字典，不是全局最少自由度定理。",
               "- 风险协方差仍是描述性 proxy；没有独立噪声校准，不能宣称概率安全保证。", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    manifest["complete"] = True
    manifest["output_files"] = ["run_summary.csv", "aggregate.csv", "per_frame_metrics.csv",
                                 "mode_selection.csv", "paired_pose_deltas.csv",
                                 "paired_comparisons.csv", "summary.json", "report.md"]
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] Scheme E research results:", out / "report.md")
    print("[DONE] Paired comparisons:", out / "paired_comparisons.csv")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "数据手套数据/captured_samples(1).xlsx")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3, help="repeated compute runs; not new independent data")
    parser.add_argument("--source", choices=("sensor", "control", "blend"), default="sensor")
    parser.add_argument("--modes", default="", help="comma-separated policy subset")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/scheme_e_research")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--process-timeout", type=float, default=180.0,
                        help="timeout per individual mode/repeat subprocess")
    parser.add_argument("--deadline-ms", type=float, default=1000.0 / 60.0,
                        help="per-frame end-to-end budget; default 60 Hz = 16.667 ms")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace output directory if non-empty")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("[INPUT]", args.samples.resolve())
        print("[MODES]", args.modes.split(",") if args.modes else DEFAULT_MODES)
        print("[REPEATS]", args.repeats, "[STEPS]", args.steps, "[PROCESS TIMEOUT]", args.process_timeout)
        for m in (args.modes.split(",") if args.modes else DEFAULT_MODES):
            print("[PLAN]", m, "config=", CONFIGS[m])
        print("[DRY RUN] no benchmark executed")
        return 0
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
