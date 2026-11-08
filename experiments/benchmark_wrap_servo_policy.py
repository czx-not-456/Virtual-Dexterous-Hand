"""Run honest paired WRAP controller ablations against the actual MuJoCo pipeline.

Run from the repository root.  Requires the user's local MuJoCo installation.
All conditions use the same dataset trial, task, physics, contact detector and
stability criteria.  Only --wrap-servo-policy changes.  Summary data are read
from real generated CSV/JSON files, never fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("original", "low_gain", "thumb_recovery", "coordinated")
COLUMNS = (
    "trial", "policy", "contact_ratio", "stable_frame_ratio", "final_stable",
    "ever_stable", "longest_stable_frames", "max_relative_translation_mm",
    "max_relative_rotation_deg", "max_object_displacement_mm",
    "mean_servo_error_mm", "control_switches", "thumb_loss_events",
    "recovery_frames", "coordination_frames", "active_object_resets",
    "mean_finger_force_on_object_xy_n", "active_frames", "file_csv", "file_summary",
)


def parse_trials(spec: str) -> list[int]:
    try:
        parts = [int(x.strip()) for x in spec.split(",")]
    except ValueError as exc:
        raise ValueError("--trials needs comma-separated integer IDs, e.g. 0,1,2") from exc
    if not parts or min(parts) < 0 or len(set(parts)) != len(parts):
        raise ValueError("--trials needs unique, nonnegative IDs")
    return parts


def truthy(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def summarize_run(csv_file: Path, summary_file: Path, *, trial: int, policy: str,
                  expected_detector: str | None = None,
                  expected_dataset: Path | None = None,
                  expected_no_auto_reset: bool | None = None) -> dict:
    if not csv_file.is_file() or not summary_file.is_file():
        raise FileNotFoundError(f"Missing actual simulation output for trial={trial}, policy={policy}: {csv_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("wrap_servo_policy") != policy or int(summary.get("dataset_trial", -1)) != trial:
        raise ValueError(f"Output provenance does not match requested condition: {summary_file}")
    if summary.get("simulation") != "mujoco" or summary.get("task_mode") != "wrap":
        raise ValueError(f"Not an actual WRAP MuJoCo result: {summary_file}")
    if expected_detector is not None and summary.get("contact_detector") != expected_detector:
        raise ValueError(f"Contact detector differs between runs: {summary_file}")
    if expected_dataset is not None and Path(summary.get("dataset") or "").resolve() != expected_dataset.resolve():
        raise ValueError(f"Dataset path differs between runs: {summary_file}")
    if expected_no_auto_reset is not None and expected_no_auto_reset and bool(summary.get("auto_reset_object_effective", True)):
        raise ValueError(f"Object auto-reset was not disabled in {summary_file}")
    with csv_file.open(newline="", encoding="utf-8-sig") as fh:
        active = [r for r in csv.DictReader(fh) if r.get("intent") == "WRAP" and truthy(r.get("task_active"))]
    if not active:
        raise ValueError(f"No active WRAP frames in {csv_file}")
    if "raw_thumb_contact" not in active[0] or "servo_thumb_recovery_active" not in active[0]:
        raise ValueError(f"Missing new contact/servo diagnostic columns in {csv_file}")
    raw_last = False
    filt_last = False
    loss_events = 0
    control_switches = 0
    for i, record in enumerate(active):
        raw = truthy(record["raw_thumb_contact"])
        filt = truthy(record["filtered_task_contact_proxy"])
        if i and raw_last and not raw:
            loss_events += 1
        if i and filt != filt_last:
            control_switches += 1
        raw_last, filt_last = raw, filt
    hand_fingers = ("thumb", "index", "middle", "ring", "little")
    def force_xy(record):
        x = sum(number(record.get(f"{finger}_force_on_object_x_n")) for finger in hand_fingers)
        y = sum(number(record.get(f"{finger}_force_on_object_y_n")) for finger in hand_fingers)
        return math.hypot(x, y)
    return {
        "trial": trial,
        "policy": policy,
        "contact_ratio": number(summary.get("contact_frame_ratio")),
        "stable_frame_ratio": number(summary.get("stable_frame_ratio")),
        "final_stable": bool(summary.get("stable_success_proxy")),
        "ever_stable": bool(summary.get("ever_hold_stable_proxy")),
        "longest_stable_frames": int(summary.get("max_stable_success_streak_frames", 0)),
        "max_relative_translation_mm": 1000 * number(summary.get("hold_max_relative_translation_span_m")),
        "max_relative_rotation_deg": number(summary.get("hold_max_relative_rotation_span_deg")),
        "max_object_displacement_mm": 1000 * max(number(r.get("object_displacement_m")) for r in active),
        "mean_servo_error_mm": 1000 * statistics.mean(number(r.get("servo_mean_error_m")) for r in active),
        "control_switches": control_switches,
        "thumb_loss_events": loss_events,
        "recovery_frames": sum(truthy(r.get("servo_thumb_recovery_active")) for r in active),
        "coordination_frames": sum(truthy(r.get("servo_wrap_coordination_active")) for r in active),
        "active_object_resets": sum(truthy(r.get("object_reset_this_step")) for r in active),
        "mean_finger_force_on_object_xy_n": statistics.mean(force_xy(r) for r in active),
        "active_frames": len(active),
        "file_csv": str(csv_file),
        "file_summary": str(summary_file),
    }


def aggregate(rows: list[dict], trials: list[int], policies: list[str]) -> list[dict]:
    """Paired per-trial contrasts; no unwarranted statistical significance tests."""
    by_key = {(r["trial"], r["policy"]): r for r in rows}
    metrics = ("contact_ratio", "stable_frame_ratio", "longest_stable_frames",
               "max_relative_translation_mm", "max_relative_rotation_deg",
               "max_object_displacement_mm", "mean_servo_error_mm",
               "mean_finger_force_on_object_xy_n", "control_switches", "thumb_loss_events")
    out = []
    for policy in policies:
        subset = [by_key[(trial, policy)] for trial in trials]
        baseline = [by_key[(trial, "original")] for trial in trials] if "original" in policies else None
        item = {"policy": policy, "n_trials": len(subset),
                "final_stable_count": sum(r["final_stable"] for r in subset),
                "ever_stable_count": sum(r["ever_stable"] for r in subset)}
        for metric in metrics:
            vals = [float(r[metric]) for r in subset]
            item[metric + "_mean"] = statistics.mean(vals)
            item[metric + "_std"] = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
            if baseline is not None:
                item[metric + "_paired_delta_mean"] = statistics.mean(
                    float(s[metric]) - float(b[metric]) for s, b in zip(subset, baseline))
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict], columns=None) -> None:
    if not rows:
        return
    cols = list(columns or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired four-way WRAP MuJoCo servo ablation")
    parser.add_argument("--trials", default="0", help="Comma-separated synthetic trial IDs; default 0")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/synthetic_glove_v1.csv")
    parser.add_argument("--contact-detector", choices=["geometry_only", "force_only", "temporal_only", "force_and_temporal"], default="geometry_only")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wrap_servo_policy_ablation")
    parser.add_argument("--summarize-only", action="store_true", help="Read existing runs; never run MuJoCo")
    parser.add_argument("--no-auto-reset-object", action="store_true", help="Disable workspace-triggered object teleports for all conditions")
    parser.add_argument("--no-inactive-object-reset", action="store_true", help="Do not teleport object home after WRAP ends; all conditions same")
    parser.add_argument("--policies", default=",".join(POLICIES), help="Policy subset; for paired deltas include original")
    args = parser.parse_args(argv)
    trials = parse_trials(args.trials)
    policies = [x.strip() for x in args.policies.split(",")]
    if not policies or len(set(policies)) != len(policies) or any(x not in POLICIES for x in policies):
        parser.error("--policies must list unique valid policy names")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    dataset = args.dataset.resolve()
    if not dataset.exists():
        parser.error(f"Dataset not found: {dataset}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_paths = []
    for trial in trials:
        for policy in policies:
            name = f"wrap_trial{trial}_{policy}"
            file_csv = output / f"{name}.csv"
            file_summary = output / f"{name}_summary.json"
            run_paths.append((trial, policy, file_csv, file_summary))
            if args.summarize_only:
                continue
            cmd = [sys.executable, "-m", "src.main", "--sim", "mujoco", "--input", "dataset",
                   "--dataset", str(dataset), "--dataset-trial", str(trial),
                   "--task", "wrap", "--steps", str(args.steps),
                   "--contact-detector", args.contact_detector,
                   "--mapping-method", "legacy", "--wrap-servo-policy", policy,
                   "--output-dir", str(output), "--run-name", name]
            if args.no_auto_reset_object:
                cmd.append("--no-auto-reset-object")
            if args.no_inactive_object_reset:
                cmd.append("--no-inactive-object-reset")
            print(f"[RUN] trial={trial} WRAP policy={policy}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT, check=False)
            if result.returncode:
                print(f"[ERROR] Failed at trial={trial}, policy={policy}; exit={result.returncode}. "
                      f"Previously completed outputs retained.", file=sys.stderr)
                return result.returncode
    rows = [summarize_run(fcsv, fsum, trial=t, policy=pol,
                          expected_detector=args.contact_detector,
                          expected_dataset=dataset,
                          expected_no_auto_reset=args.no_auto_reset_object)
            for t, pol, fcsv, fsum in run_paths]
    aggregates = aggregate(rows, trials, policies)
    write_csv(output / "per_trial.csv", rows, COLUMNS)
    write_csv(output / "aggregate.csv", aggregates)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "config_sha256": {f: sha256(ROOT / f) for f in ("configs/algorithm.yaml", "configs/simulation.yaml")},
        "trials": trials, "policies": policies, "steps": args.steps,
        "contact_detector": args.contact_detector,
        "no_auto_reset_object": args.no_auto_reset_object,
        "no_inactive_object_reset": args.no_inactive_object_reset,
        "notes": ["All methods run the real MuJoCo pipeline with fixed scenario and dataset.",
                  "Same trial index does not imply independent human subjects.",
                  "Do not count debounced controller contact as physics-grounded stable success.",
                  "No statistical significance claimed from few synthetic trials.",
                  "Geometry-only contact feedback is a controller baseline; the physical success criterion is unchanged.",
                  "Comparison is valid only within a single batch; changing reset policy changes the task."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# WRAP Servo Ablation — Actual MuJoCo results",
        "",
        f"Trials: {','.join(map(str, trials))}; contact detector: {args.contact_detector}; steps: {args.steps}; ",
        f"Workspace auto-reset disabled: {args.no_auto_reset_object}; inactive reset disabled: {args.no_inactive_object_reset}.",
        "",
        "| Policy | Mean contact rate | Stable at task end | Mean longest stable streak | Mean relative translation span | Mean thumb loss events | Mean contact-state switches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregates:
        report.append(
            f"| {r['policy']} | {r['contact_ratio_mean']:.1%} | "
            f"{r['final_stable_count']}/{r['n_trials']} | "
            f"{r['longest_stable_frames_mean']:.1f} frames | "
            f"{r['max_relative_translation_mm_mean']:.2f} mm | "
            f"{r['thumb_loss_events_mean']:.2f} | "
            f"{r['control_switches_mean']:.2f} |"
        )
    report += ["", "These values are measured on the generated synthetic replay + MuJoCo outputs, not real hardware.",
               "Contact rate does not imply stable grasp. Compare paired deltas in aggregate.csv.",
               "A fixed 1 s / 1 mm / 1° stability rule is used; results are not independent human-subject evidence.",
               "Object auto-resets and contact-force directions should be audited in the per-run CSVs."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n[RESULTS] Real simulation outputs; metrics are proxies, not hardware-grasp proof:")
    for row in aggregates:
        print(f"{row['policy']:17s} n={row['n_trials']} "
              f"contact={row['contact_ratio_mean']:.1%} "
              f"stable_end={row['final_stable_count']}/{row['n_trials']} "
              f"max_stable={row['longest_stable_frames_mean']:.1f} frames "
              f"drift={row['max_relative_translation_mm_mean']:.2f} mm "
              f"thumb_losses={row['thumb_loss_events_mean']:.2f}")
    print(f"[DONE] {output / 'per_trial.csv'}")
    print(f"[DONE] {output / 'aggregate.csv'}")
    print(f"[DONE] {output / 'manifest.json'}")
    print(f"[DONE] {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
