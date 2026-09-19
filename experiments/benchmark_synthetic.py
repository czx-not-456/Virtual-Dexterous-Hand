from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _trial_lengths(dataset: Path) -> dict[int, int]:
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    counts: dict[int, int] = {}
    with dataset.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            raise ValueError(f"dataset has no trial_id column: {dataset}")
        for row in reader:
            trial = int(row["trial_id"])
            counts[trial] = counts.get(trial, 0) + 1
    if not counts:
        raise ValueError(f"dataset contains no rows: {dataset}")
    return counts


def _invalid_result(task: str, trial: int, returncode: int, error: str) -> dict:
    return {
        "task": task,
        "trial": trial,
        "run_ok": False,
        "returncode": returncode,
        "error": error,
        "task_metric_pass": False,
    }


def _run_one(task: str, trial: int, steps: int, dataset: Path, out_dir: Path, sim: str) -> dict:
    name = f"bench_{task}_trial{trial:02d}"
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--sim",
        sim,
        "--input",
        "dataset",
        "--dataset",
        str(dataset),
        "--dataset-trial",
        str(trial),
        "--task",
        task,
        "--steps",
        str(steps),
        "--output-dir",
        str(out_dir),
        "--run-name",
        name,
    ]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        return _invalid_result(task, trial, proc.returncode, "subcommand failed")

    summary_path = out_dir / f"{name}_summary.json"
    if not summary_path.is_file():
        return _invalid_result(task, trial, 0, "missing summary")
    try:
        row = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _invalid_result(task, trial, 0, f"invalid summary: {exc}")

    required = {
        "task",
        "contact_success_proxy",
        "stable_success_proxy",
        "task_metric_pass",
        "contact_frame_ratio",
        "stable_frame_ratio",
        "active_frames",
    }
    missing = sorted(required.difference(row))
    if missing:
        return _invalid_result(task, trial, 0, f"summary missing fields: {missing}")
    if row.get("task") != task or int(row.get("active_frames", 0)) <= 0:
        return _invalid_result(task, trial, 0, "summary task mismatch or no active frames")

    row["trial"] = trial
    row["steps"] = steps
    row["run_ok"] = True
    row["returncode"] = 0
    row["error"] = ""
    return row


def _write_csv(rows: list[dict], path: Path) -> None:
    keys = [
        "task", "trial", "steps", "run_ok", "returncode", "error",
        "required_success_metric", "task_metric_pass",
        "contact_success_proxy", "stable_success_proxy", "active_frames",
        "contact_frame_ratio", "stable_frame_ratio",
        "max_contact_success_streak_frames", "max_stable_success_streak_frames",
        "first_contact_latency_ms", "first_contact_success_latency_ms",
        "first_stable_success_latency_ms", "mean_pipeline_latency_ms",
        "p95_pipeline_latency_ms", "mean_optimizer_ms", "mean_joint_rmse_rad",
        "mean_velocity_rms_rad_s", "mean_tip_contact_ratio",
        "max_orientation_drift_deg", "max_object_displacement_m",
        "mean_servo_error_m",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(rows: list[dict], path: Path) -> None:
    keys = [
        "task", "trials", "task_pass_rate", "contact_success_rate",
        "stable_success_rate", "mean_contact_ratio", "mean_stable_ratio",
        "mean_p95_latency_ms", "mean_joint_rmse_rad", "mean_tip_contact_ratio",
        "mean_max_orientation_drift_deg",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    tasks = sorted({str(r["task"]) for r in rows if r.get("run_ok")})
    out = []
    for task in tasks:
        selected = [r for r in rows if r.get("run_ok") and r.get("task") == task]
        out.append(
            {
                "task": task,
                "trials": len(selected),
                "task_pass_rate": mean(bool(r["task_metric_pass"]) for r in selected),
                "contact_success_rate": mean(bool(r["contact_success_proxy"]) for r in selected),
                "stable_success_rate": mean(bool(r["stable_success_proxy"]) for r in selected),
                "mean_contact_ratio": mean(float(r["contact_frame_ratio"]) for r in selected),
                "mean_stable_ratio": mean(float(r["stable_frame_ratio"]) for r in selected),
                "mean_p95_latency_ms": mean(float(r["p95_pipeline_latency_ms"]) for r in selected),
                "mean_joint_rmse_rad": mean(float(r["mean_joint_rmse_rad"]) for r in selected),
                "mean_tip_contact_ratio": mean(float(r["mean_tip_contact_ratio"]) for r in selected),
                "mean_max_orientation_drift_deg": mean(
                    float(r["max_orientation_drift_deg"]) for r in selected
                ),
            }
        )
    return out


def _write_report(
    agg: list[dict], path: Path, dataset: Path, step_description: str,
    command_failures: int, task_failures: int,
) -> None:
    lines = [
        "# Synthetic MuJoCo Benchmark Report",
        "",
        f"- Dataset: {dataset}",
        f"- Frames per run: {step_description}",
        f"- Command/summary failures: {command_failures}",
        f"- Completed runs that failed their required task metric: {task_failures}",
        "- Contact success and stable success are reported separately.",
        "- These are fixed-base proxies, not lift/transport success metrics.",
        "",
        "| Task | Trials | Required pass | Contact success | Stable success | Contact ratio | Stable ratio | P95 latency (ms) | Joint RMSE (rad) | Tip ratio | Max drift (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['task']} | {row['trials']} | {row['task_pass_rate']:.1%} | "
            f"{row['contact_success_rate']:.1%} | {row['stable_success_rate']:.1%} | "
            f"{row['mean_contact_ratio']:.1%} | {row['mean_stable_ratio']:.1%} | "
            f"{row['mean_p95_latency_ms']:.2f} | {row['mean_joint_rmse_rad']:.4f} | "
            f"{row['mean_tip_contact_ratio']:.3f} | "
            f"{row['mean_max_orientation_drift_deg']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(agg: list[dict], path: Path) -> None:
    table_rows = []
    for row in agg:
        table_rows.append(
            f"<tr><td>{row['task']}</td><td>{row['trials']}</td>"
            f"<td>{100 * row['task_pass_rate']:.1f}%</td>"
            f"<td>{100 * row['contact_success_rate']:.1f}%</td>"
            f"<td>{100 * row['stable_success_rate']:.1f}%</td>"
            f"<td>{100 * row['mean_contact_ratio']:.1f}%</td>"
            f"<td>{100 * row['mean_stable_ratio']:.1f}%</td>"
            f"<td>{row['mean_p95_latency_ms']:.2f}</td></tr>"
        )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Virtual Dexterous Hand Benchmark</title><style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f5f6f8;color:#20242a;padding:28px}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>虚拟灵巧手模拟数据评测</h1>
<p>接触成功与稳定成功分开报告；任务通过列使用任务配置声明的必需指标。</p>
<table><thead><tr><th>Task</th><th>Trials</th><th>Required pass</th>
<th>Contact success</th><th>Stable success</th><th>Contact ratio</th>
<th>Stable ratio</th><th>P95 latency ms</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Batch benchmark standardized MuJoCo grasp tasks with synthetic glove data"
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "synthetic_glove_v1.csv")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="frames per run; default reads and runs the complete selected trial",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--sim", choices=["mujoco", "none"], default="mujoco", help="none 仅用于检查批处理流程")
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive when provided")

    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    all_tasks = list(cfg["task_objects"].keys())
    tasks = args.tasks or all_tasks
    unknown = [task for task in tasks if task not in all_tasks]
    if unknown:
        parser.error(f"unknown tasks={unknown}; available={all_tasks}")

    lengths = _trial_lengths(args.dataset)
    trial_ids = sorted(lengths)[: args.trials]
    if len(trial_ids) < args.trials:
        parser.error(
            f"requested {args.trials} trials, dataset only has {len(lengths)}: {sorted(lengths)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        for trial in trial_ids:
            steps = int(args.steps) if args.steps is not None else lengths[trial]
            rows.append(_run_one(task, trial, steps, args.dataset, args.output_dir, args.sim))

    failures = [row for row in rows if not row.get("run_ok")]
    task_failures = [
        row for row in rows if row.get("run_ok") and not row.get("task_metric_pass")
    ]
    _write_csv(rows, args.output_dir / "benchmark_runs.csv")
    agg = _aggregate(rows)
    _write_aggregate_csv(agg, args.output_dir / "benchmark_summary.csv")
    step_description = str(args.steps) if args.steps is not None else "complete selected trial"
    _write_report(
        agg,
        args.output_dir / "BENCHMARK_REPORT.md",
        args.dataset,
        step_description,
        len(failures),
        len(task_failures),
    )
    _write_html(agg, args.output_dir / "benchmark_dashboard.html")
    print(
        f"[DONE] benchmark runs={len(rows)}, command_failures={len(failures)}, "
        f"task_metric_failures={len(task_failures)}"
    )
    print(f"[DONE] {args.output_dir / 'BENCHMARK_REPORT.md'}")
    print(f"[DONE] {args.output_dir / 'benchmark_dashboard.html'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
