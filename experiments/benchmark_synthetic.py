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
        return {"task": task, "trial": trial, "run_ok": False, "returncode": proc.returncode}
    summary_path = out_dir / f"{name}_summary.json"
    if not summary_path.is_file():
        return {"task": task, "trial": trial, "run_ok": False, "returncode": 0, "error": "missing summary"}
    row = json.loads(summary_path.read_text(encoding="utf-8"))
    row["trial"] = trial
    row["run_ok"] = True
    row["returncode"] = 0
    return row


def _write_csv(rows: list[dict], path: Path) -> None:
    keys = [
        "task", "trial", "run_ok", "success_proxy", "active_frames",
        "contact_frame_ratio", "stable_frame_ratio", "max_success_streak_frames",
        "first_contact_latency_ms", "first_success_latency_ms",
        "mean_pipeline_latency_ms", "p95_pipeline_latency_ms", "mean_optimizer_ms",
        "mean_joint_rmse_rad", "mean_velocity_rms_rad_s", "mean_tip_contact_ratio",
        "max_orientation_drift_deg", "max_object_displacement_m", "mean_servo_error_m",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)



def _write_aggregate_csv(rows: list[dict], path: Path) -> None:
    keys = [
        "task", "trials", "success_rate", "mean_contact_ratio", "mean_stable_ratio",
        "mean_p95_latency_ms", "mean_joint_rmse_rad", "mean_tip_contact_ratio",
        "mean_max_orientation_drift_deg",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _aggregate(rows: list[dict]) -> list[dict]:
    tasks = sorted({str(r["task"]) for r in rows if r.get("run_ok")})
    out = []
    for task in tasks:
        rs = [r for r in rows if r.get("run_ok") and r.get("task") == task]
        if not rs:
            continue
        successes = [bool(r.get("success_proxy", False)) for r in rs]
        out.append(
            {
                "task": task,
                "trials": len(rs),
                "success_rate": sum(successes) / len(rs),
                "mean_contact_ratio": mean(float(r.get("contact_frame_ratio", 0.0)) for r in rs),
                "mean_stable_ratio": mean(float(r.get("stable_frame_ratio", 0.0)) for r in rs),
                "mean_p95_latency_ms": mean(float(r.get("p95_pipeline_latency_ms", 0.0)) for r in rs),
                "mean_joint_rmse_rad": mean(float(r.get("mean_joint_rmse_rad", 0.0)) for r in rs),
                "mean_tip_contact_ratio": mean(float(r.get("mean_tip_contact_ratio", 0.0)) for r in rs),
                "mean_max_orientation_drift_deg": mean(float(r.get("max_orientation_drift_deg", 0.0)) for r in rs),
            }
        )
    return out


def _write_report(agg: list[dict], path: Path, dataset: Path, steps: int) -> None:
    lines = [
        "# Synthetic MuJoCo Benchmark Report",
        "",
        f"- Dataset: `{dataset}`",
        f"- Frames per run: {steps}",
        "- Success is a stable-contact proxy because the current hand base is fixed; it is not a lift/transport success metric.",
        "",
        "| Task | Trials | Success | Contact ratio | Stable ratio | P95 latency (ms) | Joint RMSE (rad) | Tip contact ratio | Max orientation drift (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        lines.append(
            f"| {r['task']} | {r['trials']} | {r['success_rate']:.1%} | "
            f"{r['mean_contact_ratio']:.1%} | {r['mean_stable_ratio']:.1%} | "
            f"{r['mean_p95_latency_ms']:.2f} | {r['mean_joint_rmse_rad']:.4f} | "
            f"{r['mean_tip_contact_ratio']:.3f} | {r['mean_max_orientation_drift_deg']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(agg: list[dict], path: Path) -> None:
    cards = []
    rows = []
    for r in agg:
        pct = max(0.0, min(100.0, 100.0 * float(r["success_rate"])))
        cards.append(
            f'''<div class="card"><h3>{r['task']}</h3><div class="big">{pct:.0f}%</div>
            <div class="bar"><span style="width:{pct:.1f}%"></span></div>
            <p>contact {100*r['mean_contact_ratio']:.1f}% · stable {100*r['mean_stable_ratio']:.1f}%</p></div>'''
        )
        rows.append(
            f"<tr><td>{r['task']}</td><td>{r['trials']}</td><td>{100*r['success_rate']:.1f}%</td>"
            f"<td>{100*r['mean_contact_ratio']:.1f}%</td><td>{100*r['mean_stable_ratio']:.1f}%</td>"
            f"<td>{r['mean_p95_latency_ms']:.2f}</td><td>{r['mean_joint_rmse_rad']:.4f}</td>"
            f"<td>{r['mean_tip_contact_ratio']:.3f}</td><td>{r['mean_max_orientation_drift_deg']:.2f}</td></tr>"
        )
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Virtual Dexterous Hand Benchmark</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f5f6f8;color:#20242a;margin:0;padding:28px}}
h1{{margin:0 0 6px}} .note{{color:#59636e;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-bottom:26px}}
.card{{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 12px #00000012}} .big{{font-size:34px;font-weight:700}}
.bar{{height:8px;background:#e5e9ef;border-radius:9px;overflow:hidden}} .bar span{{display:block;height:100%;background:#3568c0}}
table{{width:100%;border-collapse:collapse;background:white;box-shadow:0 2px 12px #00000012}} th,td{{padding:10px;border-bottom:1px solid #e6e8eb;text-align:right}} th:first-child,td:first-child{{text-align:left}}
</style></head><body><h1>虚拟灵巧手模拟数据评测看板</h1>
<div class="note">success = 连续稳定接触代理；当前固定手掌模型不宣称“抓起并搬运成功”。</div>
<div class="grid">{''.join(cards)}</div>
<table><thead><tr><th>Task</th><th>Trials</th><th>Success</th><th>Contact</th><th>Stable</th><th>P95 latency ms</th><th>Joint RMSE</th><th>Tip ratio</th><th>Orientation drift°</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>'''
    path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch benchmark standardized MuJoCo grasp tasks with synthetic glove data")
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "synthetic_glove_v1.csv")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "benchmark")
    p.add_argument("--sim", choices=["mujoco", "none"], default="mujoco", help="none 仅用于检查批处理流程")
    args = p.parse_args()

    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    all_tasks = list(cfg["task_objects"].keys())
    tasks = args.tasks or all_tasks
    unknown = [t for t in tasks if t not in all_tasks]
    if unknown:
        raise SystemExit(f"unknown tasks={unknown}; available={all_tasks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        for trial in range(args.trials):
            rows.append(_run_one(task, trial, args.steps, args.dataset, args.output_dir, args.sim))

    _write_csv(rows, args.output_dir / "benchmark_runs.csv")
    agg = _aggregate(rows)
    _write_aggregate_csv(agg, args.output_dir / "benchmark_summary.csv")
    _write_report(agg, args.output_dir / "BENCHMARK_REPORT.md", args.dataset, args.steps)
    _write_html(agg, args.output_dir / "benchmark_dashboard.html")
    failures = [r for r in rows if not r.get("run_ok")]
    print(f"[DONE] benchmark runs={len(rows)}, command_failures={len(failures)}")
    print(f"[DONE] {args.output_dir / 'BENCHMARK_REPORT.md'}")
    print(f"[DONE] {args.output_dir / 'benchmark_dashboard.html'}")


if __name__ == "__main__":
    main()
