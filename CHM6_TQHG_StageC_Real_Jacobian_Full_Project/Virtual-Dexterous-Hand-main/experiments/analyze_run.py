from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
import sys


def _float(row, key, default=0.0):
    try:
        return float(row[key])
    except Exception:
        return float(default)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Summarize a v0.4.6 run CSV")
    p.add_argument("csv", type=Path)
    args = p.parse_args()
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty csv")

    latency = [_float(r, "pipeline_latency_ms") for r in rows]
    opt = [_float(r, "optimizer_ms") for r in rows]
    drift = [_float(r, "orientation_drift_deg") for r in rows]
    tip = [_float(r, "tip_contact_ratio") for r in rows]
    stable = [str(r.get("stable_contact_proxy", "False")).lower() in {"true", "1"} for r in rows]
    task_contact = [str(r.get("task_contact_proxy", "False")).lower() in {"true", "1"} for r in rows]
    contact_success = [
        str(r.get("contact_success_proxy", "False")).lower() in {"true", "1"} for r in rows
    ]
    stable_success = [
        str(r.get("stable_success_proxy", "False")).lower() in {"true", "1"} for r in rows
    ]
    task_active = [str(r.get("task_active", "False")).lower() in {"true", "1"} for r in rows]
    contacts = [_float(r, "contact_count") for r in rows]

    intents = {}
    for r in rows:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1

    print(f"frames: {len(rows)}")
    print(f"mean pipeline latency: {mean(latency):.3f} ms")
    print(f"mean optimizer time:   {mean(opt):.3f} ms")
    print(f"max orientation drift: {max(drift):.3f} deg")
    print(f"mean tip contact ratio:{mean(tip):.3f}")
    print(f"contact frames:        {sum(c > 0 for c in contacts)}")
    print(f"stable-contact proxy:  {sum(stable)} frames")
    if any(task_active):
        active_n = sum(task_active)
        active_contact = sum(c and a for c, a in zip(task_contact, task_active))
        active_stable = sum(s and a for s, a in zip(stable, task_active))
        print(f"task active frames:     {active_n}")
        print(f"task contact ratio:     {active_contact / active_n:.1%}")
        print(f"task stable ratio:      {active_stable / active_n:.1%}")
        print(f"contact success reached:{any(contact_success)}")
        print(f"stable success reached: {any(stable_success)}")
    print("intent frames:", intents)


if __name__ == "__main__":
    main()
