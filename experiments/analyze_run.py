from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def _float(row, key, default=0.0):
    try:
        return float(row[key])
    except Exception:
        return float(default)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize a v0.3 run CSV")
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
    print("intent frames:", intents)


if __name__ == "__main__":
    main()
