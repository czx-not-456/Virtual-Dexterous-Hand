from __future__ import annotations

import argparse
import csv
from pathlib import Path


def as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize PINCH/WRAP hand-object contact episodes")
    p.add_argument("csv", type=Path)
    args = p.parse_args()

    rows = list(csv.DictReader(args.csv.open("r", encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("CSV is empty")

    for intent, proxy in (("PINCH", "pinch_contact_proxy"), ("WRAP", "wrap_contact_proxy")):
        subset = [r for r in rows if r.get("intent") == intent]
        if not subset:
            print(f"{intent}: no frames")
            continue
        success = sum(as_bool(r.get(proxy, "false")) for r in subset)
        stable = sum(as_bool(r.get("stable_contact_proxy", "false")) for r in subset)
        mean_contacts = sum(float(r.get("contact_count", 0) or 0) for r in subset) / len(subset)
        mean_tip = sum(float(r.get("tip_contact_ratio", 0) or 0) for r in subset) / len(subset)
        servo_rows = [r for r in subset if as_bool(r.get("task_servo_active", "false"))]
        if servo_rows:
            mean_servo_err = sum(float(r.get("servo_mean_error_m", 0) or 0) for r in servo_rows) / len(servo_rows)
            max_servo_err = max(float(r.get("servo_max_error_m", 0) or 0) for r in servo_rows)
            servo_text = f", servo_mean_error={mean_servo_err:.4f}m, servo_max_error={max_servo_err:.4f}m"
        else:
            servo_text = ""
        print(
            f"{intent}: frames={len(subset)}, contact_proxy={success/len(subset):.1%}, "
            f"stable_proxy={stable/len(subset):.1%}, mean_hand_contacts={mean_contacts:.2f}, "
            f"mean_tip_ratio={mean_tip:.2f}{servo_text}"
        )


if __name__ == "__main__":
    main()
