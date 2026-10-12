"""Read-only audit of CH-M6 calibration snapshots and opaque checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickletools
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    """Inspect ZIP and pickle opcodes as bytes; never unpickle model data."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        pickle_names = [name for name in entries if name.endswith("/data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError(f"unexpected checkpoint container: {path}")
        opcodes = list(pickletools.genops(archive.read(pickle_names[0])))
        globals_used = sorted({str(value) for op, value, _ in opcodes if op.name == "GLOBAL"})
        keys = sorted({str(value) for op, value, _ in opcodes
                       if op.name in {"BINUNICODE", "SHORT_BINUNICODE"}})
        storage_count = len([name for name in entries if "/data/" in name])
    return {"storage_count": storage_count, "pickle_globals": globals_used,
            "metadata_keys": keys, "deserialized": False}


def audit(delivery: Path) -> dict:
    records = sorted((delivery / "records" / "chm6_calibrations").glob("*/manifest.json"))
    snapshots = []
    devices = Counter()
    failures = []
    for manifest_path in records:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        device = manifest["glove"]
        devices[f"{device['serial_number']}_{device['hand_side']}"] += 1
        files = {
            "captured": (manifest["data"]["captured_samples"], manifest["data"]["captured_samples_sha256"]),
            "training": (manifest["data"]["training_samples"], manifest["data"]["training_samples_sha256"]),
            "base_mapping": (manifest["base_mapping"]["path"], manifest["base_mapping"]["sha256"]),
        }
        model_info = manifest.get("model")
        if model_info:
            files["model"] = (model_info["path"], model_info["sha256"])
        for kind, (relative, expected) in files.items():
            path = delivery / relative
            if not path.is_file() or sha256(path) != expected.lower():
                failures.append(f"{manifest_path.parent.name}: {kind} missing/hash mismatch")
        capture_path = delivery / files["captured"][0]
        with capture_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != manifest["sample_counts"]["captured"]:
            failures.append(f"{manifest_path.parent.name}: captured row count mismatch")
        required = [*(f"sensor_{i}" for i in range(1, 16)),
                    *(f"target_{i}" for i in range(1, 7))]
        if not rows or any(key not in rows[0] for key in required):
            failures.append(f"{manifest_path.parent.name}: expected 15 sensors and 6 targets")
        normalization = manifest.get("normalization") or {}
        if model_info and (normalization.get("method") != "minmax" or any(
            len(normalization.get(key, [])) != 15 for key in ("min", "max")
        )):
            failures.append(f"{manifest_path.parent.name}: invalid trained normalization metadata")
        base_path = delivery / files["base_mapping"][0]
        base_mapping = json.loads(base_path.read_text(encoding="utf-8"))
        if len(base_mapping.get("joint_mappings", [])) != 6:
            failures.append(f"{manifest_path.parent.name}: base mapping output count is not six")
        clips: dict[str, list[float]] = {}
        for row in rows:
            clips.setdefault(row["pose"], []).append(float(row["elapsed_ms"]))
        if any(b <= a for elapsed in clips.values() for a, b in zip(elapsed, elapsed[1:])):
            failures.append(f"{manifest_path.parent.name}: nonmonotonic elapsed_ms within pose")
        step_ms = [b - a for elapsed in clips.values() for a, b in zip(elapsed, elapsed[1:])]
        snapshots.append({
            "snapshot": manifest_path.parent.name,
            "device": device,
            "captured_frames": len(rows),
            "sensor_dimensions": 15,
            "target_dimensions": 6,
            "normalization": normalization.get("method"),
            "pose_counts": dict(Counter(row["pose"] for row in rows)),
            "pose_clip_durations_ms": {pose: elapsed[-1] - elapsed[0] for pose, elapsed in clips.items()},
            "median_interframe_ms": statistics.median(step_ms),
            "status": manifest.get("status"),
            "model": files["model"][0] if model_info else None,
            "checkpoint_container": inspect_checkpoint(delivery / files["model"][0]) if model_info else None,
            "validation_strategy": manifest.get("training_config", {}).get("validation", {}).get("strategy"),
        })
    referenced = {Path(s["model"]).name for s in snapshots if s["model"]}
    all_checkpoints = {path.name for path in (delivery / "models").glob("*.pt")}
    return {"snapshot_count": len(snapshots), "trained_checkpoint_count": sum(s["model"] is not None for s in snapshots),
            "checkpoint_file_count": len(all_checkpoints),
            "unreferenced_checkpoint_files": sorted(all_checkpoints - referenced),
            "device_count": len(devices),
            "device_snapshot_counts": dict(devices), "failures": failures,
            "note": "Captured CSV files contain separate scripted calibration pose clips. elapsed_ms restarts at each pose, so clip boundaries are not recorded continuous transitions. Checkpoint ZIP metadata was inspected without deserialization or inference.",
            "snapshots": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=ROOT / "数据手套数据")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.delivery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"snapshots={result['snapshot_count']} devices={result['device_count']} failures={len(result['failures'])}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
