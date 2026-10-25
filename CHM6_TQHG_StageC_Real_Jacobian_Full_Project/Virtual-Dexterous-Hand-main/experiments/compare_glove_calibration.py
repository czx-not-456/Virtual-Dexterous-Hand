from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glove.adapter import BaseMapping
from src.glove.captured_samples import load_captured_samples, resolve_delivery_path


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "mse_deg2": float(np.mean(error**2)),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "mae_deg": float(np.mean(np.abs(error))),
        "max_abs_deg": float(np.max(np.abs(error))),
    }


def compare_calibrations(
    samples_path: str | Path,
    base_mapping_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    samples = load_captured_samples(samples_path)
    base_mapping = BaseMapping(base_mapping_path)
    manifest_path = resolve_delivery_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = manifest.get("adaptation_result", {}).get("pose_validation", [])
    if not validation:
        raise ValueError("manifest has no adaptation_result.pose_validation")

    held_out = {sample.pose: sample for sample in samples if sample.pose_frame_index == 3}
    base_predictions: list[np.ndarray] = []
    personalized_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_pose: list[dict] = []
    for row in validation:
        pose = str(row["pose"])
        if pose not in held_out:
            raise ValueError(f"captured_samples has no fourth-frame validation sample for pose={pose!r}")
        sample = held_out[pose]
        if sample.targets_deg is None:
            raise ValueError(f"validation sample has no target columns for pose={pose!r}")
        target = np.asarray(row["target_deg"], dtype=float)
        if not np.allclose(sample.targets_deg, target, atol=1e-9):
            raise ValueError(f"manifest/captured target mismatch for pose={pose!r}")
        base = np.asarray(list(base_mapping.predict_deg(sample.sensors_deg).values()), dtype=float)
        personalized = np.asarray(row["final_prediction_deg"], dtype=float)
        if base.shape != target.shape or personalized.shape != target.shape:
            raise ValueError(f"prediction dimension mismatch for pose={pose!r}")
        base_predictions.append(base)
        personalized_predictions.append(personalized)
        targets.append(target)
        per_pose.append(
            {
                "pose": pose,
                "sample_index": sample.sample_index,
                "base": _metrics(base, target),
                "personalized": _metrics(personalized, target),
            }
        )

    target_matrix = np.vstack(targets)
    base_metrics = _metrics(np.vstack(base_predictions), target_matrix)
    personalized_metrics = _metrics(np.vstack(personalized_predictions), target_matrix)
    improvement = (
        (base_metrics["rmse_deg"] - personalized_metrics["rmse_deg"])
        / max(base_metrics["rmse_deg"], 1e-12)
        * 100.0
    )
    return {
        "evaluation_protocol": "per_pose_fourth_frame",
        "validation_samples": len(targets),
        "output_dimensions": target_matrix.shape[1],
        "base_mapping": base_metrics,
        "personalized_mapping": personalized_metrics,
        "rmse_improvement_percent": float(improvement),
        "per_pose": per_pose,
    }


def _write_pose_csv(result: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pose",
                "sample_index",
                "base_rmse_deg",
                "base_mae_deg",
                "personalized_rmse_deg",
                "personalized_mae_deg",
            ],
        )
        writer.writeheader()
        for row in result["per_pose"]:
            writer.writerow(
                {
                    "pose": row["pose"],
                    "sample_index": row["sample_index"],
                    "base_rmse_deg": row["base"]["rmse_deg"],
                    "base_mae_deg": row["base"]["mae_deg"],
                    "personalized_rmse_deg": row["personalized"]["rmse_deg"],
                    "personalized_mae_deg": row["personalized"]["mae_deg"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare delivered base and personalized glove mappings")
    parser.add_argument("--samples", type=Path, default=ROOT / "数据手套数据" / "captured_samples.xlsx")
    parser.add_argument("--base-mapping", type=Path, default=ROOT / "数据手套数据" / "base_mapping.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "数据手套数据" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "glove_calibration")
    args = parser.parse_args()

    result = compare_calibrations(args.samples, args.base_mapping, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mapping_comparison.json"
    csv_path = args.output_dir / "mapping_comparison_by_pose.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_pose_csv(result, csv_path)

    base = result["base_mapping"]
    personalized = result["personalized_mapping"]
    print(f"validation={result['evaluation_protocol']} samples={result['validation_samples']}")
    print(f"base mapping:         RMSE={base['rmse_deg']:.4f} deg, MAE={base['mae_deg']:.4f} deg")
    print(
        f"personalized mapping: RMSE={personalized['rmse_deg']:.4f} deg, "
        f"MAE={personalized['mae_deg']:.4f} deg"
    )
    print(f"RMSE improvement:     {result['rmse_improvement_percent']:.2f}%")
    print(f"[DONE] {json_path}")
    print(f"[DONE] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
