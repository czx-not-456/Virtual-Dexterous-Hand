from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Mapping

import numpy as np

from src.hand_model.frame import HumanHandFrame
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.processing.filters import MovingAverageFilter

from .calibration import AdaptiveCalibrator
from .captured_samples import CapturedSample, load_captured_samples, resolve_delivery_path
from .driver import GloveSample


_TERM = re.compile(r"([+-]?\s*\d+(?:\.\d+)?)\s*\*\s*&([1-9]\d*)")


def _load_json(path: str | Path) -> tuple[Path, dict]:
    resolved = resolve_delivery_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        return resolved, json.load(stream)


class BaseMapping:
    """Evaluate the delivered linear mapping without using ``eval``."""

    def __init__(self, path: str | Path) -> None:
        self.path, cfg = _load_json(path)
        self.cfg = cfg
        calculations = cfg.get("calculations", {})
        keys = sorted(calculations, key=lambda value: int(value))
        if keys != [str(i) for i in range(len(keys))]:
            raise ValueError("base_mapping calculations must use contiguous zero-based keys")
        self.output_names = tuple(
            item["joint_name"]
            for item in sorted(cfg["joint_mappings"], key=lambda item: int(item["joint_id"]))
        )
        if len(self.output_names) != len(keys):
            raise ValueError("base_mapping calculations and joint_mappings have different lengths")
        self._expressions = tuple(str(calculations[key]) for key in keys)

    @staticmethod
    def _evaluate(expression: str, sensors_deg: np.ndarray) -> float:
        compact = expression.replace(" ", "")
        terms = list(_TERM.finditer(compact))
        if not terms:
            raise ValueError(f"unsupported base_mapping expression: {expression!r}")
        remainder = compact
        for match in reversed(terms):
            remainder = remainder[: match.start()] + remainder[match.end() :]
        offset = 0.0 if remainder in {"", "+"} else float(remainder)
        value = offset
        for match in terms:
            coefficient = float(match.group(1).replace(" ", ""))
            sensor_index = int(match.group(2)) - 1
            if not 0 <= sensor_index < len(sensors_deg):
                raise ValueError(f"base_mapping references missing sensor {sensor_index + 1}")
            value += coefficient * float(sensors_deg[sensor_index])
        return value

    def predict_deg(self, sensors_deg: np.ndarray) -> dict[str, float]:
        sensors = np.asarray(sensors_deg, dtype=float)
        expected = int(self.cfg.get("sensor_config", {}).get("total_sensors", 15))
        if sensors.shape != (expected,):
            raise ValueError(f"base_mapping expected {expected} sensors, got {sensors.shape}")
        return {
            name: self._evaluate(expression, sensors)
            for name, expression in zip(self.output_names, self._expressions)
        }


class SimGloveAdapter:
    """Convert the existing simulated GloveSample stream to HumanHandFrame."""

    def __init__(
        self,
        human: HumanHandModel,
        calibrator: AdaptiveCalibrator,
        input_filter: MovingAverageFilter,
    ) -> None:
        self.human = human
        self.calibrator = calibrator
        self.input_filter = input_filter

    def adapt(self, sample: GloveSample) -> HumanHandFrame:
        normalized = self.input_filter.apply(self.calibrator.normalize(sample))
        return HumanHandFrame(
            timestamp=sample.timestamp,
            joint_order=tuple(self.human.joint_order),
            joint_angles_rad=self.human.angles_from_normalized(normalized),
            palm_rpy=sample.palm_rpy,
            source="sim",
        )


class GloveAdapter:
    """Convert delivered 15-sensor captures into project HumanHandFrame objects."""

    def __init__(
        self,
        human: HumanHandModel,
        robot: RobotHandModel,
        *,
        base_mapping_path: str | Path,
        manifest_path: str | Path,
        output_bindings: Mapping[str, str],
    ) -> None:
        self.human = human
        self.robot = robot
        self.base_mapping = BaseMapping(base_mapping_path)
        self.manifest_path, manifest = _load_json(manifest_path)
        normalization = manifest["normalization"]
        if normalization.get("method") != "minmax":
            raise ValueError(f"unsupported glove normalization: {normalization.get('method')!r}")
        self.sensor_min = np.asarray(normalization["min"], dtype=float)
        self.sensor_max = np.asarray(normalization["max"], dtype=float)
        if self.sensor_min.shape != (15,) or self.sensor_max.shape != (15,):
            raise ValueError("manifest normalization must contain 15 min/max values")
        self.sensor_span = self.sensor_max - self.sensor_min
        self.output_bindings = dict(output_bindings)
        missing = [name for name in robot.joint_order if name not in self.output_bindings]
        if missing:
            raise ValueError(f"glove output_bindings missing CH-M6 joints: {missing}")
        unknown = sorted(set(self.output_bindings.values()).difference(self.base_mapping.output_names))
        if unknown:
            raise ValueError(f"glove output_bindings reference unknown mapping outputs: {unknown}")

    def adapt(self, sample: CapturedSample, *, timestamp: float | None = None) -> HumanHandFrame:
        sensors = np.asarray(sample.sensors_deg, dtype=float)
        normalized_vector = np.divide(
            sensors - self.sensor_min,
            self.sensor_span,
            out=np.zeros_like(sensors),
            where=self.sensor_span > 1e-12,
        )
        normalized_vector = np.clip(normalized_vector, 0.0, 1.0)
        normalized = dict(zip(self.human.joint_order, normalized_vector.tolist()))
        mapped_deg = self.base_mapping.predict_deg(sensors)
        robot_deg = np.asarray(
            [mapped_deg[self.output_bindings[joint]] for joint in self.robot.joint_order],
            dtype=float,
        )
        robot_targets = np.clip(np.radians(robot_deg), self.robot.ranges[:, 0], self.robot.ranges[:, 1])
        return HumanHandFrame(
            timestamp=time.perf_counter() if timestamp is None else float(timestamp),
            joint_order=tuple(self.human.joint_order),
            joint_angles_rad=self.human.angles_from_normalized(normalized),
            source="glove",
            robot_joint_targets_rad=robot_targets,
            metadata={
                "sample_index": sample.sample_index,
                "pose": sample.pose,
                "pose_frame_index": sample.pose_frame_index,
                "elapsed_ms": sample.elapsed_ms,
                "base_mapping_outputs_deg": mapped_deg,
            },
        )


class CapturedGloveInput:
    """Looping offline input source backed by teacher-provided captured samples."""

    def __init__(self, path: str | Path, adapter: GloveAdapter, *, loop: bool = True) -> None:
        self.path = resolve_delivery_path(path)
        self.samples = load_captured_samples(self.path)
        self.adapter = adapter
        self.loop = bool(loop)
        self.index = 0

    def sample(self) -> HumanHandFrame:
        if self.index >= len(self.samples):
            if not self.loop:
                raise StopIteration("captured glove replay finished")
            self.index = 0
        sample = self.samples[self.index]
        self.index += 1
        return self.adapter.adapt(sample)
