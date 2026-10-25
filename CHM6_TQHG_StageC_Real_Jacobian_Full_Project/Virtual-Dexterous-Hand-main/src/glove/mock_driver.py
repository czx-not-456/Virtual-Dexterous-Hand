from __future__ import annotations

import math
import time
from dataclasses import dataclass
import numpy as np

from .driver import GloveSample


FINGERS = ("thumb", "index", "middle", "ring", "little")
JOINTS = ("mcp", "pip", "dip")


@dataclass
class MockGloveDriver:
    channels: list[str]
    sampling_hz: float = 60.0
    raw_min: float = 500.0
    raw_max: float = 3500.0
    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.frame = 0
        self.dt = 1.0 / self.sampling_hz
        self._last_sample_time = time.perf_counter()

    def _gesture_vector(self, gesture: str, phase: float = 0.0) -> dict[str, float]:
        v = {c: 0.08 for c in self.channels}
        if gesture == "OPEN":
            pass
        elif gesture == "NEUTRAL":
            for c in self.channels:
                v[c] = 0.22
        elif gesture in {"FIST", "WRAP"}:
            for finger in ("index", "middle", "ring", "little"):
                v[f"{finger}_mcp"] = 0.88
                v[f"{finger}_pip"] = 0.93
                v[f"{finger}_dip"] = 0.84
            v["thumb_mcp"], v["thumb_pip"], v["thumb_dip"] = 0.58, 0.68, 0.55
        elif gesture == "PINCH":
            # 该模式用于第一版模拟“拇指-食指精细捏合”。
            v["thumb_mcp"], v["thumb_pip"], v["thumb_dip"] = 0.52, 0.70, 0.52
            v["index_mcp"], v["index_pip"], v["index_dip"] = 0.38, 0.62, 0.46
            for finger in ("middle", "ring", "little"):
                v[f"{finger}_mcp"] = 0.16
                v[f"{finger}_pip"] = 0.13
                v[f"{finger}_dip"] = 0.11
        elif gesture == "WAVE":
            s = 0.5 + 0.35 * math.sin(phase)
            for c in self.channels:
                v[c] = float(np.clip(s, 0.05, 0.95))
        else:
            raise ValueError(f"Unknown mock gesture: {gesture}")
        return v

    def raw_from_normalized(self, normalized: dict[str, float]) -> dict[str, float]:
        span = self.raw_max - self.raw_min
        return {
            c: self.raw_min + float(np.clip(normalized[c], 0.0, 1.0)) * span
            + float(self.rng.normal(0.0, 8.0))
            for c in self.channels
        }

    def calibration_samples(self, frames_per_pose: int = 40) -> list[GloveSample]:
        out: list[GloveSample] = []
        # 申报书提出“张开、弯曲、握拳”等标准动作；这里用 OPEN/NEUTRAL/FIST 覆盖量程。
        for pose in ("OPEN", "NEUTRAL", "FIST"):
            for _ in range(frames_per_pose):
                n = self._gesture_vector(pose)
                out.append(
                    GloveSample(time.perf_counter(), self.raw_from_normalized(n), (0.0, 0.0, 0.0))
                )
        return out

    def gesture_for_frame(self, frame: int) -> str:
        period = 600
        x = frame % period
        if x < 80:
            return "OPEN"
        if x < 160:
            return "NEUTRAL"
        if x < 280:
            return "PINCH"
        if x < 360:
            return "NEUTRAL"
        if x < 520:
            return "WRAP"
        return "OPEN"

    def sample(self) -> GloveSample:
        gesture = self.gesture_for_frame(self.frame)
        n = self._gesture_vector(gesture, self.frame * self.dt * 2.0 * math.pi)
        channels = self.raw_from_normalized(n)
        # 小幅手掌运动，仅用于演示“关节 + 手掌姿态”联合数据结构。
        t = self.frame * self.dt
        palm = (0.04 * math.sin(t), 0.03 * math.sin(0.7 * t), 0.02 * math.cos(0.5 * t))
        self.frame += 1
        return GloveSample(time.perf_counter(), channels, palm)
