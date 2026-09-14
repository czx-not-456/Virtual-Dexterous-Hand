from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .feature_extractor import HandFeatures


class Intent(str, Enum):
    NEUTRAL = "NEUTRAL"
    PINCH = "PINCH"
    WRAP = "WRAP"


@dataclass
class IntentRecognizer:
    pinch_enter_m: float
    pinch_exit_m: float
    wrap_enter: float
    wrap_exit: float
    dwell_frames: int = 4
    sequence_window_frames: int = 1

    def __post_init__(self) -> None:
        self.state = Intent.NEUTRAL
        self._candidate = self.state
        self._count = 0
        self._pinch_hist: deque[float] = deque(maxlen=max(1, int(self.sequence_window_frames)))
        self._wrap_hist: deque[float] = deque(maxlen=max(1, int(self.sequence_window_frames)))

    def _smoothed(self, f: HandFeatures) -> tuple[float, float]:
        # 文献中的意图识别使用人体运动序列；本项目当前没有训练数据，
        # 因此 v0.3 只采用短时序均值，不声称复现 Bi-GRU。
        self._pinch_hist.append(float(f.pinch_distance_m))
        self._wrap_hist.append(float(f.wrap_score))
        pinch = sum(self._pinch_hist) / len(self._pinch_hist)
        wrap = sum(self._wrap_hist) / len(self._wrap_hist)
        return pinch, wrap

    def _desired(self, pinch_distance_m: float, wrap_score: float) -> Intent:
        # 先判定多指包络：WRAP 往往也会让拇指/食指距离变小。
        if self.state == Intent.WRAP:
            if wrap_score >= self.wrap_exit:
                return Intent.WRAP
        elif wrap_score >= self.wrap_enter:
            return Intent.WRAP

        if self.state == Intent.PINCH:
            if pinch_distance_m <= self.pinch_exit_m:
                return Intent.PINCH
        elif pinch_distance_m <= self.pinch_enter_m:
            return Intent.PINCH

        return Intent.NEUTRAL

    def update(self, f: HandFeatures) -> Intent:
        pinch, wrap = self._smoothed(f)
        desired = self._desired(pinch, wrap)
        if desired == self.state:
            self._candidate = self.state
            self._count = 0
            return self.state
        if desired != self._candidate:
            self._candidate = desired
            self._count = 1
        else:
            self._count += 1
        if self._count >= self.dwell_frames:
            self.state = self._candidate
            self._count = 0
        return self.state
