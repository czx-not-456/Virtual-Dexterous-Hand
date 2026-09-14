from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(slots=True)
class GloveSample:
    timestamp: float
    channels: Mapping[str, float]
    palm_rpy: tuple[float, float, float]


class GloveDriver(Protocol):
    def sample(self) -> GloveSample:
        ...
