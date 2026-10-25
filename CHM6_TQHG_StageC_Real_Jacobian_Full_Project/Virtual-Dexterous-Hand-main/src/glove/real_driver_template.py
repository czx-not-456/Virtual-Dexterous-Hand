"""真实数据手套接入模板。

申报材料没有提供 SDK/串口/网络协议，所以当前工程不能猜测真实协议。
拿到实验室示例工程后，把厂商数据解析为 GloveSample 即可。
"""
from __future__ import annotations

from .driver import GloveSample


class RealGloveDriver:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "请根据实验室数据手套 SDK/通信协议实现 RealGloveDriver；"
            "上层算法只要求 sample() 返回 GloveSample。"
        )

    def sample(self) -> GloveSample:  # pragma: no cover
        raise NotImplementedError
