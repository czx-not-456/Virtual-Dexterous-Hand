from __future__ import annotations

import json
import socket
import numpy as np


class UnityUDPBridge:
    """可选的 Unity UDP 输出桥。

    当前版本只定义最小协议：JSON 中发送 joint_positions 与 intent。
    Unity 端协议在申报材料中未给出，因此这里不假定最终实现。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5005) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, q: np.ndarray, intent: str) -> None:
        payload = {"joint_positions": np.asarray(q, dtype=float).tolist(), "intent": intent}
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
