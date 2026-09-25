"""把掼蛋 wss 的**原始帧全量**转储下来（默认关闭，靠环境变量开）。

为什么需要：原先那份抓包只存了每帧前 256 字节，而帧真实长度到 4497 字节 ——
发牌（310B）、结算、进贡这些**大消息的正文从来没被看过**。所以
「网络里到底有没有级别」这个结论至今悬空，而级别错了逢人配就全错。

和 `net/addon.py` 的分工：addon 解出**语义事件**（谁出了什么牌），
这里存**原始字节**（供事后做全量普查，不再受截断影响）。
两个一起挂：`mitmdump -s net/addon.py -s net/rawdump.py`。

跑法（推荐用 launcher，它管证书与代理的启停与还原）：
    set GUANDAN_RAW=1 && python -m net.launcher
"""

import json
import os
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST_KEY = "hlxyxws"
OUT = os.path.join(_PROJ, "net", "raw.jsonl")


class RawDump:
    def __init__(self, out_path=OUT):
        self.out_path = out_path
        self._fh = None
        self.stats = {"frames": 0, "bytes": 0, "max": 0}

    def _emit(self, **ev):
        if self._fh is None:
            self._fh = open(self.out_path, "a", encoding="utf-8", buffering=1)
        ev["t"] = round(time.time(), 3)
        self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def websocket_message(self, flow):
        try:
            if HOST_KEY not in str(flow.request.pretty_host):
                return
            m = flow.websocket.messages[-1]
            body = bytes(m.content)
            # **全量**，不截断。n 是真实长度，配上全量 hex 才能自检有没有丢。
            self._emit(k="frame", dir="C→S" if m.from_client else "S→C",
                       n=len(body), hex=body.hex())
            self.stats["frames"] += 1
            self.stats["bytes"] += len(body)
            self.stats["max"] = max(self.stats["max"], len(body))
        except Exception as exc:                      # noqa: BLE001
            print(f"[raw] 转储异常 {type(exc).__name__}: {exc}", flush=True)

    def websocket_end(self, flow):
        if HOST_KEY in str(flow.request.pretty_host):
            print(f"[raw] 断开，累计 {self.stats}", flush=True)

    def done(self):
        if self.stats["frames"]:
            print(f"[raw] 收工 {self.stats} -> {self.out_path}", flush=True)


addons = [RawDump()]
