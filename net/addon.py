"""mitmproxy 插件：把掼蛋牌局事件实时写成一行行 JSON。

只干一件事 —— 监听游戏服（hlxyxws）**服务器 → 客户端**的 WebSocket，
解出牌局事件后追加到 `net/events.jsonl`。面板在另一个进程跟读这个文件。

为什么用文件而不是 socket：mitmproxy 跑在独立的 venv 里，面板跑在项目 venv 里，
文件是两个进程之间最省事的通道；而且断线重连、面板重启都不丢数据，
出问题还能直接拿文件复盘。延迟只有轮询间隔（~50ms），
相对于原来那条 20 秒的日志路线可以忽略。

注意：**这里不 import mitmproxy**，所以这个模块在项目 venv 里也能直接 import
做离线测试。
"""

import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from net import cards, protocol          # noqa: E402

EVENTS = os.path.join(_PROJ, "net", "events.jsonl")
HOST_KEY = "hlxyxws"


class GuandanTap:
    """mitmproxy 的插件对象，靠 addons = [GuandanTap()] 交给它。"""

    def __init__(self, out_path=EVENTS):
        self.out_path = out_path
        self._fh = None
        self.stats = {"frames": 0, "plays": 0, "hands": 0, "passes": 0,
                      "unparsed": 0}

    # ------------------------------------------------------------ 内部

    def _emit(self, **ev):
        if self._fh is None:
            # 面板可能还没建好目录
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            self._fh = open(self.out_path, "a", encoding="utf-8", buffering=1)
        ev["t"] = round(time.time(), 3)
        self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def handle(self, body: bytes) -> dict:
        """解一帧，返回事件 dict（解不出来返回 None）。抽出来是为了能离线测。"""
        self.stats["frames"] += 1
        msg = protocol.parse(body)
        if not msg:
            self.stats["unparsed"] += 1
            return None

        if msg["msgid"] == 3005:
            play = protocol.decode_play(msg["fields"])
            if play:
                self.stats["plays"] += 1
                return {
                    "type": "play",
                    "seat": play["seat"],
                    "cards": play["cards"],
                    "names": cards.decode_all(play["cards"]),
                    "card_type": play["card_type"],
                    "next": play["next"],
                    "left": play["left"],
                    "left_cards": play.get("left_cards"),
                }
        elif msg["msgid"] == 3019:
            hand = protocol.decode_hand(msg["fields"])
            if hand:
                self.stats["hands"] += 1
                return {
                    "type": "hand",
                    "cards": hand["cards"],
                    "names": cards.decode_all(hand["cards"]),
                }
        elif msg["msgid"] == 3006:
            p = protocol.decode_pass(msg["fields"])
            if p:
                self.stats["passes"] += 1
                return {"type": "pass", "seat": p["seat"], "next": p["next"]}
        return None

    # ------------------------------------------------------ mitmproxy 钩子

    def websocket_message(self, flow):
        try:
            if HOST_KEY not in str(flow.request.pretty_host):
                return
            m = flow.websocket.messages[-1]
            if m.from_client:
                return                      # 上行是密文，不碰
            ev = self.handle(bytes(m.content))
            if ev:
                self._emit(**ev)
        except Exception as exc:            # noqa: BLE001 - 插件里崩了会拖垮抓包
            print(f"[tap] 解析异常 {type(exc).__name__}: {exc}", flush=True)

    def websocket_end(self, flow):
        if HOST_KEY in str(flow.request.pretty_host):
            self._emit(type="disconnect")
            print(f"[tap] 游戏服断开，累计 {self.stats}", flush=True)


addons = [GuandanTap()]
