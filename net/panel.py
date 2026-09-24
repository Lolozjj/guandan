"""实时面板：跟读事件流，把牌局状态画在终端里。

两种跑法：
    python -m net.panel              # 跟读实时事件流（要 mitmproxy 在跑）
    python -m net.panel --replay     # 用抓包文件回放一遍（离线，看看长什么样）

现在还是朴素文本版 —— 先把数据和实时性验证好，再接项目里现成的面板 UI。
"""

import argparse
import json
import os
import sys
import time

from . import cards, protocol
from .addon import EVENTS
from .state import GameState

CLEAR = "\033[2J\033[H"
DIM = "\033[2m"
RESET = "\033[0m"


class Tailer:
    """从文件末尾开始跟读 JSONL。处理「文件被重建 / 变小」。"""

    def __init__(self, path, from_end=True):
        self.path = path
        self.pos = os.path.getsize(path) if (from_end and os.path.exists(path)) else 0

    def read(self):
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size < self.pos:                 # 被重建了
            self.pos = 0
        if size == self.pos:
            return []
        out = []
        with open(self.path, "rb") as fh:
            fh.seek(self.pos)
            raw = fh.read()
        cut = raw.rfind(b"\n")
        if cut == -1:
            return []
        self.pos += cut + 1
        for line in raw[:cut + 1].decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out


def apply_event(st: GameState, ev: dict) -> str:
    """把一个事件喂进状态机，返回给面板显示的一行提示。"""
    t = ev.get("type")
    if t == "play":
        st.on_play(ev["seat"], ev["cards"], ev.get("card_type", 0),
                   ev.get("next", 0), ev.get("left", 0),
                   ev.get("left_cards"))
        who = st.seat_label(ev["seat"])
        return f"{who} 出 {' '.join(ev.get('names') or cards.decode_all(ev['cards']))}"
    if t == "hand":
        st.on_hand(ev["cards"])
        return f"手牌同步 {len(ev['cards'])} 张"
    if t == "pass":
        st.on_pass(ev["seat"], ev.get("next"))
        return f"{st.seat_label(ev['seat'])} 要不起"
    if t == "disconnect":
        return "游戏服断开"
    return ""


def draw(st: GameState, hint: str, n_ev: int, live: bool) -> None:
    mode = "实时" if live else "回放"
    print(CLEAR, end="")
    print(f"{DIM}[{mode}] 已收 {n_ev} 个事件   {hint}{RESET}")
    print(st.render())


def run_live(st: GameState, path: str) -> None:
    tail = Tailer(path)
    n, hint = 0, "等游戏服数据…"
    draw(st, hint, n, True)
    while True:
        events = tail.read()
        if events:
            for ev in events:
                h = apply_event(st, ev)
                if h:
                    hint = h
                n += 1
            draw(st, hint, n, True)
        else:
            time.sleep(0.05)


def run_replay(st: GameState, capture: str, delay: float) -> None:
    """把抓包文件当实时流慢慢喂，看面板长什么样。"""
    frames = []
    with open(capture, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") == "ws_msg" and "hlxyxws" in str(r.get("host")) \
                    and r.get("dir") == "S→C":
                frames.append(r)
    frames.sort(key=lambda r: r["t"])
    n, hint = 0, "回放开始"
    for r in frames:
        body = bytes.fromhex(r["hex"])
        msg = protocol.parse(body)
        if msg and msg["msgid"] == 3005:
            p = protocol.decode_play(msg["fields"])
            if p:
                ev = {"type": "play", "seat": p["seat"], "cards": p["cards"],
                      "names": cards.decode_all(p["cards"]),
                      "card_type": p["card_type"], "next": p["next"],
                      "left": p["left"]}
                hint = apply_event(st, ev); n += 1
        elif msg and msg["msgid"] == 3019:
            h = protocol.decode_hand(msg["fields"])
            if h:
                hint = apply_event(st, {"type": "hand", "cards": h["cards"]}); n += 1
        if n:
            draw(st, hint, n, False)
            time.sleep(delay)
    print("\n回放结束。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="用抓包文件回放")
    ap.add_argument("--capture", default=r"C:\Users\17837\mitmtool\capture.jsonl")
    ap.add_argument("--delay", type=float, default=0.35, help="回放时每步停多久")
    ap.add_argument("--events", default=EVENTS)
    args = ap.parse_args()

    st = GameState()
    try:
        if args.replay:
            run_replay(st, args.capture, args.delay)
        else:
            run_live(st, args.events)
    except KeyboardInterrupt:
        print("\n面板已退出。")


if __name__ == "__main__":
    sys.exit(main())
