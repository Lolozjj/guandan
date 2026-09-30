"""实时面板：跟读事件流，把牌局状态画在终端里。

两种跑法：
    python -m guandan.ui.panel              # 跟读实时事件流（要 mitmproxy 在跑）
    python -m guandan.ui.panel --replay     # 用抓包文件回放一遍（离线，看看长什么样）

现在还是朴素文本版 —— 先把数据和实时性验证好，再接项目里现成的面板 UI。
"""

import argparse
import json
import os
import sys
import time

from guandan.capture import cards, protocol
from guandan.capture.addon import EVENTS
from guandan.capture.levelwatch import LevelWatcher
from guandan.capture.state import GameState

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
    if t == "leader":
        st.on_leader(ev["seat"])
        return f"开桌：{st.seat_label(ev['seat'])} 领出"
    if t == "seat":
        st.on_seat(ev["seat"])
        return f"座位同步：我是 {ev['seat']} 号"
    if t == "hand":
        st.on_hand(ev["cards"], ev.get("seat"))
        return f"手牌同步 {len(ev['cards'])} 张"
    if t == "pass":
        st.on_pass(ev["seat"], ev.get("next"))
        return f"{st.seat_label(ev['seat'])} 要不起"
    if t == "disconnect":
        return "游戏服断开"
    return ""


def draw(st: GameState, hint: str, n_ev: int, live: bool,
         shadow_line: str = "") -> None:
    mode = "实时" if live else "回放"
    print(CLEAR, end="")
    print(f"{DIM}[{mode}] 已收 {n_ev} 个事件   {hint}{RESET}")
    if shadow_line:
        print(f"{DIM}{shadow_line}{RESET}")
    print(st.render())




def run_live(st: GameState, path: str, level: int = None, shadow_log=None,
             seconds: float = 0) -> None:
    """实时跟读事件流。

    `level` 给离线用；实时那条路自己从游戏日志读级别（`LevelWatcher` —— 网络里
    没有本局级别，而级别一局之内不变，日志那 ~20 秒延迟无妨）。
    `seconds` 只给测试用（>0 时跑这么久就返回）。
    """
    tail, lvl = Tailer(path), LevelWatcher()
    n, hint = 0, "等游戏服数据…"
    if level:
        st.level = level
        st.level_src = "--level（手输）"
        if shadow_log is not None:
            shadow_log.note_level()
    t0 = time.time()
    draw(st, hint, n, True, shadow_log.panel_text() if shadow_log else "")
    try:
        while True:
            events = tail.read()
            if events:
                for ev in events:
                    h = apply_event(st, ev)
                    if h:
                        hint = h
                    n += 1
                    if shadow_log is not None:
                        shadow_log.after_event(st, ev)
                draw(st, hint, n, True, shadow_log.panel_text() if shadow_log else "")
            else:
                time.sleep(0.05)
            lv = lvl.poll()
            if lv is not None:
                st.level = lv
                st.level_src = lvl.level_src
                hint = f"级别更新：打{st.level_name()}（{lvl.level_src}）"
                if shadow_log is not None:
                    shadow_log.note_level()
            if seconds and time.time() - t0 >= seconds:
                return
    finally:
        # Ctrl-C 也要收尾：没回填的决策点不落盘的话，「分歧点」的统计会有偏
        if shadow_log is not None:
            shadow_log.close()


def run_replay(st: GameState, capture: str, delay: float, level: int = None,
               shadow_log=None) -> None:
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
    if level:
        st.level = level
        st.level_src = "--level（手输）"
    for r in frames:
        body = bytes.fromhex(r["hex"])
        msg = protocol.parse(body)
        ev = None
        if msg and msg["msgid"] == 3005:
            p = protocol.decode_play(msg["fields"])
            if p:
                # `left_cards` 一定要带上：**只有自己的出牌有它**，座位就是靠它认出来的
                # （不带的话回放里永远不知道该把哪家当我，影子模式会一路跳过）
                ev = {"type": "play", "seat": p["seat"], "cards": p["cards"],
                      "card_type": p["card_type"], "next": p["next"],
                      "left": p["left"], "left_cards": p.get("left_cards")}
        elif msg and msg["msgid"] == 3006:
            q = protocol.decode_pass(msg["fields"])
            if q:
                ev = {"type": "pass", "seat": q["seat"], "next": q["next"]}
        elif msg and msg["msgid"] == 3004:
            # 开桌那条：谁领出。回放也要走同一条路，否则回放里我领出那局没建议
            s2 = protocol.decode_leader(msg["fields"])
            if s2 is not None:
                ev = {"type": "leader", "seat": s2}
        elif msg and msg["msgid"] == 3019:
            h = protocol.decode_hand(msg["fields"])
            if h:
                ev = {"type": "hand", "cards": h["cards"]}
        if ev is not None:
            hint = apply_event(st, ev)
            n += 1
            if shadow_log is not None:
                shadow_log.after_event(st, ev)
        if n:
            draw(st, hint, n, False, shadow_log.panel_text() if shadow_log else "")
            time.sleep(delay)
    if shadow_log is not None:
        shadow_log.close()
    print("\n回放结束。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="用抓包文件回放")
    ap.add_argument("--capture", default=r"C:\Users\17837\mitmtool\capture.jsonl")
    ap.add_argument("--delay", type=float, default=0.35, help="回放时每步停多久")
    ap.add_argument("--events", default=EVENTS)
    ap.add_argument("--level", type=int, default=None,
                    help="回放/离线时直接给级别（网络里没有本局级别）")
    ap.add_argument("--no-advice", action="store_true", help="关掉影子模式")
    ap.add_argument("--no-show-advice", action="store_true",
                    help="算建议但不显示（记录里会如实写 advice_shown:false）")
    args = ap.parse_args()

    st = GameState()
    # 建记录器只有**一处**（`shadow.open_shadow`）—— launcher 也走它，
    # 免得「实机入口忘了接」这种事再发生一次
    sh = None
    if not args.no_advice:
        from guandan.advice import shadow
        sh = shadow.open_shadow(show_advice=not args.no_show_advice)
        print(sh.last_line)
    try:
        if args.replay:
            run_replay(st, args.capture, args.delay, level=args.level,
                       shadow_log=sh)
        else:
            run_live(st, args.events, level=args.level, shadow_log=sh)
    except KeyboardInterrupt:
        print("\n面板已退出。")


if __name__ == "__main__":
    sys.exit(main())
