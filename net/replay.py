"""把抓包文件当"实时流"回放一遍，用日志当真值检验整条链路。

跑这个不需要证书、不需要联网、不用开游戏 —— 纯离线。
这是第二步的验收：证明「网络报文 → 协议解码 → 状态机 → 面板」串起来是对的。

用法：
    python -m net.replay
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

from . import cards, protocol
from .state import GameState

DEFAULT_CAPTURE = r"C:\Users\17837\mitmtool\capture.jsonl"
LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2}):(\d{3})")
GIVE_RE = re.compile(r"NotifyGiveCards 后台通知客户端出牌结果 info = (\{.*)")


def load_frames(path):
    """取游戏服「服务器 → 客户端」的报文，按时序排好。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") != "ws_msg":
                continue
            if "hlxyxws" not in str(r.get("host")) or r.get("dir") != "S→C":
                continue
            out.append((r["t"], r["n"], bytes.fromhex(r["hex"])))
    out.sort(key=lambda x: x[0])
    return out


def log_plays(t0, t1):
    """日志里窗口内的出牌真值。"""
    res = []
    for f in glob.glob(os.path.join(LOG_DIR, "*.log")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TS_RE.match(line)
                if not m:
                    continue
                t = datetime.datetime.strptime(
                    f"{m.group(1)} {m.group(2)}.{m.group(3)}",
                    "%Y-%m-%d %H:%M:%S.%f")
                if not (t0 <= t.timestamp() <= t1):
                    continue
                g = GIVE_RE.search(line)
                if not g:
                    continue
                try:
                    d = json.loads(g.group(1))
                except ValueError:
                    continue
                res.append((d.get("SeatID"), list(d.get("CardList") or [])))
    return res


def feed(frames, st, verbose=False):
    """把报文喂给状态机，返回喂进去的事件统计。"""
    stat = {"出牌": 0, "手牌": 0, "解析失败": 0}
    for t, n, body in frames:
        m = protocol.parse(body)
        if not m:
            stat["解析失败"] += 1
            continue
        if m["msgid"] == 3005:
            p = protocol.decode_play(m["fields"])
            if p:
                st.on_play(p["seat"], p["cards"], p["card_type"],
                           p["next"], p["left"], p.get("left_cards"))
                stat["出牌"] += 1
        elif m["msgid"] == 3019:
            h = protocol.decode_hand(m["fields"])
            if h:
                st.on_hand(h["cards"])
                stat["手牌"] += 1
    return stat


def check_turn(frames):
    """轮次预测校验 —— 这是对「要不起也要推进轮次」的硬检验。

    规则：每个出牌事件发生时，状态机预测的 turn 应该**正好就是这个出牌人**
    （第一次出牌时状态机还不知道，跳过）。用日志当真值，不用等实机就能验。
    """
    st = GameState()
    total = right = 0
    wrong = []
    for t, n, body in frames:
        m = protocol.parse(body)
        if not m:
            continue
        if m["msgid"] == 3005:
            p = protocol.decode_play(m["fields"])
            if not p:
                continue
            if st.turn is not None:
                total += 1
                if st.turn == p["seat"]:
                    right += 1
                else:
                    wrong.append({"出牌人": p["seat"], "预测": st.turn,
                                  "时间": datetime.datetime.fromtimestamp(
                                      t).strftime("%H:%M:%S")})
            st.on_play(p["seat"], p["cards"], p["card_type"],
                       p["next"], p["left"], p.get("left_cards"))
        elif m["msgid"] == 3006:
            q = protocol.decode_pass(m["fields"])
            if q:
                st.on_pass(q["seat"], q["next"])
        elif m["msgid"] == 3019:
            h = protocol.decode_hand(m["fields"])
            if h:
                st.on_hand(h["cards"])
    return total, right, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--show", action="store_true", help="打印最终面板")
    args = ap.parse_args()

    if not os.path.exists(args.capture):
        sys.exit(f"[FAIL] 找不到 {args.capture}")

    frames = load_frames(args.capture)
    if not frames:
        sys.exit("[FAIL] 抓包里没有游戏服的下行报文")
    t0, t1 = frames[0][0], frames[-1][0]
    print(f"报文 {len(frames)} 条，"
          f"窗口 {datetime.datetime.fromtimestamp(t0):%H:%M:%S}"
          f" ~ {datetime.datetime.fromtimestamp(t1):%H:%M:%S}")

    st = GameState()
    stat = feed(frames, st)
    print(f"喂进状态机: 出牌 {stat['出牌']}、手牌 {stat['手牌']}、"
          f"解析失败 {stat['解析失败']}")

    # ---- 与日志对照：出牌序列 ----
    truth = log_plays(t0 - 5, t1 + 15)
    got = [(p.seat, p.cards) for p in st.plays]
    print(f"\n[出牌序列] 日志 {len(truth)} 条 / 状态机 {len(got)} 条")
    same = 0
    for i in range(min(len(truth), len(got))):
        if truth[i] == got[i]:
            same += 1
    print(f"  逐条对齐（同序同内容）: {same}/{min(len(truth), len(got))}"
          f" = {same / max(1, min(len(truth), len(got))):.0%}")
    for i in range(min(3, len(truth), len(got))):
        mark = "✓" if truth[i] == got[i] else "✗"
        print(f"    {mark} 日志 seat{truth[i][0]} {cards.decode_all(truth[i][1])}")
        if truth[i] != got[i]:
            print(f"      状态机 seat{got[i][0]} {cards.decode_all(got[i][1])}")

    # ---- 与日志对照：我的手牌 ----
    print(f"\n[我的手牌] 状态机手上 {len(st.hand)} 张")
    print("  " + " ".join(st.hand_grouped()))

    if args.show:
        print("\n" + "=" * 60)
        print(st.render())
        print("=" * 60)

    ok = same == min(len(truth), len(got)) and min(len(truth), len(got)) > 0
    print(f"\n[结论] 链路{'可用 ✓' if ok else '需调整 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
