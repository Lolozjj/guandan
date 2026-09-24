"""离线验收：拿抓包数据 + 游戏日志，检验协议解码对不对。

这是第一步的验收脚本 —— **不需要联网、不需要装证书、不碰游戏**，
只用已经抓下来的报文和游戏自己写的日志。

用法：
    python -m net.verify
    python -m net.verify --capture <capture.jsonl 路径>
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re
import sys

from . import cards, protocol

DEFAULT_CAPTURE = r"C:\Users\17837\mitmtool\capture.jsonl"
LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2}):(\d{3})")
GIVE_RE = re.compile(r"NotifyGiveCards 后台通知客户端出牌结果 info = (\{.*)")
DEAL_RE = re.compile(r"SendCardsService set roundID:([\d,]*) k : (\{.*)")


def load_frames(path, host="hlxyxws", direction="S→C"):
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") != "ws_msg":
                continue
            if host not in str(r.get("host")) or r.get("dir") != direction:
                continue
            body = bytes.fromhex(r["hex"])
            truncated = len(body) < r["n"]
            frames.append({"t": r["t"], "n": r["n"], "body": body,
                           "truncated": truncated})
    return frames


def log_events(kinds=("give", "deal")):
    """从游戏日志里取真值。日志就是我们的标准答案。"""
    gives, deals = [], []
    for f in glob.glob(os.path.join(LOG_DIR, "*.log")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TS_RE.match(line)
                if not m:
                    continue
                t = datetime.datetime.strptime(
                    f"{m.group(1)} {m.group(2)}.{m.group(3)}",
                    "%Y-%m-%d %H:%M:%S.%f")
                g = GIVE_RE.search(line)
                if g:
                    try:
                        gives.append((t, json.loads(g.group(1))))
                    except ValueError:
                        pass
                d = DEAL_RE.search(line)
                if d:
                    try:
                        deals.append((t, json.loads(d.group(2))))
                    except ValueError:
                        pass
    return gives, deals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--window", type=float, default=6.0,
                    help="日志事件与报文的配对时间窗（秒）")
    args = ap.parse_args()

    if not os.path.exists(args.capture):
        sys.exit(f"[FAIL] 找不到抓包文件：{args.capture}")

    frames = load_frames(args.capture)
    full = [f for f in frames if not f["truncated"]]
    print(f"游戏服 S→C 报文 {len(frames)} 条"
          f"（完整 {len(full)}，被截断 {len(frames) - len(full)}）")

    # ---- 1. 帧边界自检 ----
    ok_len = sum(1 for f in frames if protocol.declared_length(f["body"]) == f["n"])
    print(f"\n[1] 帧长自检（前 2 字节大端帧长 == 实际长度）: "
          f"{ok_len}/{len(frames)} = {ok_len / len(frames):.0%}")

    # ---- 2. 解码覆盖率 ----
    parsed = []
    for f in frames:
        m = protocol.parse(f["body"])
        if m:
            parsed.append((f, m))
    print(f"[2] 能解出协议结构的报文: {len(parsed)}/{len(frames)} "
          f"= {len(parsed) / len(frames):.0%}")

    names = collections.Counter(m["name"] for _, m in parsed)
    print(f"    msgid 分布: {dict(names.most_common(10))}")

    # ---- 3. 出牌事件 vs 日志 ----
    # 日志里有两天、上千条出牌，只有抓包窗口内那些才是可比的 —— 先裁窗口，
    # 否则命中率会被无关的历史事件稀释成假的低分。
    gives, deals = log_events()
    t0 = min(f["t"] for f in frames) - 5
    t1 = max(f["t"] for f in frames) + max(args.window, 15)
    all_gives, all_deals = gives, deals
    gives = [(t, g) for t, g in gives if t0 <= t.timestamp() <= t1]
    deals = [(t, d) for t, d in deals if t0 <= t.timestamp() <= t1]
    print(f"\n    抓包窗口 {datetime.datetime.fromtimestamp(t0):%H:%M:%S}"
          f" ~ {datetime.datetime.fromtimestamp(t1):%H:%M:%S}；"
          f"日志全量 出牌{len(all_gives)} 发牌{len(all_deals)}，"
          f"窗口内 出牌{len(gives)} 发牌{len(deals)}")

    plays = []
    for f, m in parsed:
        p = protocol.decode_play(m["fields"])
        if p:
            plays.append((f["t"], p))
    print(f"\n[3] 解出出牌事件 {len(plays)} 条；窗口内日志 {len(gives)} 条")

    matched = 0
    checked = 0
    for t, g in gives:
        want = g.get("CardList") or []
        if len(want) < 1:
            continue
        checked += 1
        te = t.timestamp()
        for ft, p in plays:
            if abs(ft - te) <= args.window and p["cards"] == list(want):
                matched += 1
                break
    print(f"    与日志逐张对照：{matched}/{checked} 命中 "
          f"= {matched / max(1, checked):.0%}")

    # ---- 4. 字段级核对（取一条命中的细看）----
    print("\n[4] 字段级核对样本：")
    shown = 0
    for t, g in gives:
        want = list(g.get("CardList") or [])
        if not want:
            continue
        te = t.timestamp()
        for ft, p in plays:
            if abs(ft - te) > args.window or p["cards"] != want:
                continue
            diff = {k: (p.get(k), g.get(j)) for k, j in
                    (("seat", "SeatID"), ("next", "NextTurnSeatID"),
                     ("card_type", "CardType"), ("left", "LeftCardLen"),
                     ("count", "CardLen"))
                    if p.get(k) != g.get(j)}
            flag = "全对" if not diff else f"不一致 {diff}"
            print(f"    {t.strftime('%H:%M:%S')} seat{p['seat']} "
                  f"{p['count']}张 {cards.decode_all(p['cards'])}  -> {flag}")
            shown += 1
            break
        if shown >= 6:
            break

    # ---- 5. 手牌 ----
    print("\n[5] 手牌核对：")
    hands = []
    for f, m in parsed:
        h = protocol.decode_hand(m["fields"])
        if h:
            hands.append((f["t"], h))
    print(f"    解出手牌报文 {len(hands)} 条；日志里发牌 {len(deals)} 次")
    hit = 0
    for t, d in deals:
        want = set(d.get("Cards") or [])
        if len(want) < 20:
            continue
        te = t.timestamp()
        for ft, h in hands:
            if abs(ft - te) <= 15 and set(h["cards"]) == want:
                hit += 1
                break
    print(f"    与日志发牌对照：{hit}/{len(deals)} 命中")

    print("\n[结论] 协议解码" + ("可用 ✓" if matched >= checked * 0.6
                              else "还需调整 ✗"))


if __name__ == "__main__":
    main()
