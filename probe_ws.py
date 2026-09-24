"""对局探针：同时盯两件事。

1. 微信小程序宿主进程（WeChatAppEx.exe）的网络连接——找**长命**的连接，
   判断游戏那条 wss 到底走不走系统代理（走 Clash 127.0.0.1:7897）。
   XHR 加载素材是短连接，WebSocket 会一直挂着，靠存活时长区分。
   这一条决定 MITM 抓包这条路通不通。

2. 游戏日志的落盘节奏——空闲时实测是严格 20 秒一批，对局中会不会更快？
   顺便量出每条牌局事件从发生到出现在磁盘上的真实滞后。

stdout 每行一条事件（给监控用）；每次落盘都追加到 audit/ws_probe.jsonl（不打印，留档）。
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

USER = r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users\67b5ef56e08ab757e0cd7cac86e2366d"
LOGDIR = os.path.join(USER, r"applet\local\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")
OUT = r"C:\Users\17837\PycharmProjects\yolo\audit\ws_probe.jsonl"
PROXY_PORT = "7897"

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2}):(\d{3})")

PATTERNS = [
    ("发牌", re.compile(r"SendCardsService set roundID:([\d,]+) k : (\{.*)")),
    ("出牌", re.compile(r"NotifyGiveCards 后台通知客户端出牌结果 info = (\{.*)")),
    ("人读", re.compile(r"玩家seatId = (\d+)出的牌: (.*)")),
    ("要不起", re.compile(r"OrUpService 其他玩家要不起, 其SeatID = (\d+)")),
    ("进贡", re.compile(r"TributeService 其他玩家进贡, 其SeatID = (\d+)")),
    ("还贡", re.compile(r"ReturnTributeServie 其他玩家还贡, 其SeatID = (\d+)")),
    ("我出", re.compile(r"GiveCardsService 自己实际出牌 = \[(\d+)张\]\|(.*?)\|")),
    ("结算", re.compile(r"结算协议 = \{")),
]


def line_ts(line):
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(
            m.group(1) + " " + m.group(2) + "." + m.group(3),
            "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


class Tail:
    """跟着小时文件滚动读日志，只消费完整行。"""

    def __init__(self):
        self.path = None
        self.pos = 0
        self.in_game = False
        self.last_flush = None
        self.last_size = 0

    def poll(self):
        files = glob.glob(os.path.join(LOGDIR, "*.log"))
        if not files:
            return [], None
        f = max(files, key=os.path.getmtime)
        note = None
        if self.path != f:
            if self.path is None:
                self.pos = os.path.getsize(f)      # 首次：从当下接上，不读两天历史
                self.last_size = self.pos
                note = "接入 %s" % os.path.basename(f)
            else:
                self.pos = 0                        # 整点换文件
                self.last_size = 0
                note = "换文件 → %s" % os.path.basename(f)
            self.path = f
        try:
            size = os.path.getsize(f)
        except OSError:
            return [], note
        if size < self.pos:                         # 被截断
            self.pos = 0
        if size == self.pos:
            return [], note
        with open(f, "rb") as fh:
            fh.seek(self.pos)
            raw = fh.read()
        cut = raw.rfind(b"\n")
        if cut == -1:
            return [], note
        self.pos += cut + 1
        return raw[:cut + 1].decode("utf-8", "replace").splitlines(), note


class Conns:
    """只报告活得久的连接——短命的是素材 XHR，长命的是 WebSocket。"""

    LONG = 15.0

    def __init__(self):
        self.pids = {}
        self.pid_age = 0.0
        self.live = {}

    def wechat_pids(self):
        if time.time() - self.pid_age > 30:
            self.pid_age = time.time()
            try:
                out = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                                     capture_output=True, text=True,
                                     errors="replace").stdout
            except OSError:
                return self.pids
            self.pids = {}
            for line in out.splitlines():
                p = [x.strip('"') for x in line.split('","')]
                if len(p) >= 2 and "wechatappex" in p[0].lower():
                    self.pids[p[1]] = p[0]
        return self.pids

    def sample(self):
        pids = self.wechat_pids()
        if not pids:
            return []
        try:
            out = subprocess.run(["netstat", "-ano"], capture_output=True,
                                 text=True, errors="replace").stdout
        except OSError:
            return []
        now = time.time()
        current = {}
        for l in out.splitlines():
            f = l.split()
            if len(f) < 5 or not f[0].upper().startswith("TCP"):
                continue
            pid, local, peer, st = f[-1], f[1], f[2], f[3]
            if pid not in pids or st != "ESTABLISHED":
                continue
            if "127.0.0.1" in peer and PROXY_PORT not in peer:
                continue                            # 进程内部通信，不看
            current[local + "->" + peer] = peer

        events = []
        for key, peer in current.items():
            info = self.live.setdefault(
                key, {"seen": now, "reported": False, "peer": peer})
            age = now - info["seen"]
            if age >= self.LONG and not info["reported"]:
                info["reported"] = True
                via = PROXY_PORT in peer
                events.append(
                    "[网络] 长连接出现：%s %s 已存活 %.0fs —— %s" % (
                        "经系统代理" if via else "直连外网", peer, age,
                        "疑似游戏 WebSocket，MITM 有望抓到" if via
                        else "绕过了代理，MITM 抓不到"))
        for key in list(self.live):
            if key not in current:
                info = self.live.pop(key)
                if info["reported"]:
                    events.append("[网络] 长连接 %s 关闭，共存活 %.0fs" % (
                        info["peer"], now - info["seen"]))
        return events


def summarize(events):
    """把一批日志事件压成一行。"""
    bits = []
    for kind, g in events:
        try:
            if kind == "发牌":
                d = json.loads(g[1])
                bits.append("发牌 打%s %s张 先出seat%s" % (
                    d.get("Trump"), d.get("CardLen"), d.get("nWhoIsFirstOut")))
            elif kind == "出牌":
                d = json.loads(g[0])
                bits.append("出牌 seat%s %s张(型%s) → 轮到seat%s" % (
                    d.get("SeatID"), d.get("CardLen"),
                    d.get("CardType"), d.get("NextTurnSeatID")))
            elif kind == "人读":
                bits.append("出的是 %s" % g[1][:44])
            elif kind == "我出":
                bits.append("我出了 %s张 %s" % (g[0], g[1][:30]))
            elif kind in ("要不起", "进贡", "还贡"):
                bits.append("%s seat%s" % (kind, g[0]))
            elif kind == "结算":
                bits.append("本局结算")
        except Exception as e:                       # noqa: BLE001
            bits.append("%s(解析失败:%s)" % (kind, type(e).__name__))
    seen, out = set(), []
    for b in bits:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return " | ".join(out[:8])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    jf = open(OUT, "a", encoding="utf-8")
    tail = Tail()
    conns = Conns()
    next_beat = time.time() + 90
    next_conn = 0.0
    quiet_since = time.time()

    print("探针就绪：盯日志落盘节奏 + 微信长连接（代理端口 %s）。开一局吧。"
          % PROXY_PORT, flush=True)

    while True:
        now = time.time()

        if now >= next_conn:
            next_conn = now + 30
            for e in conns.sample():
                print(e, flush=True)
                quiet_since = now
                jf.write(json.dumps({"t": now, "kind": "net", "msg": e},
                                    ensure_ascii=False) + "\n")
                jf.flush()

        lines, note = tail.poll()
        if note:
            print("[日志] %s" % note, flush=True)
            quiet_since = now
        if lines:
            ev, newest_ts, ev_ts = [], None, []
            for line in lines:
                ts = line_ts(line)
                if ts:
                    newest_ts = ts
                for kind, rx in PATTERNS:
                    m = rx.search(line)
                    if not m:
                        continue
                    ev.append((kind, m.groups()))
                    if ts:
                        ev_ts.append(ts)
                    if kind == "发牌":
                        tail.in_game = True
                    elif kind == "结算":
                        tail.in_game = False

            gap = None if tail.last_flush is None else now - tail.last_flush
            tail.last_flush = now
            lag = None if newest_ts is None else \
                (datetime.now() - newest_ts).total_seconds()
            # 真正决定面板可用性的数字：这一批里**最早那手牌**等了多久才落盘
            lag_old = None if not ev_ts else \
                (datetime.now() - min(ev_ts)).total_seconds()
            lag_new = None if not ev_ts else \
                (datetime.now() - max(ev_ts)).total_seconds()
            delta = os.path.getsize(tail.path) - tail.last_size
            tail.last_size = os.path.getsize(tail.path)

            jf.write(json.dumps(
                {"t": now, "kind": "flush", "gap": gap, "lag": lag,
                 "lag_old_ev": lag_old, "lag_new_ev": lag_new,
                 "delta": delta, "bytes": len(lines), "in_game": tail.in_game,
                 "n_events": len(ev)},
                ensure_ascii=False) + "\n")
            jf.flush()

            if ev:
                quiet_since = now
                print("[落盘] 间隔%s 最老事件滞后%.1fs 最新%.1fs +%dB | %s" % (
                    "-" if gap is None else "%.1fs" % gap,
                    -1.0 if lag_old is None else lag_old,
                    -1.0 if lag_new is None else lag_new,
                    delta, summarize(ev)), flush=True)

        if now >= next_beat:
            next_beat = now + 90
            if now - quiet_since > 90:
                print("[心跳] 探针在跑，90s 内没有新事件。看到这条说明监控还活着。",
                      flush=True)

        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
