"""把游戏日志解析成结构化对局。

日志只保留 2 天，且按小时切文件，所以解析要按文件名排序、跨文件连续扫。
**解析不出来必须报错，不能返回空列表** —— 否则验收会空跑还报绿
（本项目吃过一次「静默出错」的亏，见 HANDOFF 验收协议）。

实测补充（brief 之外）：发牌行同一毫秒还会多出一行
``SendCardsService set roundID:S…``（例：``S7380R1T1669t6AB4E78ES0A``）。
它与发牌行共享前缀，但既没有 ``k : {json}`` 载荷、也不是新的一局，只是同一局
的伴随行。若把它当成发牌行，一局会凭空裂成两局（实测 63 局会被数成 126 局），
或因为解不出 JSON 而直接抛错。这里显式识别并跳过；除此之外任何「像发牌行却
认不出」的行，一律明着 raise。
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2}):(\d{3})")
_DEAL_PREFIX = re.compile(r"SendCardsService set roundID")
_DEAL = re.compile(r"SendCardsService set roundID:.* k : (\{.*)")
# 伴随行：`roundID:S…`，行尾就是该 token，没有 k : {json} 载荷。
_DEAL_COMPANION = re.compile(r"SendCardsService set roundID:S[0-9A-Za-z]+\s*$")
_PLAY = re.compile(r"NotifyGiveCards 后台通知客户端出牌结果 info = (\{.*)")
_SETTLE = re.compile(r"EVA1B001结算协议 = (\{.*)")


@dataclass
class PlayRec:
    seat: int
    cards: list[int]
    card_type: int
    left: int        # 这一手之后该家还剩几张
    nxt: int         # 服务器给的下一手座位；-1 表示本局结束


@dataclass
class GameLog:
    t0: datetime
    trump: int                 # 级别（A=1, 2..10, J=11, Q=12, K=13）
    my_cards: list[int]        # 发牌给我自己的 27 张
    plays: list[PlayRec] = field(default_factory=list)
    settle: Optional[dict] = None
    unparsed: int = 0          # 前缀命中却解不出 JSON 的出牌/结算行条数（丢数据要看得见）


def _ts(line):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}.{m.group(3)}",
                             "%Y-%m-%d %H:%M:%S.%f")


def _json(pattern, line):
    m = pattern.search(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def load_games(log_dir: str = LOG_DIR) -> list[GameLog]:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(
            f"日志目录不存在：{log_dir}\n"
            f"日志只保留 2 天，可能被轮转删了 —— 别把它当成「0 局」继续跑。")

    files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not files:
        raise RuntimeError(f"目录在但一个 .log 都没有：{log_dir}")

    games, cur = [], None
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _DEAL_PREFIX.search(line):
                    # 伴随行与发牌行共享前缀但不是一局；必须在「收上一局」之前
                    # 判掉，否则当前局会被错误地截断成两局。
                    if _DEAL_COMPANION.search(line):
                        continue
                    if cur is not None:
                        games.append(cur)      # 上一局没有结算，也收着
                    d = _json(_DEAL, line)
                    if not d:
                        raise RuntimeError(f"发牌行解不出 JSON：{line[:200]}")
                    cur = GameLog(t0=_ts(line) or datetime.min,
                                  trump=int(d["Trump"]),
                                  my_cards=list(d["Cards"]))
                    continue
                if cur is None:
                    continue
                if _PLAY.search(line):
                    d = _json(_PLAY, line)
                    if d is None:
                        # 前缀命中却解不出 JSON：真丢了一条，必须计数暴露出来，
                        # 不许静默。注意「JSON 能解析但没有 CardList」（如「要不起」）
                        # 是正常消息、不是丢数据，不能计进来。
                        cur.unparsed += 1
                    elif d.get("CardList"):
                        cur.plays.append(PlayRec(
                            seat=int(d["SeatID"]),
                            cards=list(d["CardList"]),
                            card_type=int(d.get("CardType", 0)),
                            left=int(d.get("LeftCardLen", 0)),
                            nxt=int(d.get("NextTurnSeatID", -1))))
                    continue
                if _SETTLE.search(line):
                    d = _json(_SETTLE, line)
                    if d is None:
                        cur.unparsed += 1      # 同上：丢了一条结算行要看得见
                    else:
                        cur.settle = d
                        games.append(cur)
                        cur = None
    if cur is not None:
        games.append(cur)
    return games


def conserved(g: GameLog) -> bool:
    """出过的牌 + 结算剩的牌 == 108（两副牌）。"""
    if not g.settle:
        return False
    played = sum(len(p.cards) for p in g.plays)
    left = sum(len(e.get("Cards") or []) for e in (g.settle.get("LeftCards") or []))
    return played + left == 108


# ---------------------------------------------------------------- 语料快照
#
# 为什么需要：验收脚本读的是游戏日志，而**日志会被轮转删除**（只留约 2 天）。
# 一旦日志没了，`load_games` 会（按设计）抛错，Plan 1 的回归防线就整个跑不起来 ——
# 而 Plan 2/3 每次改模拟器都要靠它。快照让这道防线不依赖日志还活着。
#
# 注意 `load_games` 的语义**不因此改变**（它照旧只读实时日志、目录没了就抛错）：
# 它的测试要真的在测实时解析。「优先快照」这条策略放在 `load_corpus` 里，
# 并且会把实际用了哪一份报出来 —— 换源必须可见。

SNAPSHOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "game_corpus.json")

#: 最近一次 `load_corpus` 用的是哪一份语料（给人看的字符串）
LAST_SOURCE = None

#: 快照里保留的结算字段 —— 只留用得到的，别把整份 PlayerInfo 塞进仓库
_SETTLE_KEEP = ("Rank", "LeftCards", "UpgradeInfo")


def to_jsonable(g: GameLog) -> dict:
    settle = None
    if g.settle:
        settle = {k: g.settle[k] for k in _SETTLE_KEEP if k in g.settle}
    return {
        "t0": g.t0.isoformat(),
        "trump": g.trump,
        "my_cards": list(g.my_cards),
        "plays": [[p.seat, list(p.cards), p.card_type, p.left, p.nxt]
                  for p in g.plays],
        "settle": settle,
        "unparsed": g.unparsed,
    }


def from_jsonable(d: dict) -> GameLog:
    g = GameLog(t0=datetime.fromisoformat(d["t0"]),
                trump=int(d["trump"]),
                my_cards=list(d["my_cards"]),
                unparsed=int(d.get("unparsed", 0)))
    g.plays = [PlayRec(seat=int(a), cards=list(b), card_type=int(c),
                       left=int(e), nxt=int(f))
               for a, b, c, e, f in d["plays"]]
    g.settle = d.get("settle")
    return g


def save_snapshot(games, path: str = SNAPSHOT) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump([to_jsonable(g) for g in games], fh, ensure_ascii=False)
        fh.write("\n")
    return path


def load_snapshot(path: str = SNAPSHOT) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"没有语料快照：{path}\n"
            f"先生成：.venv/Scripts/python.exe -m tools.snapshot_logs")
    with open(path, encoding="utf-8") as fh:
        return [from_jsonable(d) for d in json.load(fh)]


def load_corpus(log_dir: str = None, snapshot: str = SNAPSHOT) -> list:
    """验收用：**优先快照**（日志会被轮转删），没有才退回实时日志。

    用哪一份记在 `LAST_SOURCE` 里，调用方应当报出来 —— 换源必须可见，
    不能让人以为在验实时日志、其实验的是几个月前的快照。
    """
    global LAST_SOURCE
    if log_dir is None and snapshot and os.path.exists(snapshot):
        LAST_SOURCE = f"语料快照 {os.path.relpath(snapshot)}（实时日志可能已被轮转删除）"
        return load_snapshot(snapshot)
    LAST_SOURCE = f"实时日志 {log_dir or LOG_DIR}"
    return load_games(log_dir or LOG_DIR)
