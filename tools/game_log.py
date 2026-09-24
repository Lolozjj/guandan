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
                    if d and d.get("CardList"):
                        cur.plays.append(PlayRec(
                            seat=int(d["SeatID"]),
                            cards=list(d["CardList"]),
                            card_type=int(d.get("CardType", 0)),
                            left=int(d.get("LeftCardLen", 0)),
                            nxt=int(d.get("NextTurnSeatID", -1))))
                    continue
                if _SETTLE.search(line):
                    d = _json(_SETTLE, line)
                    if d:
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
