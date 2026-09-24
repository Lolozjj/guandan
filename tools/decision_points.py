"""从结构化对局重建每个决策点：谁、手上什么、桌上什么、实际出了什么。

手牌重建原理：每家起手牌 = 出过的 ∪ 结算剩的。进贡/还贡在第一手之前完成，
所以重建出来的就是「第一手之前」的手牌，之后只减不增。

桌面判定刻意**不看牌型**（那是 net/sim/meld.py 的事，这里用了就循环依赖）。
只认两种「领出」：同一座位又出牌（其余三家都要不起，他重新领出）；或队友接风
（上一手的主人已经出完，队友接着领出）。

**不能拿服务器的 `NextTurnSeatID` 判领出** —— 服务器算下一手时会跳过已经出完的
座位，所以 nxt 指到队友既可能是接风，也可能只是跳过了一个出完的座位（那时队友
其实是在压牌）。实测（55 局 / 1706 手）：`nxt == 桌面主人` 从不命中（0 次，
因为 nxt 不会绕回自己）；按 nxt 判会误清 34 手、漏清 27 手，其中 27 手会被重建
成「用对子去压三带二」这种非法响应。改成只看座位与「主人是否出完」后，两类都归零。

等级（`GameLog.trump`）原样透传，本模块不做归一 —— 遇到 1 就是 1、遇到 14 就是 14，
怎么解释交给下游 net/sim/meld.py（本模块一改，下游就没法自己定了）。

认不出的局面一律明着 raise：重建出「出了手上没有的牌」这种矛盾必须炸，
不能当没看见 —— 静默出错会让下游验收报假绿。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tools.game_log import GameLog


@dataclass
class Snapshot:
    seat: int
    hand: list                 # 该座位此刻手上的牌（含他即将打出的）
    table: Optional[list]      # 桌面待压的牌；None = 他领出
    actual: list               # 他实际打出的牌（真值）
    level: int
    t: datetime


def initial_hands(g: GameLog) -> dict[int, set[int]]:
    """每家起手牌 = 他出过的所有牌 ∪ 他结算时剩的牌。"""
    hands = {i: set() for i in range(4)}
    for p in g.plays:
        hands.setdefault(p.seat, set())
        hands[p.seat] |= set(p.cards)
    if g.settle:
        for i, e in enumerate(g.settle.get("LeftCards") or []):
            hands.setdefault(i, set())
            hands[i] |= set(e.get("Cards") or [])
    return hands


def decision_points(g: GameLog) -> list[Snapshot]:
    remaining = {i: set(v) for i, v in initial_hands(g).items()}
    snaps = []
    table = None
    table_seat = None
    prev_left = None           # 上一手之后主人还剩几张；0 = 他刚出完

    for p in g.plays:
        if table is not None:
            partner = (table_seat + 2) % 4
            # 清桌只有两种情况：
            #   1) 同一座位又出牌了 —— 其余三家都要不起，他重新领出
            #   2) 队友接风 —— 上一手的主人已经出完，队友接着领出
            # 注意 2) 的判据是「主人出完了」（prev_left == 0），不是「nxt 指到队友」。
            if p.seat == table_seat or (p.seat == partner and prev_left == 0):
                table = None

        snaps.append(Snapshot(seat=p.seat,
                              hand=sorted(remaining.get(p.seat, set())),
                              table=list(table) if table else None,
                              actual=list(p.cards),
                              level=g.trump,
                              t=g.t0))

        remaining.setdefault(p.seat, set())
        missing = set(p.cards) - remaining[p.seat]
        if missing:
            raise RuntimeError(
                f"{g.t0} 座位{p.seat} 打出了手上没有的牌 {sorted(missing)} —— "
                f"重建错了，不要静默跳过")
        remaining[p.seat] -= set(p.cards)
        if p.left == 0:
            remaining[p.seat] = set()

        table = list(p.cards)
        table_seat = p.seat
        prev_left = p.left
    return snaps
