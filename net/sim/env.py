"""掼蛋自对弈环境（spec §4）。

**这个模块唯一的、不可动摇的纪律：喂给策略的状态里不许有对手手牌。**

做法不是靠注释提醒，是靠**类型**：`Observation` 这个 dataclass 里根本没有
「四家手牌」这个字段 —— 它只有可观测的东西（我的手牌、各家出过的牌、各家剩几张、
桌面、轮到谁、级别）。编码器 `encode_state` 只吃 `Observation`，
所以**它在结构上就无法泄漏**。`tests/test_env_leak.py` 再用自动化测试钉死一遍。

泄漏的后果（spec §3）：会训出靠偷看才成立的打法 —— 训练分数漂亮、真机全废，
**而且静默失效**（同本项目「合成 val 骗过一次」的教训）。

座位一律**相对化**：索引 0 = 自己、1 = 下家、2 = 对家、3 = 上家。
四个座位共享一套权重，所以相对化不是可选项。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from net import cards
from net.sim import meld, rules

#: spec §4.1 的状态维度，逐项对齐：
#:     我的手牌 108 + 已出过的牌 4×108 + 各家剩几张 4 + 桌面待压 108
#:     + 桌面牌型 10 + 桌面主点数 15 + 谁要不起 4 + 轮到谁 4 + 级别 15
STATE_DIM = 108 + 4 * 108 + 4 + 108 + 10 + 15 + 4 + 4 + 15
assert STATE_DIM == 700

_OFF_HAND = 0
_OFF_PLAYED = _OFF_HAND + 108
_OFF_LEFT = _OFF_PLAYED + 4 * 108
_OFF_TABLE = _OFF_LEFT + 4
_OFF_KIND = _OFF_TABLE + 108
_OFF_RANK = _OFF_KIND + 10
_OFF_PASSED = _OFF_RANK + 15
_OFF_TURN = _OFF_PASSED + 4
_OFF_LEVEL = _OFF_TURN + 4

#: 桌面/动作里的「主点数」编码：点数 1..13 -> 0..12；
#: **级牌 -> 13**（`meld.POINT_LEVEL`）；王 -> 14（`POINT_SMALL` / `POINT_BIG` 合到一格，
#: 因为王当桌面牌型时只有「一对王」这一种，大小王不会再区分）。
_RANK_SLOTS = 15


def _rank_slot(point: int) -> int:
    """主点数 -> 0..14 的槽位：普通点数 1..13 -> 0..12、**级牌 -> 13**、王 -> 14。

    ⚠️ 别写成 `point >= POINT_LEVEL` 一档到底 —— 那样**级牌会掉进王的槽位**
    （POINT_LEVEL=14 而王是 15/16），于是「桌面是级牌」与「桌面是一对王」
    在编码里分不开，而它们是两个完全不同的层级。
    """
    if point >= meld.POINT_SMALL:          # 15 / 16 = 王
        return _RANK_SLOTS - 1             # 14
    if point >= meld.POINT_LEVEL:          # 14 = 级牌
        return _RANK_SLOTS - 2             # 13
    return min(max(point, 1), _RANK_SLOTS - 2) - 1


@dataclass(frozen=True)
class Observation:
    """策略能看见的东西。

    **故意没有「四家手牌」这个字段。** 这不是省略，是设计 ——
    只要它不在这个类型里，编码器就没有办法把它编进去。
    """
    seat: int                       # 出牌人（相对化的原点，**绝对**座位号）
    hand: frozenset                 # 我的手牌
    played: Tuple[frozenset, ...]   # 4 家各自出过的牌（公开信息）
    left: Tuple[int, ...]           # 4 家各剩几张（公开信息）
    table: Tuple[int, ...]          # 桌面待压的牌；空元组 = 我领出
    table_kind: int                 # 桌面牌型（0 = 无）
    table_rank: int                 # 桌面主点数（`meld.point_value` 口径；-1 = 无）
    passed: Tuple[bool, ...]        # 本轮谁「要不起」（公开信息）
    turn: int                       # 轮到谁（绝对座位）
    level: int


def _rel(seq, seat: int):
    """把按绝对座位排的序列转成相对座位（0 = 自己、1 = 下家 …）。"""
    return tuple(seq[(seat + i) % 4] for i in range(4))


def encode_state(obs: Observation) -> np.ndarray:
    """`Observation` -> 700 维 float32。**只吃 `Observation`**，别给它开别的入口。"""
    v = np.zeros(STATE_DIM, dtype=np.float32)

    for c in obs.hand:
        v[_OFF_HAND + cards.slot(c)] = 1.0

    for i, cards_i in enumerate(_rel(obs.played, obs.seat)):
        for c in cards_i:
            v[_OFF_PLAYED + i * 108 + cards.slot(c)] = 1.0

    for i, n in enumerate(_rel(obs.left, obs.seat)):
        v[_OFF_LEFT + i] = n / 27.0            # 归一，别让 27 这种量级和 0/1 混在一起

    for c in obs.table:
        v[_OFF_TABLE + cards.slot(c)] = 1.0

    if obs.table_kind:
        v[_OFF_KIND + obs.table_kind - 1] = 1.0
    if obs.table_rank >= 0:
        v[_OFF_RANK + _rank_slot(obs.table_rank)] = 1.0

    for i, p in enumerate(_rel(obs.passed, obs.seat)):
        v[_OFF_PASSED + i] = 1.0 if p else 0.0

    v[_OFF_TURN + (obs.turn - obs.seat) % 4] = 1.0
    v[_OFF_LEVEL + obs.level] = 1.0            # 1..13 用第 1..13 格；0 与 14 恒为 0

    return v


def observe(hand: "rules.Hand", seat: int, played, table_meld: Optional["meld.Meld"]):
    """从**明牌**的 `Hand` 里切出 `seat` 视角的可观测状态。

    **这是明牌与策略之间唯一的窄口**：进来的是整个 `Hand`（四家都看得见），
    出去的 `Observation` 里只有公开信息 + 自己的手牌。

    `played` 由调用方维护（`{座位: 出过的牌}`），因为 `Hand` 只记 `steps`，
    不按座位分桶 —— 分桶是面板/记牌器的口径，不是规则的口径。
    """
    table = ()
    kind, rank = 0, -1
    if table_meld is not None:
        table = tuple(table_meld.cards)
        kind = table_meld.kind
        rank = table_meld.rank
    return Observation(
        seat=seat,
        hand=frozenset(hand.hands[seat]),
        played=tuple(frozenset(played.get(s, ())) for s in rules.SEATS),
        left=tuple(len(hand.hands[s]) for s in rules.SEATS),
        table=table, table_kind=kind, table_rank=rank,
        passed=tuple(s in hand.passed for s in rules.SEATS),
        turn=hand.turn,
        level=hand.level,
    )
