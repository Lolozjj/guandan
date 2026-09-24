"""掼蛋牌型真源 —— 合法着法枚举与大小比较。

按**牌 ID**（int）工作，复用 net/cards.py 的编码：
    parts(cid) -> (idx, suit, deck)
    idx 口径 A=1、2..10、J=11、Q=12、K=13、小王=14、大王=15

为什么不用牌名字符串：live/rules.py 的 6 个 bug 有一半来自字符串处理
（"10" vs "T"），见 spec §2.3。这里一律用整数。

牌型表从游戏协议的 card_type 字段统计得到（465 手真牌，spec §2.2）。
炸弹顺序由用户口述 + 24 对真实证据交叉验证（0 条矛盾）：
    4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 天王炸
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from net import cards

SINGLE, PAIR, TRIPLE = 1, 2, 3
STRAIGHT, TRIPLE_PAIR = 4, 5
PAIR_RUN, PLATE = 6, 7
BOMB, STRAIGHT_FLUSH, BOMB6 = 8, 9, 10

JOKER_SMALL, JOKER_BIG = 14, 15

# 非序列牌型的点数比较值：级牌 > A > K > ... > 2
_POINT = {**{i: i - 1 for i in range(2, 11)}, 11: 10, 12: 11, 13: 12, 1: 13}
POINT_LEVEL, POINT_SMALL, POINT_BIG = 14, 15, 16

_MIN_BOMB = 4
_MAX_BOMB = 8          # 两副牌，一个点数最多 8 张

_BOMB_CLASS_BY_SIZE = {4: 1, 5: 2, 6: 4, 7: 5, 8: 6}
CLASS_FLUSH = 3
CLASS_JOKER_BOMB = 7

# --- 序列类牌型（顺子 / 连对 / 钢板）--------------------------------------
# 规模是定死的：数据里 4 张只有炸弹（没有二连对），顺子也只出现 5 张的。
_SEQ_LEN = 5
_PAIR_RUN_LEN = 3
_PLATE_LEN = 2
_NAT_MAX = 14          # A 当大牌时的自然值；也是序列能给到的最大值


def norm_level(level: Optional[int]) -> Optional[int]:
    """级别归一。**日志里偶尔用 14 表示 A**（net/cards.py 的 sort_key 也处理过这条），
    不归一的话 14 会被当成小王，级牌判定与顺子权重全错。
    """
    if level == 14:
        return 1
    return level


def point_value(idx: int, level: Optional[int]) -> int:
    level = norm_level(level)
    if idx == JOKER_BIG:
        return POINT_BIG
    if idx == JOKER_SMALL:
        return POINT_SMALL
    if level is not None and idx == level:
        return POINT_LEVEL
    return _POINT[idx]


def is_wild(cid: int, level: Optional[int]) -> bool:
    """级牌红桃 = 逢人配（万能牌）。"""
    level = norm_level(level)
    if level is None:
        return False
    idx, suit, _ = cards.parts(cid)
    return idx == level and suit == "♥"


def nat_values(idx: int) -> tuple:
    """这个点数在序列里的自然值。**A 可作最小也可作最大**，王不参与序列。

    A2345 与 10JQKA 都合法，所以 A 同时给 1 和 14；这比「比较时特判 A」干净，
    因为枚举端（_seq_lookup）和比较端（rank 取顶端自然值）只有这一个真源。

    王返回空元组 —— 顺子/连对/钢板的「王不能进序列」就落在这一条上，
    枚举端不需要再写一遍 `idx < JOKER_SMALL` 判断。
    """
    if idx == 1:
        return (1, _NAT_MAX)
    if 2 <= idx <= 13:
        return (idx,)
    return ()


@dataclass(frozen=True)
class Meld:
    kind: int
    size: int
    rank: int                       # 比较主键（同 kind 内可比）
    cards: tuple
    wild_used: int = 0

    @property
    def is_bomb(self) -> bool:
        return bomb_class(self) is not None


def _all_jokers(ids) -> bool:
    """四张牌全是王（大小王各两张，即天王炸）。

    先过 `cards.is_card`：单测里的 Meld 允许用占位整数当 cards，比较关系
    不该为此炸掉 KeyError。真实牌局里 cards 全是合法牌 ID，这一层零影响。
    """
    if len(ids) != 4:
        return False
    for c in ids:
        if not cards.is_card(c):
            return False
        if cards.parts(c)[0] not in (JOKER_SMALL, JOKER_BIG):
            return False
    return True


def bomb_class(m: Meld) -> Optional[int]:
    """炸弹层级；不是炸弹返回 None。天王炸最高。"""
    if m.kind == STRAIGHT_FLUSH:
        return CLASS_FLUSH
    if _all_jokers(m.cards):
        return CLASS_JOKER_BOMB
    if m.kind in (BOMB, BOMB6):
        cls = _BOMB_CLASS_BY_SIZE.get(m.size)
        if cls is None:
            raise ValueError(f"不认识的炸弹张数 {m.size}（牌 {m.cards}）")
        return cls
    return None


def beats(a: Meld, b: Meld) -> bool:
    """a 能不能压过 b。"""
    ca, cb = bomb_class(a), bomb_class(b)
    if ca is not None and cb is not None:
        return ca > cb if ca != cb else a.rank > b.rank
    if ca is not None:
        return True                 # 炸弹压普通牌型
    if cb is not None:
        return False                # 普通牌型压不了炸弹
    if a.kind != b.kind or a.size != b.size:
        return False                # 牌型不同不能压
    return a.rank > b.rank


def _by_idx(hand: Sequence[int]) -> dict:
    g = {}
    for c in hand:
        g.setdefault(cards.parts(c)[0], []).append(c)
    return g


def _melds_basic(hand: Sequence[int], level: Optional[int]) -> list:
    out = []
    for idx, ids in _by_idx(hand).items():
        pv = point_value(idx, level)
        out.append(Meld(SINGLE, 1, pv, (ids[0],)))
        if idx >= JOKER_SMALL:
            # 王不能当普通点数用：两个小王是一对，但不能凑三张/炸弹
            if len(ids) >= 2:
                out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
            continue
        if len(ids) >= 2:
            out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
        if len(ids) >= 3:
            out.append(Meld(TRIPLE, 3, pv, tuple(ids[:3])))
        for n in range(_MIN_BOMB, min(len(ids), _MAX_BOMB) + 1):
            out.append(Meld(BOMB, n, pv, tuple(ids[:n])))
    return out


def _melds_triple_pair(level, triples, pairs) -> list:
    out = []
    for t_idx, t_cards in triples:
        for p_idx, p_cards in pairs:
            if p_idx == t_idx:
                continue
            out.append(Meld(TRIPLE_PAIR, 5, point_value(t_idx, level),
                            tuple(t_cards) + tuple(p_cards)))
    return out


def _melds_joker_bomb(hand: Sequence[int]) -> list:
    """四大天王（大小王各两张）。最高层级，bomb_class 靠 _all_jokers 认。

    王在 _melds_basic 里是 continue 掉的（不能凑三张/普通炸弹），
    所以这里必须单独补一条 —— 漏了它，用户永远拿不到「出天王炸」的建议。
    """
    js = [c for c in hand if cards.parts(c)[0] in (JOKER_SMALL, JOKER_BIG)]
    if len(js) >= 4:
        return [Meld(BOMB, 4, 0, tuple(sorted(js[:4])))]
    return []


def _seq_lookup(g: dict) -> dict:
    """自然值 -> 能占这个位置的牌（都按 `_by_idx` 的顺序，取 [0] 就是最小的那组）。

    A 同时落进 1 和 14 两格，王两格都不落 —— 全部由 `nat_values` 决定。
    同一个窗口内 1 与 14 不可能同时出现（窗口最长 5 个连续自然值，
    取值域 1..14），所以 A 不会被同一个顺子用两次。
    """
    nat = {}
    for idx, ids in g.items():
        for nv in nat_values(idx):
            nat.setdefault(nv, []).extend(ids)
    return nat


def _window(nat: dict, top: int, span: int):
    """[top-span+1, top] 这一段连续自然值；只要有一格没牌就返回 None。

    张数够不够（连对要 2、钢板要 3）由调用方自己判 —— 顺子一格一张就够。
    """
    nats = list(range(top - span + 1, top + 1))
    if any(not nat.get(n) for n in nats):
        return None
    return nats


def _melds_straights(g: dict, level) -> list:
    """顺子（恰好 5 张连续）与同花顺。比较主键 = 顶端自然值。"""
    nat = _seq_lookup(g)
    out = []
    for top in range(_SEQ_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _SEQ_LEN)
        if nats is None:
            continue
        ids = [nat[n][0] for n in nats]
        out.append(Meld(STRAIGHT, _SEQ_LEN, top, tuple(ids)))
        # 同花顺 = 顺子且五张同花色；王的花色是空串，进不了这里（也进不了顺子）
        if len({cards.parts(c)[1] for c in ids}) == 1:
            out.append(Meld(STRAIGHT_FLUSH, _SEQ_LEN, top, tuple(ids)))
    return out


def _melds_pair_run(g: dict, level) -> list:
    """连对（恰好 3 个连续对子）。4 张只有炸弹，所以没有二连对。"""
    nat = _seq_lookup(g)
    out = []
    for top in range(_PAIR_RUN_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _PAIR_RUN_LEN)
        if nats is None:
            continue
        if any(len(nat[n]) < 2 for n in nats):
            continue
        ids = [c for n in nats for c in nat[n][:2]]
        out.append(Meld(PAIR_RUN, _PAIR_RUN_LEN * 2, top, tuple(ids)))
    return out


def _melds_plate(g: dict, level) -> list:
    """钢板（恰好 2 个连续三张）。"""
    nat = _seq_lookup(g)
    out = []
    for top in range(_PLATE_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _PLATE_LEN)
        if nats is None:
            continue
        if any(len(nat[n]) < 3 for n in nats):
            continue
        ids = [c for n in nats for c in nat[n][:3]]
        out.append(Meld(PLATE, _PLATE_LEN * 3, top, tuple(ids)))
    return out


def melds_from(hand: Sequence[int], level: Optional[int] = None) -> list:
    """枚举手牌能组成的所有牌型。

    单/对/三/三带二/炸弹/天王炸 + 顺子/同花顺/连对/钢板。
    逢人配在这里**当作它自己那张级牌**参与枚举（Task 6 再加替代能力）。
    """
    level = norm_level(level)
    g = _by_idx(hand)
    out = _melds_basic(hand, level)
    # 三张必须排除王：每种王只有两张，凑不出三张。
    triples = [(i, v[:3]) for i, v in g.items()
               if i < JOKER_SMALL and len(v) >= 3]
    # 对子**不排除王**：游戏允许王当三带二里的对子
    # （真实数据：三个 2 带两张小王）。_by_idx 按 idx 分组，
    # 所以小王只跟小王成对、大王只跟大王成对，不会跨 idx 凑一对。
    pairs = [(i, v[:2]) for i, v in g.items() if len(v) >= 2]
    out += _melds_triple_pair(level, triples, pairs)
    out += _melds_joker_bomb(hand)      # 天王炸（王在 _melds_basic 里是 continue 掉的）
    out += _melds_straights(g, level)
    out += _melds_pair_run(g, level)
    out += _melds_plate(g, level)
    return out


def legal_moves(hand: Sequence[int], table: Optional[Meld],
                level: Optional[int] = None) -> list:
    """现在能出的所有牌。table 为 None 表示我领出（此时不产生「过」）。"""
    moves = melds_from(hand, level)
    if table is None:
        return moves
    return [m for m in moves if beats(m, table)]
