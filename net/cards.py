"""掼蛋牌 ID ↔ 牌面的互相转换。

编码规则不是猜的，是从游戏自己的日志里逐张对照出来的
（`牌SOLE[|67|323|307|51]{牌花色点数[4张]|3♦|3♦|3♣...}` 这种行，
1891 张比对一致）：

    id = 花色基址 + 点数
    ♠=16  ♥=32  ♣=48  ♦=64        点数 A=1, 2..10, J=11, Q=12, K=13
    第二副牌 +256
    小王=14  大王=15（第二副为 270 / 271）

注意第二副的王是 270/271，不是 14/15 —— 第一版解码就是漏了这条，错了 52 张。
"""

_SUIT = {16: "♠", 32: "♥", 48: "♣", 64: "♦"}
_RANK = {1: "A", 11: "J", 12: "Q", 13: "K", **{i: str(i) for i in range(2, 11)}}
_JOKER = {14: ("小王", 1), 15: ("大王", 1), 270: ("小王", 2), 271: ("大王", 2)}

MAX_ID = 271 + 62          # 第二副最大的牌（第二副 K♦ = 333）


def is_card(cid: int) -> bool:
    """这个整数是不是一个合法牌 ID。

    用来把「牌数组」和「别的整数数组」分开 —— 这是解码的关键判据，
    比看字段号可靠，因为帧头长度会变、字段路径会跟着漂。
    """
    if cid in _JOKER:
        return True
    if not 0 < cid <= 333:
        return False
    low = cid % 256
    base = low // 16 * 16
    return base in _SUIT and (low - base) in _RANK


def decode(cid: int) -> str:
    """牌 ID -> 人读牌面。77 -> K♦，332 -> Q♦(二副)。"""
    if cid in _JOKER:
        name, deck = _JOKER[cid]
        return name if deck == 1 else f"{name}(二副)"
    deck = cid // 256 + 1
    low = cid % 256
    base = low // 16 * 16
    name = _RANK[low - base] + _SUIT[base]
    return name if deck == 1 else f"{name}(二副)"


def decode_all(ids) -> list:
    return [decode(c) for c in ids]


# ---------------------------------------------------------------- 大小排序

# 掼蛋的大小顺序，**越大越靠前**：
#     大王 > 小王 > 级牌 > A > K > Q > J > 10 > 9 > … > 2
# 注意级牌是**跟着当前级别走的**（打 5 的时候四个 5 就排在 A 前面），
# 所以排序必须带 level，不能只按牌 ID 排。
_RANK_ORDER = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
               10: 9, 11: 10, 12: 11, 13: 12, 1: 13}
_SUIT_ORDER = {"♠": 0, "♥": 1, "♣": 2, "♦": 3}
_TIER_JOKER_BIG, _TIER_JOKER_SMALL, _TIER_LEVEL, _TIER_NORMAL = 3, 2, 1, 0


def parts(cid: int):
    """牌 ID -> (点数索引, 花色, 第几副)。

    点数索引跟游戏日志的 Trump 字段一致：A=1、2..10、J=11、Q=12、K=13，
    王用 14（小王）/ 15（大王）。
    """
    if cid in _JOKER:
        name, deck = _JOKER[cid]
        return (14 if name == "小王" else 15), "", deck
    deck = cid // 256 + 1
    low = cid % 256
    base = low // 16 * 16
    return low - base, _SUIT[base], deck


def sort_key(cid: int, level: int = None):
    """排序键 —— 面板按它**降序**排就是掼蛋的顺序。

    **注意这是给「降序」用的键**：花色和副数都要取负，
    否则降序会把 ♠♥♣♦ 反过来变成 ♦♣♥♠、把二副排到正牌前面。
    （这个坑我踩过：花色反了，牌面看着"排过序"但顺序不对。）
    """
    idx, suit, deck = parts(cid)
    d = -deck
    if idx == 15:
        return (_TIER_JOKER_BIG, 0, 0, d)
    if idx == 14:
        return (_TIER_JOKER_SMALL, 0, 0, d)
    lv = level
    if lv == 14:           # 日志里偶尔用 14 表示 A
        lv = 1
    if lv is not None and idx == lv:
        # 级牌之间也要按花色排 —— 这里漏了花色的话，几个级牌的先后是乱的。
        return (_TIER_LEVEL, 0, -_SUIT_ORDER.get(suit, 9), d)
    return (_TIER_NORMAL, _RANK_ORDER.get(idx, 0),
            -_SUIT_ORDER.get(suit, 9), d)


def sort_ids(ids, level: int = None) -> list:
    """按掼蛋大小排好（大的在前）。"""
    return sorted(ids, key=lambda c: sort_key(c, level), reverse=True)


def names_sorted(ids, level: int = None) -> list:
    """按掼蛋大小排好并转成牌面名。"""
    return [decode(c) for c in sort_ids(ids, level)]


# ---------------------------------------------------------------- 编码位
#
# 给「把一手牌编码成定长向量」用的：两副牌 108 张 -> 0..107。
# 布局：每副 54 张 = ♠A..♠K(0..12) + ♥(13..25) + ♣(26..38) + ♦(39..51) + 小王(52) + 大王(53)，
# 第二副整体 +54。**花色是第一档分组**，这样同花色的牌落在连续区间里。

_SUIT_ORDER_4 = ("♠", "♥", "♣", "♦")
SLOTS = 108


def slot(cid: int) -> int:
    """牌 ID -> 0..107 的编码位。非法的牌 ID **直接炸**（不静默给个默认位）。"""
    if cid in _JOKER:
        name, deck = _JOKER[cid]
        return (deck - 1) * 54 + (52 if name == "小王" else 53)
    if not is_card(cid):
        raise ValueError(f"不是合法牌 ID：{cid}")
    deck = cid // 256
    low = cid % 256
    base = low // 16 * 16
    return deck * 54 + _SUIT_ORDER_4.index(_SUIT[base]) * 13 + (low - base - 1)
