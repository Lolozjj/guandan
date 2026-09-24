"""掼蛋牌型校验 —— 用来判断识别出来的出牌「合不合理」。

为什么需要（用户提的，很对）：模型有时会把「三个 10 带对三」这种 5 张牌
认成「两个 3 + 一个 10」这种 3 张牌。**这种组合在掼蛋里根本不存在** ——
也就是说，识别结果自己就暴露了它是错的。

实测那次：机器人3 那 5 张牌里最左边两张被「出牌」按钮**完全盖住**，
模型连 0.03 门槛都输出不了。按钮底下的牌，任何算法都读不出来，
所以正确的做法不是硬猜，而是**认出「这个结果不合法」，然后老实说不知道**。

合法牌型：
    单张 / 对子 / 三张 / 三带二 / 三带一(3+1) / 顺子(5) / 同花顺(5)
    连对(2 或 3 连对) / 钢板(2 个连续三张) / 炸弹(4 张及以上同点数)
    四大天王(4 张王)
另外**级牌红桃是万能牌**（逢人配），可以当任意牌用，所以判合法性时要把它
当通配符去凑。
"""
from __future__ import annotations

RANK_SEQ = "23456789TJQKA"          # 从小到大
SUIT_OF = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def _rank(cls: str) -> str | None:
    """牌的类别 -> 点数；大小王返回 None。"""
    if cls.startswith("JOKER"):
        return None
    return cls[1:]


def _suit(cls: str) -> str | None:
    return None if cls.startswith("JOKER") else cls[0]


def _idx(rank: str) -> int:
    return RANK_SEQ.index(rank)


def _is_bomb(ranks: list[str]) -> str | None:
    if len(ranks) >= 4 and len(set(ranks)) == 1:
        return "%d 张炸" % len(ranks)
    return None


def _legal_exact(cards: list[str], level: str) -> str | None:
    """不含万能牌时的牌型判定。cards 是类别列表。"""
    jokers = [c for c in cards if c.startswith("JOKER")]
    norm = [c for c in cards if not c.startswith("JOKER")]
    n = len(cards)

    if jokers:
        # 只有「四张王」是合法牌型；大小王不能被当普通牌用
        if n == 4 and len(jokers) == 4:
            return "四大天王"
        if n == 1:
            return "单张"
        if n == 2 and len(jokers) == 2:
            return None          # 两个王不是对子
        return None

    ranks = [_rank(c) for c in norm]
    suits = [_suit(c) for c in norm]
    uniq = set(ranks)

    if n == 1:
        return "单张"
    if n == 2:
        return "对子" if len(uniq) == 1 else None
    if n == 3:
        return "三张" if len(uniq) == 1 else None

    bomb = _is_bomb(ranks)
    if bomb:
        return bomb

    if n == 4:
        # 两个连续对子（木板）
        if len(uniq) == 2:
            a, b = sorted(_idx(r) for r in uniq)
            if b - a == 1:
                return "连对"
        return None

    if n == 5:
        # 三带二 / 三带一
        cnt = {}
        for r in ranks:
            cnt[r] = cnt.get(r, 0) + 1
        if sorted(cnt.values()) == [2, 3]:
            return "三带二"
        # 顺子 / 同花顺
        idx = sorted(_idx(r) for r in uniq)
        if len(uniq) == 5 and idx[-1] - idx[0] == 4:
            return "同花顺" if len(set(suits)) == 1 else "顺子"
        return None

    if n == 6:
        cnt = {}
        for r in ranks:
            cnt[r] = cnt.get(r, 0) + 1
        vals = sorted(cnt.values())
        idx = sorted(_idx(r) for r in uniq)
        if vals == [2, 2, 2]:
            if idx[-1] - idx[0] == 2 and len(uniq) == 3:
                return "三连对"
        if vals == [3, 3] and len(uniq) == 2 and idx[-1] - idx[0] == 1:
            return "钢板"
        return None

    return None


def classify(cards: list[str], level: str = "2") -> str | None:
    """判牌型。合法返回牌型名，不合法返回 None。

    level 是当前级牌点数；级牌红桃（逢人配）当万能牌，会尝试各种补法。
    """
    if not cards:
        return None
    wild_cls = "H" + level
    n_wild = sum(1 for c in cards if c == wild_cls)

    if n_wild == 0:
        return _legal_exact(list(cards), level)

    # 先按原样判一次 —— 级牌红桃也可能就是在当它自己那张牌用
    # （比如 ♠3♣3♥3 本身就是三个 3），这时不该标「含逢人配」
    direct = _legal_exact(list(cards), level)
    if direct:
        return direct

    rest = [c for c in cards if c != wild_cls]
    if n_wild == 1:
        # 万能牌当哪张都行：补上任意一种牌再看合不合法
        for r in RANK_SEQ:
            for s in "SCDH":
                got = _legal_exact(rest + [s + r], level)
                if got:
                    return got + "（含逢人配）"
        return None
    # 两张及以上万能牌：组合数很少，直接穷举
    import itertools
    pool = [s + r for r in RANK_SEQ for s in "SCDH"]
    for combo in itertools.combinations_with_replacement(pool, n_wild):
        got = _legal_exact(rest + list(combo), level)
        if got:
            return got + "（含逢人配）"
    return None


def describe(cards: list[str], level: str = "2") -> str:
    """给面板显示用：合法就写牌型，不合法就明确说不合法。"""
    t = classify(cards, level)
    if t:
        return t
    return "不合法"
