"""A1b 的变体模式：**默认必须逐位不变**，开了必须是**超集**（现状代表排第一）。

契约一句话：`variants=True` 只**多给选项**，不改规则、不改形状集合 ——
否则"多给选项值多少"这个实验就不成立了（变成"换了套玩法"）。
"""
from guandan.sim import env, meld, rules

A = meld.cid_from_name


def _level_bomb_hand():
    """打 5 的手牌：8 张 5（含 2 张 ♥ = 逢人配）+ 一张 9♠。"""
    return sorted([A("S5"), A("S5", deck=2), A("C5"), A("C5", deck=2),
                   A("D5"), A("D5", deck=2), A("H5"), A("H5", deck=2), A("S9")])


def _key(ms):
    return {(m.kind, tuple(sorted(m.cards))) for m in ms}


def test_default_is_unchanged_and_variants_are_a_superset():
    hand = _level_bomb_hand()
    base = _key(meld.melds_from(hand, 5))
    more = _key(meld.melds_from(hand, 5, variants=True))
    assert base <= more, "开了变体必须是超集（现状代表排第一）"
    assert len(more) > len(base), "这手牌明明有花色变体可选"


def test_variants_close_the_wildcard_gap():
    """A1 探针查出的缺口：级牌炸的代表**必然带 ♥ = 逢人配** —— 开了变体必须能不用它。"""
    hand = _level_bomb_hand()
    bombs = [m for m in meld.melds_from(hand, 5, variants=True)
             if m.kind == meld.BOMB and m.size == 4]
    assert any(not any(meld.is_wild(c, 5) for c in m.cards) for m in bombs), \
        "变体里没有「不用逢人配的 4 炸」—— 缺口没补上"


def test_wild_used_counts_substitutions_not_hearts():
    """`wild_used` 数的是**替身**，不是"牌组里有几张 ♥级牌"。

    打 5 时：4 张 5 里的 ♥5 就是**它自己那个点数** ⇒ 0 张替身（与老的 `_melds_basic` 一致）；
    而拿 ♥5 顶一张 9 凑成对 9 ⇒ 1 张替身。
    ⚠️ 这条钉的是我自己踩过的坑：变体里按"有几张 ♥"算，会让同一条候选的 `wild_used`
    在"基础代表"与"变体"之间自相矛盾（0 vs 1）。
    """
    for m in meld.melds_from(_level_bomb_hand(), 5, variants=True):
        if m.kind == meld.BOMB and m.size == 4:
            assert m.wild_used == 0, f"级牌炸里的 ♥5 是自己那个点数，不算替身：{m}"
    hand9 = sorted([A("S9"), A("H5"), A("S3")])
    pair9 = [m for m in meld.melds_from(hand9, 5, variants=True)
             if m.kind == meld.PAIR and meld.cards.parts(m.cards[0])[0] == 9]
    assert pair9 and pair9[0].wild_used == 1, "顶了一张 9 就该算 1 张替身"


def test_shape_set_is_unchanged():
    """验收① 按形状比对：开了变体**不该多出新的形状**。"""
    hand = _level_bomb_hand()
    base = {(m.kind, m.size, m.rank) for m in meld.melds_from(hand, 5)}
    more = {(m.kind, m.size, m.rank) for m in meld.melds_from(hand, 5, variants=True)}
    assert base == more


def test_cache_key_includes_variants():
    """⚠️ `variants` 不进缓存键的话：先开再关会拿到**开的那一份**（静默错）。"""
    hand = _level_bomb_hand()
    first = meld.melds_from(hand, 5, variants=True)
    second = meld.melds_from(hand, 5)
    assert len(second) < len(first), "缓存把 variants 这个键混掉了"


def test_env_expands_only_the_named_seats():
    """`expand_seats` 只影响名单里的座位；别的座位与 `Hand.actions` 默认口径一致。"""
    e = env.GuandanEnv(seed=7)
    e.reset()
    seat = e.hand.turn
    assert e.expand_seats == frozenset(), "默认必须是空集（老行为）"
    base = len(e.legal())
    e.expand_seats = frozenset({seat})
    assert len(e.legal()) >= base, "名单里的座位应当拿到超集"
    other = next(s for s in rules.SEATS if s != seat)
    e.expand_seats = frozenset()
    e.hand.turn = other                       # 只改"轮到谁"，看候选口径
    assert len(e.legal()) == len(e.hand.actions(other))
