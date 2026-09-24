from net.sim import meld
from net import cards

_TABLE = {}
for _cid in range(1, 334):
    if cards.is_card(_cid):
        _TABLE[cards.decode(_cid)] = _cid
_TABLE.update({"小王": 14, "大王": 15, "小王(二副)": 270, "大王(二副)": 271})


def C(*names):
    """牌面名 -> 牌 ID。只在本测试文件里用，meld.py 里不许有字符串。"""
    return [_TABLE[n] for n in names]


def test_point_value_level_card_beats_ace():
    assert meld.point_value(5, 5) > meld.point_value(1, 5)     # 打5，5 比 A 大
    assert meld.point_value(1, 5) > meld.point_value(13, 5)    # A 比 K 大


def test_norm_level_ace_written_as_14():
    """日志里 A 可能写成 14，必须归一成 1，否则会被当成小王。"""
    assert meld.norm_level(14) == 1
    assert meld.norm_level(5) == 5
    assert meld.point_value(1, 14) == meld.point_value(1, 1)
    assert meld.point_value(14, 14) == meld.point_value(14, 1)   # 小王不变


def test_point_value_level_two_and_ace():
    """边界：级牌正好是 2 或 A。"""
    assert meld.point_value(2, 2) > meld.point_value(1, 2)     # 打2，2 最大
    assert meld.point_value(1, 1) > meld.point_value(13, 1)    # 打A，A 最大
    assert meld.point_value(15, None) > meld.point_value(14, None)


def test_single_pair_triple():
    hand = C("5♦", "5♣", "5♠", "7♥")
    kinds = {}
    for m in meld.melds_from(hand, level=2):
        kinds.setdefault(m.kind, []).append(m.size)
    assert kinds[meld.SINGLE] == [1, 1]
    assert kinds[meld.PAIR] == [2]
    assert kinds[meld.TRIPLE] == [3]


def test_two_decks_are_two_cards():
    """两副牌的同名牌是两张，不能当一张。"""
    hand = C("5♦", "5♦(二副)")
    pairs = [m for m in meld.melds_from(hand, level=2) if m.kind == meld.PAIR]
    assert len(pairs) == 1
    assert len(pairs[0].cards) == 2 and len(set(pairs[0].cards)) == 2


def test_bomb_sizes():
    hand = C("5♦", "5♦(二副)", "5♣", "5♣(二副)", "5♠", "5♠(二副)", "5♥")
    sizes = sorted(m.size for m in meld.melds_from(hand, level=9)
                   if m.kind == meld.BOMB)
    assert sizes == [4, 5, 6, 7]


def test_triple_pair():
    hand = C("5♦", "5♣", "5♠", "3♥", "3♣")
    tp = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.TRIPLE_PAIR]
    assert len(tp) == 1 and tp[0].size == 5


def _mk(kind, size, rank, ids):
    return meld.Meld(kind=kind, size=size, rank=rank, cards=tuple(ids))


def test_bomb_order_matches_user_spec():
    """4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 天王炸"""
    four = _mk(meld.BOMB, 4, 5, (1, 2, 3, 4))
    five = _mk(meld.BOMB, 5, 5, (1, 2, 3, 4, 5))
    six = _mk(meld.BOMB, 6, 5, (1, 2, 3, 4, 5, 6))
    seven = _mk(meld.BOMB, 7, 5, (1, 2, 3, 4, 5, 6, 7))
    eight = _mk(meld.BOMB, 8, 5, tuple(range(1, 9)))
    flush = _mk(meld.STRAIGHT_FLUSH, 5, 10, (1, 2, 3, 4, 5))
    joker = _mk(meld.BOMB, 4, 0, (14, 15, 270, 271))
    assert meld.beats(five, four)
    assert meld.beats(flush, five)
    assert meld.beats(six, flush)
    assert meld.beats(seven, six)
    assert meld.beats(eight, seven)
    assert meld.beats(joker, eight)
    assert not meld.beats(four, five)


def test_four_jokers_is_the_top_bomb():
    """四大天王必须能枚举出来，且是最高层级。

    漏了这条，用户永远拿不到「出天王炸」的建议 —— 数据里 465 手没出现过，
    所以只有这条测试能挡住它。
    """
    hand = C("小王", "小王(二副)", "大王", "大王(二副)")
    top = [m for m in meld.melds_from(hand, level=9)
           if meld.bomb_class(m) == 7]
    assert len(top) == 1
    assert len(top[0].cards) == 4
    assert meld.beats(top[0], _mk(meld.BOMB, 8, 5, tuple(range(1, 9))))


def test_bomb_beats_normal_and_not_reverse():
    pair = _mk(meld.PAIR, 2, 13, (1, 2))
    bomb = _mk(meld.BOMB, 4, 2, (3, 4, 5, 6))
    assert meld.beats(bomb, pair)
    assert not meld.beats(pair, bomb)


def test_same_kind_compares_rank_only():
    low = _mk(meld.PAIR, 2, 5, (1, 2))
    high = _mk(meld.PAIR, 2, 6, (3, 4))
    assert meld.beats(high, low)
    assert not meld.beats(low, high)
    assert not meld.beats(low, low)          # 一样大不能压


def test_different_kind_does_not_beat():
    pair = _mk(meld.PAIR, 2, 5, (1, 2))
    single = _mk(meld.SINGLE, 1, 13, (1,))
    assert not meld.beats(single, pair)


def test_unknown_bomb_size_raises():
    """认不出的炸弹张数必须报错，不许静默。"""
    import pytest
    weird = _mk(meld.BOMB, 3, 5, (1, 2, 3))
    with pytest.raises(ValueError):
        meld.bomb_class(weird)


def test_legal_moves_when_leading_is_not_empty():
    """领出时合法着法非空。"""
    hand = C("5♦", "7♥")
    moves = meld.legal_moves(hand, table=None, level=2)
    assert moves, "领出必须能出牌"
    assert all(m.size >= 1 for m in moves)


def test_legal_moves_against_table():
    """桌面是对 9，只有更大的对子和炸弹能出。"""
    hand = C("5♦", "5♣", "J♦", "J♣", "K♠")
    table = _mk(meld.PAIR, 2, meld.point_value(9, 2), (16 + 9, 32 + 9))
    moves = meld.legal_moves(hand, table=table, level=2)
    assert moves
    for m in moves:
        assert meld.beats(m, table)
    assert not any(m.kind == meld.SINGLE for m in moves)


def test_triple_pair_allows_joker_pair():
    """王可以当三带二里的对子。

    真实抓包数据里出现过这一手（card_type=5）：
        2♦ 2♦(二副) 2♣ 小王 小王(二副)
    pairs 若把王排除掉，这手的三带二候选是空的，Task 7 验收①会直接报红。
    """
    hand = C("2♦", "2♦(二副)", "2♣", "小王", "小王(二副)")
    tp = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.TRIPLE_PAIR]
    assert len(tp) == 1
    m = tp[0]
    assert set(m.cards) == set(hand)            # 正好是这 5 张
    assert m.size == 5
    assert m.rank == meld.point_value(2, 9)     # rank 取三张的点数，不是王的


def test_jokers_cannot_form_triples_or_cross_pairs():
    """防止改过头的否定断言：王只按同类成对，且永远凑不出三张。"""
    two = C("小王", "小王(二副)")
    moves = meld.melds_from(two, level=9)
    assert not any(m.kind == meld.TRIPLE for m in moves)
    assert not any(m.kind == meld.TRIPLE_PAIR for m in moves)

    both = C("小王", "大王")                    # 不同 idx，凑不成一对
    assert not any(m.kind == meld.PAIR for m in meld.melds_from(both, level=9))
