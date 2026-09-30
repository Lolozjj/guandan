from guandan.sim import meld
from tests.test_meld_basic import C


def test_nat_values_ace_is_both_ends():
    assert set(meld.nat_values(1)) == {1, 14}
    assert meld.nat_values(10) == (10,)
    assert meld.nat_values(14) == ()       # 小王不进序列
    assert meld.nat_values(15) == ()       # 大王不进序列


def test_ace_low_straight_is_legal():
    """A2345 是合法顺子（数据里有 2 手），这是老 YOLO 适配层（已删）的 bug #2。"""
    hand = C("A♥", "2♦", "3♥", "4♠", "5♥")
    st = [m for m in meld.melds_from(hand, level=9) if m.kind == meld.STRAIGHT]
    assert len(st) == 1
    assert st[0].rank == 5              # 顶端是 5


def test_ace_high_straight():
    hand = C("10♦", "J♦", "Q♦", "K♠", "A♦")
    st = [m for m in meld.melds_from(hand, level=9) if m.kind == meld.STRAIGHT]
    assert len(st) == 1 and st[0].rank == 14


def test_joker_cannot_join_straight():
    hand = C("10♦", "J♦", "Q♦", "K♠", "大王")
    assert not [m for m in meld.melds_from(hand, level=9)
                if m.kind == meld.STRAIGHT]


def test_straight_flush_requires_same_suit():
    same = C("5♠", "6♠", "7♠", "8♠", "9♠")
    mixed = C("5♠", "6♠", "7♠", "8♥", "9♠")
    assert meld.STRAIGHT_FLUSH in [m.kind for m in meld.melds_from(same, level=2)]
    km = [m.kind for m in meld.melds_from(mixed, level=2)]
    assert meld.STRAIGHT_FLUSH not in km
    assert meld.STRAIGHT in km


def test_straight_flush_survives_card_order():
    """同花顺不能因为「同点数的杂色牌排在前面」而漏掉。

    两副牌下每个点数必有两张不同花色，所以这是常态而非边角。
    漏掉不只是少一个建议 —— Task 7 的验收①会把真实打出的同花顺报成枚举不出。
    """
    has_flush = C("5♥", "5♠", "6♠", "7♠", "8♠", "9♠")     # 5♠ 排在后面
    ordered = C("5♠", "5♥", "6♠", "7♠", "8♠", "9♠")
    for hand, name in ((has_flush, "杂色在前"), (ordered, "同花在前")):
        sf = [m for m in meld.melds_from(hand, level=2)
              if m.kind == meld.STRAIGHT_FLUSH]
        assert len(sf) == 1, f"{name} 的手牌漏了同花顺"
        assert sf[0].rank == 9
        assert {meld.cards.parts(c)[1] for c in sf[0].cards} == {"♠"}


def test_ace_low_straight_flush_survives_card_order():
    """A 低窗同样：A♦ 排在 A♥ 前面时不能漏掉 A♥2♥3♥4♥5♥。"""
    hand = C("A♦", "A♥", "2♥", "3♥", "4♥", "5♥")
    sf = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.STRAIGHT_FLUSH]
    assert len(sf) == 1 and sf[0].rank == 5


def test_every_straight_flush_card_set_is_enumerable():
    """同顶端的两条同花顺（不同花色）必须**都**枚举出来，不能只留一个代表。

    Task 7 验收① 是按**牌张集合**比对的，不是按 (kind, size, rank)：

        tools/accept_meld.py:  sorted(m.cards) == sorted(s.actual)

    所以「同顶端只留一个代表」会把真实打出的另一种花色报成「枚举不出」。
    这里手上同时有 ♠ 和 ♥ 两条顶端 9 的同花顺，两条都得能查到。
    """
    hand = C("5♠", "5♥", "6♠", "6♥", "7♠", "7♥", "8♠", "8♥", "9♠", "9♥")
    found = {frozenset(m.cards) for m in meld.melds_from(hand, level=2)
             if m.kind == meld.STRAIGHT_FLUSH}
    assert frozenset(C("5♠", "6♠", "7♠", "8♠", "9♠")) in found
    assert frozenset(C("5♥", "6♥", "7♥", "8♥", "9♥")) in found


def test_pair_run_is_exactly_three_pairs():
    three = C("4♦", "4♣", "5♦", "5♠", "6♣", "6♠")
    two = C("4♦", "4♣", "5♦", "5♠")
    assert [m for m in meld.melds_from(three, level=2)
            if m.kind == meld.PAIR_RUN]
    assert not [m for m in meld.melds_from(two, level=2)
                if m.kind == meld.PAIR_RUN]


def test_plate_is_two_consecutive_triples():
    plate = C("3♥", "3♦", "3♣", "4♦", "4♣", "4♠")
    assert [m for m in meld.melds_from(plate, level=2) if m.kind == meld.PLATE]
    not_plate = C("3♥", "3♦", "3♣", "5♦", "5♣", "5♠")
    assert not [m for m in meld.melds_from(not_plate, level=2)
                if m.kind == meld.PLATE]


def test_straight_rank_ordering_uses_top():
    low = C("A♥", "2♦", "3♥", "4♠", "5♥")
    high = C("6♦", "7♦", "8♦", "9♠", "10♥")
    a = [m for m in meld.melds_from(low, level=9) if m.kind == meld.STRAIGHT][0]
    b = [m for m in meld.melds_from(high, level=9) if m.kind == meld.STRAIGHT][0]
    assert meld.beats(b, a)


def test_ace_low_pair_run():
    """三连对也允许 A 当小牌：A-2-3。"""
    hand = C("A♥", "A♦", "2♥", "2♦", "3♠", "3♣")
    assert [m for m in meld.melds_from(hand, level=9) if m.kind == meld.PAIR_RUN]
