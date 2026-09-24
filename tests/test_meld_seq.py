from net.sim import meld
from tests.test_meld_basic import C


def test_nat_values_ace_is_both_ends():
    assert set(meld.nat_values(1)) == {1, 14}
    assert meld.nat_values(10) == (10,)
    assert meld.nat_values(14) == ()       # 小王不进序列
    assert meld.nat_values(15) == ()       # 大王不进序列


def test_ace_low_straight_is_legal():
    """A2345 是合法顺子（数据里有 2 手），这是 live/rules.py 的 bug #2。"""
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
