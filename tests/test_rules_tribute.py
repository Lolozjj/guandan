import pytest

from net import cards
from net.sim import meld, rules

A = meld.cid_from_name
BIG1, BIG2 = A("JOKER_B"), A("JOKER_B", deck=2)          # 两张大王


def _h(level=2, **seat_cards):
    hands = [set() for _ in range(4)]
    for name, ids in seat_cards.items():
        hands[int(name[1:])] = set(ids)
    return rules.Hand(hands=hands, level=level, turn=0)


def test_tributers_single_when_the_two_losers_are_on_different_teams():
    # 名次：座位1=1、2=2、0=3、3=4 -> 第 3(座位0)、4(座位3) 名**不同队** -> 单贡
    assert rules.tributers([3, 1, 2, 4]) == [3]


def test_tributers_double_when_both_losers_are_one_team():
    # 名次：座位0=1、2=2、3=3、1=4 -> 第 3、4 名都是 1/3 队 -> 双贡
    assert sorted(rules.tributers([1, 4, 2, 3])) == [1, 3]


def test_single_tribute_gives_the_biggest_card_and_returns_the_smallest_ten_or_under():
    # 名次：座位2=1、1=2、0=3、3=4 -> 第 3、4 名不同队 -> 单贡，座位 3 贡给座位 2
    h = _h(s0=[A("S9")], s1=[A("S8")], s2=[A("S3"), A("SK"), A("H5")], s3=[A("SA"), A("D4")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert t.kind == "single"
    assert t.gave == {3: A("SA")}                  # 贡手上最大的牌
    assert t.returned == {2: A("S3")}              # 还 2..10 里最小的
    assert A("SA") in h.hands[2] and A("S3") in h.hands[3]
    assert t.leader == 2                           # 头游先出（基线，见实现里的注释）


def test_ace_is_not_a_valid_return_even_though_its_rank_index_is_1():
    """**「≤10」是点数 2..10，不是「索引 ≤10」。**

    牌 ID 体系里 A=1，所以 `idx <= 10` 会把 **A 也算成「≤10」** ——
    而 A 恰恰是除王与级牌之外最大的牌，等于白送。

    ⚠️ 这里**必须用 level=5**：打 2 的时候 2♦ 自己是级牌，`_smallest_returnable`
    会把它排到最后，测不出 A 的问题。打 5 时 2♦ 是普通小牌，才是干净的对照。
    """
    h = _h(level=5, s0=[A("S9")], s1=[A("S8")],
           s2=[A("SA"), A("SK"), A("HJ"), A("D2")], s3=[A("HA"), A("D4")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert cards.parts(t.returned[2])[0] == 2      # 只有 2♦ 合格
    assert cards.parts(t.returned[2])[0] != 1      # **A 绝不能被当成「≤10」还出去**


def test_resist_when_the_losing_team_holds_two_big_jokers():
    """抗贡判据：**输的那一队**手上合计 ≥2 张大王（不是只看进贡的那一个人）。

    证据：日志里 `TributeSectionStatrt seatId:1|大王| seatId:3|大王|` 出现 9 次，
    每次都是「同队两个座位各持一张大王」，**单贡时也只报这两个座位** ——
    说明判的是**队**而不是进贡的那个人。"""
    h = _h(s0=[A("S5")], s1=[BIG1, A("S3")], s2=[A("S6")], s3=[BIG2, A("S4")])
    t = rules.apply_tribute(h, prev_ranks=[1, 3, 2, 4])      # 1、3 是 3、4 名，同队
    assert t.kind == "resist" and t.gave == {} and t.returned == {}
    assert h.hands[1] == {BIG1, A("S3")} and h.hands[3] == {BIG2, A("S4")}   # 牌没动
    assert t.leader == 0


def test_level_card_outranks_ace_when_deciding_the_biggest_card():
    """**打 2 的时候 2 是级牌，比 A 大。** 自己比较点数会在这里栽跟头 ——
    所以取值必须走 `meld.point_value`。"""
    h = _h(s0=[A("S9")], s1=[A("S8")], s2=[A("S3")], s3=[A("SA"), A("D2")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert t.gave == {3: A("D2")}


def test_no_tribute_when_there_is_no_previous_hand():
    h = _h(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    t = rules.apply_tribute(h, prev_ranks=None)
    assert t.kind == "none" and t.gave == {} and t.returned == {}
