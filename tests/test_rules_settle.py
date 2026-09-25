import pytest

from net import cards
from net.sim import meld, rules

A = meld.cid_from_name


def _finish(order):
    """造一个已经打完的局面：order 是出完顺序，剩下的人按剩牌数排。"""
    h = rules.Hand(hands=[set() for _ in range(4)], level=2, turn=0)
    h.order = list(order)
    h.over = True
    return h


def test_ranks_follow_finish_order_then_leftover_cards():
    # 座位 2、0 出完（双上），1 剩 1 张、3 剩 5 张 -> 1 是第 3 名
    h = _finish([2, 0])
    h.hands[1] = {A("S3")}
    h.hands[3] = {A("S4"), A("S5"), A("S6"), A("S7"), A("S8")}
    assert h.ranks() == [2, 3, 1, 4]


def test_ranks_tiebreak_is_stable_by_seat():
    """剩牌同数时按座位小者排前 —— 只是为了让结果可复现。
    **这样定出来的 3/4 名对 reward 无影响**（双上时败方谁 3 谁 4 不改变形态），
    实测 25 局里有 2 局与「剩牌少者排前」不符，见 spec §13.5。"""
    h = _finish([2, 0])
    h.hands[1] = {A("S3"), A("S4")}
    h.hands[3] = {A("S5"), A("S6")}
    assert h.ranks() == [2, 3, 1, 4]


def test_ranks_after_a_real_double_win():
    """走完一整段真实轮转再算名次 —— 上面 `_finish` 那几条是手搭局面，
    这一条验的是「rules 自己走出来的局，名次算得对」。"""
    h = rules.Hand(hands=[{A("S3")}, {A("S9")}, {A("S5")}, {A("ST")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))       # 座位 0 出完
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)   # 三家要不起 -> 接风给队友
    assert h.turn == 2
    h.play(2, meld.as_meld([A("S5")], 2))       # 队友也出完 -> 双上，立即终局
    assert h.is_over() and h.order == [0, 2]
    assert h.ranks() == [1, 3, 2, 4]
    assert rules.points(h.ranks()) == 3          # 座位 0、2 包了前两名 -> +3


def test_ranks_after_a_real_three_seat_finish():
    """出完 3 家、赢家是 1、3 名 -> 升级点 2（实测 30/30 局是这种 3 出完的形态）。"""
    h = rules.Hand(hands=[{A("S3")}, {A("S6")}, {A("S2")}, {A("SA")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))
    h.play(3, meld.as_meld([A("SA")], 2))       # 真实轮转里座位 3 接着来
    assert not h.is_over()
    h.play(2, meld.as_meld([A("S2")], 2))       # 打 2 时 2 是级牌，压得过 A
    assert h.is_over() and h.order == [0, 3, 2]
    assert h.ranks() == [1, 4, 3, 2]
    assert rules.points(h.ranks()) == 2


def test_ranks_requires_the_hand_to_be_over():
    h = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    with pytest.raises(rules.IllegalPlay):
        h.ranks()


@pytest.mark.parametrize("ranks,expect", [
    ([1, 3, 2, 4], 3),      # 座位 0、2 是 1、2 名 = 双上 -> +3
    ([1, 4, 2, 3], 3),      # 换一种写法，还是双上
    ([1, 2, 3, 4], 2),      # 座位 0、2 是 1、3 名 -> +2
    ([1, 2, 4, 3], 1),      # 座位 0、2 是 1、4 名 -> +1
    ([4, 1, 3, 2], 3),      # 座位 1、3 双上（名次 1、2）
    ([2, 1, 3, 4], 1),      # 座位 1、3 是 1、4 名
])
def test_points(ranks, expect):
    assert rules.winner_team(ranks) in (0, 1)
    assert rules.points(ranks) == expect


def test_reward_is_zero_sum_from_the_seat_perspective():
    """四个座位共享一套策略 -> 奖励必须从**出牌人所在队**的视角看，且零和。
    （斗地主式 DMC 的前提；视角搞错会让同一局给四个座位同一个目标。）"""
    ranks = [1, 3, 2, 4]                       # 0、2 队赢，双上
    assert rules.reward(ranks, 0) == 3 and rules.reward(ranks, 2) == 3
    assert rules.reward(ranks, 1) == -3 and rules.reward(ranks, 3) == -3
    assert sum(rules.reward(ranks, s) for s in rules.SEATS) == 0
