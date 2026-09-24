"""live/rules.py 适配层的测试（Task 8）。

这里的断言不是「测新引擎」——引擎的测试在 tests/test_meld_*.py。
这里是**钉住旧实现的 6 个已证实缺陷**：每条在旧实现下红、在适配层下绿。

另外几条是适配层自己的契约（见 task-8 的硬约束「认不出必须明着报错」），
以及一条旧实现本来就做对、不能被这次改动弄丢的（同花顺不能降级成顺子）。
"""
import pytest

from live import rules
from net import cards
from net.sim import meld


# ---------------------------------------------------------- 6 个缺陷
# 说明：旧实现下真会红的只有 4 条（ten / ace-low / joker-pair / 二连对），
# `test_illegal_returns_none` 与 `test_describe_falls_back` 旧实现本来就过 ——
# 它们的作用是**守住适配层别把不合法的判成合法**（回归网，不是缺陷证据）。


def test_ten_is_handled():
    """旧实现遇到 10 会直接崩（bug #1）。"""
    assert rules.classify(["S10", "H10", "D10"], "2") == "三张"


def test_ace_low_straight():
    """旧实现判非法（bug #2）。"""
    assert rules.classify(["SA", "H2", "D3", "C4", "S5"], "9") == "顺子"


def test_two_small_jokers_are_a_pair():
    """旧实现说「两个王不是对子」（bug #3）。"""
    got = rules.classify(["JOKER_SMALL", "JOKER_SMALL"], "9")
    assert got is not None and "对" in got


def test_three_twos_with_two_jokers_is_triple_pair():
    """bug #3 的真实着法那一半：三个 2 + 两张小王是**三带二**（旧实现判 None）。

    王不能凑三张，但**能当三带二里的对子**（net/sim/meld.py 的 `pairs` 不排除王）。
    """
    assert rules.classify(
        ["S2", "D2", "C2", "JOKER_SMALL", "JOKER_SMALL"], "9") == "三带二"


def test_two_pair_run_is_illegal():
    """旧实现把 4 张二连对判合法（bug #4），游戏里 4 张只有炸弹。"""
    assert rules.classify(["S4", "H4", "S5", "H5"], "2") is None


def test_illegal_returns_none():
    assert rules.classify(["S5", "H7", "D9"], "2") is None


def test_describe_falls_back():
    assert rules.describe(["S5", "H7", "D9"], "2") == "不合法"


# ------------------------------------------------- 旧实现做对、别弄丢的
# bug #6 的另一半：逢人配的能力**不能低估**。旧实现在这里其实是错的
# （`wild_cls = "H"+level` 只认 "H2" 这种写法，碰到 "10" 直接崩），
# 但「同花顺优先于顺子」这一条旧实现是对的，换适配层不能把它降级。


def test_wild_makes_four_bomb():
    """真实着法：2♥2♥2♠10♥，打 10，是 4 张炸（不是三张、不是连对）。"""
    assert rules.classify(["S2", "H2", "D2", "H10"], "10") == "4 张炸（含逢人配）"


def test_wild_straight_flush_beats_plain_straight():
    """真实着法：9♣10♣J♣Q♣+♥2（打 2）是同花顺 —— 同一组牌也能解释成顺子，
    必须取更强的那条（同花顺是炸弹，顺子不是，面板和比大小都靠它）。"""
    got = rules.classify(["C9", "C10", "CJ", "CQ", "H2"], "2")
    assert got == "同花顺（含逢人配）"


def test_natural_straight_flush_not_downgraded():
    assert rules.classify(["S5", "S6", "S7", "S8", "S9"], "2") == "同花顺"


def test_second_deck_suffix_is_stripped():
    """live/ 的牌名带 (二副) 后缀，适配层要剥掉再转 ID。"""
    assert rules.classify(["S10", "H10", "D10", "S10(二副)"], "2") == "4 张炸"


# ------------------------------------------------- 认不出必须明着报错
# 硬约束：不许静默返回 None（那会把「认不出」冒充成「不合法」），
# 也不许把 "JOKER_FOO" 这种没见过的名字当小王。


@pytest.mark.parametrize("name", ["SX", "X10", "S", "", "JOKER_MIDDLE", "3S", "10S",
                                  "S1", "S11", "S14", "S37"])
def test_unknown_card_name_raises(name):
    with pytest.raises(ValueError):
        meld.cid_from_name(name)


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        rules.classify(["S5"], "T")


def test_unknown_deck_raises():
    with pytest.raises(ValueError):
        meld.cid_from_name("S3", 3)


def test_bad_name_does_not_silently_return_none():
    """认不出的牌名要抛异常，**不能**被 classify 吞成 None。"""
    with pytest.raises(ValueError):
        rules.classify(["S5", "H7", "ZZ9"], "2")


# ------------------------------------------------- 编码本身
# 适配层是「牌名 <-> ID」的唯一转换点，转错了整个 live 线都错，
# 所以四个花色 x 十三个点数逐张对一遍 net/cards.py 的编码。


@pytest.mark.parametrize("letter,suit", [("S", "♠"), ("H", "♥"), ("C", "♣"), ("D", "♦")])
@pytest.mark.parametrize("rank,idx", [("A", 1)] + [(str(i), i) for i in range(2, 11)]
                          + [("J", 11), ("Q", 12), ("K", 13)])
def test_every_card_name_maps_to_its_id(letter, suit, rank, idx):
    cid = meld.cid_from_name(letter + rank)
    assert cards.parts(cid) == (idx, suit, 1)
    assert meld.cid_from_name(letter + rank, 2) == cid + 256
    assert cards.parts(meld.cid_from_name(letter + rank, 2)) == (idx, suit, 2)


def test_jokers():
    assert cards.parts(meld.cid_from_name("JOKER_SMALL")) == (14, "", 1)
    assert cards.parts(meld.cid_from_name("JOKER_BIG")) == (15, "", 1)
    assert meld.cid_from_name("JOKER_SMALL", 2) == 270
    assert meld.cid_from_name("JOKER_BIG", 2) == 271
