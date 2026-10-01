"""败局解剖（A6）里**会影响结论**的那一步：手牌强度代理。

`hands_left` 是"这一队还要几手能走完"的粗估（越小越强）。它算反了，
A6 的分层表会整个倒过来 —— 而它**跑得通、不报错**，所以必须有牙。
"""
from guandan.rl.rule_policy import hand_partition
from guandan.sim import meld
from tools.defeat_autopsy import hands_left

A = meld.cid_from_name


def test_scattered_cards_need_more_hands_than_a_made_one():
    """一手顺子 vs 五张散牌：散牌的手数必须**更多**（方向不能反）。"""
    straight = [A("S5"), A("S6"), A("S7"), A("S8"), A("S9")]
    scat = [A("S3"), A("H5"), A("C7"), A("D9"), A("SJ")]
    level = 2
    assert len(hand_partition(straight, level)) < len(hand_partition(scat, level))
    assert hands_left([straight], level) < hands_left([scat], level)


def test_empty_hand_is_zero():
    assert hands_left([[]], 2) == 0


def test_hands_left_sums_over_the_team():
    """一队两个人的手数要**相加**（A6 比的是两队之和）。"""
    a = [A("S5"), A("S6"), A("S7"), A("S8"), A("S9")]
    b = [A("S3"), A("H5"), A("C7")]
    assert hands_left([a, b], 2) == (len(hand_partition(a, 2)) + len(hand_partition(b, 2)))
