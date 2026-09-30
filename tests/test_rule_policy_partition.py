"""`hand_partition` 要认**序列类**牌型（顺子 / 三连对 / 钢板）。

它是 [源 2]「还剩 2~3 手就先走非火力那手（留炸）」的唯一输入。
原来的口径（移植自那份开源 AI 的 `utils.partition`）**只认对子/三张/炸**，
于是手里有顺子时会把 5 张算成 5 张单牌 ⇒ **系统性高估手数** ⇒ [源 2] 几乎不触发。

⚠️ 它仍然是**粗估**（贪心、不认逢人配），只用来跟 2/3 比大小，
**不是**「这手牌最少几手走完」的精确答案。
"""
from guandan.sim import meld
from tests.test_rule_policy import C
from guandan.rl import rule_policy as rp


def _kinds(hands):
    return sorted(h.kind for h in hands)


def test_a_five_card_run_is_one_hand():
    hands = rp.hand_partition(C("S3", "H4", "D5", "C6", "S7"), 8)
    assert len(hands) == 1 and hands[0].kind == meld.STRAIGHT


def test_a_three_pair_run_is_one_hand():
    """三连对：三个连续点数各两张。"""
    hands = rp.hand_partition(C("S3", "H3", "S4", "H4", "S5", "H5"), 8)
    assert len(hands) == 1 and hands[0].kind == meld.PAIR_RUN


def test_a_plate_is_one_hand():
    """钢板：两个连续点数各三张。"""
    hands = rp.hand_partition(C("S3", "H3", "D3", "S4", "H4", "D4"), 8)
    assert len(hands) == 1 and hands[0].kind == meld.PLATE


def test_a_run_plus_a_pair_is_two_hands():
    hands = rp.hand_partition(C("S3", "H4", "D5", "C6", "S7", "S9", "H9"), 8)
    assert len(hands) == 2
    assert meld.STRAIGHT in _kinds(hands) and meld.PAIR in _kinds(hands)


def test_the_old_behaviour_is_untouched_for_hands_without_runs():
    """没有序列可抽时，口径**一字不变**（三张配一对、对子、单张、≥4 算炸）。"""
    assert [h.kind for h in rp.hand_partition(C("S6", "H6", "D6", "S4", "D4"), 8)] \
        == [meld.TRIPLE_PAIR]
    assert [h.kind for h in rp.hand_partition(C("S6", "H6", "D6", "C6"), 8)] \
        == [meld.BOMB]
    assert len(rp.hand_partition(C("S6", "H7", "D8", "C9"), 8)) == 4        # 4 张单牌


def test_a_run_is_not_double_counted_against_a_pair():
    """抽走顺子之后，剩下的牌**不许**被再用一次（同一个牌 ID 只能在一手里）。"""
    hand = C("S3", "H4", "D5", "C6", "S7")
    hands = rp.hand_partition(hand, 8)
    used = [c for h in hands for c in h.cards]
    assert sorted(used) == sorted(hand), "牌被重复使用或漏掉了"
