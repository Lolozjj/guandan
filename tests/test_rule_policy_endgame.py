"""残局：**对手快走完的时候，领出该出什么**。

用户 2026-09-29 之后查到的一条**反向**行为：

    我领出、对手只剩 1 张、我手里全是单张「3♠ 4♥ 5♦」
      [源 5] 知道该卡（风险刻度 1.0 ≥ DANGER）
      卡的办法是「领张数 ≠ 1 的牌」—— 但我手里全是单张，卡不了
      → 落到兜底 = 最小的合法牌 = **出 3♠**   ✗ 把牌权送给对手

正确是出 **5♦**（最大那张）：对手只有 1 张，压不过就接不上，牌权还在我手上。
人不会把最小的单张递给只剩一张的对手。

⚠️ 另两条是**护栏**：喂队友优先于卡对手、能卡形状时仍按形状卡。
"""
from net.sim import meld
from tests.test_rule_policy import C, _idx, _obs
from train import rule_policy as rp


def _lead_acts(hand):
    return meld.melds_from(sorted(hand), 8)


def test_leads_the_biggest_single_when_shape_denial_is_impossible():
    """**用户报的那条**：全是单张时，卡不住形状就卡强度 —— 出最大的。"""
    hand = C("S3", "H4", "D5")
    o = _obs(seat=0, hand=hand, left=(3, 1, 20, 13))     # 座位1 只剩 1 张
    acts = _lead_acts(hand)
    assert rp.finish_risk(1, rp.unseen_pool(o)) >= rp.DANGER
    assert rp.rule_choose(o, acts) == _idx(acts, C("D5")), "把最小的单张递给只剩一张的对手"


def test_still_denies_by_shape_when_a_non_single_exists():
    """有对子就领对子（形状卡）—— **不许**退化成出最大的单张。"""
    hand = C("S3", "H4", "D5", "S9", "H9")
    o = _obs(seat=0, hand=hand, left=(5, 1, 20, 13))
    acts = _lead_acts(hand)
    assert rp.rule_choose(o, acts) == _idx(acts, C("S9", "H9"))


def test_feeding_the_ally_beats_denying_the_opponent():
    """队友也只剩 1 张 -> **喂队友优先**（出最小单张让他接上），不是卡对手。"""
    hand = C("S3", "H4", "D5")
    o = _obs(seat=0, hand=hand, left=(3, 13, 1, 20))     # 座位2（队友）剩 1 张
    acts = _lead_acts(hand)
    assert rp.rule_choose(o, acts) == _idx(acts, C("S3"))
