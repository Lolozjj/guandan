"""A4 的**风格尺子**：默认必须逐位不变，三个旋钮必须真的咬得住。

⚠️ 默认变了就是灾难：`rule_policy()` 是**尺子**，它一动，历史上所有 `vs 规则式`
的读数全部作废。所以这里第一条就是「默认 == NORMAL」。
"""
import pytest

from guandan.rl import rule_policy as rp
from guandan.sim import meld
from tests.test_rule_policy import _acts, _idx, _obs

C = lambda *n: [meld.cid_from_name(x) for x in n]      # noqa: E731


def test_default_style_is_the_incumbent():
    """默认值必须就是现役的那些数（动了它 = 历史读数作废）。"""
    assert rp.NORMAL.danger == rp.DANGER
    assert (rp.NORMAL.bomb_rank, rp.NORMAL.hold_fire) == (10, True)
    assert set(rp.STYLES) == {"normal", "bomb", "hold"}


def test_rule_choose_defaults_to_normal_on_real_states():
    """真实决策点上 `rule_choose(obs, acts)` 与显式 `NORMAL` **逐位相同**。"""
    hand = C("S5", "H5", "D5", "S6", "H6", "S7", "S8", "S9", "H9", "SK")
    obs = _obs(hand=hand, level=8)
    acts = _acts(C("S6", "S7", "S8", "S9", "ST"), C("S5", "H5", "D5"),
                 C("S9", "H9"), C("SK"))
    assert rp.rule_choose(obs, acts) == rp.rule_choose(obs, acts, rp.NORMAL)


def _pass_idx(acts):
    """「过」那个下标 —— `_idx` 只认 Meld，`None` 得单独取。"""
    return acts.index(None)


def test_bomb_rank_knob_bites():
    """`bomb_rank` 低 => 对着**小台面**也动炸；默认只在主点 >10 的台面动炸。

    构造：台面是三个 6（主点 6），我手上**只有**一个 4 张的炸能压，
    且对手不危险（各家剩 10 张）⇒ 走的是 `_follow` 结尾那条。
    """
    table = C("S6", "H6", "D6")
    hand = C("S4", "H4", "D4", "C4", "S3")
    obs = _obs(hand=hand, level=8, table=table, table_kind=meld.TRIPLE,
               table_rank=meld.point_value(6, 8), left=(10, 10, 10, 10))
    acts = _acts(C("S4", "H4", "D4", "C4"), with_pass=True)     # 只有炸 + 过
    assert rp.rule_choose(obs, acts, rp.NORMAL) == _pass_idx(acts)             # 默认：过
    assert rp.rule_choose(obs, acts, rp.BOMB) == _idx(acts, C("S4", "H4", "D4", "C4"))


def test_danger_knob_bites():
    """`danger` 高 => 对手快走完也不拦（龟）；默认会拦。

    构造的关键：**只有炸能压**。`_follow` 里"必须拦"那一支会动炸，而结尾那支
    因为台面主点不高不会动炸 ⇒ 两边才分得开。
    对手剩 4 张、未见牌池里某个点数还有 ≥4 张 ⇒ `finish_risk` = 0.6：
    默认门槛 0.5 拦、龟的门槛 0.8 不拦。
    """
    table = C("S6", "H6", "D6")
    hand = C("S4", "H4", "D4", "C4", "S3")
    obs = _obs(hand=hand, level=8, table=table, table_kind=meld.TRIPLE,
               table_rank=meld.point_value(6, 8), left=(2, 4, 20, 20))
    acts = _acts(C("S4", "H4", "D4", "C4"), with_pass=True)
    from guandan.sim.features import unseen_pool
    assert 0.5 <= rp.finish_risk(4, unseen_pool(obs)) < 0.8, "这一例的风险必须夹在 0.5~0.8"
    assert rp.rule_choose(obs, acts, rp.NORMAL) == _idx(acts, C("S4", "H4", "D4", "C4"))
    assert rp.rule_choose(obs, acts, rp.HOLD) == _pass_idx(acts)


def test_policy_factory_accepts_a_name():
    """训练/尺子按**名字**取风格（别在别处再写一份字典）。"""
    p = rp.rule_policy("bomb")
    hand = C("S5", "H5", "D5", "S6", "H6", "S7", "S8", "S9", "H9", "SK")
    obs = _obs(hand=hand, level=8)
    acts = _acts(C("S6", "S7", "S8", "S9", "ST"), C("S5", "H5", "D5"))
    assert 0 <= p(obs, acts) < len(acts)
    with pytest.raises(KeyError):
        rp.rule_policy("没有这个风格")
