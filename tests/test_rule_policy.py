"""规则式对手（`guandan/rl/rule_policy.py`）—— 每条规则一条测试，**都能脱离牌局跑**。

为什么要有这个对手：现在的固定对手是「能压就压最小」，它**会压自己队友**、
**从不主动炸**、**不算剩牌**。这几条测试就是钉「它现在不会那样了」。

规则出处见模块 docstring（主体移植自 `github.com/QinlinChen/guandan-ai`，
用户 2026-09-28 补了「按剩的张数估一手走完的风险」那条）。
"""
import pytest

from guandan.sim import meld
from guandan.sim.env import Observation
from guandan.rl import rule_policy as rp


def C(*names):
    """牌面名 -> 牌 ID 的**列表**（`meld.cid_from_name` 的 'S6'/'ST' 写法）。

    不用 `tests/test_meld_basic.C` —— 那个吃的是 `cards.decode` 的**显示格式**
    （点数在前），两套格式不一样（`meld.py` 里专门写了这条）。
    """
    return [meld.cid_from_name(n) for n in names]


def _obs(seat=0, hand=(), played=None, left=None, table=(), table_kind=0,
         table_rank=-1, level=8):
    """拼一个 `Observation`。`played` 给 `{座位: [牌]}`（列表，见 `C` 的约定）。"""
    pl = {s: [] for s in range(4)}
    pl.update(played or {})
    return Observation(
        seat=seat, hand=frozenset(hand),
        played=tuple(frozenset(pl[s]) for s in range(4)),
        left=tuple(left or (27, 27, 27, 27)),
        table=tuple(table), table_kind=table_kind, table_rank=table_rank,
        passed=(False, False, False, False), turn=seat, level=level)


def _acts(*groups, level=8, with_pass=False):
    """候选：`_acts([牌组1], [牌组2], with_pass=True)`。"""
    out = [meld.as_meld(g, level) for g in groups]
    assert all(m is not None for m in out), "测试里的牌组本身得是合法牌型"
    return (out + [None]) if with_pass else out


def _idx(acts, group):
    want = set(group)
    for i, m in enumerate(acts):
        if m is not None and set(m.cards) == want:
            return i
    raise AssertionError("acts 里没有这一手")


# ---------------------------------------------------------------- 公开信息

def test_table_owner_is_derived_from_public_info():
    """「台面是谁出的」没有单独字段，从「他出过的牌**包含**台面」推 —— 纯公开信息。"""
    t = C("S6", "H6", "D6", "S4")
    assert rp.table_owner(_obs(played={1: t}, table=t)) == 1
    assert rp.table_owner(_obs(played={2: t}, table=t)) == 2
    assert rp.table_owner(_obs(table=())) is None            # 领出


def test_unseen_pool_is_108_minus_my_hand_minus_everything_played():
    mine = C("S8", "H8")
    o = _obs(hand=mine, played={0: C("S3"), 3: C("D4", "H4")})
    pool = rp.unseen_pool(o)
    assert sum(pool.values()) == 108 - 2 - 3
    assert pool.get(8) == 8 - 2          # 两张 8 在我手上
    assert pool.get(4) == 8 - 2          # 两张 4 出过了
    assert pool.get(3) == 8 - 1


# ---------------------------------------------------------------- 风险估计

def test_one_card_left_is_certain():
    assert rp.finish_risk(1, {}) == 1.0
    assert rp.finish_risk(0, {}) == 1.0


def test_finish_risk_covers_three_five_and_six_cards():
    """⚠️ **用户补的那条**：威胁不是「剩 1~2 张」——
    3 张（三张）、5 张（三带二）、6 张（三连对）**都可能一手走完**。"""
    rich = {i: 8 for i in range(1, 14)}
    for n in (1, 2, 3, 5, 6):
        assert rp.finish_risk(n, rich) >= 0.5, f"剩 {n} 张该算危险"
    assert rp.finish_risk(20, rich) < 0.5, "剩 20 张不可能一手走完"
    thin = {5: 1, 7: 1}                  # 凑不出对子/三张
    assert rp.finish_risk(2, thin) < 0.5
    assert rp.finish_risk(3, thin) < 0.5


def test_five_cards_need_a_triple_pair_or_a_run():
    assert rp.finish_risk(5, {3: 8, 9: 8}) >= 0.5              # 三带二凑得出
    assert rp.finish_risk(5, {2: 8, 3: 8, 4: 8, 5: 8, 6: 8}) >= 0.5   # 顺子凑得出
    assert rp.finish_risk(5, {2: 1, 9: 1}) < 0.5


def test_six_cards_can_be_a_pair_run_or_a_plate():
    assert rp.finish_risk(6, {3: 8, 4: 8, 5: 8}) >= 0.5        # 三连对
    assert rp.finish_risk(6, {3: 8, 4: 8}) >= 0.5              # 钢板
    assert rp.finish_risk(6, {2: 1, 3: 1}) < 0.5


# ---------------------------------------------------------------- 手数估算

def test_partition_pairs_a_triple_with_a_pair_into_one_hand():
    hands = rp.hand_partition(C("S6", "H6", "D6", "S4", "D4"), 8)
    assert len(hands) == 1 and hands[0].kind == meld.TRIPLE_PAIR


def test_partition_counts_four_of_a_kind_as_a_bomb():
    hands = rp.hand_partition(C("S6", "H6", "D6", "C6"), 8)
    assert len(hands) == 1 and hands[0].is_bomb


# ---------------------------------------------------------------- 跟牌

def _follow_obs(opp_left, table, table_kind, table_rank, *, owner=2, hand=()):
    """台面是 `owner` 出的；`opp_left` 是**对手**剩的张数（两家一样，简化）。"""
    return _obs(seat=0, hand=hand, table=table, table_kind=table_kind,
                table_rank=table_rank, played={owner: list(table)},
                left=(len(hand), opp_left, 20, opp_left))


def test_does_not_beat_the_ally_when_they_are_winning_with_a_bomb():
    """队友赢着这一手、且他那手是**炸** ⇒ 过。（贪心会压 —— 它不看这是谁的牌。）"""
    bomb = C("SK", "HK", "DK", "CK")
    o = _follow_obs(20, bomb, meld.BOMB, 12, owner=2)
    acts = _acts(C("S2", "H2", "D2", "C2"), with_pass=True)     # 我有更大的炸
    assert rp.ally_of(0) == 2 and rp.table_owner(o) == 2
    assert rp.rule_choose(o, acts) == acts.index(None), "队友的炸，不该压"


def test_does_not_beat_the_ally_when_their_hand_is_already_big():
    """队友压到主点 ≥ K 了 —— 也别添乱（源规则里那条 `rank ≥ K`）。"""
    t = C("SK", "HK", "DK", "S4", "D4")
    o = _follow_obs(20, t, meld.TRIPLE_PAIR, 12, owner=2)
    acts = _acts(C("SA", "HA", "DA", "S5", "D5"), with_pass=True)
    assert rp.rule_choose(o, acts) == acts.index(None)


def test_keeps_the_bomb_when_a_plain_card_can_beat_it():
    """不危险时：有普通牌能压就压普通牌，**不动炸**（留火力）。"""
    t = C("S6", "H6", "D6", "S4", "D4")
    o = _follow_obs(20, t, meld.TRIPLE_PAIR, 5, owner=1)
    acts = _acts(C("S7", "H7", "D7", "S5", "D5"), C("S3", "H3", "D3", "C3"),
                 with_pass=True)
    assert rp.rule_choose(o, acts) == _idx(acts, C("S7", "H7", "D7", "S5", "D5"))


def test_bombs_when_an_opponent_is_one_hand_from_finishing():
    """**用户那条规则的正面**：对手剩 5 张（池子里凑得出三带二）⇒ 必须拦 ⇒ 动炸。"""
    t = C("S6", "H6", "D6", "S4", "D4")
    acts = _acts(C("S3", "H3", "D3", "C3"), with_pass=True)
    o = _follow_obs(5, t, meld.TRIPLE_PAIR, 5, owner=1)
    assert rp.finish_risk(5, rp.unseen_pool(o)) >= rp.DANGER
    assert rp.rule_choose(o, acts) == _idx(acts, C("S3", "H3", "D3", "C3"))


def test_passes_the_same_table_when_no_opponent_is_close_to_finishing():
    """**反面**：同样的台面、同样的炸，但对手还剩 20 张 ⇒ 留着炸，过。"""
    t = C("S6", "H6", "D6", "S4", "D4")
    acts = _acts(C("S3", "H3", "D3", "C3"), with_pass=True)
    o = _follow_obs(20, t, meld.TRIPLE_PAIR, 5, owner=1)
    assert rp.finish_risk(20, rp.unseen_pool(o)) < rp.DANGER
    assert rp.rule_choose(o, acts) == acts.index(None)


# ---------------------------------------------------------------- 领出

def test_leads_a_nonfire_hand_first_when_only_two_or_three_hands_left():
    """剩 2~3 手、既有火力又有非火力 ⇒ 先走非火力那手（**留炸**）。"""
    hand = C("S6", "H6", "D6", "C6", "S9", "H9")      # 炸 + 一对 = 2 手
    o = _obs(hand=hand)
    acts = [m for m in meld.melds_from(list(hand), 8)]
    got = rp.rule_choose(o, acts)
    assert not rp.is_fire(acts[got]), "该先走非火力那手"
    assert acts[got].kind == meld.PAIR


def test_feeds_the_ally_a_single_when_the_ally_has_one_card_left():
    """队友剩 1 张 ⇒ 领出**最小单张**喂他。"""
    hand = C("S3", "S9", "H9", "D9", "C9")
    o = _obs(hand=hand, left=(len(hand), 20, 1, 20))     # 队友(座位2)剩 1 张
    acts = meld.melds_from(list(hand), 8)
    got = rp.rule_choose(o, acts)
    assert acts[got].kind == meld.SINGLE and set(acts[got].cards) == set(C("S3"))


def test_denies_an_opponent_by_leading_a_different_size():
    """卡对手：领出的**张数不等于他剩的张数** —— 他一手就走不完了。

    跟牌必须同牌型同张数，所以这是「不让他走」最直接的一手。
    """
    # ⚠️ 手上必须**有一步走不完**的东西 —— 否则「能一把走完」那条先赢，
    #    测试就测不到「卡对手」这条了（第一版就是这么写错的）
    hand = C("S3", "S4", "S5", "S6", "S7", "S9")
    o = _obs(hand=hand, left=(len(hand), 5, 20, 20))    # 对手(座位1)剩 5 张
    pool = rp.unseen_pool(o)
    assert rp.finish_risk(5, pool) >= rp.DANGER
    acts = meld.melds_from(list(hand), 8)
    got = rp.rule_choose(o, acts)
    assert acts[got].size != 5, "不该领出正好 5 张的牌型"


def test_passes_even_when_the_ally_played_something_small():
    """**队友赢着这一手、只出了一张小牌 —— 也让**（贪心会压，人类不会）。

    这条钉的是**偏离源规则**的那次改动，依据见 `rule_policy._follow` 里的实测表：
    三种「让队友」的写法胜率都在噪声内 ⇒ 取最像人的那个（压队友 100% → 1.2%）。
    """
    t = C("S6", "H6", "D6", "S4", "D4")            # 队友出的三带二，主点很小的 5
    o = _follow_obs(20, t, meld.TRIPLE_PAIR, 5, owner=2)
    acts = _acts(C("S7", "H7", "D7", "S5", "D5"), with_pass=True)
    assert rp.rule_choose(o, acts) == acts.index(None), "队友的牌，再小也不压"


def test_takes_over_from_the_ally_only_to_go_out():
    """例外：压上去**正好走完** —— 那就抢（那是净赚一手）。"""
    t = C("S6", "H6", "D6", "S4", "D4")
    o = _follow_obs(20, t, meld.TRIPLE_PAIR, 5, owner=2,
                    hand=C("S7", "H7", "D7", "S5", "D5"))     # 整手正好能压
    acts = _acts(C("S7", "H7", "D7", "S5", "D5"), with_pass=True)
    assert rp.rule_choose(o, acts) == _idx(acts, C("S7", "H7", "D7", "S5", "D5"))
