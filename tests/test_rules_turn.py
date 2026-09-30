import pytest

from guandan.capture import cards
from guandan.sim import meld, rules

A = meld.cid_from_name


def _hand(level=2, first=0, **seat_cards):
    """用手写牌面搭局面。`rules.Hand(hands=[...], level=..., turn=...)` 是唯一入口。"""
    hands = [set() for _ in range(4)]
    for name, ids in seat_cards.items():
        hands[int(name[1:])] = set(ids)
    return rules.Hand(hands=hands, level=level, turn=first)


def test_next_seat_is_the_previous_number_not_the_next_one():
    """**出牌顺序是 0 → 3 → 2 → 1**（55 局实测，见本任务开头）。

    这一条单独钉住，因为它是整个轮转的地基：写成 `(s+1)%4` 的话
    55 局真实对局**一局都走不通**。"""
    assert rules.NEXT == (3, 0, 1, 2)
    for s in rules.SEATS:
        assert rules.NEXT[rules.NEXT[rules.NEXT[rules.NEXT[s]]]] == s   # 四步回环


def test_leader_must_play_and_cannot_pass():
    h = _hand(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    assert None not in h.actions(0)                 # 领出不含「过」
    with pytest.raises(rules.IllegalPlay):
        h.pass_turn(0)


def test_follower_may_pass_even_when_able_to_beat():
    """**「能压也可以过」** —— HANDOFF 的开工前第 2 条只写了「压不过 = 只能过」，漏了这一半。
    掼蛋里跟牌的人永远可以过，哪怕手上压得过。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S5")], s3=[A("S4"), A("H4")])
    h.play(0, meld.as_meld([A("S3")], 2))
    assert h.turn == 3                              # 下家是 3，不是 1
    acts = h.actions(3)
    assert None in acts                             # 「过」永远在
    assert any(m is not None and set(m.cards) == {A("S4")} for m in acts)   # 4♠ 单张压得过 3♠
    # 对子压不过单张（牌型不同），所以它不该在候选里
    assert not any(m is not None and set(m.cards) == {A("S4"), A("H4")} for m in acts)


def test_when_nothing_beats_the_table_actions_is_exactly_pass():
    """`meld.legal_moves` 压不过时返回 `[]` —— env 必须读成「只能过」。
    **这是 HANDOFF「Plan 2 开工前第 2 条」要钉住的那件事。**

    注意别拿 2 当小牌：**打 2 的时候 2 是级牌，比 A 还大**。所以这里用 4♠（非级牌）
    去面对 A♠，才是真的压不过。"""
    h = _hand(s0=[A("SA")], s1=[A("S6")], s2=[A("S5")], s3=[A("S4")])
    h.play(0, meld.as_meld([A("SA")], 2))
    assert h.turn == 3
    assert h.actions(3) == [None]


def test_play_rejects_illegal_shape_and_missing_cards():
    h = _hand(s0=[A("S3"), A("S4")], s1=[A("S6")], s2=[A("S5")], s3=[A("H7")])
    with pytest.raises(rules.IllegalPlay):
        h.play(0, meld.as_meld([A("H9")], 2))       # 手上没有这张
    with pytest.raises(rules.IllegalPlay):
        h.play(0, meld.Meld(kind=meld.STRAIGHT, size=5, rank=3,
                            cards=(A("S3"), A("S4"))))   # 牌面根本不是顺子


def test_wrong_seat_cannot_act():
    h = _hand(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    with pytest.raises(rules.IllegalPlay):
        h.play(3, meld.as_meld([A("S6")], 2))


def test_a_new_play_reopens_the_round_for_everyone():
    """**每出一手，`passed` 清空** —— 这一手等于重新开一轮，三家重新有机会。

    实测依据：不清空的话，55 局里只有 4 局能走通（38 局会在中途被判成「压不过桌面」）。

    **每家给 2 张牌**：1 张牌的话座位 0、2 一出完就是双上、这一手直接结束，
    `passed` 还没被用上就没了。
    """
    h = _hand(s0=[A("S3"), A("S8")], s1=[A("S6"), A("S7")],
              s2=[A("S5"), A("S9")], s3=[A("S4"), A("ST")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.pass_turn(3)
    assert h.passed == {3}
    h.play(2, meld.as_meld([A("S5")], 2))           # 座位 2 压过 —— 新一轮
    assert h.passed == set(), "出了一手之后 passed 必须清空"
    assert 3 not in h.passed, "座位 3 之前要不起过，新一轮里必须重新有机会"
    h.pass_turn(1); h.pass_turn(0)                  # 轮转 2 → 1 → 0 → 3
    assert h.turn == 3, "轮到座位 3 —— 他先前过过，现在又能出手了"


# ---- 下面四条对应 Review Focus 的第 2、3、4 条
#      （spec 隐含要求，但没有哪一处的测试天然覆盖到）

def test_two_wilds_complete_a_triple_and_never_duplicate_a_candidate():
    """**Review Focus #2：手上同时有 2 张逢人配。**

    两件事必须成立：① 候选里**没有完全重复**的着法（重复 = 同一个着法占两格，
    它的 Q 值被两个样本分别训练）；② 两张逢人配确实能凑出「三个 9」
    （漏了 = 模型永远看不到某个合法着法）。"""
    wild = [A("H2"), A("H2", deck=2)]          # 打 2 时红桃 2 就是逢人配
    h = _hand(s0=wild + [A("S9"), A("S4"), A("S5")],
              s1=[A("S6")], s2=[A("S7")], s3=[A("S8")])
    acts = h.actions(0)
    keys = [(m.kind, tuple(sorted(m.cards))) for m in acts]
    assert len(keys) == len(set(keys)), "候选里出现了完全重复的着法"
    triples = [m for m in acts
               if m.kind == meld.TRIPLE and m.rank == meld.point_value(9, 2)]
    assert triples, "两张逢人配 + 一张 9♠ 应该能凑出「三个 9」"
    assert triples[0].wild_used == 2


def test_partner_inherits_the_lead_when_the_owner_goes_out():
    """**Review Focus #3：只剩 1 张牌的座位出完后，接风给队友。**

    判据是「**主人已经出完**」，不是服务器给的 `NextTurnSeatID` ——
    后者会跳过出完的座位，实测按它判会误清 34 手、漏清 27 手
    （`tools/decision_points.py` 的 docstring 记了这次实测）。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S5")], s3=[A("S2")])
    h.play(0, meld.as_meld([A("S3")], 2))       # 座位 0 出完
    assert h.order == [0]
    assert h.turn == 3                          # 下家是 3
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)
    assert h.table is None, "三家都要不起 -> 必须清桌"
    assert h.turn == 2, "**接风**：主人出完了，领出权给他的队友（座位 2）"


def test_double_win_ends_the_hand_with_two_seats_still_holding_cards():
    """**Review Focus #4：双上立即终局（实测 25/25 局：只出完 2 家）。**

    若把局终条件写成「3 家出完」，这 25 局会被多打几手、名次也可能错。"""
    h = _hand(s0=[A("S3")], s1=[A("S9")], s2=[A("S5")], s3=[A("ST")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)
    assert h.turn == 2
    h.play(2, meld.as_meld([A("S5")], 2))       # 队友也出完 -> 双上
    assert h.is_over(), "同一队包了前两名必须**立即**终局"
    assert h.order == [0, 2]                    # 名次怎么算在 Task 2 里验


def test_three_seats_out_ends_the_hand_when_the_winners_are_not_first_and_second():
    """**Review Focus #4 的另一半**：1、3 名 / 1、4 名时要打到 **3 家出完**（实测 30/30 局）。

    出完顺序按真实轮转（0 → 3 → 2），所以这里 0 是 1 名、3 是 2 名、2 是 3 名
    —— 1、3 名归座位 0 与 2，是「1、3 形态」。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S2")], s3=[A("SA")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.play(3, meld.as_meld([A("SA")], 2))       # 轮转里座位 3 接着来
    assert not h.is_over(), "只出完 2 家、且分属两队 -> 还没完"
    h.play(2, meld.as_meld([A("S2")], 2))       # 打 2 时 2 是级牌，压得过 A
    assert h.is_over(), "出完 3 家必须终局"
    assert h.order == [0, 3, 2]                 # 名次与升级点在 Task 2 里验


def test_the_wild_can_be_placed_in_the_triple_even_when_naturals_suffice():
    """**手里多一张牌，不该让一手合法的牌消失。**

    打 6（逢人配 = ♥6）时 `A♠A♥ + 2♠2♥ + ♥6` 这 5 张，游戏读成
    「三个 A 带一对 2」（rank 13）—— 用逢人配顶第三张 A。它能压过 rank<13 的三带二。

    但手里的**第三张 A** 会让这条读法凭空消失：枚举三带二时自然牌是**贪婪**占位的
    （`g[t][:3]` 把三张 A 全用掉），于是「A,A + 逢人配」这条永远产不出来，
    `actions()` 面对 rank=5 的三带二**一个候选都不给**。手里的第三张 2 则没这个问题。

    实测（2026-09-25，最终评审发现）：第三张 A -> 候选为空；第三张 2 -> 候选 rank 13。
    """
    target = {A("SA"), A("HA"), A("S2"), A("H2"), A("H6")}
    for extra, name in ((A("CA"), "第三张 A"), (A("C2"), "第三张 2")):
        h = _hand(level=6, s0=target | {extra},
                  s1=[A("S9")], s2=[A("ST")], s3=[A("SJ")])
        h.table = meld.Meld(meld.TRIPLE_PAIR, 5, 5,
                            (A("C3"), A("D3"), A("C3", deck=2), A("S4"), A("H4")))
        h.table_seat = 3
        ranks = [m.rank for m in h.actions(0)
                 if m is not None and set(m.cards) == target]
        assert ranks, f"（{name}）这 5 张能压过 rank=5 的三带二，候选里却一个都没有"
        assert max(ranks) == 13, f"（{name}）最强读法应当是 rank 13，实际 {sorted(ranks)}"


def test_two_wilds_plus_two_ranks_do_not_duplicate_a_candidate():
    """**Review Focus #2 的另一半：候选不许有重复。**

    光「两张逢人配能凑出三个 9」还不够 —— 那半边在原来的手牌上**测不出重复**
    （删掉 `melds_from` 的去重它照样绿）。真正会产生重复的是这组牌：
    两张逢人配 + **两个各 2 张的点数**（`9♠9♥` 与 `4♠4♥`）。
    实测去重前 `_melds_wild` 为同一组 5 张牌产出 **4 条**三带二
    （「三个 9 带一对 4」与「三个 4 带一对 9」各自成组）——
    没有去重的话，一个物理上只能出一次的牌会占掉动作空间的 4 格。
    """
    h = _hand(s0=[A("H2"), A("H2", deck=2), A("S9"), A("H9"), A("S4"), A("H4")],
              s1=[A("S6")], s2=[A("S7")], s3=[A("S8")])
    keys = [(m.kind, tuple(sorted(m.cards))) for m in h.actions(0)]
    assert len(keys) == len(set(keys)), "候选里出现了完全重复的着法"
