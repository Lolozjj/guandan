"""动作空间探针（A1）的读数要可复现 —— 尤其那两条**已知缺口**。

⚠️ 后两条测试钉的是**当前的缺口**（不是期望行为）：探针跑出来「真实着法有 8.28%
表达不出来」。**哪天动作空间修好了，它们会红 —— 那正是提醒去更新
`plans/2026-09-30-beat-70-roadmap.md` 的 A1 结论**，而不是把测试改绿了事。
"""
from guandan.sim import meld
from tools import action_space_probe as probe

A = meld.cid_from_name


def test_suit_choices_count_flavors_not_decks():
    """「同形状有几种选择」按**花色多重集**算 —— 两副的同名牌不算两种选择。"""
    hand = sorted([A("S5"), A("S5", deck=2), A("H5")])
    assert probe.suit_multisets(hand, 5, 1) == {("♠",), ("♥",)}


def test_level_rank_bomb_forces_the_wildcard():
    """已知缺口：级牌炸的代表**必然包含 ♥ = 逢人配**，哪怕手上有 6 张非红桃的。

    「用四张非红桃级牌炸、把逢人配留着」这个选项**不在候选集合里**。
    """
    level = 5
    hand = sorted([A("S5"), A("S5", deck=2), A("C5"), A("C5", deck=2),
                   A("D5"), A("D5", deck=2), A("H5"), A("H5", deck=2)])
    bombs = [m for m in meld.melds_from(hand, level)
             if m.kind == meld.BOMB and m.size == 4]
    assert bombs, "枚举不出 4 炸"
    assert any(meld.is_wild(c, level) for c in bombs[0].cards), (
        "这条钉的是「代表里带了逢人配」这个已知缺口；"
        "若已修好，请更新 plans 的 A1 结论")


def test_single_only_offers_the_first_card_of_the_rank():
    """同理：同一点数的单张只给 `ids[0]`（牌 ID 最小的那个 = ♠）。"""
    hand = sorted([A("S5"), A("H5"), A("D5")])
    singles = [m for m in meld.melds_from(hand, 9)
               if m.kind == meld.SINGLE and meld.cards.parts(m.cards[0])[0] == 5]
    assert len(singles) == 1, "契约：每个形状只给一条代表"
    assert meld.cards.parts(singles[0].cards[0])[1] == "♠"
