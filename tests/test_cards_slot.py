from guandan.capture import cards


def test_slot_is_a_bijection_over_the_108_cards():
    """两副牌正好 108 张，slot 必须一一对应 —— 编码位撞了就是静默丢牌。"""
    deck = [c for c in range(0, 334) if cards.is_card(c)]
    assert len(deck) == 108
    assert sorted(cards.slot(c) for c in deck) == list(range(108))


def test_slot_known_values():
    """钉住几张具体的牌，防止基址被改。"""
    from guandan.sim import meld
    A = meld.cid_from_name
    assert cards.slot(A("SA")) == 0             # 第一副 ♠A
    assert cards.slot(A("SK")) == 12            # 花色内按点数排
    assert cards.slot(A("HA")) == 13            # 花色是第一档分组
    assert cards.slot(A("DK", deck=2)) == 105   # 第二副整体 +54
    assert cards.slot(A("JOKER_S")) == 52       # 小王（第一副）
    assert cards.slot(A("JOKER_B", deck=2)) == 107
