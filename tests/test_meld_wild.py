from net.sim import meld
from tests.test_meld_basic import C


def test_wild_alone_is_its_own_card():
    """打 5 时 ♥5 本来就是一张 5。"""
    hand = C("5♥")
    assert [m for m in meld.melds_from(hand, level=5) if m.kind == meld.SINGLE]


def test_two_wilds_are_a_pair_of_level_cards():
    hand = C("5♥", "5♥(二副)")
    pairs = [m for m in meld.melds_from(hand, level=5) if m.kind == meld.PAIR]
    assert pairs and all(m.wild_used == 0 for m in pairs)


def test_wild_fills_a_bomb():
    """数据里 2♥2♥2♠10♥ 在打10时是四个2的炸。"""
    hand = C("2♥", "2♥(二副)", "2♠", "10♥")
    bombs = [m for m in meld.melds_from(hand, level=10) if m.kind == meld.BOMB]
    assert any(m.size == 4 and m.rank == meld.point_value(2, 10) for m in bombs)


def test_wild_fills_straight_flush():
    """数据里 9♣10♣J♣Q♣ + ♥2 在打2时是同花顺。"""
    hand = C("9♣", "10♣", "J♣", "Q♣", "2♥")
    sf = [m for m in meld.melds_from(hand, level=2)
          if m.kind == meld.STRAIGHT_FLUSH]
    assert sf, "逢人配应当能补成同花顺"


def test_wild_never_becomes_a_joker():
    hand = C("2♥", "大王", "大王(二副)", "小王")
    for m in meld.melds_from(hand, level=2):
        if m.kind == meld.BOMB and m.size == 4:
            assert all(meld.cards.parts(c)[0] in (14, 15) for c in m.cards), \
                "王炸里不能混进逢人配"


def test_no_wild_at_all_still_works():
    hand = C("5♦", "5♣", "5♠")
    assert [m for m in meld.melds_from(hand, level=9) if m.kind == meld.TRIPLE]


def test_only_one_wild_available():
    hand = C("5♦", "5♣", "2♥")
    triples = [m for m in meld.melds_from(hand, level=2) if m.kind == meld.TRIPLE]
    assert triples and triples[0].wild_used == 1


def test_results_are_not_duplicated():
    hand = C("5♦", "5♣", "5♠", "2♥")
    keys = [tuple(sorted(m.cards)) for m in meld.melds_from(hand, level=2)]
    assert len(keys) == len(set(keys)), "同一组牌重复出现了"


# --- 以下是 brief 之外补的：补牌必须报出**具体牌张** -------------------------
# Task 7 验收① 的 as_meld() 是拿真实那一手的牌组做**精确集合**比对的：
#     sorted(m.cards) == sorted(s.actual)
# 补出来的 Meld 若只放天然那几张、不把逢人配本身算进 cards，验收① 会把
# 真实打出的每一手带逢人配的牌都报成「判不出牌型」。真数据里 98 手带逢人配的
# 着法中有 90 手如此（打2时的四个8炸、打9时的 6♦8♦9♦10♦+♥9 同花顺……）。


def test_wild_assisted_meld_reports_concrete_cards():
    """补出来的炸弹要把逢人配那张牌算进 cards，且张数等于 size。"""
    hand = C("2♥", "2♥(二副)", "2♠", "10♥")
    bomb = [m for m in meld.melds_from(hand, level=10)
            if m.kind == meld.BOMB and m.size == 4][0]
    assert set(bomb.cards) == set(hand)      # 正好是这 4 张
    assert len(bomb.cards) == len(set(bomb.cards))
    assert bomb.wild_used == 1


def test_wild_fills_triple_pair_pair_half():
    """三张齐、对子差一张：逢人配补对子那一半。

    真实数据（net/events.jsonl）里的一手 card_type=5：
        6♠(二副) 6♥ 6♥(二副) 4♦(二副) + ♥2（打 2）
    「三张已齐、只缺对子」是最常见的一种，不能被补牌条件挡掉。
    """
    hand = C("6♠(二副)", "6♥", "6♥(二副)", "4♦(二副)", "2♥")
    tp = [m for m in meld.melds_from(hand, level=2)
          if m.kind == meld.TRIPLE_PAIR]
    assert tp, "三张齐、对子缺一张的三带二没枚举出来"
    assert set(tp[0].cards) == set(hand)

    # 三张那一半靠补牌、对子天然那种同样要有
    other = C("5♦", "5♣", "4♥", "4♠", "2♥")
    tp2 = [m for m in meld.melds_from(other, level=2)
           if m.kind == meld.TRIPLE_PAIR]
    assert tp2 and set(tp2[0].cards) == set(other)


def test_wild_fills_triple_pair_when_both_halves_short():
    """三张与对子都只有两张，逢人配补成三张。

    真实数据里的一手 card_type=5：
        10♥(二副) 10♣(二副) + 2♠(二副) 2♦(二副) + ♥J（打 J）
    """
    hand = C("10♥(二副)", "10♣(二副)", "2♠(二副)", "2♦(二副)", "J♥(二副)")
    tp = [m for m in meld.melds_from(hand, level=11)
          if m.kind == meld.TRIPLE_PAIR]
    assert tp, "两边都差一张的三带二没枚举出来"
    assert set(tp[0].cards) == set(hand)
    assert tp[0].wild_used == 1


# --- 炸弹张数上限：8 张天然 + 最多 2 张逢人配 = 10（R19）-------------------
# ⚠️ **9炸/10炸 在炸弹阶梯上的位置是自然延伸，数据未验**：
#    4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 有用户口述 + 24 对真实「对压」证据，
#    9炸/10炸 只有「存在」证据（下面这一手），**没有**对压证据。


def test_nine_card_bomb_from_real_play():
    """真数据里的一手 9 张炸：8 张 J 加一张逢人配 ♥3（打 3，card_type=10）。

    `_MAX_BOMB` 原来是 8（两副牌一个点数最多 8 张），这一手枚举不出来 ——
    tools/accept_meld.py 验收① 会把它报成「真实出的牌本身判不出牌型」。
    """
    hand = C("J♠", "J♠(二副)", "J♥", "J♥(二副)",
             "J♣", "J♣(二副)", "J♦", "J♦(二副)", "3♥")
    bombs = [m for m in meld.melds_from(hand, level=3) if m.kind == meld.BOMB]
    nine = [m for m in bombs if m.size == 9]
    assert nine, "8 张 J + 逢人配的 9 张炸没枚举出来"
    assert nine[0].wild_used == 1
    assert set(nine[0].cards) == set(hand)
    assert nine[0].rank == meld.point_value(11, 3)

    eight = meld.Meld(meld.BOMB, 8, meld.point_value(11, 3),
                      tuple(C("J♠", "J♠(二副)", "J♥", "J♥(二副)",
                              "J♣", "J♣(二副)", "J♦", "J♦(二副)")))
    joker = meld.Meld(meld.BOMB, 4, 0,
                      tuple(C("小王", "小王(二副)", "大王", "大王(二副)")))
    assert meld.beats(nine[0], eight), "9 炸应当压得过 8 炸"
    assert meld.beats(joker, nine[0]), "天王炸仍应压得过 9 炸"
    assert not meld.beats(nine[0], joker), "9 炸不该压得过天王炸"


def test_ten_card_bomb_is_the_ceiling():
    """上界 = 8 张天然 + 2 张逢人配 = 10；再多一张逢人配也不存在。"""
    hand = C("J♠", "J♠(二副)", "J♥", "J♥(二副)", "J♣", "J♣(二副)",
             "J♦", "J♦(二副)", "3♥", "3♥(二副)")
    bombs = [m for m in meld.melds_from(hand, level=3) if m.kind == meld.BOMB]
    ten = [m for m in bombs if m.size == 10]
    assert ten, "8 张 J + 2 张逢人配的 10 张炸没枚举出来"
    assert ten[0].wild_used == 2
    assert set(ten[0].cards) == set(hand)
    assert max(m.size for m in bombs) == 10
