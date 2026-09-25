import random
import collections

import pytest

from net import cards
from net.sim import rules


def test_full_deck_is_two_complete_decks():
    assert len(rules.FULL_DECK) == 108
    assert len(set(rules.FULL_DECK)) == 108                 # 没有重复 ID
    idx = collections.Counter(cards.parts(c)[0] for c in rules.FULL_DECK)
    assert all(n == 8 for k, n in idx.items() if k not in (14, 15))   # 每个点数两副共 8 张
    assert idx[14] == 2 and idx[15] == 2                     # 大小王各 2 张


@pytest.mark.parametrize("seed", range(20))
def test_deal_gives_everyone_27_distinct_cards_and_conserves_108(seed):
    rng = random.Random(seed)
    h = rules.deal(rng, level=2)
    sizes = sorted(len(h.hands[s]) for s in rules.SEATS)
    assert sizes == [27, 27, 27, 27]
    union = set().union(*[h.hands[s] for s in rules.SEATS])
    assert len(union) == 108                                 # 同一张牌不许出现在两家


def test_new_hand_with_explicit_hands_keeps_them_verbatim():
    """回放真实对局时要能手给四家牌 —— 这时**不许**洗牌。"""
    hands = [{1, 2, 3}, {4, 5}, set(), {6}]
    h = rules.new_hand(random.Random(0), level=7, hands=hands, first=2)
    assert [set(x) for x in h.hands] == [set(x) for x in hands]
    assert h.level == 7 and h.turn == 2


def test_level_14_is_rejected_not_silently_treated_as_a_joker():
    """日志里 A 有时写 14。**归一必须走 `meld.norm_level`**，
    绝不能让 14 顺着接口流进来 —— 那会被 `cards.parts` 当成小王。
    `new_hand` 是唯一入口，所以在这里拦。"""
    with pytest.raises(ValueError):
        rules.new_hand(random.Random(0), level=14, hands=[{1}] * 4)
