"""配合护栏（`rl/coop.py`）：只在「我领出 + 队友剩 1~2 张」时把首选换成他需要的牌型。"""
from guandan.rl.coop import coop_index
from guandan.sim import meld


class _M:
    def __init__(self, kind, size=1, rank=5):
        self.kind, self.size, self.rank = kind, size, rank
        self.is_bomb = False
        self.wild_used = 0
        self.cards = (1,)


def test_leading_and_mate_short_switches_to_needed_kind():
    acts = [_M(meld.SINGLE, 1, 9), _M(meld.PAIR, 2, 3)]     # 首选是单张（index 0）
    q = [5.0, 1.0]
    # 队友剩 2 张 ⇒ 应该改成对子（index 1），哪怕 Q 更低
    assert coop_index(q, acts, left_mate=2, table=(), chosen_i=0) == 1
    # 队友剩 1 张 ⇒ 单张已经对了 ⇒ 不动
    q2 = [5.0, 1.0]
    assert coop_index(q2, acts, left_mate=1, table=(), chosen_i=0) == 0


def test_following_never_intervenes():
    """⚠️ 跟牌（桌面非空）时**不许**介入 —— 那时没有"喂谁"的自由。"""
    acts = [_M(meld.SINGLE, 1, 9), _M(meld.PAIR, 2, 3)]
    assert coop_index([5.0, 1.0], acts, left_mate=2, table=(1,), chosen_i=0) == 0


def test_no_candidate_of_needed_kind_means_no_change():
    acts = [_M(meld.SINGLE, 1, 9), _M(meld.TRIPLE, 3, 3)]     # 没有对子
    assert coop_index([5.0, 1.0], acts, left_mate=2, table=(), chosen_i=0) == 0


def test_mate_with_many_cards_no_change():
    acts = [_M(meld.SINGLE, 1, 9), _M(meld.PAIR, 2, 3)]
    assert coop_index([5.0, 1.0], acts, left_mate=7, table=(), chosen_i=0) == 0
