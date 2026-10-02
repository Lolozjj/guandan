"""领出也管（`leads=True`）：2026-10-03 发现"带 bug 的版本"顺带管领出时联赛 +1.80pp，
而只跟牌的版本只有 +0.08pp ⇒ 这一维值得**明确地**做成开关再复核。"""
from guandan.rl import tidy


class _M:
    def __init__(self, kind, size=1, rank=5, bomb=False, wild=0):
        self.kind, self.size, self.rank = kind, size, rank
        self.is_bomb, self.wild_used = bomb, wild
        self.cards = (1,)


def test_leads_off_by_default_keeps_bomb_leads():
    acts = [_M(1, 1, 5), _M(8, 4, 8, bomb=True)]
    q = [1.0, 3.0]
    # 领出（has_table=False）+ leads 关 ⇒ 一动不动
    assert tidy.tidy_index(q, acts, 1, has_table=False, margin=0.25) == 1


def test_leads_on_downgrades_weak_bomb_lead():
    acts = [_M(1, 1, 5), _M(8, 4, 8, bomb=True)]
    q = [1.0, 1.1]                       # 只差 0.1（margin=0.25 之内）
    assert tidy.tidy_index(q, acts, 1, has_table=False, margin=0.25, leads=True) == 0


def test_leads_on_respects_margin():
    acts = [_M(1, 1, 5), _M(8, 4, 8, bomb=True)]
    q = [1.0, 9.0]                       # 差 8 ⇒ 网络很坚决 ⇒ 不动
    assert tidy.tidy_index(q, acts, 1, has_table=False, margin=0.25, leads=True) == 1
