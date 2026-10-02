"""信念模块的最小测试：**抽出来的粒子必须自洽**，以及"最便宜的能压选择"的口径。

这两个是它的全部根据：粒子自洽（否则信念是假的），
`cheapest_beating` 正确（否则行为先验算的是别的东西）。
"""
import random

from guandan.rl.belief import Belief, can_beat, cheapest_beating, unseen_ids
from guandan.sim import env, meld, rules

A = meld.cid_from_name if hasattr(meld, "cid_from_name") else None


def _obs(seed=7):
    e = env.GuandanEnv(seed=seed)
    return e, e.reset()


def test_particles_are_consistent():
    """每个粒子里：三家手数之和 = 未见牌张数；三家手牌互不相交、合起来正好是未见牌。"""
    e, obs = _obs()
    b = Belief(obs, k=8, rng=random.Random(0))
    pool = set(unseen_ids(obs))
    for h in b.hands:
        assert set(h) == set(b.others)
        assert sum(len(v) for v in h.values()) == len(pool)
        union = set().union(*h.values())
        assert union == pool, "三家手牌必须恰好分完未见牌"
        assert len(set().union(*h.values())) == sum(len(v) for v in h.values()), "不能重叠"


def test_sizes_match_left_counts():
    e, obs = _obs(seed=11)
    b = Belief(obs, k=4, rng=random.Random(0))
    for h in b.hands:
        for s in b.others:
            assert len(h[s]) == obs.left[s]


def test_cheapest_beating_and_can_beat():
    """能压就给出最便宜的那一手；压不过给 None。"""
    table = meld.as_meld([A("S3")], 2)          # 桌面一张 3
    assert cheapest_beating([A("S4")], table, 2) is not None
    assert can_beat([A("S4")], table, 2)
    # ⚠️ 别拿 ♠2 当"压不过"的例子：打 2 时它是**级牌**，比 3 大（我第一版就写错了）。
    # 同点数压不过才算数：
    assert cheapest_beating([A("S3")], table, 2) is None
    assert not can_beat([A("S3")], table, 2)
    # 有得选时要**最便宜**的：手里有 4 和 K，选 4
    m = cheapest_beating([A("S4"), A("SK")], table, 2)
    assert m is not None and A("S4") in m.cards


def test_reweight_shrinks_when_pass_was_possible():
    """能压却过牌 ⇒ 权重打折（机械正确性；信息量本身已被探针证否）。"""
    e, obs = _obs(seed=13)
    b = Belief(obs, k=16, rng=random.Random(0), alpha=0.5)
    table = meld.as_meld([A("S3")], 2)          # 桌面很小 ⇒ 几乎人人都能压
    seat = b.others[0]
    b.reweight([(seat, table, None, [])])
    assert all(w <= 1.0 for w in b.w)
    assert min(b.w) < 1.0, "至少有一部分粒子被判成『本来能压却过了』"
    assert 0.0 < b.ess() <= b.k
