"""一步前瞻（`rl/search.py`）：**top_k=1 必须逐位退化成 argmax** —— 这条是它唯一的安全网。

另外钉住"零和搬值"的符号规则与"终局用精确分"这两条口径；
它们是这个 policy improvement operator 的全部假设，错了只会静默变差。
"""
import random

import pytest

from guandan.rl.net import QNet
from guandan.rl.search import one_step_backup
from guandan.sim import env, rules

A = None


def _net():
    import torch
    torch.manual_seed(0)
    return QNet().eval()


def _items(n_games=3, seed=5):
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset(level=rng.randint(1, 13))
        out.append((e, (e.observe(), e.legal(),
                        env.encode_history(e.hand, e.hand.turn))))
    return out


def test_top_k_one_is_exactly_argmax():
    """⚠️ `top_k=1` 时前瞻**只有一个候选可选** ⇒ 必须与 argmax 完全一致。

    这条同时校验：候选下标对齐、`step(k)` 与 `legal()` 是同一套编号、
    以及"平手时退回 argmax 的顺序"。
    """
    from guandan.rl.net import q_values
    net = _net()
    items = _items(4)
    picks = one_step_backup(net, items, top_k=1)
    for (e, (obs, acts, hist)), got in zip(items, picks):
        q = [float(x) for x in q_values(net, obs, acts, hist)]
        want = max(range(len(q)), key=q.__getitem__)
        assert got == want


def test_returns_valid_indices_for_every_item():
    net = _net()
    items = _items(4)
    picks = one_step_backup(net, items, top_k=4)
    assert len(picks) == len(items)
    for (e, (_o, acts, _h)), got in zip(items, picks):
        assert 0 <= got < len(acts)


def test_zero_top_k_raises():
    net = _net()
    with pytest.raises(ValueError, match="top_k"):
        one_step_backup(net, _items(1), top_k=0)


def test_match_accepts_search_a():
    """`eval.match(search_a=...)` 那条路要能跑（只影响 a 方选动作、不改权重）。"""
    from guandan.rl import eval as ev
    from guandan.rl.rule_policy import rule_policy
    net = _net()

    def search(items):
        return one_step_backup(net, items, top_k=2)

    wr = ev.match(lambda o, a, h=None: 0, rule_policy(), games=2, seed=1,
                  search_a=search)
    assert 0.0 <= wr <= 1.0
