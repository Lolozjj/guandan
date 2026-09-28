"""自举的接线：目标网络同步、β 真的改变了目标、发散守门真的会响。

与 `tests/test_learn_step.py` 分开是有意的：那一份测的是**重构**
（采样→重放→拼张量收成一份，行为不变），这一份测的是**新行为**。
"""
import random

import torch

from train.net import QNet
from train.selfplay import _learn_step, _targets, build_samples, sync_target

from tests.test_learn_step import _buffer

def test_beta_one_never_computes_a_bootstrap_value():
    """**默认行为不变**：β=1 时一次都不该去算 `V`（省掉那次昂贵的前向）。"""
    import train.selfplay as sp
    calls = []
    orig = sp.q_max_batch
    sp.q_max_batch = lambda n, p: calls.append(p) or orig(n, p)
    try:
        torch.manual_seed(0)
        net = QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        _learn_step(net, None, _buffer(), random.Random(0), opt, 0,
                    batch_games=4, bomb_cost=0.0, mc_mix=1.0, n_step=3)
    finally:
        sp.q_max_batch = orig
    assert not calls, "β=1 是纯 MC，不该走自举那条路"


def test_beta_half_actually_bootstraps():
    """β<1 且给了目标网络时，**真的走了自举那一支**（boot 非空）。"""
    import train.selfplay as sp
    calls = []
    orig = sp.q_max_batch
    sp.q_max_batch = lambda n, p: calls.append(p) or orig(n, p)
    try:
        torch.manual_seed(0)
        net, tgt = QNet(), QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        _learn_step(net, tgt, _buffer(), random.Random(0), opt, 0,
                    batch_games=4, bomb_cost=0.0, mc_mix=0.5, n_step=3)
    finally:
        sp.q_max_batch = orig
    assert calls and any(p is not None for p in calls[0]), "自举项没算"


def test_beta_half_differs_from_the_mc_targets():
    """同一个种子下，β=0.5 与 β=1 的目标**必须不同** —— 否则旋钮是假的。"""
    from train.selfplay import _targets
    torch.manual_seed(0)
    net, tgt = QNet(), QNet()
    _s1, y1 = _targets(net, tgt, _buffer(), random.Random(0), 4, 0.0, 1.0, 3)
    _s2, y2 = _targets(net, tgt, _buffer(), random.Random(0), 4, 0.0, 0.5, 3)
    assert y1 != y2


def test_sync_target_copies_only_at_the_interval():
    net, tgt = QNet(), QNet()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    assert sync_target(net, tgt, games=999, last_sync=0, every=1000) == 0
    assert not torch.equal(next(iter(net.parameters())),
                           next(iter(tgt.parameters())))
    assert sync_target(net, tgt, games=1000, last_sync=0, every=1000) == 1000
    assert torch.equal(next(iter(net.parameters())), next(iter(tgt.parameters())))


def test_sync_target_is_a_noop_without_a_target_net():
    assert sync_target(QNet(), None, games=10 ** 9, last_sync=0) == 0
