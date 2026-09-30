"""自举的接线：目标网络同步、β 真的改变了目标、发散守门真的会响。

与 `tests/test_learn_step.py` 分开是有意的：那一份测的是**重构**
（采样→重放→拼张量收成一份，行为不变），这一份测的是**新行为**。
"""
import random

import pytest
import torch

from guandan.rl.net import QNet
import guandan.rl.selfplay as sp
from guandan.rl.selfplay import _learn_step, _targets, build_samples, sync_target

from tests.test_learn_step import _buffer

def test_beta_one_never_computes_a_bootstrap_value():
    """**默认行为不变**：β=1 时一次都不该去算 `V`（省掉那次昂贵的前向）。"""
    import guandan.rl.selfplay as sp
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
    import guandan.rl.selfplay as sp
    calls = []
    orig = sp.q_max_batch
    sp.q_max_batch = lambda n, p: calls.append(p) or orig(n, p)
    try:
        torch.manual_seed(0)
        net, tgt = QNet(), QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        _learn_step(net, tgt, _buffer(), random.Random(0), opt, 0,
                    batch_games=4, bomb_cost=0.0, mc_mix=0.5, n_step=2)
    finally:
        sp.q_max_batch = orig
    assert calls and any(p is not None for p in calls[0]), "自举项没算"


def test_beta_half_differs_from_the_mc_targets():
    """同一个种子下，β=0.5 与 β=1 的目标**必须不同** —— 否则旋钮是假的。"""
    from guandan.rl.selfplay import _targets
    torch.manual_seed(0)
    net, tgt = QNet(), QNet()
    _s1, y1 = _targets(tgt, _buffer(), random.Random(0), 4, 0.0, 1.0, 2)
    _s2, y2 = _targets(tgt, _buffer(), random.Random(0), 4, 0.0, 0.5, 2)
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


# ---- 2026-09-28 评审挑出来的三条 ----


def test_sync_target_rejects_a_nonpositive_interval():
    """`--tgt-sync 0` 会**静默**变成「永不同步」（目标网络永远停在热启动权重上，
    正是 spec §3.3 要防的「靶子不动」）。配置错了必须响。"""
    with pytest.raises(ValueError):
        sync_target(QNet(), QNet(), games=10 ** 6, last_sync=0, every=0)


def test_targets_use_the_target_net():
    """扰动**目标网络** → 目标值必须跟着变（证明它真的被用上了）。

    而「误用在线网络」这件事现在由**签名**排除：`_targets` 根本收不到在线网络
    （2026-09-28 评审的 I2/M2 —— 老签名里那个没用到的 `net` 参数会让人以为
    「在线网络也参与算目标」，顺手把它误用成 `q_max_batch(net, ...)` 时全测试都绿）。
    """
    buf = _buffer()
    torch.manual_seed(0)
    tgt = QNet()
    _s1, y1 = _targets(tgt, buf, random.Random(0), 4, 0.0, 0.5, 2)
    with torch.no_grad():
        for p in tgt.parameters():
            p.add_(10.0)
    _s2, y2 = _targets(tgt, buf, random.Random(0), 4, 0.0, 0.5, 2)
    assert y1 != y2, "扰动目标网络后目标值没变 —— 自举根本没用到它"


@pytest.mark.parametrize("kw", [
    {"mc_mix": 0.5, "n_step": 0},     # 要自举却往后看 0 步：说了不做
    {"mc_mix": 0.5, "n_step": -1},
    {"mc_mix": 1.5},
    {"tgt_sync": 0},                  # 0 = 永不同步
])
def test_contradictory_config_is_rejected(kw):
    """日志会写「自举 β=0.5」而实际一个自举项都不算 —— 那种状态必须响亮地拒掉。"""
    # `eval_games=1` 让收尾评测变便宜（否则这条测试要跑 600 局评测）
    with pytest.raises(ValueError):
        sp.train(seconds=0, eval_games=1, eval_every=10 ** 9, **kw)
