"""`log π(a|s)` 与熵：**必须带梯度**（不带的话训练一步都不动，日志上看不出来）。

这是整个策略式方案的命门所在的文件 —— 现有 `_flat_scores` 是 `no_grad` 的
（它服务于"算目标值"那类用途），照抄它会让 loss 变成常数、梯度为 `None`。
"""
import math
import random

import pytest
import torch

from train import replay
from train.net import (ENT_FLOOR_FRAC, QNet, check_entropy, check_logits,
                       log_prob_and_entropy, q_argmax_batch, q_values)
from train.policies import greedy_policy


def _samples(n=3, seed=0, level=5):
    """`replay.expand` 的产出形状：`(obs, acts, 选中下标, 出牌人, hist)`。"""
    out = []
    for k in range(n):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), level=level,
                                              capture=False)
        pts, _y2, _b = replay.expand(rec)
        out += pts[:4]
    return out


def test_log_prob_matches_a_hand_computed_softmax():
    """与「单局面单独算」逐点一致 —— 同 `q_max_batch` 那条测试的路子。"""
    net = QNet().eval()
    samples = _samples()
    lp, _ent, _zmax = log_prob_and_entropy(net, samples)
    assert len(lp) == len(samples)
    for (obs, acts, i, _seat, hist), got in zip(samples, lp):
        z = q_values(net, obs, acts, hist)
        want = torch.log_softmax(z, dim=0)[i]
        assert float(got.detach()) == pytest.approx(float(want), abs=1e-5)


def test_entropy_is_the_softmax_entropy():
    net = QNet().eval()
    samples = _samples()
    _lp, ent, _zmax = log_prob_and_entropy(net, samples)
    for (obs, acts, _i, _seat, hist), got in zip(samples, ent):
        z = q_values(net, obs, acts, hist)
        p = torch.softmax(z, dim=0)
        want = float(-(p * torch.log(p)).sum())
        assert float(got.detach()) == pytest.approx(want, abs=1e-5)


def test_zmax_is_reported_for_the_divergence_guard():
    net = QNet().eval()
    samples = _samples(1)
    _lp, _ent, zmax = log_prob_and_entropy(net, samples)
    assert zmax > 0.0


def test_log_prob_carries_gradient():
    """⚠️ **这一条是整个方案的命门**：`log π` 必须带梯度。"""
    net = QNet()
    lp, ent, _zmax = log_prob_and_entropy(net, _samples())
    assert lp.requires_grad and ent.requires_grad


def test_one_optimizer_step_actually_changes_the_parameters():
    net = QNet()
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    samples = _samples()
    lp, ent, _zmax = log_prob_and_entropy(net, samples)
    # ⚠️ 长度必须与样本数一致（计划里我写成固定的 4 个，那是错的）
    loss = -(lp * torch.tensor([2.0] * len(samples))).mean() - 0.01 * ent.mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    changed = sum(1 for a, b in zip(before, net.parameters()) if not torch.equal(a, b))
    assert changed, "训一步参数没变 —— 梯度断在哪儿了"


def test_gradient_contrasts_the_chosen_action_against_the_others(monkeypatch):
    """**机制测试**：`R > b` 时，**梯度**对选中项为负、对其余为正（符号相反）。

    ⚠️ 两处我自己写错后校正的地方（都值得记）：
    1. 第一版从输出层的偏置取梯度 —— **错的**：输出层是 `Linear(d, 1)`，
       偏置只有 1 维，根本没有"每个候选一个梯度"。改成**直接盯 logits**。
    2. 第二版把符号断反了 —— `loss = −log π(a)·adv`，所以**梯度**是
       `(π − e_a)·adv`：选中项为**负**、其余为**正**；
       而参数更新走 `−梯度` ⇒ 把选中的**抬上去**、把其余的压下去 ✓ **这才是对比**。

    对比正是策略梯度与回归的分水岭：回归把所有候选都往同一个 `R` 拉 ⇒ 梯度**同号**
    ⇒ 没有对比 ⇒ 就是"标签不区分动作"的数学形态。
    """
    import train.net as net_mod
    z = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    monkeypatch.setattr(net_mod, "_flat_scores",
                        lambda net, pending, grad=False: (z, [3]))
    samples = [("o", ["a", "b", "c"], 2, 0, "h")]        # 选中第 2 个
    lp, _ent, _zmax = net_mod.log_prob_and_entropy(object(), samples)
    (-(lp * torch.tensor([2.0])).mean()).backward()
    g = z.grad
    assert float(g[2]) < 0, "选中项的梯度该为负（更新时被抬上去）"
    assert float(g[0]) > 0 and float(g[1]) > 0, "其余的该为正（更新时被压下去）"
    # 再钉一层：实际更新方向 = −梯度 ⇒ 选中项抬、其余压
    z.data -= 0.1 * g
    assert float(z[2].detach()) > 2.0 > float(z[0].detach()), "更新后选中的该更大、其余更小"


def test_grad_depends_on_which_action_was_taken(monkeypatch):
    """换个「实际出的那一手」，梯度必须不同 —— **回归做不到这一点**（所有候选同目标）。"""
    import train.net as net_mod
    grads = []
    for take in (0, 2):
        z = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        monkeypatch.setattr(net_mod, "_flat_scores",
                            lambda net, pending, grad=False, _z=z: (_z, [3]))
        lp, _e, _x = net_mod.log_prob_and_entropy(
            object(), [("o", ["a", "b", "c"], take, 0, "h")])
        (-(lp * torch.tensor([2.0])).mean()).backward()
        grads.append(z.grad.detach().clone())
    assert not torch.allclose(grads[0], grads[1])


def test_the_existing_helpers_still_run_under_no_grad():
    """`grad` 默认 False ⇒ 老调用方（`q_argmax_batch` 等）行为不变。"""
    net = QNet().eval()
    first = _samples()[0]
    q_argmax_batch(net, [(first[0], first[1], first[4])])


def test_entropy_guard_raises_when_collapsed():
    """熵塌到「候选数均匀熵」的 5% 以下 ⇒ 响亮地炸（等于偷偷退化成 argmax）。"""
    check_entropy(0.9 * math.log(10), math.log(10))          # 正常：不炸
    with pytest.raises(RuntimeError):
        check_entropy(0.001, math.log(10))
    with pytest.raises(RuntimeError):                        # NaN 也要拦住
        check_entropy(float("nan"), math.log(10))


def test_entropy_floor_is_a_small_fraction():
    assert 0.0 < ENT_FLOOR_FRAC <= 0.2


def test_logits_guard_does_not_fire_on_normal_logits():
    """⚠️ **回归测试**：几十的 logits 是**正常的**，不许炸。

    2026-09-29 真踩过：把 `Q_ABS_MAX = 30` 照搬到 logits 上，PG 跑到 8,256 局被**误杀**
    —— 当时 `loss=0.152`、熵 0.84，一切正常。logits 是**对数几率**，没有有界尺度。
    """
    check_logits(31.13, games=8256, loss=0.152)      # 就是那次误杀的那个值
    check_logits(500.0, games=1, loss=0.0)


def test_logits_guard_fires_on_overflow_and_nan():
    with pytest.raises(RuntimeError):
        check_logits(1e9, games=1, loss=0.0)
    with pytest.raises(RuntimeError):
        check_logits(float("nan"), games=1, loss=float("nan"))
    with pytest.raises(RuntimeError):
        check_logits(float("inf"), games=1, loss=0.0)
