"""`q_max_batch`：自举项 `V(s) = max_a Q(s,a)`。**返回 float，不是 tensor。**"""
import random

import pytest
import torch

from guandan.sim import env
from guandan.rl import replay
from guandan.rl.net import (QNet, check_q_scale, q_argmax_batch, q_max_batch,
                       q_values)
from guandan.rl.policies import greedy_policy


def _pending(games=2, seed=0):
    """一批形状正确的 `(obs, acts, hist)`。"""
    out = []
    for k in range(games):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), capture=False)
        e = env.GuandanEnv(seed=0)
        e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
        out.append((e.observe(), e.legal(), env.encode_history(e.hand, e.hand.turn)))
    return out


def test_matches_the_single_state_path():
    """批量算出来的必须**逐个等于** `q_values(...).max()` —— 最强的等价性测试。"""
    net = QNet().eval()
    pend = _pending()
    for (obs, acts, hist), v in zip(pend, q_max_batch(net, pend)):
        assert v == pytest.approx(float(q_values(net, obs, acts, hist).max()), abs=1e-5)


def test_returns_floats_not_tensors():
    """**类型上就带不了梯度** —— 目标项不许 backprop 到任何东西。"""
    assert isinstance(q_max_batch(QNet().eval(), _pending(games=1))[0], float)


def test_none_entries_are_preserved():
    """`None` 的位置原样返回 `None`（那些点整项退回 MC），**不许挪位**。"""
    pend = _pending(games=1)
    got = q_max_batch(QNet().eval(), [None, pend[0], None])
    assert got[0] is None and got[2] is None and isinstance(got[1], float)


def test_all_none_does_not_touch_the_network():
    assert q_max_batch(QNet().eval(), [None, None]) == [None, None]


def test_two_different_nets_give_different_values():
    """防「拿错了网络」—— 目标网络接错成在线网络时，这条会红。"""
    torch.manual_seed(0)
    a, b = QNet().eval(), QNet().eval()
    pend = _pending(games=1)
    assert q_max_batch(a, pend)[0] != pytest.approx(q_max_batch(b, pend)[0])


def test_forward_runs_under_no_grad():
    """**这条要有牙**：把 `_flat_scores` 里的 `torch.no_grad()` 删掉，它必须红。

    （原来那条「`p.grad is None`」是**假牙** —— 只前向不 backward，`grad` 本来就是
    None，删掉 `no_grad` 它照样绿。2026-09-28 评审实测过。）
    """
    net = QNet()
    seen = []
    orig = QNet.forward

    def spy(self, *a, **kw):
        seen.append(torch.is_grad_enabled())
        return orig(self, *a, **kw)

    QNet.forward = spy
    try:
        q_max_batch(net, _pending(games=1))
    finally:
        QNet.forward = orig
    assert seen, "前向没被走到？"
    assert not any(seen), "前向跑在开启梯度的上下文里 —— 目标项会被卷进梯度图"


def test_argmax_batch_is_unchanged_by_the_refactor():
    """重构不许改变老行为 —— 与 `q_values` 的 argmax 逐个比。"""
    net = QNet().eval()
    pend = _pending()
    for (obs, acts, hist), i in zip(pend, q_argmax_batch(net, pend)):
        assert i == int(q_values(net, obs, acts, hist).argmax())


def test_check_q_scale_raises_above_the_limit():
    with pytest.raises(RuntimeError):
        check_q_scale(30.1, games=1000, loss=1.0)


def test_check_q_scale_passes_at_the_limit():
    check_q_scale(30.0, games=1000, loss=1.0)


def test_check_q_scale_raises_on_nan():
    """NaN 与任何数比较都是 False —— 写成 `x > limit` 会让它悄悄溜过去。"""
    with pytest.raises(RuntimeError):
        check_q_scale(float("nan"), games=1000, loss=float("nan"))
