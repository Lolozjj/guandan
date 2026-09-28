"""行为策略：**从 π 采样**（不是 argmax、也不带 ε）。

on-policy 的定义就落在这个函数上：PG 的数据必须来自 π 自己。
"""
import random

import torch

import train.net as net_mod
from tests.test_pg_logprob import _samples
from train.net import QNet, policy_sample_batch


def test_sample_follows_pi(monkeypatch):
    """频率要贴近 π —— 固定 logits，采 4000 次逐个比。"""
    z = torch.tensor([0.0, 1.0, 2.0])
    monkeypatch.setattr(net_mod, "_flat_scores",
                        lambda net, pending, grad=False: (z, [3]))
    p = torch.softmax(z, 0)
    rng = random.Random(0)
    cnt = [0, 0, 0]
    for _ in range(4000):
        cnt[policy_sample_batch(object(), [("o", ["a", "b", "c"], "h")], rng)[0]] += 1
    for got, want in zip(cnt, p):
        assert abs(got / 4000 - float(want)) < 0.03, (cnt, p.tolist())


def test_sample_is_reproducible_for_the_same_rng():
    """可复现性挂在**调用方的 `rng`** 上（不走 torch 的生成器）——
    整条链连「哪一队当对手」都用同一个 rng 抽，这里必须一致。"""
    net = QNet().eval()
    pend = [(o, a, h) for o, a, _i, _s, h in _samples(2)]
    assert (policy_sample_batch(net, pend, random.Random(7))
            == policy_sample_batch(net, pend, random.Random(7)))


def test_sample_returns_valid_indices():
    net = QNet().eval()
    pend = [(o, a, h) for o, a, _i, _s, h in _samples(2)]
    got = policy_sample_batch(net, pend, random.Random(0))
    assert len(got) == len(pend)
    assert all(0 <= j < len(a) for j, (_o, a, _h) in zip(got, pend))


def test_sample_covers_more_than_the_argmax():
    """同一个局面采多次，必须出现**不止一个**结果 —— 否则它其实在 argmax。"""
    net = QNet().eval()
    pend = [(o, a, h) for o, a, _i, _s, h in _samples(1)]
    rng = random.Random(0)
    seen = {policy_sample_batch(net, pend, rng)[0] for _ in range(60)}
    assert len(seen) > 1, "60 次采样只出一个结果 —— 行为策略没在采样"
