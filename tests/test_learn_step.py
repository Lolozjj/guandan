"""训练步的共享实现 —— 采样/重放/拼张量这一段不许在两条路线里各写一份。"""
import random

import torch

from train import replay
from train.net import QNet
from train.policies import greedy_policy
from train.selfplay import _learn_step, build_samples


def _buffer(n=8, seed=0):
    buf = replay.ReplayBuffer(capacity_games=n)
    for k in range(n):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), capture=False)
        buf.add(rec)
    return buf


def test_build_samples_is_aligned():
    s, y, b = build_samples(_buffer(), random.Random(0), 4)
    assert len(s) == len(y) > 0
    assert b == [], "n_step=0 时根本不算自举 —— boot 是空表（与 expand 一致）"


def test_build_samples_uses_the_fresh_cache():
    """`fresh` 命中时不该再去重放（重放一局约 18ms，白花）。"""
    buf = _buffer(n=1)                 # 只有一局 -> 采样必然采到它
    rec = buf._games[0]
    pts = [("P", "A", 0, 0, "H")]
    s, y, b = build_samples(buf, random.Random(0), 1,
                            fresh={id(rec): (pts, [-3.0])})
    assert (s, y) == (pts, [-3.0]) and b == []


def test_build_samples_ignores_fresh_when_bootstrapping():
    """**β<1 时快捷缓存必须让路** —— 缓存里没有自举值，
    用它就等于让这一部分样本悄悄退回纯 MC（loss 曲线上看不出来）。"""
    buf = _buffer()
    fresh = {id(r): ([("P", "A", 0, 0, "H")], [-3.0]) for r in buf._games}
    s, y, b = build_samples(buf, random.Random(0), 4, fresh=fresh, n_step=3)
    assert len(b) == len(s) > 4, "带了 n_step 就必须走 expand（boot 非空）"


def test_learn_step_returns_a_finite_loss():
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = _learn_step(net, _buffer(), random.Random(0), opt,
                       batch_games=4, bomb_cost=0.0)
    assert isinstance(loss, float) and loss == loss


def test_learn_step_is_deterministic_for_one_seed():
    """同一批牌 + 同一种子 → 同一条轨迹（重放是确定性的）。"""
    def once():
        torch.manual_seed(0)
        net = QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        return _learn_step(net, _buffer(), random.Random(7), opt,
                           batch_games=4, bomb_cost=0.0)
    assert once() == once()
