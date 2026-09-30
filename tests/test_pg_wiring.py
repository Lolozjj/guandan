"""`--algo pg` 的接线：行为策略必须是 π、ε 必须彻底失效、日志必须说清。

⚠️ 每一处都**共用同一个 `QNet` 实例** —— 每调一次 `QNet()` 都是新的随机初始化，
那样两臂的差异就来自权重而不是接线了（这个坑本仓库踩过一次，测试会假红）。
"""
import random

import pytest
import torch

import guandan.rl.selfplay as sp
from guandan.rl.net import QNet
from guandan.rl.selfplay import generate_batch


def test_pg_mode_completely_ignores_eps():
    """⚠️ **on-policy 的命门**：PG 模式下 ε 取什么值都必须不影响打出来的牌。

    行为策略是 π ⇒ 数据才是 π 的。若 ε 还在偷偷起作用，策略梯度估的是
    **另一个分布**的梯度 —— 静默学歪，loss 曲线完全看不出来。
    """
    net = QNet().eval()                        # ⚠️ 共用一个网络
    def run(eps):
        return [rec.actions for rec, _p, _y in generate_batch(
            net, random.Random(3), eps, 4, capture=True, opp_mix=0.0, sample=True)]
    assert run(0.1) == run(0.9), "ε 在 PG 模式下还在起作用"


def test_pg_mode_actually_samples():
    """行为策略必须**不是** argmax（否则等于没换）：同一批牌，采样与 argmax 不能逐点全同。"""
    from guandan.rl.net import q_argmax_batch
    net = QNet().eval()
    caps, chosen = [], []
    for _rec, pts, _y in generate_batch(net, random.Random(3), 0.0, 4, capture=True,
                                        opp_mix=0.0, sample=True):
        caps += pts
        chosen += [i for _o, _a, i, _s, _h in pts]
    picks = q_argmax_batch(net, [(o, a, h) for o, a, _i, _s, h in caps])
    assert chosen != picks, "采样与 argmax 完全一样 —— 行为策略没换成 π"


def test_pg_header_is_honest(tmp_path):
    """**换源必须可见**：PG 模式要说清 ε 不适用、β_ent 是多少、不用 buffer。"""
    lines = []
    sp.train(seconds=1, algo="pg", log=lines.append, eval_games=1,
             eval_every=10 ** 9, out_dir=str(tmp_path))
    head = "\n".join(lines[:4])
    assert "pg" in head.lower(), head
    assert "ε 不适用" in head, head
    assert "β_ent" in head, head
    assert "不用 replay buffer" in head, head


def test_dmc_is_still_the_default():
    import inspect
    for fn in (sp.train, sp.train_parallel):
        assert inspect.signature(fn).parameters["algo"].default == "dmc"


def test_cli_forwards_the_pg_knobs(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0, "games": 0,
                "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--algo", "pg", "--beta-ent", "0.02", "--weight-sync-games", "200"])
    assert (seen["algo"], seen["beta_ent"], seen["weight_sync_games"]) == ("pg", 0.02, 200)


def test_unknown_algo_is_rejected(monkeypatch):
    """⚠️ **评审 M1**：`--algo ppo` 原来会**静默走 DMC**，而日志对非 pg 一个字都不提
    ⇒ 一次手滑的「PG 臂」其实是对照臂，没有任何东西会响。"""
    with pytest.raises(ValueError):
        sp.main(["1", "--algo", "ppo"])
    with pytest.raises(ValueError):
        sp.main(["1", "--algo", "PG"])


def test_weight_sync_games_zero_is_rejected(monkeypatch):
    """0 会让 `games - last_sync >= 0` 恒真 ⇒ 每步广播 7.8 MB（评审 M7）。"""
    with pytest.raises(ValueError):
        sp.main(["1", "--weight-sync-games", "0"])
