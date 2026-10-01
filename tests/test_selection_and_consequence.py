"""点③（多种子挑 `best.pt`）与点②（动作侧特征开关）——两个都是"改口径/改对照"的改动。

点③错在**默认值悄悄变**：`--rule-eval-seeds` 默认必须是 1（老行为），只有显式传才变。
点②错在**关掉特征却改了宽度**：那会让老存档装不上、缓存键漂。
"""
import inspect

import pytest
import torch

import guandan.rl.selfplay as sp
from guandan.sim import env, features, meld

A = meld.cid_from_name


def _fake_train(seen):
    def f(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}
    return f


# ---------------------------------------------------------------- 点③

def test_rule_eval_seeds_default_is_one():
    """默认必须还是 1 种子（老行为）—— 口径不许悄悄变。"""
    assert inspect.signature(sp.train).parameters["rule_eval_seeds"].default == 1
    assert inspect.signature(sp.train_parallel).parameters["rule_eval_seeds"].default == 1
    seen = {}
    orig = sp.train
    sp.train = _fake_train(seen)
    try:
        sp.main(["1"])
        assert "rule_eval_seeds" not in seen
        sp.main(["1", "--rule-eval-seeds", "3"])
        assert seen["rule_eval_seeds"] == 3
    finally:
        sp.train = orig


def test_maybe_eval_uses_multiple_seeds_and_averages(monkeypatch, tmp_path):
    """`rule_seeds>1` 时必须**跑多个种子并取均值**（这就是压选点偏差的那一手）。"""
    seen_seeds = []

    def fake_match(pol, opp, games=200, seed=0, **kw):
        seen_seeds.append(seed)
        return {1004: 0.60, 1005: 0.70, 1006: 0.80}.get(seed, 0.5)

    monkeypatch.setattr(sp, "match", fake_match)
    monkeypatch.setattr(sp, "net_play", lambda n: n)
    monkeypatch.setattr(sp, "rule_policy", lambda: None)
    monkeypatch.setattr(sp, "random_policy", lambda rng: None)
    monkeypatch.setattr(sp, "greedy_policy", None)
    net = torch.nn.Linear(2, 1)
    curve = []
    best = sp._maybe_eval(net, 0, curve, -1.0, 4, 1, 1, str(tmp_path), lambda *_: None,
                          snap_every=0, pool_size=0, rule_games=4, rule_seeds=3)
    # 前两个是 vs 随机(1001) / vs 贪心(1002)；规则式这把必须是 1004,1005,1006
    assert [s for s in seen_seeds if s >= 1004] == [1004, 1005, 1006]
    assert best == pytest.approx((0.60 + 0.70 + 0.80) / 3)
    assert curve[-1][3] == pytest.approx(0.70)


# ---------------------------------------------------------------- 点②

def test_consequence_switch_keeps_the_width_and_zeroes_the_block(monkeypatch):
    """关掉特征：**宽度不变**（`ACTION_DIM` 还是 146），那 3 维恒 0、前 143 维不变。"""
    hand = [A("S3"), A("H3"), A("D3"), A("S5")]
    m = meld.as_meld([A("S3"), A("H3"), A("D3")], 2)
    on = env.encode_action_now(m, 2, hand)
    assert on.shape == (env.ACTION_DIM,) and on[env.ACTION_BASE_DIM] == 1.0
    monkeypatch.setattr(features, "CONSEQUENCE_ENABLED", False)
    off = env.encode_action_now(m, 2, hand)
    assert off.shape == (env.ACTION_DIM,), "关掉特征**不许**改宽度"
    assert off[env.ACTION_BASE_DIM:].sum() == 0.0
    assert (off[:env.ACTION_BASE_DIM] == on[:env.ACTION_BASE_DIM]).all()


def test_worker_cfg_carries_the_consequence_switch():
    """多进程那条路也要能关（worker 自己编动作，口径必须与 learner 一致）。"""
    from guandan.rl import worker
    cfg = worker.worker_cfg(1, 0.1, 0.5, 0.8, 4, consequence=False)
    assert cfg["consequence"] is False
    assert worker.worker_cfg(1, 0.1, 0.5, 0.8, 4)["consequence"] is True
