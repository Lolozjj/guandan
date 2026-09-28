"""`expand(n=)` 产出的自举源必须是**真的** `s_{t+n}`，而且与 points 对齐。"""
import random

import numpy as np
import pytest

from net.sim import env
from train import replay
from train.policies import greedy_policy


def _rec(seed=0, level=5):
    rec, _pts, _y = replay.play_capturing(greedy_policy, random.Random(seed),
                                          level=level, capture=False)
    return rec


def _kept_mask(rec):
    """哪些步是「保留的决策点」（`learn` 为 None 时四家都保留）。"""
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    out = []
    for i in rec.actions:
        out.append(learn is None or e.hand.turn in learn)
        e.step(i)
    return out


def test_n_zero_returns_no_boot_and_the_old_points():
    """`n=0`（默认）与老行为逐点相等 —— `boot` 是空表。"""
    rec = _rec()
    pts, y, boot = replay.expand(rec)
    assert boot == []
    assert len(pts) == len(y) == len(rec.actions)


def test_boot_is_aligned_with_points():
    rec = _rec()
    pts, y, boot = replay.expand(rec, n=3)
    assert len(boot) == len(pts) == len(y)
    assert any(b is not None for b in boot), "至少有一个点该有自举源"
    assert all(b is None or len(b) == 3 for b in boot), "源是 (obs, acts, hist)"


def test_boot_is_really_the_state_n_steps_later():
    """**不靠自证**：另外手工走一遍 env，比对第 t+n 步的 obs 与历史。

    ⚠️ 这条是本次改动的核心正确性 —— 数错一步（t+n-1 或 t+n+1）在训练里
    只会表现为「学得慢一点」，任何 loss 曲线都看不出来。
    """
    rec = _rec(seed=3)
    # 手工重放，逐步记下「每一步开始前」的 (obs, hist)
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    steps = []
    for i in rec.actions:
        steps.append((e.observe(), env.encode_history(e.hand, e.hand.turn)))
        e.step(i)
    kept = [t for t, ok in enumerate(_kept_mask(rec)) if ok]
    for n in (1, 2, 3):
        _pts, _y, boot = replay.expand(rec, n=n)
        for j, b in enumerate(boot):
            t = kept[j]
            if t + n >= len(steps):
                assert b is None, f"越过终局那一步该退回 MC（t={t} n={n}）"
                continue
            obs, hist = steps[t + n]
            assert b is not None, f"该有自举源（t={t} n={n}）"
            assert np.array_equal(b[0], obs), f"obs 对不上（t={t} n={n}）"
            assert np.array_equal(b[2], hist), f"历史对不上（t={t} n={n}）"


def test_negative_n_raises():
    with pytest.raises(ValueError):
        replay.expand(_rec(), n=-1)
