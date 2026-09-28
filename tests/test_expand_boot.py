"""`expand(n=)` 产出的自举源必须是**真的** `s_{t+n}`，而且与 points 对齐。"""
import random

import numpy as np
import pytest

from net.sim import env, rules
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
    assert all(b is None or len(b) == 4 for b in boot), "源是 (obs, acts, hist, 出手人)"


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
            # ⚠️ `b is None` 还有一种原因：视角/对手过滤（见 _boot_source_ok）。
            # 所以这里只查「非空时**必须**是那个局面」，不查「必须非空」——
            # 「必须非空」由 `test_odd_n_barely_bootstraps...` 按比例管。
            if b is None:
                continue
            assert np.array_equal(b[0], obs), f"obs 对不上（t={t} n={n}）"
            assert np.array_equal(b[2], hist), f"历史对不上（t={t} n={n}）"


def test_negative_n_raises():
    with pytest.raises(ValueError):
        replay.expand(_rec(), n=-1)


def _seats_and_boots(rec, n):
    """手工重放一遍，返回每一步的出手人，以及 `expand` 在同样参数下的 boot。"""
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    seats = []
    for i in rec.actions:
        seats.append(e.hand.turn)
        e.step(i)
    _pts, _y, boot = replay.expand(rec, n=n)
    return seats, boot


def test_boot_never_comes_from_the_other_team():
    """⚠️ **视角**：`Q(s,a)` 是「**出手人那一队**」的收益，而 `V(s_{t+n})` 取的是
    `s_{t+n}` 处出手人的视角。出手顺序是 0→3→2→1，所以 n 是奇数时那个位置
    在**对家** —— 不筛掉的话自举项的符号是反的，`beta*R + (1-beta)*V` 两项互相抵消，
    标签被往 0 拉（2026-09-28 评审抓到的 Critical；实测 n=3 时 corr(V, 标签) = -0.15
    而 n=2/4 是 +0.27）。
    """
    for n in (1, 2, 3, 4):
        for seed in (0, 5):
            rec = _rec(seed=seed)
            kept = [t for t, ok in enumerate(_kept_mask(rec)) if ok]
            seats, boot = _seats_and_boots(rec, n)
            for j, b in enumerate(boot):
                if b is None:
                    continue
                src_seat = b[3]
                assert src_seat == seats[kept[j] + n], "第 4 位必须是自举源的出手人"
                assert rules.TEAM[src_seat] == rules.TEAM[seats[kept[j]]], (
                    f"自举源落在对家（n={n}，第 {kept[j]} 步）—— 符号会反")


def test_odd_n_barely_bootstraps_and_even_n_mostly_does():
    """奇偶这把闸的**量化**后果 —— 所以默认值必须是偶数。"""
    def hit_rate(n, games=6):
        hits = tot = 0
        for seed in range(games):
            _pts, _y, boot = replay.expand(_rec(seed=seed), n=n)
            tot += len(boot)
            hits += sum(1 for b in boot if b is not None)
        return hits / max(1, tot)

    assert hit_rate(1) < 0.25 and hit_rate(3) < 0.25
    assert hit_rate(2) > 0.8 and hit_rate(4) > 0.8


def test_boot_never_comes_from_the_fixed_opponent():
    """固定对手那两家的状态**不是我们的值** —— 与 `expand` 过滤 points 同一个道理
    （「拿它当老师」）。`learn=(0, 2)` 时自举源只许是 0 或 2。"""
    rec = _rec(seed=1)
    from dataclasses import replace
    rec = replace(rec, learn=(0, 2))
    for n in (2, 4):
        _pts, _y, boot = replay.expand(rec, n=n)
        for b in boot:
            if b is not None:
                assert b[3] in (0, 2), f"自举源是固定对手（座位 {b[3]}）"
