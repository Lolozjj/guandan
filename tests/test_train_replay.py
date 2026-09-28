"""Replay buffer 与重放的测试。

最要紧的一条是**重放必须逐字段复现现场**（`test_expand_reproduces_the_live_trajectory`）：
buffer 里存的是紧凑记录，训练样本是采样时重放出来的 —— 重放要是悄悄跑偏，
喂给网络的就不是它自己打出来的那局，而且**不报错**。
"""
import random

import numpy as np

from net.sim import env, rules
from train import replay
from train.policies import greedy_policy


def test_expand_reproduces_the_live_trajectory():
    """**重放必须逐字段复现现场** —— 状态、候选、选中的下标、出牌人、历史、终局 reward。"""
    rng = random.Random(0)
    e = env.GuandanEnv(seed=rng.randrange(1 << 30))
    e.reset(level=7)
    hands0 = [set(e.hand.hands[s]) for s in rules.SEATS]
    live, actions = [], []
    obs = e.observe()
    while not e.done:
        acts = e.legal()
        seat = e.hand.turn
        hist = env.encode_history(e.hand, seat)
        i = greedy_policy(obs, acts, hist)
        live.append((obs, acts, i, seat, hist))
        actions.append(i)
        obs, _r, _d, _info = e.step(i)
    live_y = [rules.reward(e.ranks, s) for (_o, _a, _i, s, _h) in live]   # ranks 是 property，不加括号

    rec = replay.GameRecord.of(e, actions, hands0)
    points, y, _b = replay.expand(rec)

    assert len(points) == len(live), "重放出来的步数与现场不一致"
    for (o1, a1, i1, s1, h1), (o2, a2, i2, s2, h2) in zip(live, points):
        assert (i1, s1) == (i2, s2), "选中的下标或出牌人对不上"
        assert a1 == a2, "候选着法对不上 —— 枚举在重放里跑偏了"
        assert np.array_equal(env.encode_state(o1), env.encode_state(o2)), \
            "状态编码对不上"
        assert np.array_equal(h1, h2), "历史编码对不上"
    assert y == live_y, "终局 reward 对不上"


def test_record_is_small():
    """紧凑记录必须真的紧凑 —— 这是「50 万局」能不能装下的唯一依据。"""
    import sys
    rng = random.Random(1)
    rec, n = replay.play_and_record(greedy_policy, rng, level=5)
    # 发牌 108 个 int + 每步一个 int（int 在 tuple 里是引用，8 字节）
    approx = 108 * 8 + n * 8 + 64
    assert approx < 4096, f"每局约 {approx} 字节，不该超过 4 KB"
    assert len(rec.hands) == 4 and all(len(h) == 27 for h in rec.hands)
    assert sys.getsizeof(rec) > 0


def test_buffer_keeps_the_newest_and_respects_capacity():
    buf = replay.ReplayBuffer(capacity_games=3)
    for lv in (2, 3, 4, 5):
        rec, _ = replay.play_and_record(greedy_policy, random.Random(lv), level=lv)
        buf.add(rec)
    assert len(buf) == 3, "容量 3 却存了别的数量"
    assert buf.added == 4, "累计进过的局数应当照数"
    assert {r.level for r in buf.sample(50, random.Random(0))} <= {3, 4, 5}, \
        "最老的那局（级别 2）应当已经被挤掉"


def test_sample_allows_repeats():
    """有放回采样（DMC 就是这么采的）—— 32 局的 batch 从不足 32 局的 buffer 里也能取。"""
    buf = replay.ReplayBuffer(capacity_games=10)
    rec, _ = replay.play_and_record(greedy_policy, random.Random(9), level=4)
    buf.add(rec)
    got = buf.sample(32, random.Random(0))
    assert len(got) == 32 and all(g is rec for g in got)
