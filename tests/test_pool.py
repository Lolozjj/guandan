"""对手池的记账与采样 —— 纯函数，不碰进程、不碰 torch。

规格：`docs/superpowers/specs/2026-09-27-opponent-pool-design.md` §3.2。

**为什么是 `p·(1−p)`**：`p` 是学习者对该成员的胜率。`p → 1`（已经打穿）没信息，
`p → 0`（完全打不过）没梯度，**只有接近五五开才学得到东西**。
两头都趋 0，再加一个均匀下限防饿死、兜住预热期与陈旧的胜率估计。
"""
import random

import pytest

from train import pool


def test_pfsp_peaks_at_even_matchups():
    """专挑五五开的：p=0.5 权重最大，打穿/打不过的两头都小。"""
    w = pool.pfsp_weights({0: 0.5, 1: 0.05, 2: 0.95, 3: 0.5},
                          games={0: 999, 1: 999, 2: 999, 3: 999})
    assert w[0] == pytest.approx(w[3])
    assert w[0] > w[1] and w[0] > w[2]
    assert sum(w.values()) == pytest.approx(1.0)


def test_pfsp_keeps_a_uniform_floor_at_both_extremes():
    """两头都趋 0 但**不为 0**（防饿死），且两个极端对称。"""
    w = pool.pfsp_weights({0: 1.0, 1: 0.0}, games={0: 999, 1: 999})
    assert w[0] > 0 and w[1] > 0
    assert w[0] == pytest.approx(w[1], rel=1e-9)


def test_pfsp_is_uniform_during_warmup():
    """局数不够的成员按**均匀**算 —— 它的 p 还是噪声（§3.2 预热）。"""
    w = pool.pfsp_weights({0: 0.5, 1: 0.0}, games={0: 999, 1: 3}, min_games=20)
    assert w[0] == pytest.approx(w[1])


def test_winrate_window_forgets_the_past():
    """胜率必须滑窗 —— 学习者在变强，老胜率会过期（§3.2）。"""
    wr = pool.WinRates(window=10)
    for _ in range(10):
        wr.record(0, True)                     # 先连赢 10 局
    assert wr.rate(0) > 0.8
    for _ in range(10):
        wr.record(0, False)                    # 再连输 10 局，窗口里只剩这些
    assert wr.rate(0) < 0.2, "窗口没把老的胜绩冲掉"


def test_pick_opponent_reserves_a_share_for_greedy():
    """混合局里贪心拿**固定份额**，且不进 PFSP（它是判据的尺子 + 防池子退化）。"""
    rng = random.Random(0)
    ids = [0, 1, 2]
    w = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    picks = [pool.pick_opponent(rng, ids, w, greedy_share=0.2) for _ in range(2000)]
    n_greedy = sum(1 for p in picks if p == ("greedy",))
    assert 0.16 < n_greedy / 2000 < 0.24, f"贪心份额 {n_greedy / 2000:.2%}"
    assert any(p[0] == "member" for p in picks), "池成员一次都没被抽到"


def test_pick_opponent_does_not_fall_back_to_greedy_before_the_first_broadcast():
    """还没收到第一次 PFSP 广播（权重全 0）时，按**均匀抽一个成员**，别退回贪心。

    退回贪心会让「池子刚起步那一段」偷偷变成纯贪心局，而日志上看不出来。
    """
    rng = random.Random(0)
    picks = [pool.pick_opponent(rng, [0, 1], {}, greedy_share=0.0) for _ in range(200)]
    assert all(p[0] == "member" for p in picks), "权重未到时退回了贪心"
    assert {p[1] for p in picks} == {0, 1}, "只抽到了一个成员"


def test_effective_member_count_flags_a_collapsed_pool():
    """池子塌成一个成员要**看得出来**（§1.4）。"""
    assert pool.effective_members({0: 1.0}) == pytest.approx(1.0)
    assert pool.effective_members({0: 0.5, 1: 0.5}) == pytest.approx(2.0)
    assert pool.effective_members({0: 0.9, 1: 0.1}) < 1.3


def test_record_carries_the_opponent():
    """记录里要能读出「这一局打的是谁」—— PFSP 靠它归因（§3.2）。

    ⚠️ `opp` 的第二个元素**口径不统一**：对 `member` 是成员 id，对其余是队号。
    所以判类型要先看 `opp[0]`。
    """
    import torch
    from train import selfplay
    from train.net import QNet
    torch.manual_seed(0)
    out = selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=8,
                                  capture=False, opp_mix=1.0, greedy_share=1.0)
    assert all(rec.opp and rec.opp[0] == "greedy" for rec, _p, _y in out)
    out = selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=4,
                                  capture=False, opp_mix=0.0)
    assert all(rec.opp is None for rec, _p, _y in out)
