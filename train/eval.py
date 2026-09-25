"""评测：两支策略对打，返回「a 队」的胜率。

spec §7 的四行判据里，前三行（vs 随机 / vs 贪心 / vs 人类着法）需要的都是这件事，
所以只写一份。

**座位每局对调**（偶数局 a 坐 0/2、奇数局 a 坐 1/3）：四个座位共享一套权重、
状态也按出牌人相对化，理论上不该有座位偏置 —— 但对调是**零成本的保险**，
而且「测出来的胜率」将来要被当成结论引用的，不该留一个没人验过的假设。
"""
from __future__ import annotations

import random

from net.sim import env, rules


def match(policy_a, policy_b, games: int = 200, seed: int = 0,
          level: int = None) -> float:
    """a 队对 b 队的胜率（`games` 局，座位对调）。

    `level=None` 时每局随机 1..13（与训练时的分布一致）。
    """
    rng = random.Random(seed)
    wins = 0
    for i in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset(level=level)
        a_on_team0 = (i % 2 == 0)
        obs = e.observe()
        while not e.done:
            seat = e.hand.turn
            acts = e.legal()
            # hist 由这里算（策略自己拿不到明牌，见 train/policies.py 的说明）
            hist = env.encode_history(e.hand, seat)
            on_a = (rules.TEAM[seat] == 0) == a_on_team0
            i_act = (policy_a if on_a else policy_b)(obs, acts, hist)
            obs, _r, _d, _info = e.step(i_act)
        winner = rules.winner_team(e.ranks)
        if (winner == 0) == a_on_team0:
            wins += 1
    return wins / games


def win_rate_vs(policy, opponent, games: int = 200, seed: int = 0,
                level: int = None) -> float:
    """`policy` 对 `opponent` 的胜率 —— 名字更直白的包装，判据那几行读起来顺。"""
    return match(policy, opponent, games=games, seed=seed, level=level)
