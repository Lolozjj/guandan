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

    **所有局同步推进**（各走一步、攒成一批再问网络），理由与自对弈那边一样：
    一次前向只算一个局面时 GPU 的批处理完全浪费（spec §14.3）。
    带 `batch_choose` 的策略（`train/policies.py` 的 `batch_net_policy`）会被批量调用；
    随机 / 贪心这类便宜的策略仍然逐决策点调用。

    `level=None` 时每局随机 1..13（与训练时的分布一致）。
    """
    rng = random.Random(seed)
    envs, a_on_team0 = [], []
    for i in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset(level=level)
        envs.append(e)
        a_on_team0.append(i % 2 == 0)          # 座位对调

    alive = list(range(games))
    while alive:
        pending = []
        for k in alive:
            e = envs[k]
            pending.append((e.observe(), e.legal(),
                            env.encode_history(e.hand, e.hand.turn)))

        groups = {}
        for j, (obs, acts, hist) in enumerate(pending):
            seat = envs[alive[j]].hand.turn
            on_a = (rules.TEAM[seat] == 0) == a_on_team0[alive[j]]
            groups.setdefault(on_a, []).append(j)

        picks = [None] * len(pending)
        for on_a, js in groups.items():
            pol = policy_a if on_a else policy_b
            bc = getattr(pol, "batch_choose", None)
            if bc is not None:
                got = bc([pending[j] for j in js])
            else:
                got = [pol(*pending[j]) for j in js]
            for j, idx in zip(js, got):
                picks[j] = idx

        nxt = []
        for k, idx in zip(alive, picks):
            envs[k].step(idx)
            if not envs[k].done:
                nxt.append(k)
        alive = nxt

    wins = sum(1 for k in range(games)
               if (rules.winner_team(envs[k].ranks) == 0) == a_on_team0[k])
    return wins / games


def win_rate_vs(policy, opponent, games: int = 200, seed: int = 0,
                level: int = None) -> float:
    """`policy` 对 `opponent` 的胜率 —— 名字更直白的包装，判据那几行读起来顺。"""
    return match(policy, opponent, games=games, seed=seed, level=level)
