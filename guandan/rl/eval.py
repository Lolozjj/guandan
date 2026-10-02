"""评测：两支策略对打，返回「a 队」的胜率。

spec §7 的四行判据里，前三行（vs 随机 / vs 贪心 / vs 人类着法）需要的都是这件事，
所以只写一份。

**座位每局对调**（偶数局 a 坐 0/2、奇数局 a 坐 1/3）：四个座位共享一套权重、
状态也按出牌人相对化，理论上不该有座位偏置 —— 但对调是**零成本的保险**，
而且「测出来的胜率」将来要被当成结论引用的，不该留一个没人验过的假设。
"""
from __future__ import annotations

import random

from guandan.sim import env, rules


def match(policy_a, policy_b, games: int = 200, seed: int = 0,
          level: int = None, expand_a: bool = False, search_a=None) -> float:
    """a 队对 b 队的胜率（`games` 局，座位对调）。

    **所有局同步推进**（各走一步、攒成一批再问网络），理由与自对弈那边一样：
    一次前向只算一个局面时 GPU 的批处理完全浪费（spec §14.3）。
    带 `batch_choose` 的策略（`guandan/rl/policies.py` 的 `batch_net_policy`）会被批量调用；
    随机 / 贪心这类便宜的策略仍然逐决策点调用。

    `level=None` 时每局随机 1..13（与训练时的分布一致）。

    `expand_a=True`：**只给 `policy_a` 那一侧**的候选补花色变体（A1b 实验，
    `env.expand_seats`）—— `policy_b`（尺子）保持老候选，所以测的是
    「多给选项值多少」，不是「换了套规则」。

    `search_a`：可选的**推理时搜索**（见 `rl/search.py` 的 `one_step_backup`）。
    给了之后 `policy_a` 那一侧改由它选动作（批量接口：
    `search_a([(env, (obs, acts, hist)), ...]) -> [下标, ...]`）。
    它**只影响选动作，不改权重** —— 所以这是"多算一次前向"值多少的直接测量。
    """
    rng = random.Random(seed)
    envs, a_on_team0 = [], []
    for i in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset(level=level)
        if expand_a:
            team = 0 if i % 2 == 0 else 1
            e.expand_seats = frozenset(s for s in rules.SEATS if rules.TEAM[s] == team)
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
            if on_a and search_a is not None:
                got = search_a([(envs[alive[j]], pending[j]) for j in js])
            else:
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


def bomb_waste(policy, games: int = 40, seed: int = 0, opponent=None):
    """**炸弹浪费率**：在「能用普通牌压过桌面」的局面里，策略却选了炸弹的比例。

    返回 `(浪费次数, 能用普通牌压的总次数)`。

    `opponent=None` 时**四家都用 `policy`**（自己打自己）；给了就只让 `policy`
    打一队（逐局换边），另一队交给它。

    ⚠️ 这两个数**不一样，而且都要看**（实测 2026-09-26）：同一版权重自己打自己
    是 6%，而面对贪心是 17% —— 人不爱炸，模型对不爱炸的对手更爱用炸，
    **用户看到的是后一个数**。别只量前一个就下结论。

    为什么要有这个尺子：`vs 贪心` 看不见它 —— **贪心从不主动炸**，
    所以「有普通牌却出炸」在胜率上几乎不受惩罚。用户 2026-09-26 实机发现
    这个毛病（88 个级别正确的决策点里 13 次），根因是**只跟自己打**：
    自对弈里对手也爱炸，「不炸就被炸」成了均衡，于是浪费从来没被罚过。
    """
    return _bomb_stats(policy, games, seed, opponent)[:2]


def bomb_opportunity(has_table, acts) -> bool:
    """这一步有没有「用普通牌压」的机会：**桌上有牌要压**、且**手里有非炸弹候选**。

    `acts` 已经按 `beats` 过滤过，所以「非炸弹候选」就是能压的普通牌。
    """
    return bool(has_table) and any(x is not None and not x.is_bomb for x in acts)


def is_wasted_bomb(has_table, acts, chosen) -> bool:
    """**白炸**：有机会用普通牌压，却用了炸弹 —— 「炸弹浪费率」的唯一判定。

    ⚠️ **只此一份**：`_bomb_stats`（统计口径）与 `tools/show_game.py`（战报里标出
    具体哪一手）共用它。两处各写一份就是「副本会漂」，而这条正是用户最初的抱怨
    （「有普通牌能压却出炸」），口径漂了会把人骗得很惨。
    """
    return (bomb_opportunity(has_table, acts)
            and chosen is not None and chosen.is_bomb)


def _bomb_stats(policy, games: int = 40, seed: int = 0, opponent=None):
    """跑 N 局，**一次**量出四样：`(浪费数, 能压的普通牌机会数, 用炸手数, 局数)`。

    `bomb_waste` 取前两个、`bomb_rate` 取后两个 —— **一次走局、两个视图**。
    两处各写一份走局就是「副本会漂」（本仓库为此反复吃过亏）。

    `opponent` 的座位轮换与 `bomb_waste` 的老口径一致（逐局换边）；
    `mine` 为真时才计数 —— 对手的炸弹不算在策略头上。
    """
    waste = chance = bombs = 0
    rng = random.Random(seed)
    for game in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        while not e.done:
            obs, acts = e.observe(), e.legal()
            hist = env.encode_history(e.hand, e.hand.turn)
            seat = e.hand.turn
            mine = opponent is None or (seat % 2) == (0 if game % 2 == 0 else 1)
            pol = policy if mine else opponent
            i = pol(obs, acts, hist)
            m = acts[i]
            if mine:
                if m is not None and m.is_bomb:
                    bombs += 1
                if bomb_opportunity(obs.table, acts):
                    chance += 1
                    if is_wasted_bomb(obs.table, acts, m):
                        waste += 1
            e.step(i)
    return waste, chance, bombs, games


def bomb_rate(policy, games: int = 40, seed: int = 0, opponent=None):
    """**用炸率**：策略主动炸了多少手。返回 `(炸的手数, 局数)`。

    ⚠️ 为什么必须和 `bomb_waste` 一起看：挂了炸弹代价最典型的失败**不是没效果，
    是「从此一刀切不炸」** —— 那不比乱炸好，而且用户看面板会觉得建议变蠢。
    `bomb_waste` 只回答「有得选的时候选错了吗」，回答不了「还炸不炸」。
    """
    return _bomb_stats(policy, games, seed, opponent)[2:]


# ---------------------------------------------------------------- 万能牌（逢人配）
#
# 2026-10-01 用户实机反馈的第二类毛病：「明明 8888 就够，他要用万能牌凑成 88888」。
# 与「白炸」同构：**桌上有牌要压**（所以每个合法候选都能压过）+ **存在不用万能牌的候选**
# ⇒ 用万能牌就不是必需的。`Meld.wild_used` 是引擎给的口径（含补上的逢人配张数），
# 不用自己数牌，也就不会数错。

def wild_opportunity(has_table, acts) -> bool:
    """这一步有没有「不用万能牌也能压」的机会：桌上有牌 + 有 `wild_used == 0` 的候选。"""
    return bool(has_table) and any(x is not None and not x.wild_used for x in acts)


def is_wasted_wild(has_table, acts, chosen) -> bool:
    """**白用万能牌**：有机会不用万能牌压，却用了（`wild_used > 0`）。"""
    return (wild_opportunity(has_table, acts)
            and chosen is not None and bool(chosen.wild_used))


def is_upgraded_wild(has_table, acts, chosen) -> bool:
    """**过度升级**（用户举的那一手）：用万能牌把**同一牌型**做得更大。

    判定：有机会不用万能牌压，却用了；而且候选里**存在同牌型且不用万能牌**的一手
    （例如天然 8888 就在候选里，它却打了 8+8+8+8+逢人配 的 5 张炸）。
    """
    if not is_wasted_wild(has_table, acts, chosen):
        return False
    return any(x is not None and not x.wild_used and x.kind == chosen.kind for x in acts)


def is_wild_enlarged_bomb(has_table, acts, chosen) -> bool:
    """用户原话那一手：「炸个 8888 就行了，却用万能牌凑成五个 8」。

    判定：选了**炸弹且用了万能牌**，而候选里**存在不用万能牌的炸弹**
    （⇒ 天然炸弹就够压，万能牌是白搭进去把它做大的）。
    """
    if chosen is None or not chosen.is_bomb or not chosen.wild_used:
        return False
    return any(x is not None and x.is_bomb and not x.wild_used for x in acts)


def _wild_stats(policy, games: int = 40, seed: int = 0, opponent=None):
    """跑 N 局，一次量出万用牌三样：`(浪费, 过度升级, 机会, 用万能牌手数, 局数)`。

    与 `_bomb_stats` 共用同一套走局/换边口径（**别各写一份**）。
    """
    waste = upgrade = chance = used = 0
    rng = random.Random(seed)
    for game in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        while not e.done:
            obs, acts = e.observe(), e.legal()
            hist = env.encode_history(e.hand, e.hand.turn)
            seat = e.hand.turn
            mine = opponent is None or (seat % 2) == (0 if game % 2 == 0 else 1)
            pol = policy if mine else opponent
            i = pol(obs, acts, hist)
            m = acts[i]
            if mine:
                if m is not None and m.wild_used:
                    used += 1
                if wild_opportunity(obs.table, acts):
                    chance += 1
                    if is_wasted_wild(obs.table, acts, m):
                        waste += 1
                    if is_upgraded_wild(obs.table, acts, m):
                        upgrade += 1
            e.step(i)
    return waste, upgrade, chance, used, games


def wild_waste(policy, games: int = 40, seed: int = 0, opponent=None):
    """**万能牌浪费率**：有机会不用万能牌压，却用了。返回 `(浪费, 机会)`。"""
    return _wild_stats(policy, games, seed, opponent)[:2]


def wild_upgrade(policy, games: int = 40, seed: int = 0, opponent=None):
    """**万能牌过度升级率**：用万能牌把同一牌型做得更大。返回 `(过度升级, 机会)`。"""
    w = _wild_stats(policy, games, seed, opponent)
    return w[1], w[2]


def wild_use_rate(policy, games: int = 40, seed: int = 0, opponent=None):
    """**用万能牌率**：主动用了几手。返回 `(手数, 局数)`。与 `bomb_rate` 对称。"""
    w = _wild_stats(policy, games, seed, opponent)
    return w[3], w[4]


def win_rate_vs(policy, opponent, games: int = 200, seed: int = 0,
                level: int = None) -> float:
    """`policy` 对 `opponent` 的胜率 —— 名字更直白的包装，判据那几行读起来顺。"""
    return match(policy, opponent, games=games, seed=seed, level=level)
