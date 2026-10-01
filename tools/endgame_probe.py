"""点④的**便宜预判**：残局里到底有没有"网络没学到、但可以被学到"的东西？

背景（A6b 的读数）：进残局（所有手牌 ≤8）时手握出牌权也只有 **67%** 赢；
输局里最大一类是**胶着局**。所以残局是唯一还有明确弱点靶子的地方。

**这一版只做预判，不做训练**。做法：

1. 用现役权重打若干局（我方 = 网络 argmax，对手 = 规则式），
   在每个"所有手牌 ≤ `--endgame-n`"的**第一个时刻**抓快照（deepcopy）。
2. 对每个残局局面，取网络 Q 的前 K 个候选；对每个候选做 M 次
   **确定性 rollout**：把没露面的牌随机分给另外三家（张数与公开信息一致），
   然后让我方继续用网络、对手用规则式把这一局打完 ⇒ 得到该候选的均值回报。
3. 报三件事：**网络的 top-1 与 rollout 的 top-1 一致率**、
   **网络 top-1 相对 rollout-best 的价值损失（点）**、以及 rollout-best 是否根本不在网络前 K 里。

**判据（起跑之前定）**：价值损失 < **0.05 点**（一局典型价值 ≈2 点）⇒ **杀**
（残局没有可学的结构，别再往这个方向投算力）；≥ 0.2 点 ⇒ 值得为残局单独造标签。

    .venv/Scripts/python.exe -m tools.endgame_probe --games 60 --positions 40
"""
from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys

import numpy as np

from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, features, rules

DEFAULT_ENDGAME_N = 8


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _hidden_seats(seat: int) -> list:
    return [s for s in rules.SEATS if s != seat]


def _unseen_ids(obs) -> list:
    """**没露面**的牌（真实 card id）——另外三家手上只能是这些。

    ⚠️ 不能用 `features.unseen_pool` 的**点数直方图**去发牌：那会把"点数"当"牌"发出去，
    而引擎要的是 card id（`cards.parts(1)` 直接 KeyError）。公开信息里"哪些点数的牌没露面"
    与我们这里需要的"哪些**牌**没露面"是同一批，但**表示不同**。
    """
    seen = set(obs.hand)
    for s in rules.SEATS:
        seen |= set(obs.played[s])
    return [cid for cid, _idx in features._DECK if cid not in seen]


def _determinize(e, obs, rng) -> None:
    """把没露面的牌**随机**分给另外三家（张数与 `obs.left` 一致）——就地改 `e`。"""
    pool = _unseen_ids(obs)
    need = sum(obs.left[s] for s in _hidden_seats(obs.seat))
    if len(pool) != need:
        raise ValueError(
            f"未见牌 {len(pool)} 张 != 三家手牌合计 {need} 张 —— 公开信息不自洽")
    rng.shuffle(pool)
    k = 0
    for s in _hidden_seats(obs.seat):
        n = obs.left[s]
        e.hand.hands[s] = set(pool[k:k + n])
        k += n


def _rollout_value(net, e_snap, k_act: int, rng, my_team: int, max_steps: int = 200,
                   determinize: bool = True) -> float:
    """从快照走一步 `k_act`，之后我方用网络、对手用规则式，返回**我方终局回报**。

    `determinize=True`：先把暗牌随机分一遍（用于**估计**候选价值）。
    `determinize=False`：用**真实**暗牌（用于**干预**实验 —— 与真实对局同一副牌，可配对）。
    """
    e = copy.deepcopy(e_snap)
    if determinize:
        _determinize(e, e.observe(), rng)
    acts = e.legal()
    if not 0 <= k_act < len(acts):
        raise IndexError(f"候选下标 {k_act} 越界（{len(acts)} 个）")
    seat0 = e.hand.turn
    e.step(k_act)
    steps = 0
    while not e.done and steps < max_steps:
        seat = e.hand.turn
        obs, acts = e.observe(), e.legal()
        if rules.TEAM[seat] == my_team:
            hist = env.encode_history(e.hand, seat)
            i = int(np.argmax(q_values(net, obs, acts, hist)))
        else:
            i = rule_choose(obs, acts)
        e.step(i)
        steps += 1
    if not e.done:
        return 0.0                              # 走不完就当中性（不该发生）
    return float(rules.reward(e.ranks, seat0))


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="残局预判：确定性 rollout vs 网络排序")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=3, help="只看网络 Q 的前 K 个候选")
    ap.add_argument("--samples", type=int, default=8, help="每个候选的确定性采样次数")
    ap.add_argument("--endgame-n", type=int, default=DEFAULT_ENDGAME_N)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    net = load(a.weights)
    rng = random.Random(a.seed)
    snaps: list = []
    games_done = 0
    while len(snaps) < a.positions and games_done < a.games:
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        team0 = games_done % 2                        # 座位对调
        games_done += 1
        taken = False
        while not e.done:
            seat = e.hand.turn
            mine = rules.TEAM[seat] == team0
            if (not taken and mine
                    and max(len(e.hand.hands[s]) for s in rules.SEATS) <= a.endgame_n):
                snaps.append((copy.deepcopy(e), team0))
                taken = True
            obs, acts = e.observe(), e.legal()
            if mine:
                i = int(np.argmax(q_values(net, obs, acts,
                                           env.encode_history(e.hand, seat))))
            else:
                i = rule_choose(obs, acts)
            e.step(i)
    print(f"抓到残局快照 {len(snaps)} 个（打了 {games_done} 局）")

    agree = 0
    losses = []
    picks = []
    not_in_topk = 0
    for e_snap, team0 in snaps:
        obs, acts = e_snap.observe(), e_snap.legal()
        seat = e_snap.hand.turn
        q = [float(x) for x in q_values(net, obs, acts,
                                       env.encode_history(e_snap.hand, seat))]
        order = sorted(range(len(acts)), key=lambda i: -q[i])[:a.top_k]
        vals = []
        for k in order:
            v = statistics.mean([_rollout_value(net, e_snap, k, rng, team0)
                                 for _ in range(a.samples)])
            vals.append((v, k))
        best_v, best_k = max(vals)
        net_k = order[0]
        picks.append((e_snap, team0, net_k, best_k))
        net_v = dict((k, v) for v, k in vals)[net_k]
        losses.append(best_v - net_v)
        agree += (best_k == net_k)
        if best_k not in order:
            not_in_topk += 1

    n = len(snaps)
    if not n:
        print("一个残局快照都没抓到，别下结论")
        return 0
    big = [x for x in losses if x > 0.2]
    ties = [x for x in losses if x <= 0.05]
    print(f"\n一致率（网络 top-1 == rollout top-1）：{agree}/{n} = {agree / n:.1%}")
    print(f"价值损失（rollout-best − 网络 top-1）：中位 {statistics.median(losses):+.3f} 点，"
          f"均值 {statistics.mean(losses):+.3f} 点（一局典型价值 ≈2 点）")
    print(f"  其中「损失 > 0.2 点」的位置 {len(big)}/{n} = {len(big) / n:.1%}；"
          f"「损失 ≤ 0.05 点」{len(ties)}/{n} = {len(ties) / n:.1%}")

    # ---- 干预实验：把 rollout-best **真的在真局里走一遍**（配对，同一副真实暗牌）----
    diffs = []
    for e_snap, team0, net_k, best_k in picks:
        if net_k == best_k:
            continue
        va = _rollout_value(net, e_snap, net_k, rng, team0, determinize=False)
        vb = _rollout_value(net, e_snap, best_k, rng, team0, determinize=False)
        diffs.append(vb - va)
    if diffs:
        m, sd = statistics.mean(diffs), (statistics.stdev(diffs) if len(diffs) > 1 else 0.0)
        t = m / (sd / (len(diffs) ** 0.5)) if sd > 0 else float("inf")
        print(f"\n【干预】换掉的分歧点 {len(diffs)} 个："
              f"「rollout-best − 网络选择」的终局分差 均值 {m:+.3f} 点，sd {sd:.2f}，t = {t:+.2f}")
        print("  ⇒ 用**真实暗牌**配对，这条比上面的采样估计更硬："
              "若 |t| 很小 ⇒ 网络在残局已经把该拿的分拿到了。")

    print("\n判读（预登记）：损失 < 0.05 点 ⇒ **杀**（残局没有可学的结构）；"
          "≥ 0.2 点 ⇒ 值得为残局单独造标签")
    return 0


if __name__ == "__main__":
    sys.exit(main())
