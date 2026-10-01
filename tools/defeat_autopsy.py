"""败局解剖（路线图 A6）：那 30% 的败局里，有多少是**决策可以避免**的？

做法（复用 `tools/opp_headroom.py` 的反事实装置，**不另写一份**）：

1. 学生 vs 规则式跑 N 局（座位按局号对调，与评测器同口径）；
2. **每一个败局**里，用蓄水池抽 1 个「学生与规则式分歧」的决策点；
3. 在那个点分叉：A 走学生的着法、B 走规则式的着法，**两条都用学生继续打完**
   （`_rollout` 的口径：把那一手隔离出来，其余全同），Δ = B − A（学生视角的得分差）；
4. 按**起手强度**给每局分层（`hand_partition` 的手数，越少越强）。

⚠️ 口径要说清：分叉之后**四方都换成学生**再打（规则式不再参与）。
   所以 Δ 衡量的不是"对手会怎么变"，而是"**这一手本身值多少分**"。

预登记判据（`plans/2026-09-30-beat-70-roadmap.md` §四）：

    可避免比例 > 20%  -> 决策问题（走 A2 显式特征 / A3 队友多样性）
    可避免比例 < 10%  -> 多为发牌/尺子问题（走 A6b 结构指标 / C2 换尺子）
    10%~20%           -> 两者都做，先做便宜的（A2）

跑法：
    .venv/Scripts/python.exe -m tools.defeat_autopsy --games 300
"""
from __future__ import annotations

import argparse
import collections
import copy
import math
import random
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl.net import QNet, q_values
from guandan.rl.rule_policy import hand_partition, rule_choose
from guandan.rl.selfplay import net_play
from guandan.sim import env, rules
from tools.opp_headroom import _rollout          # 反事实的 rollout 只有一份

#: 预登记判据
AVOIDABLE_HIGH = 0.20
AVOIDABLE_LOW = 0.10


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    n.load_state_dict(d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def hands_left(hands, level) -> int:
    """一队的起手「还要几手能走完」之和 —— **越小越强**。

    `hand_partition` 是贪心估、**不认逢人配**、会高估手数（它自己的 docstring 写了），
    所以这里只当**粗尺子**用：比较两队的相对强弱，不读绝对值。
    """
    return sum(len(hand_partition(sorted(h), level)) for h in hands)


def autopsy(net, games: int, seed: int) -> dict:
    pol = net_play(net)
    rng = random.Random(seed)
    #: ⚠️ 抽样用**独立的 rng**：共用 `rng` 会让每次蓄水池抽样都从发牌序列里抠掉一个数，
    #: 于是第 2 局之后的发牌与 `eval.match(seed=同)` 岔开 —— 结构数就没法与尺子对账。
    pick = random.Random(seed * 7919 + 13)
    wins = losses = 0
    names = collections.Counter()
    deltas, pairs = [], []
    strength_rows = []                   # (强度差, 是否赢)
    loss_strength = []                   # 败局里的强度差
    reasons = collections.Counter()      # 败局的终局形态
    forks = 0
    for g in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        learner_team = g % 2
        level = e.hand.level
        start = [sorted(h) for h in e.hand.hands]
        my_left = hands_left([start[s] for s in rules.SEATS if rules.TEAM[s] == learner_team], level)
        op_left = hands_left([start[s] for s in rules.SEATS if rules.TEAM[s] != learner_team], level)
        diff = op_left - my_left                     # >0 = 我方起手更强

        fork, k = None, 0
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            if rules.TEAM[seat] == learner_team:
                qs = [float(x) for x in q_values(net, obs, acts,
                                                env.encode_history(e.hand, seat))]
                i_q = max(range(len(qs)), key=qs.__getitem__)
                if len(acts) >= 2:
                    i_r = rule_choose(obs, acts)
                    if i_r != i_q:                   # 只看**分歧点**（没分歧就无事可做）
                        k += 1
                        if pick.random() < 1.0 / k:  # 蓄水池：每局等概率取一个
                            fork = (copy.deepcopy(e), i_q, i_r, seat)
                idx = i_q
            else:
                idx = rule_choose(obs, acts)
            e.step(idx)

        won = rules.reward(e.ranks, 0 if learner_team == 0 else 1) > 0
        rank_of = {s: e.ranks[s] for s in rules.SEATS}
        my_ranks = sorted(rank_of[s] for s in rules.SEATS if rules.TEAM[s] == learner_team)
        names["双上" if my_ranks == [1, 2] else
              "1-3" if my_ranks == [1, 3] else
              "2-4" if my_ranks == [2, 4] else "末游(3-4)"] += 1
        strength_rows.append((diff, won))
        if won:
            wins += 1
        else:
            losses += 1
            loss_strength.append(diff)
            if fork is not None:
                forks += 1
                fe, i_q, i_r, seat = fork
                a, b = copy.deepcopy(fe), copy.deepcopy(fe)
                a.step(i_q)
                b.step(i_r)
                ra, rb = _rollout(a, pol, seat), _rollout(b, pol, seat)
                deltas.append(rb - ra)
                pairs.append((ra, rb))
                reasons["分叉后仍输" if rb <= 0 else "分叉后翻盘"] += 1
    return {"games": games, "wins": wins, "losses": losses, "names": names,
            "deltas": deltas, "pairs": pairs, "forks": forks,
            "strength_rows": strength_rows, "loss_strength": loss_strength,
            "reasons": reasons}


def _bins(rows) -> None:
    """强度差 -> 胜率（分箱）。"""
    buckets: dict = collections.defaultdict(lambda: [0, 0])
    for diff, won in rows:
        key = max(-6, min(6, diff))          # 极值并到 ±6 桶里
        buckets[key][0] += 1
        buckets[key][1] += 1 if won else 0
    print("  强度差（对手手数−我方手数，>0 = 我方更强） -> 胜率：")
    for key in sorted(buckets):
        n, w = buckets[key]
        if n < 5:
            continue
        tag = "  <- 势均力敌" if abs(key) <= 1 else ""
        print(f"    {key:+3d}: {w:4d}/{n:4d} = {w / n:6.1%}{tag}")


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="败局解剖：30% 的败局有多少可避免")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1002)
    a = ap.parse_args(argv)

    path = a.weights or str(paths.BEST)
    net, g = load(path)
    print(f"学生：{path}" + (f"（{g:,} 局）" if g else ""))
    print(f"对局：{a.games} 局 vs 规则式（座位按局号对调）/ seed={a.seed}\n")

    r = autopsy(net, a.games, a.seed)
    print(f"【结构】学生胜 {r['wins']} / 负 {r['losses']}"
          f"（胜率 {r['wins'] / r['games']:.1%}）")
    print("  名次形态：" + "  ".join(f"{k} {v}" for k, v in r["names"].most_common()))
    print()
    _bins(r["strength_rows"])

    print(f"\n【反事实】败局 {r['losses']} 个，其中抽到分歧点并分叉 {r['forks']} 个")
    dd = r["deltas"]
    if len(dd) < 2:
        print("  样本不够，答不了。")
        return 1
    flip = sum(1 for ra, rb in r["pairs"] if ra < 0 and rb > 0) / len(dd)
    better = sum(1 for x in dd if x > 0) / len(dd)
    same = sum(1 for x in dd if x == 0) / len(dd)
    m = statistics.mean(dd)
    sd = statistics.stdev(dd)
    t = m / (sd / math.sqrt(len(dd)))
    print(f"  **单点翻盘率 {flip:.1%}**（那一手换成规则式的着法，终局从输变赢）")
    print(f"  变好 {better:.1%} / 毫无影响 {same:.1%} / 变差 {1 - better - same:.1%}")
    print(f"  平均 Δ = {m:+.3f} 点（sd {sd:.2f}，t = {t:+.2f}；一局典型价值 ≈2 点）")
    print("  分叉后仍输 " + str(r["reasons"].get("分叉后仍输", 0))
          + " / 翻盘 " + str(r["reasons"].get("分叉后翻盘", 0)))
    if r["loss_strength"]:
        weak = sum(1 for d in r["loss_strength"] if d < 0) / len(r["loss_strength"])
        print(f"  败局里我方起手更弱的占 {weak:.1%}"
              f"（起手强度差中位 {statistics.median(r['loss_strength']):+.1f}）")

    print("\n判读（预登记判据）：")
    if flip > AVOIDABLE_HIGH:
        print(f"  **可避免比例 {flip:.1%} > {AVOIDABLE_HIGH:.0%} ⇒ 决策问题**"
              f" ⇒ 走 A2（显式特征）/ A3（队友多样性）")
    elif flip < AVOIDABLE_LOW:
        print(f"  **可避免比例 {flip:.1%} < {AVOIDABLE_LOW:.0%} ⇒ 多为发牌/尺子问题**"
              f" ⇒ 走 A6b（结构指标）/ C2（换更强的尺子）")
    else:
        print(f"  **可避免比例 {flip:.1%} 落在 10%~20%** ⇒ 两者都做，先做便宜的（A2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
