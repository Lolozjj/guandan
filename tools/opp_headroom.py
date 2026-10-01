"""**「换个更强的对手 / 拿规则式当老师」还剩多少可换** —— 一条探针，答值不值得跑训练。

为什么要有它：分析（R7→R10 分析（已归档到 `master` 分支） §4.3）里挂着一条
「换更强的对手 / 模仿学习」，它当初被否的依据是「规则式太弱」，而那个数是
**尺子 bug** 造成的 —— 前提塌了，路线回到「待评估」。
但「待评估」不等于「值得跑」：跑一条臂要 2.5~9 小时。

探针分三层，一层比一层硬：

1. **难度谱** —— 学生打 随机 / 贪心 / 规则式 各是多少。
   学习信号最强在胜率 ~50%；**饱和的对手（99%）贡献≈0**。
2. **分歧率 + 分位** —— 学生与规则式在多少个决策点上选了不同的牌，
   以及规则式的着法在学生自己的 Q 里排到第几。这只答「有没有东西不一样」。
3. **反事实（决定性）** —— 在分歧点上**把局面分叉**：一条走学生的、一条走规则式的，
   **两条都用学生继续打完**，比终局得分。
   正 = 听规则式的更好。
   ⚠️ 这是唯一能真正答「拿它当 teacher 会不会变好」的一层 —— 前两层都答不了。

⚠️ **别用「Q 差」当判据**：学生选的就是 argmax，所以
`Q(学生的) − Q(规则式的) ≥ 0` 是**恒成立**的，它只反映差距大小，**证不了学生对**。
本工具把它降级成一行参考数，判读不看它。

用法：

    .venv/Scripts/python.exe -m tools.opp_headroom
    .venv/Scripts/python.exe -m tools.opp_headroom --games 400 --seed 1002
    .venv/Scripts/python.exe -m tools.opp_headroom --weights models/best.pt
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import sys

import torch

from guandan.advice import advise
from guandan.sim import env, rules
from guandan.rl.eval import match
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.policies import greedy_policy, random_policy
from guandan.rl.rule_policy import rule_choose, rule_policy
from guandan.rl.selfplay import net_play
from guandan.console import utf8_stdout

#: 反事实那一层：**正 = 听规则式的更好**。0.05 点 ≈ 一局典型价值的 2.4%
#: （实测每座位终局 |reward| 均值 2.05 点）。
MATTERS = 0.05


def load(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def rank_pct(qs, idx) -> float:
    """`qs[idx]` 在 `qs` 里排第几，归一成 **1.0 = 最好的那个、0.0 = 最差的**。

    用「比它好的有几个」数，**不是**按名次比例算 —— 后者在并列时会漂。
    `len(qs) < 2` 时返回 1.0（只有一个候选，谈不上分歧）。
    """
    if len(qs) < 2:
        return 1.0
    better = sum(1 for q in qs if q > qs[idx])
    return 1.0 - better / (len(qs) - 1)


def _rollout(e, pol, seat: int) -> float:
    """从当前局面用 `pol` 打到底，返回**做出分叉那一手的那家**所在队的终局得分。"""
    while not e.done:
        e.step(pol(e.observe(), e.legal(), env.encode_history(e.hand, e.hand.turn)))
    return rules.reward(e.ranks, seat)


def probe(net, games: int, seed: int, forks: int = None) -> dict:
    """学生（team0/team1 各半）打规则式。一次跑完三层：

    - 分歧率 / 分位 / Q 差（在**学生该走、且多候选**的点上数）
    - 反事实对（**每局等概率抽一个分歧点**分叉，避免一局多票）
    """
    pol = net_play(net)
    rng = random.Random(seed)
    dec = dis = 0
    pcts, gaps, deltas, wins = [], [], [], 0
    for g in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        learner_team = g % 2                       # 座位对调
        fork, k, alive = None, 0, True
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            if rules.TEAM[seat] == learner_team:
                qs = q_values(net, obs, acts, env.encode_history(e.hand, e.hand.turn))
                cand = [float(x) for x in qs]
                i_q = max(range(len(cand)), key=cand.__getitem__)
                if len(cand) >= 2:                 # 只有一个候选 ⇒ 没有选择，不记
                    i_r = rule_choose(obs, acts)
                    dec += 1
                    if i_r != i_q:
                        dis += 1
                        pcts.append(rank_pct(cand, i_r))
                        gaps.append(cand[i_q] - cand[i_r])
                        k += 1
                        if rng.random() < 1.0 / k:  # 蓄水池：每局等概率取一个分歧点
                            fork = (copy.deepcopy(e), i_q, i_r, seat)
                idx = i_q
            else:
                idx = rule_choose(obs, acts)
            e.step(idx)
            alive = not e.done
        if rules.winner_team(e.ranks) == learner_team:
            wins += 1
        if fork is not None:
            fe, i_q, i_r, seat = fork
            a, b = copy.deepcopy(fe), copy.deepcopy(fe)
            a.step(i_q)                            # 走学生的
            b.step(i_r)                            # 走规则式的
            deltas.append(_rollout(b, pol, seat) - _rollout(a, pol, seat))
    return {"dec": dec, "dis": dis, "pcts": pcts, "gaps": gaps,
            "deltas": deltas, "win": wins / games}


def ladder(pol, games: int, seed: int) -> list:
    """学生打各家固定策略的胜率（座位对调、固定牌堆）。"""
    return [(name, match(pol, other, games=games, seed=seed))
            for name, other in (("随机", random_policy(random.Random(101))),
                                ("贪心", greedy_policy),
                                ("规则式", rule_policy()))]


def main(argv=None):
    utf8_stdout()
    ap = argparse.ArgumentParser(description="「换更强的对手 / 拿规则式当老师」还剩多少可换")
    ap.add_argument("--weights", default=None, help="默认挑面板会加载的那份")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1002)
    a = ap.parse_args(argv)

    path = a.weights or advise.newest_weights()
    net, g = load(path)
    print(f"学生：{path}" + (f"（{g:,} 局）" if g else ""))
    print(f"探针：{a.games} 局 / seed={a.seed}")

    print()
    print("【第一层 · 难度谱】学生打各家（座位对调）")
    print("  信号最强在 ~50%；**越高越没东西可学**")
    for name, wr in ladder(net_play(net), a.games, a.seed):
        if wr > 0.9:
            tag = "饱和（贡献≈0）"
        elif wr > 0.8:
            tag = "接近饱和"
        elif 0.4 <= wr <= 0.8:
            tag = "**有信号**"
        else:
            tag = "被压着打"
        print(f"  {name:6s} {wr:6.1%}   {tag}")
    print("  自对弈   50.0%   （按定义；已在训练里占 50% 的局）")

    d = probe(net, a.games, a.seed)
    rate = d["dis"] / d["dec"] if d["dec"] else float("nan")
    print()
    print("【第二层 · 有没有东西不一样】学生该走的决策点上（多候选的才算）")
    print(f"  决策点 {d['dec']:,} 个，其中 {d['dis']:,} 个与规则式**分歧**"
          f" → 分歧率 **{rate:.1%}**")
    if d["pcts"]:
        print(f"  分歧时，规则式的着法在**学生自己的 Q** 里平均分位 "
              f"{statistics.mean(d['pcts']):.1%}（1.0 = 学生的首选）")
        print(f"  （参考：学生眼里的差距 {statistics.mean(d['gaps']):.3f} 点 —— "
              f"⚠️ 恒 ≥0，因为学生选的就是 argmax，**证明不了学生对**）")
    print(f"  同批对局里学生胜率 {d['win']:.1%}")

    print()
    print("【第三层 · 反事实（决定性）】分歧点上分叉，两条都用学生继续打完")
    dd = d["deltas"]
    if len(dd) < 2:
        print("  样本不够，答不了。")
        return 1
    m = statistics.mean(dd)
    sd = statistics.stdev(dd)
    t = m / (sd / math.sqrt(len(dd)))
    better = sum(1 for x in dd if x > 0) / len(dd)
    worse = sum(1 for x in dd if x < 0) / len(dd)
    print(f"  {len(dd):,} 个分歧点（每局抽一个）：规则式**更好** {better:.1%} / **一样** {1-better-worse:.1%} / **更差** {worse:.1%}")
    print(f"  平均 Δ = **{m:+.3f} 点**（sd {sd:.2f}，t = {t:+.2f}；"
          f"一局典型价值 2.05 点）")

    print()
    print("判读（**单侧**：要判的是「有没有正效应」，不是「是否等价」）：")
    se = sd / math.sqrt(len(dd))
    hi = m + 2 * se
    if t > 3 and m > MATTERS:
        print("  **规则式的着法确实更好** —— 学生在分歧点上有系统性错误，")
        print("  模仿/蒸馏这条路**值得一试**（这是唯一能推翻「不值得」的证据）。")
    elif t < -3:
        print("  **学生自己的选择明确更好** —— 拿规则式当 teacher 会把模型**带差**，")
        print("  这条路明确不值得。它继续当尺子/对手是对的。")
    elif hi < MATTERS:
        print(f"  **没有正效应** —— 正效应那一侧的 95% 上界只有 {hi:+.3f} 点（< {MATTERS}），")
        print("  也就是「听规则式的更好」被排除在**有意义**的范围之外。")
        print("  ⇒ **不值得**。⚠️ 这是「没测出正效应」，不是「证明了等价」——")
        print("     真要证等价得加大样本；决策只需要前者。")
    else:
        print("  **样本不够** —— 正效应的上界还没压到门槛以内，加 --games 再跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
