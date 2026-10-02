"""Step A：**信念到底有没有信息量？**（便宜的先杀 —— 没有就不往下做搜索）

做法：用现役权重打若干局（我方 = Q-argmax，对手 = 规则式），
**从发牌开始**跑一颗粒子滤波（`guandan/rl/belief.py`），
在每个采样点上比较两套信念对"每张暗牌在谁手上"的命中率：

- **均匀基线**：当场重建的粒子（只用公开信息：我的手牌 + 各家已出 + 各家手数）；
- **滤波（带行为软似然）**：从发牌跑到现在的那一颗。

⚠️ 两者**用的公开信息一样**，差别只在"行为线索"（能压却过牌、没出最便宜的那一手）
⇒ 命中率的差就是**行为线索的信息量**，这正是判据（kill 线）要的那个数。

    .venv/Scripts/python.exe -m tools.belief_probe --games 20 --positions 200
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

import numpy as np

from guandan.console import utf8_stdout
from guandan.rl.belief import Belief
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, rules


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _accuracy(belief, truth, others, cards) -> float:
    """每张暗牌的**持有者命中率**：边际 argmax 是否等于真实持有者。"""
    hit = tot = 0
    for c in cards:
        s = truth.get(c)
        if s is None:
            continue
        tot += 1
        hit += int(belief.predict_holder(c) == s)
    return hit / tot if tot else float("nan")


def _logp_true(belief, truth, cards) -> float:
    """**真实持有者**在该信念边际下的平均对数概率（越接近 0 越好，均匀时 ≈ log(1/3)）。"""
    import math
    tot, n = 0.0, 0
    for c in cards:
        s = truth.get(c)
        if s is None:
            continue
        pr = max(belief.holder_probs(c).get(s, 0.0), 1e-9)
        tot += math.log(pr)
        n += 1
    return tot / n if n else float("nan")


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="信念信息量探针（Step A）")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--particles", type=int, default=24)
    ap.add_argument("--every", type=int, default=8, help="每几步采一个点")
    ap.add_argument("--seed", type=int, default=1002)
    ap.add_argument("--alpha", type=float, default=None, help="能压却过牌的折扣（越小越强）")
    ap.add_argument("--beta", type=float, default=None, help="没出最便宜的折扣")
    ap.add_argument("--window", type=int, default=12, help="看最近多少条事件")
    a = ap.parse_args(argv)

    net = load(a.weights)
    rng = random.Random(a.seed)
    acc_u, acc_f, esses = [], [], []
    lp_u, lp_f = [], []
    games_done = 0
    while len(acc_f) < a.positions and games_done < a.games:
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        team0 = games_done % 2
        games_done += 1
        rng_b = random.Random(a.seed + games_done)
        events = []            # (seat, table_meld, chosen_or_None, played_after) 时间序
        played_since = {s: [] for s in rules.SEATS}
        step = 0
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            mine = rules.TEAM[seat] == team0
            # ⚠️ 信念要的是 **Meld**（`e.hand.table`），不是 `obs.table`（那是个牌张元组）
            table = e.hand.table
            # 采样点：比较两套信念
            if step % a.every == 0 and len(acc_f) < a.positions:
                truth = {}
                hidden = []
                for s in rules.SEATS:
                    if s == seat:
                        continue
                    for c in e.hand.hands[s]:
                        truth[c] = s
                        hidden.append(c)
                if hidden:
                    kw = {}
                    if a.alpha is not None:
                        kw["alpha"] = a.alpha
                    if a.beta is not None:
                        kw["beta"] = a.beta
                    uni = Belief(obs, k=a.particles, rng=random.Random(a.seed * 31 + step))
                    # 加权版：同一批抽法 + 用**历史行为事件**重新加权
                    wt = Belief(obs, k=a.particles, rng=random.Random(a.seed * 37 + step), **kw)
                    evs = [(s_, tb, ch, list(played_since[s_])) for (s_, tb, ch, _pa) in events]
                    wt.reweight(evs, window=a.window)
                    acc_u.append(_accuracy(uni, truth, None, hidden))
                    acc_f.append(_accuracy(wt, truth, None, hidden))
                    esses.append(wt.ess())
                    # 更敏感的读数：**真实持有者**在信念边际下的平均对数概率
                    lp_u.append(_logp_true(uni, truth, hidden))
                    lp_f.append(_logp_true(wt, truth, hidden))
            # 出牌（我方用 Q，对手用规则式）
            if mine:
                hist = env.encode_history(e.hand, seat)
                i = int(np.argmax(q_values(net, obs, acts, hist)))
            else:
                i = rule_choose(obs, acts)
            m = acts[i]
            # 把这一步喂给信念（**只喂别人**：我自己的手牌本来就知道）
            if not mine:
                # 记事件：喂给信念的是「当时候重建手牌」所需的信息
                events.append((seat, table, m, None))
            if m is not None:
                played_since[seat].extend(m.cards)   # 展开成一张张的 card id（不是每手一个元组）
            e.step(i)
            step += 1

    n = len(acc_f)
    if not n:
        sys.exit("一个采样点都没有")
    mu, mf = statistics.mean(acc_u), statistics.mean(acc_f)
    ess_med = statistics.median(esses)
    print(f"采样点 {n} 个（{games_done} 局，每局每 {a.every} 步采一个；粒子 {a.particles} 个）\n")
    print(f"  均匀基线（只用公开信息）   每张暗牌命中率 **{mu:.1%}**")
    print(f"  滤波（+ 行为软似然）       每张暗牌命中率 **{mf:.1%}**")
    print(f"  ⇒ 行为线索的信息量 = **{(mf - mu) * 100:+.1f} 个百分点**")
    print(f"  ESS 中位 {ess_med:.1f} / {a.particles} = {ess_med / a.particles:.0%}")
    print(f"  真实持有者的平均对数概率：均匀 {statistics.mean(lp_u):+.3f} → "
          f"加权 {statistics.mean(lp_f):+.3f}（差 {statistics.mean(lp_f) - statistics.mean(lp_u):+.4f}）")
    print("  （均匀时理论值 ≈ log(1/3) = −1.099；越接近 0 说明信念越集中在真相上）")
    print("\n判据（预登记）：命中率 ≥ +2pp 且 ESS ≥ 20% ⇒ 继续 Step B；否则**杀**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
