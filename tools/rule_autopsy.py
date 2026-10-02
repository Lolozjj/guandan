"""把**规则式当学生**解剖：在它犹豫（与网络分歧）的地方，反事实地"听网络的话"，能涨多少？

背景（用户问过"继续优化规则式的逻辑有没有搞头"）：规则式在这个项目里身兼两职 ——
**主尺子**与**训练对手**。改它会作废历史数字，所以"优化规则式"要有**证据**：
它到底强不强？弱在哪一步？

这一个工具量的是**决策层面的天花板**：开局（规则式执某一队、对手用网络）打若干局，
在每个"规则式与网络给出不同选择"的决策点上，反事实地分叉 ——
A：**照规则式走**（真实发生的事）；B：**照网络走**。
两边都用同一套后续策略打到终局，比较终局回报 ⇒ 这一手的**因果价值**。

- 若 Δ 显著为正 ⇒ 规则式有决策层面的余量（"听网络"能涨），且余量集中在某类局面上；
- 若 Δ ≈ 0 ⇒ **规则式已经够好**，它输的分不在决策，而在牌/配合 ——
  那"优化规则式逻辑"就**不该**指望涨分（与本仓库"对手池/风格多样性都无诊断靶子"的结论一致）。

    .venv/Scripts/python.exe -m tools.rule_autopsy --games 200
"""
from __future__ import annotations

import argparse
import collections
import copy
import random
import statistics
import sys

import numpy as np

from guandan.console import utf8_stdout
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


def _rollout(e, net, rule_team: int, seat0: int, max_steps: int = 400) -> float:
    """从 `e` 打完这一局：**规则式那一队继续用规则式**、网络那一队继续用网络。"""
    steps = 0
    while not e.done and steps < max_steps:
        obs, acts = e.observe(), e.legal()
        seat = e.hand.turn
        if rules.TEAM[seat] == rule_team:
            i = rule_choose(obs, acts)
        else:
            i = int(np.argmax(q_values(net, obs, acts, env.encode_history(e.hand, seat))))
        e.step(i)
        steps += 1
    return float(rules.reward(e.ranks, seat0)) if e.done else 0.0


def _phase(left: int) -> str:
    return "开局(≥20 张)" if left >= 20 else "中盘(10~19)" if left >= 10 else "残局(<10)"


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="把规则式当学生：反事实听网络")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1002)
    a = ap.parse_args(argv)

    net = load(a.weights)
    rng = random.Random(a.seed)
    pick = random.Random(a.seed * 7919 + 13)          # 抽样用独立 rng（别扰动发牌序列）
    wins = losses = 0
    disagree = 0
    decisions = 0
    deltas, by_phase = [], collections.defaultdict(list)
    for g in range(a.games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        rule_team = g % 2
        fork, k = None, 0
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            if rules.TEAM[seat] == rule_team:
                decisions += 1
                i_r = rule_choose(obs, acts)
                qs = [float(x) for x in q_values(net, obs, acts,
                                                 env.encode_history(e.hand, seat))]
                i_q = max(range(len(qs)), key=qs.__getitem__)
                if i_r != i_q:
                    disagree += 1
                    k += 1
                    if pick.random() < 1.0 / k:       # 蓄水池：每局等概率取一个分歧点
                        left = min(len(e.hand.hands[s]) for s in rules.SEATS
                                   if rules.TEAM[s] == rule_team)
                        fork = (copy.deepcopy(e), i_r, i_q, seat, _phase(left))
                idx = i_r
            else:
                idx = int(np.argmax(q_values(net, obs, acts,
                                             env.encode_history(e.hand, seat))))
            e.step(idx)
        if rules.reward(e.ranks, 0 if rule_team == 0 else 1) > 0:
            wins += 1
        else:
            losses += 1
        if fork is not None:
            fe, i_r, i_q, seat, ph = fork
            a_, b_ = copy.deepcopy(fe), copy.deepcopy(fe)
            a_.step(i_r); b_.step(i_q)
            ra = _rollout(a_, net, rule_team, seat)
            rb = _rollout(b_, net, rule_team, seat)
            deltas.append(rb - ra)
            by_phase[ph].append(rb - ra)

    n = len(deltas)
    print(f"规则式 vs 网络：{a.games} 局，规则式胜 {wins}（{wins / a.games:.1%}）")
    print(f"规则式的决策点 {decisions} 个，其中与网络**分歧** {disagree} 个"
          f"（{disagree / max(decisions, 1):.1%}）")
    if not n:
        print("没抓到分歧点，无法判读")
        return 0
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    t = m / (sd / (n ** 0.5)) if sd else float("inf")
    print(f"\n【反事实】在 {n} 个分歧点上「听网络 − 照规则式」的终局分差："
          f"均值 {m:+.3f} 点，sd {sd:.2f}，t = {t:+.2f}")
    print(f"  其中「听网络更好」的比例 {sum(1 for x in deltas if x > 0) / n:.1%}；"
          f"「明显更好(>0.5 点)」{sum(1 for x in deltas if x > 0.5) / n:.1%}")
    print("\n按阶段拆：")
    for ph in ("开局(≥20 张)", "中盘(10~19)", "残局(<10)"):
        v = by_phase.get(ph)
        if v:
            print(f"  {ph:10s} n={len(v):4d}  均值 {statistics.mean(v):+.3f} 点")
    print("\n判读：均值 ≈0 且 t 小 ⇒ **规则式在这个层面的余量很小**，"
          "优化它的逻辑不该指望涨分（它输的更多在牌与配合上）；"
          "均值明显为正 ⇒ 决策层面有真余量，且上面那三行指出了它在哪个阶段最需要修。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
