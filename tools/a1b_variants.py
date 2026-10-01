"""A1b-1：给**学生那一侧**的候选补花色变体，**不重训**，看现役权重的胜率动不动。

预登记在 `plans/2026-09-30-beat-70-roadmap.md` §九：

    Δ >= +1.0pp 且 t >= 3   -> 动作空间缺口**有真实代价**（A1b-2：改核心枚举 + 重训一臂）
    Δ =  +0.5 ~ +1.0pp      -> **重测/加局数**，不许当成"有涨"
    Δ <= +0.5pp             -> **无代价** —— 降级（转 A2 特征 + A6b 结构指标）

做法：同一批种子、同一副牌，跑两遍 `eval.match`：对照臂 `expand_a=False`、
处理臂 `expand_a=True`（**只给学生多给选项，规则式那把尺子不变**）。

附带必报两条：
  - **候选膨胀**（平均候选数之比）
  - **学生真的选了新变体的比例** —— 一次都没选，说明 Q 在变体之间没有区分力。

跑法：
    .venv/Scripts/python.exe -m tools.a1b_variants
    .venv/Scripts/python.exe -m tools.a1b_variants --games 400 --seeds 1002,1003
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.policies import greedy_policy
from guandan.rl.rule_policy import rule_policy
from guandan.rl.selfplay import generate_batch, net_play
from guandan.sim import env, meld

DEFAULT_SEEDS = [1002, 1003, 1004, 1005, 1006, 1007]

#: 预登记判据
PASS_PP = 1.0
FAIL_PP = 0.5
PASS_T = 3.0


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def _cands(obs, variants: bool) -> list:
    """从 `Observation` 复原候选（口径与 `rules.Hand.actions` 一致）。

    跟牌时补上那条 `None`（能压也可以过）—— 与 `advise.candidates` 同口径。
    """
    table = meld.as_meld(list(obs.table), obs.level) if obs.table else None
    moves = meld.melds_from(sorted(obs.hand), obs.level, variants=variants)
    if table is None:
        return list(moves)
    return [m for m in moves if meld.beats(m, table)] + [None]


def _key(m) -> tuple:
    return () if m is None else tuple(sorted(m.cards))


def usage(net, games: int = 6, seed: int = 11) -> dict:
    """**学生到底用没用新选项**（离线诊断：同一批决策点，两种候选集合下各取一次 argmax）。"""
    rng = random.Random(seed)
    base_n = exp_n = 0
    switched = 0                      # 两种候选下 argmax 的**牌组**不同
    picked_new = 0                    # 选了**只在变体集合里**才有的牌组
    total = 0
    for rec, pts, _y in generate_batch(net, rng, 0.0, games, capture=True, opp_mix=0.0):
        for obs, acts, _i, _seat, hist in pts:
            exp = _cands(obs, True)
            if len(exp) <= len(acts):
                continue
            total += 1
            base_n += len(acts)
            exp_n += len(exp)
            qa = [float(x) for x in q_values(net, obs, acts, hist)]
            qb = [float(x) for x in q_values(net, obs, exp, hist)]
            ma, mb = acts[max(range(len(qa)), key=qa.__getitem__)], \
                exp[max(range(len(qb)), key=qb.__getitem__)]
            if _key(ma) != _key(mb):
                switched += 1
                if _key(mb) not in {_key(m) for m in acts}:
                    picked_new += 1
    return {"total": total, "base_n": base_n, "exp_n": exp_n,
            "switched": switched, "picked_new": picked_new}


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="A1b-1：扩候选（不重训）值多少分")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--usage-games", type=int, default=6)
    a = ap.parse_args(argv)

    path = a.weights or str(paths.BEST)
    net, g = load(path)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"学生：{path}" + (f"（{g:,} 局）" if g else ""))
    print(f"每种子 {a.games} 局 vs 规则式（座位对调）；处理臂 = **只给学生**扩候选\n")

    base, exp, diffs = [], [], []
    for s in seeds:
        wb = ev.match(net_play(net), rule_policy(), games=a.games, seed=s)
        we = ev.match(net_play(net), rule_policy(), games=a.games, seed=s,
                      expand_a=True)
        base.append(wb)
        exp.append(we)
        diffs.append(we - wb)
        print(f"  seed={s}  对照 {wb:6.1%}   扩候选 {we:6.1%}   Δ {(we - wb) * 100:+5.2f}pp")

    mb, me = statistics.mean(base), statistics.mean(exp)
    d, sd = statistics.mean(diffs), statistics.stdev(diffs)
    t = d / (sd / math.sqrt(len(diffs)))
    print(f"\n配对均值：对照 {mb:.1%} → 扩候选 {me:.1%}   "
          f"**Δ = {d * 100:+.2f}pp**（sd {sd * 100:.1f}pp，t = {t:+.2f}，n={len(diffs)} 种子）")

    u = usage(net, games=a.usage_games)
    if u["total"]:
        print(f"\n诊断（{u['total']} 个决策点，自对弈离线）：")
        print(f"  候选数：平均 {u['base_n'] / u['total']:.1f} → "
              f"{u['exp_n'] / u['total']:.1f}（×{u['exp_n'] / max(1, u['base_n']):.2f}）")
        print(f"  argmax 换人 {u['switched'] / u['total']:.1%}；"
              f"其中**选了只在变体里才有的牌组** {u['picked_new'] / u['total']:.1%}")
    else:
        print("\n诊断：这批自对弈里没碰到「有变体可选」的决策点（样本太小）")

    print("\n判读（预登记判据）：")
    pp = d * 100
    if pp >= PASS_PP and t >= PASS_T:
        print(f"  **Δ {pp:+.2f}pp / t={t:+.2f} ⇒ 动作空间缺口有真实代价**"
              f" ⇒ 走 A1b-2（改核心枚举 + 重训一臂）")
    elif pp <= FAIL_PP:
        print(f"  **Δ {pp:+.2f}pp ⇒ 无代价** ⇒ 动作空间这条线降级，"
              f"转 A2（显式特征）+ A6b（结构指标）")
    else:
        print(f"  **Δ {pp:+.2f}pp 落在 0.5~1.0pp 之间 ⇒ 重测/加局数**，不许当「有涨」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
