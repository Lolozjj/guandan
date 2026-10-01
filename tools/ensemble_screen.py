"""点①筛选：**集成**（输出平均 / 权重平均）到底值不值。

判据（`plans/2026-10-01-overnight-9h.md` §一）：12 种子 × 300 局**配对**，
与"单件最佳"比；Δ > 0 且 t ≥ 2 ⇒ 接进 `advise.py`。

    .venv/Scripts/python.exe -m tools.ensemble_screen
    .venv/Scripts/python.exe -m tools.ensemble_screen --members a.pt,b.pt --seeds 1002,1003
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.ensemble import ensemble_policy, load_net, soup
from guandan.rl.rule_policy import rule_policy
from guandan.rl.selfplay import net_play
from tools.ruler import paired

#: 默认成员：现役（O2b/best）、另一条干净 ε 的臂（O3b/best）、以及上一代 R11
DEFAULT_MEMBERS = ["models/best.pt",
                   "runs/ab/O3b_loweps/best.pt",
                   "models/best_r11_backup.pt"]
DEFAULT_SEEDS = [1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013]


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="集成 vs 单件（配对）")
    ap.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    ap.add_argument("--base", default=None, help="配对基准（默认 = 第一个成员）")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--games", type=int, default=300)
    a = ap.parse_args(argv)

    paths = [p for p in a.members.split(",") if p.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    base = a.base or paths[0]
    nets = {p: load_net(p) for p in paths}
    print(f"成员 {len(paths)} 个：" + "，".join(paths))
    print(f"{len(seeds)} 种子 × {a.games} 局；配对基准 = {base}\n")

    policies = {}
    for p in paths:                                  # 单件
        policies[p] = net_play(nets[p])
    policies["**输出集成**"] = ensemble_policy([nets[p] for p in paths])
    policies["**权重平均(soup)**"] = net_play(soup([nets[p] for p in paths]))

    rows = {}
    for name, pol in policies.items():
        wr = [ev.match(pol, rule_policy(), games=a.games, seed=s) for s in seeds]
        rows[name] = wr
        m = statistics.mean(wr)
        print(f"  {name:26s} {m:6.1%}")

    if base not in rows:
        print(f"⚠️ 基准 {base} 不在跑过的臂里，跳过配对差")
        return 0
    print(f"\n配对差（基准 = {base}，逐种子对消）")
    print("-" * 74)
    for name, wr in rows.items():
        if name == base:
            continue
        d = [x - y for x, y in zip(wr, rows[base])]
        m, sd, t = paired(d)
        verdict = ("**成立**" if abs(t) > 3 else
                   "看着像，种子不够" if abs(t) > 1.5 else "分辨不出")
        print(f"{name:26s} {m * 100:+6.2f}pp  sd={sd * 100:4.1f}pp  t={t:+5.2f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
