"""**两件权重 × 分风格**的配对比较（判据③要的就是这个）。

`tools/style_profile.py` 只报"一个权重在不同风格上的差"；`tools/ruler.py` 只有 normal 一把。
判"动作计价有没有把胜率赔掉"需要**同一把风格尺子上两个权重逐种子对消** —— 就是本工具。

    .venv/Scripts/python.exe -m tools.style_paired runs/ab/W1_costs/best.pt runs/ab/O3b_loweps/best.pt \
        --seeds 12 --games 300
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan.console import utf8_stdout
from guandan.rl.rule_policy import STYLES
from tools.style_profile import load, profile


def _paired(d):
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    t = m / (sd / (len(d) ** 0.5)) if sd else float("inf")
    return m, sd, t


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="两件权重 × 分风格配对")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1002)
    ap.add_argument("--games", type=int, default=300)
    a = ap.parse_args(argv)

    styles = [x for x in a.styles.split(",") if x.strip()]
    for s in styles:
        if s not in STYLES:
            sys.exit(f"认不出的风格 {s!r}（只有 {sorted(STYLES)}）")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    net_a, _g_a = load(a.a)
    net_b, _g_b = load(a.b)      # `style_profile.load` 返回 `(net, games)`，别当中只返回网络
    print(f"A = {a.a}\nB = {a.b}\n{a.seeds} 种子 × {a.games} 局，逐种子配对（A − B）\n")
    for st in styles:
        pa = profile(net_a, [st], seeds, a.games)[st]
        pb = profile(net_b, [st], seeds, a.games)[st]
        d = [x - y for x, y in zip(pa, pb)]
        m, sd, t = _paired(d)
        verdict = ("**成立**" if abs(t) > 3 else
                   "看着像，种子不够" if abs(t) > 1.5 else "分辨不出")
        print(f"  {st:8s} A {statistics.mean(pa):6.1%}   B {statistics.mean(pb):6.1%}"
              f"   Δ {m * 100:+6.2f}pp  sd={sd * 100:4.1f}pp  t={t:+5.2f}  {verdict}")
    print("\n判据③（预登记）：normal 与 hold 上 Δ ≥ −1.0pp 才算「没把胜率赔光」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
