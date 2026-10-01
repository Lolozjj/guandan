"""**ε 噪声定价**：给现役权重加"随机出牌"的比例，看尺子掉多少。

为什么需要它：热启动的臂结束时 ε 还有 0.13（默认退火 25 万局，而臂只有 22 万局）。
若"R11 + 13% 随机"就掉 ~3pp，那 **O1 的 −3.25pp 大部分不是"学坏了"，
而是"交付的是一个噪声未退净的策略"** ⇒ 修法是**更快退火**，不是换目标。

    .venv/Scripts/python.exe -m tools.eps_noise models/best.pt --eps 0.13
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state, q_argmax_batch
from guandan.rl.rule_policy import rule_policy
from tools.ruler import paired

DEFAULT_SEEDS = [1002, 1003, 1004]


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="给权重加 ε 随机，量它值多少分")
    ap.add_argument("weights", nargs="?", default=None)
    ap.add_argument("--eps", type=float, default=0.13)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    a = ap.parse_args(argv)

    p = a.weights or str(paths.BEST)
    net, g = load(p)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"权重：{p}" + (f"（{g:,} 局）" if g else "") + f"；ε = {a.eps:.2f}\n")

    def make_pol(rng):
        def pol(obs, acts, hist=None):
            if rng.random() < a.eps:
                return rng.randrange(len(acts))
            return q_argmax_batch(net, [(obs, acts, hist)])[0]
        return pol

    base, exp, diffs = [], [], []
    for s in seeds:
        wb = ev.match(_argmax_pol(net), rule_policy(), games=a.games, seed=s)
        we = ev.match(make_pol(random.Random(5000 + s)), rule_policy(),
                      games=a.games, seed=s)
        base.append(wb)
        exp.append(we)
        diffs.append(we - wb)
        print(f"  seed={s}  纯 argmax {wb:6.1%}   +ε{a.eps:.2f} {we:6.1%}   Δ {(we - wb) * 100:+5.2f}pp")
    m, sd, t = paired(diffs)
    print(f"\n配对均值：{statistics.mean(base):.1%} → {statistics.mean(exp):.1%}"
          f"   **Δ = {m * 100:+.2f}pp**（sd {sd * 100:.1f}pp，t = {t:+.2f}）")
    print("\n读法：**这条 δ 就是「ε 噪声的价格」** —— 热启动臂结束时的 ε 若在 0.1~0.15，"
          "它的尺子读数就自带这个折扣。")
    return 0


def _argmax_pol(net):
    def pol(obs, acts, hist=None):
        return q_argmax_batch(net, [(obs, acts, hist)])[0]
    return pol


if __name__ == "__main__":
    sys.exit(main())
