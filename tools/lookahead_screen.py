"""P2 筛选：**一步前瞻**（推理时搜索）到底值不值。

判据（`plans/2026-09-30-overnight-8h.md` §一）：同一批种子配对，
`search_a=one_step_backup(top_k)` 相对纯 argmax 的差。
**只影响选动作、不改权重** ⇒ 这是"多算前向"值多少的直接测量。

    .venv/Scripts/python.exe -m tools.lookahead_screen --games 200 --seeds 1002,1003,1004
    .venv/Scripts/python.exe -m tools.lookahead_screen --top-k 1     # 必须与 argmax 完全相同（自检）
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.rule_policy import rule_policy
from guandan.rl.search import one_step_backup, rollout_backup
from guandan.rl.selfplay import net_play
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
    ap = argparse.ArgumentParser(description="一步前瞻 vs argmax（配对）")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--mode", default="rollout",
                    choices=("rollout", "signed", "unsigned"),
                    help="rollout=推到轮到我队再估值（默认）；signed/unsigned=朴素一步（诊断用）")
    a = ap.parse_args(argv)

    p = a.weights or str(paths.BEST)
    net, g = load(p)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"权重：{p}" + (f"（{g:,} 局）" if g else ""))
    print(f"每种子 {a.games} 局 vs 规则式；模式 {a.mode}，top_k={a.top_k}\n")

    def search(items):
        if a.mode == "rollout":
            return rollout_backup(net, items, top_k=a.top_k)
        return one_step_backup(net, items, top_k=a.top_k,
                               signed=(a.mode == "signed"))

    base, exp, diffs = [], [], []
    for s in seeds:
        wb = ev.match(net_play(net), rule_policy(), games=a.games, seed=s)
        we = ev.match(net_play(net), rule_policy(), games=a.games, seed=s,
                      search_a=search)
        base.append(wb)
        exp.append(we)
        diffs.append(we - wb)
        print(f"  seed={s}  argmax {wb:6.1%}   一步前瞻 {we:6.1%}   Δ {(we - wb) * 100:+5.2f}pp")

    m, sd, t = paired(diffs)
    print(f"\n配对均值：argmax {statistics.mean(base):.1%} → 前瞻 {statistics.mean(exp):.1%}"
          f"   **Δ = {m * 100:+.2f}pp**（sd {sd * 100:.1f}pp，t = {t:+.2f}）")
    print("\n判读：")
    if a.top_k == 1:
        print("  ⚠️ top_k=1 是**自检**：它必须与 argmax 逐位相同（Δ 应当 = 0.00pp）。")
        print("     不为 0 说明前瞻的实现有 bug（或者两条路用了不同的 rng 消费）。")
    elif abs(t) > 3 and m > 0:
        print("  **前瞻有信号（t>3 且为正）⇒ 值得接进 `advise.py`**")
    elif abs(t) > 3 and m < 0:
        print("  **前瞻更差（t>3 且为负）** ⇒ 记录后放弃，不动 advise")
    else:
        print("  分辨不出 ⇒ 只记录，不动 `advise.py`（别用「看着像」改产品）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
