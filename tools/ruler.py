"""**多臂 × 多种子配对测量** —— 一把尺子量完所有候选，再算配对差。

为什么要有它：`tools/ab_compare.py` 是「一臂一行、各量各的」，适合单臂体检；
但**比两臂谁强**不能看两次独立测量的差 —— 单种子 400 局的 sd 约 2.5pp，
比大多数要判的效应还大（台账记过：2 个种子的趋势读数分辨不了 3~4pp）。

配对能把这层噪声消掉：**同种子 = 同一副牌、同样谁坐哪**，
所以两臂的差是逐局对消的，sd 掉一个量级。

口径（引用时别丢）：

- 强度：`match(臂, 规则式, games, seed)`，座位对调、**同步推进、固定牌堆**。
- `vs 规则式` 是第二把尺子：`vs 贪心` 已饱和（现役 ~93%），
  而规则式「像人」（不压队友、留炸、算剩牌），留了 40pp 余量。
- 单种子 400 局 sd ≈ 2.1~2.5pp；**≥3 个种子**才谈得上结论。

用法：

    # 默认量 runs/*/best.pt，4 个种子
    .venv/Scripts/python.exe -m tools.ruler
    .venv/Scripts/python.exe -m tools.ruler runs/R8_rule/best.pt runs/R10_rule/best.pt
    .venv/Scripts/python.exe -m tools.ruler --ckpt-name pool/snap_300000.pt runs/R10_rule
    .venv/Scripts/python.exe -m tools.ruler --base runs/R8_rule/best.pt <ckpts...>
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import statistics
import sys

import torch

from guandan.rl.eval import match
from guandan.rl.net import QNet
from guandan.rl.rule_policy import rule_policy
from guandan.rl.selfplay import net_play
from tools.accept_meld import _utf8_stdout

#: 默认种子。1002 是历史沿用的那把（与 `ab_compare` / 台账全部旧数同源），
#: 所以它必须在里面 —— 换了就对不上历史。
DEFAULT_SEEDS = [1002, 1003, 1004, 1005]


def load(path: str):
    """权重 -> (策略, 元信息)。策略走 `net_play`，与上线同一条路。"""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    n.load_state_dict(d["net"] if isinstance(d, dict) else d)
    n.eval()
    return net_play(n), (d if isinstance(d, dict) else {})


def label_of(path: str) -> str:
    """`runs/R8_rule/best.pt` -> `R8_rule/best.pt`。

    `pool/` 是中间层，去掉它 —— 快照的标签要留住**臂名**
    （`R10_rule/snap_300000.pt`），不然混着看时分不清是哪条臂的。
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "pool":
        return f"{parts[-3]}/{parts[-1]}"
    return "/".join(parts[-2:])


def paired(diffs):
    """配对差的 (均值, sd, t)。`diffs` 是逐种子的「A-B」。

    t = 均值 / (sd/√n)。**n 只有种子数**，所以 4 个种子时 |t|>3 才算稳
    （约等于 p<0.05 的双侧门槛，小样本下 t 分布比正态厚）。
    n<2 时 sd 与 t 都是 nan —— 一个种子算不出离散度，别硬报。
    """
    n = len(diffs)
    if n < 2:
        return (diffs[0] if n else float("nan"), float("nan"), float("nan"))
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf") * (1 if m > 0 else -1 if m < 0 else 0)
    return m, sd, t


def run(paths, seeds, games, base):
    rows = {}
    for p in paths:
        pol, meta = load(p)
        wr = []
        for s in seeds:
            r = match(pol, rule_policy(), games=games, seed=s)
            wr.append(r)
            print(f"   {label_of(p):28s} seed={s}  {r:6.1%}")
        rows[p] = {"wr": wr, "games": meta.get("games"),
                   "self": meta.get("winrate_greedy")}
    return rows


def report(rows, seeds, base):
    print("\n" + "=" * 74)
    print(f"{'臂':30s} {'局数':>8s} {'vs规则式':>9s} {'sd':>6s}  {'自报vs贪心':>10s}")
    print("-" * 74)
    for p, r in rows.items():
        m = statistics.mean(r["wr"])
        sd = statistics.stdev(r["wr"]) if len(r["wr"]) > 1 else float("nan")
        g = f"{r['games']:,}" if r["games"] is not None else "?"
        s = f"{r['self']:.1%}" if r["self"] is not None else "-"
        print(f"{label_of(p):30s} {g:>8s} {m:>8.1%} {sd*100:>5.1f}pp {s:>10s}")

    if base and base in rows:
        print(f"\n配对差（基准 = {label_of(base)}，逐种子对消）")
        print("-" * 74)
        for p, r in rows.items():
            if p == base:
                continue
            d = [a - b for a, b in zip(r["wr"], rows[base]["wr"])]
            m, sd, t = paired(d)
            verdict = ("**成立**" if abs(t) > 3 else
                       "看着像，种子不够" if abs(t) > 1.5 else "分辨不出")
            print(f"{label_of(p):30s} {m*100:+6.2f}pp  sd={sd*100:4.1f}pp  "
                  f"t={t:+5.2f}  {verdict}")
    return rows


def main(argv=None):
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="多臂 × 多种子配对测量（vs 规则式）")
    ap.add_argument("ckpts", nargs="*", help="权重路径；留空 = runs/*/best.pt")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                    help="逗号分隔的种子（配对靠它，别只给一个）")
    ap.add_argument("--games", type=int, default=400,
                    help="每个种子的局数（400 = 与全部历史数同口径）")
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="给的是**臂目录**时，取目录下哪个权重")
    ap.add_argument("--base", default=None, help="配对差的基准权重")
    a = ap.parse_args(argv)

    paths = []
    for c in (a.ckpts or sorted(glob.glob("runs/*/best.pt"))):
        if os.path.isdir(c):
            c = os.path.join(c, a.ckpt_name)
        if not os.path.exists(c):
            print(f"!! 跳过（不存在）: {c}")
            continue
        paths.append(c)
    if not paths:
        sys.exit("没有可量的权重")
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    print(f"量 {len(paths)} 个权重 × {len(seeds)} 个种子 × {a.games} 局 "
          f"（规则式较慢，约 {len(paths)*len(seeds)*a.games*0.028/60:.0f} 分钟）\n")
    rows = run(paths, seeds, a.games, a.base)
    base = a.base
    if base is None and len(paths) > 1:
        base = paths[0]
    report(rows, seeds, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
