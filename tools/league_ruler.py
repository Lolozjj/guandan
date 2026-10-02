"""**联赛尺子**：模型 vs 冻结的检查点 —— 两个都够强的对手，才压得出战术/信息维度的差距。

动机（`plans/2026-10-02-league-ruler.md`）：脚本风格**既不惩罚白炸**（"擦浪费"要付 −2.1pp），
**也不惩罚信息劣势**（新做的 `info` 风格触发 11.9% 决策点却零优势）——
因为它们的**强度**不够。唯一现成的强对手是**我们自己**。

用法：

    # 两件权重对打（A 相对 B 的胜率，逐局换边）
    .venv/Scripts/python.exe -m tools.league_ruler --a models/best.pt --b models/best_r11_backup.pt

    # A 侧用推理策略（例如"擦浪费"），看它在**强对手**面前值不值
    .venv/Scripts/python.exe -m tools.league_ruler --a tidy:margin=0.25 --weights models/best.pt \
        --b models/best.pt

- **走局只用 `rl/eval.py::match`**（"a 队对 b 队、逐局换边"的语义已在那里），不另写一份；
- **浪费率只用 `_bomb_stats` / `_wild_stats`**（与战报、尺子同一份判定）；
- 逐种子配对：同一种子的两臂用同一批发牌 ⇒ 报配对差与 t。
"""
from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np
import torch

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.tidy import tidy_net_policy


def load_net(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def make_policy(spec: str, weights: str = None):
    """`spec`：`xxx.pt`（检查点）或 `tidy:margin=0.25`（推理策略，基于 `weights`）。

    返回 `(policy, 人类可读的名字)` —— 名字要**打出来**（换源纪律）。
    """
    if spec.startswith("combined"):
        from guandan.rl.coop import combined_net_policy
        base = load_net("models/best.pt")
        return (combined_net_policy(base, margin=0.25, leads=True),
                f"{spec}（配合护栏 + 擦浪费，基于 models/best.pt）")
    if spec.startswith("coop"):
        from guandan.rl.coop import coop_net_policy
        base = load_net(weights or "models/best.pt")
        return coop_net_policy(base), f"{spec}（喂队友护栏，基于 {weights or 'models/best.pt'}）"
    if spec.startswith("tidy"):
        base = load_net(weights or "models/best.pt")
        mg = 0.0
        if "margin=" in spec:
            mg = float(spec.split("margin=")[1].split(",")[0])
        return (tidy_net_policy(base, margin=mg, leads=("leads=1" in spec)),
                f"{spec}（基于 {weights or 'models/best.pt'}）")
    if spec.startswith("mixed"):
        from guandan.rl.mix import mixed_net_policy
        base = load_net(weights or "models/best.pt")
        mg, tp = 0.15, 0.05
        if "margin=" in spec:
            mg = float(spec.split("margin=")[1].split(",")[0])
        if "temp=" in spec:
            tp = float(spec.split("temp=")[1].split(",")[0])
        return (mixed_net_policy(base, margin=mg, temp=tp),
                f"{spec}（混合策略，基于 {weights or 'models/best.pt'}）")
    if spec.startswith("raw"):
        base = load_net(weights or "models/best.pt")
        return tidy_net_policy(base, bombs=False, wilds=False), f"raw（基于 {weights}）"
    n = load_net(spec)
    return tidy_net_policy(n, bombs=False, wilds=False), spec      # 检查点本体（不擦）


def _paired(d):
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return m, sd, (m / (sd / len(d) ** 0.5) if sd else float("inf"))


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="联赛尺子：模型 vs 冻结检查点")
    ap.add_argument("--a", required=True, help="A 侧：路径.pt 或 tidy:margin=0.25")
    ap.add_argument("--b", required=True, help="B 侧：同上")
    ap.add_argument("--weights", default=None, help="`tidy:` 这类策略的底权重")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1002)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--meters", type=int, default=120, help="量浪费率的局数（每种子）")
    ap.add_argument("--a2", default=None,
                    help="第二套 A 侧策略：与 A **同种子**对同一个 B，用来做配对比较")
    a = ap.parse_args(argv)

    pol_a, name_a = make_policy(a.a, a.weights)
    pol_b, name_b = make_policy(a.b, a.weights)
    pol_a2, name_a2 = make_policy(a.a2, a.weights) if a.a2 else (None, None)
    print(f"A  = {name_a}" + (f"\nA2 = {name_a2}" if a.a2 else "") + f"\nB  = {name_b}\n"
          f"{a.seeds} 种子 × {a.games} 局（逐局换边）；浪费率每种子 {a.meters} 局\n")

    def run(pol, tag):
        wr, waste, wild, bombs = [], [], [], []
        for k in range(a.seeds):
            s = a.seed0 + k
            wr.append(ev.match(pol, pol_b, games=a.games, seed=s))
            w = ev._bomb_stats(pol, a.meters, s, opponent=pol_b)
            ww = ev._wild_stats(pol, a.meters, s, opponent=pol_b)
            waste.append(w[0] / max(w[1], 1))
            bombs.append(w[2])
            wild.append(ww[0] / max(ww[2], 1))
        print(f"  {tag}: 联赛胜率 {statistics.mean(wr):6.1%}"
              f"（逐种子 sd {statistics.stdev(wr) * 100:.1f}pp）  "
              f"白炸 {statistics.mean(waste):5.1%}  用炸 {sum(bombs):4d} 手")
        return wr, waste

    wr_a, wa_a = run(pol_a, "A ")
    if pol_a2 is not None:
        wr_2, wa_2 = run(pol_a2, "A2")

    # 单臂：对 50% 做单样本 t（联赛赛没有天然配对）
    mean_a = statistics.mean(wr_a)
    sda = statistics.stdev(wr_a) if len(wr_a) > 1 else 0.0
    t50 = (mean_a - 0.5) / (sda / len(wr_a) ** 0.5) if sda else float("inf")
    print(f"\nA 的联赛胜率 **{mean_a:.1%}**：相对 50% 的 t = **{t50:+.2f}**"
          f"（{'显著不同 ✓' if abs(t50) >= 2 else '分辨不出'}）")
    if pol_a2 is not None:
        d = [x - y for x, y in zip(wr_2, wr_a)]
        m, sd, tt = _paired(d)
        dw = [x - y for x, y in zip(wa_2, wa_a)]
        mw, sdw, tw = _paired(dw)
        print(f"A2 − A（同种子配对）：胜率 **{m * 100:+.2f}pp**，sd {sd * 100:.1f}pp，"
              f"t={tt:+.2f}；白炸率 {mw * 100:+.2f}pp（t={tw:+.2f}）")
    print("\n判读：胜率 >50% ⇒ A 更强；≈50% ⇒ 分辨不出（那就是「两件一样强」）。"
          "\n  ⚠️ 与脚本尺子的数**不可横比**（对手不同）—— 要比的是**同一改动在两把尺子上的方向**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
