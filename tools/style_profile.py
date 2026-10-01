"""A4 的**新尺子**：同一个权重，打不同**风格**的规则式，看鲁棒性。

为什么需要它：我们现在只有两把尺子（贪心、规则式），而训练里 50% 的局都在打规则式
⇒「打不打得过规则式」与「换个风格还打不打得过」是两件事，**后者没有尺子就测不出来**。
`rule_policy.Style` 给出三个脚本级风格（同一条规则链，只动门槛）：

    normal  现役规则式（原值）
    bomb    炸侠：几乎见牌就炸、残局先动炸
    hold    龟：只在对手**非常**可能一手走完时才拦，几乎不主动动炸

⚠️ 它是**尺子**，不是训练目标：风格分低不等于弱（规则式自己打炸侠也会掉），
所以报告里**同时给规则式的自我对局**当参照系 —— 参照系不贴出来，读数就没法解释。

跑法：
    .venv/Scripts/python.exe -m tools.style_profile models/best.pt
    .venv/Scripts/python.exe -m tools.style_profile runs/ab/A2_features/last.pt --games 200
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.rule_policy import STYLES, rule_policy
from guandan.rl.selfplay import net_play

DEFAULT_SEEDS = [1002, 1003, 1004]
DEFAULT_GAMES = 200


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def profile(net, styles, seeds, games) -> dict:
    """`{风格: [逐种子胜率]}`。座位按局号对调（`match` 内部做）。"""
    out = {s: [] for s in styles}
    for s in styles:
        for seed in seeds:
            out[s].append(ev.match(net_play(net), rule_policy(s), games=games, seed=seed))
    return out


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="A4：权重 × 对手风格 的胜率矩阵")
    ap.add_argument("ckpts", nargs="+", help="权重路径（可给多个）")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--styles", default=",".join(STYLES),
                    help=f"逗号分隔；可选 {sorted(STYLES)}")
    a = ap.parse_args(argv)

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    styles = [x for x in a.styles.split(",") if x.strip()]
    for s in styles:
        if s not in STYLES:
            sys.exit(f"认不出的风格 {s!r}（只有 {sorted(STYLES)}）")

    rows = {}
    for p in a.ckpts:
        net, g = load(p)
        rows[p] = (g, profile(net, styles, seeds, a.games))

    # 参照系：规则式自己打这些风格（每种子同一副牌，口径一致）
    ref = {s: [] for s in styles}
    for s in styles:
        for seed in seeds:
            ref[s].append(ev.match(rule_policy(), rule_policy(s), games=a.games, seed=seed))

    print(f"\n{a.games} 局 × {len(seeds)} 种子；基准行 = **规则式打自己这些风格**（解释读数用）\n")
    head = f"{'权重':28s} {'局数':>9s} " + " ".join(f"{s:>8s}" for s in styles)
    print(head)
    print("-" * len(head))
    print(f"{'（规则式自己）':28s} {'-':>9s} " +
          " ".join(f"{statistics.mean(ref[s]):>7.1%} " for s in styles))
    for p, (g, wr) in rows.items():
        name = "/".join(p.replace("\\", "/").split("/")[-2:])
        print(f"{name:28s} {(f'{g:,}' if g else '?'):>9s} " +
              " ".join(f"{statistics.mean(wr[s]):>7.1%} " for s in styles))
    print("\n⚠️ 风格分之间**不要直接横比**（难度不同）—— 要比的是「同一个权重在不同风格上的差」，\n"
          "   以及「它相对规则式自己那一行的差」。修的时候也别只盯 normal 那一列。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
