"""**擦浪费**的代价/收益：配对量"原样 vs 擦浪费"两套策略的胜率，外加两类的浪费率。

为什么必须量增益和代价两面（本仓库的教训）：
- 挂了炸弹代价最典型的失败**不是没效果，是「从此一刀切不炸」** —— 那不比乱炸好；
- 所以这里同时报：白炸率（该降）、用炸手数（别归零）、以及**配对胜率**（别掉）。

    .venv/Scripts/python.exe -m tools.tidy_screen --weights models/best.pt --seeds 12 --games 300
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.rule_policy import STYLES, rule_policy
from guandan.rl.tidy import tidy_net_policy
from guandan.rl.eval import match
from tools.mistake_profile import net_play


def load_net(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _paired(d):
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    t = m / (sd / (len(d) ** 0.5)) if sd else float("inf")
    return m, sd, t


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="擦浪费：配对胜率 + 浪费率 + 用炸手数")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1002)
    a = ap.parse_args(argv)

    net = load_net(a.weights)
    raw, tidy = net_play(net), tidy_net_policy(net)
    rule = rule_policy()
    print(f"权重 {a.weights}；{a.seeds} 个种子 × {a.games} 局，配对（同一副牌逐局对消）\n")

    wr_raw, wr_tidy = [], []
    for k in range(a.seeds):
        s = a.seed0 + k
        wr_raw.append(match(raw, rule, games=a.games, seed=s))
        wr_tidy.append(match(tidy, rule, games=a.games, seed=s))
        print(f"  seed {s}: 原样 {wr_raw[-1]:6.1%}   擦浪费 {wr_tidy[-1]:6.1%}"
              f"   Δ {(wr_tidy[-1] - wr_raw[-1]) * 100:+5.2f}pp")
    d = [x - y for x, y in zip(wr_tidy, wr_raw)]
    m, sd, t = _paired(d)
    print(f"\n平均：原样 {statistics.mean(wr_raw):.1%}   擦浪费 {statistics.mean(wr_tidy):.1%}")
    print(f"配对差（擦浪费 − 原样）：{m * 100:+.2f}pp  sd={sd * 100:.1f}pp  t={t:+.2f}")
    print("⇒ 若 |t| 小（分辨不出）⇒ **擦浪费在胜率上不付代价**，而行为更合人意（面板可默认开）。")

    # ---- 分风格再判一次：**惩罚浪费的尺子是 `hold`（龟）**，不是 normal ----
    print("\n分风格配对（同一副牌逐种子对消）—— 白炸在 normal 上不被惩罚，"
          "但在会存炸的 hold 上应当被惩罚：")
    for st in STYLES:
        opp = rule_policy(style=st)
        r = [match(raw, opp, games=a.games, seed=a.seed0 + k) for k in range(a.seeds)]
        q = [match(tidy, opp, games=a.games, seed=a.seed0 + k) for k in range(a.seeds)]
        dd = [x - y for x, y in zip(q, r)]
        mm, ss, tt = _paired(dd)
        print(f"  {st:8s} 原样 {statistics.mean(r):6.1%}   擦浪费 {statistics.mean(q):6.1%}"
              f"   Δ {mm * 100:+5.2f}pp  t={tt:+5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
