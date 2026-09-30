"""量「动作边际」：同一局面下候选之间的 Q 差 —— 动作通道到底用了多少。

判据 2（spec §1.2）：现役存档 `1407` 是 **0.123**（标签尺度 ±3）。自举如果起作用，
Q 对动作的区分度就该变大。

⚠️ **必须和「vs 贪心」一起读**：边际变宽而胜率掉，那是数值发散，不是变好。

做法：`generate_batch(..., eps=0, opp_mix=0)` 自对弈若干局并抓决策点（四家都抓），
对**每个决策点**跑一遍 `q_values`（全部候选），取排序后前两名之差。
种子固定 → 结果可复现。

⚠️ 这个脚本原来是在临时目录里算的（2026-09-27），清理完数字就不可复现了
（评审 M8）—— 所以它必须住在 `tools/` 里。

用法：
    .venv/Scripts/python.exe -m tools.action_margin models/best.pt
"""
from __future__ import annotations

import random
import statistics
import sys

import torch

from guandan import paths
from guandan.rl.net import QNet, q_values
from guandan.rl.selfplay import generate_batch
from guandan.console import utf8_stdout


def margins_of(net, pending):
    """`pending = [(obs, acts, hist), ...]` -> 每个决策点的 `top1 - top2`。"""
    out = []
    for obs, acts, hist in pending:
        if len(acts) < 2:
            continue
        q = q_values(net, obs, acts, hist)
        top = torch.topk(q, k=2).values
        out.append(float(top[0] - top[1]))
    return out


def measure(net, games: int = 32, seed: int = 0):
    """返回 `(决策点数, 统计)`。"""
    rng = random.Random(seed)
    pending = []
    for _rec, caps, _y in generate_batch(net, rng, 0.0, games, capture=True,
                                         opp_mix=0.0):
        pending += [(o, a, h) for o, a, _i, _s, h in caps]
    m = margins_of(net, pending)
    # 候选间离散度（max-min）也报：与现用方案 §四.2 那张表同口径（0.401 / 0.123）
    spread = []
    for obs, acts, hist in pending:
        if len(acts) < 2:
            continue
        q = q_values(net, obs, acts, hist)
        spread.append(float(q.max() - q.min()))
    return len(pending), {
        "median_margin": statistics.median(m),
        "p90_margin": sorted(m)[int(0.9 * (len(m) - 1))],
        "tie_share": sum(1 for x in m if x < 0.01) / max(1, len(m)),
        "median_spread": statistics.median(spread) if spread else 0.0,
        "n_margins": len(m),
    }


def main(argv=None) -> int:
    utf8_stdout()
    argv = sys.argv[1:] if argv is None else argv
    for p in (argv or [str(paths.BEST)]):
        d = torch.load(p, map_location="cpu", weights_only=False)
        net = QNet()
        net.load_state_dict(d["net"])
        net.eval()
        n, st = measure(net)
        print(f"== {p}（{d.get('games', 0):,} 局）")
        print(f"   决策点 {n} 个，参与统计 {st['n_margins']}")
        print(f"   首选次选之差 中位 {st['median_margin']:.3f}  "
              f"p90 {st['p90_margin']:.3f}  打平(<0.01) {st['tie_share']:.1%}")
        print(f"   候选间离散度 中位 {st['median_spread']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
