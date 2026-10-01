"""把多个存档**平均**成一个（model soup）—— 零训练、可复现、**不按评测挑**。

为什么这条值得固化成工具（2026-10-01 夜实测）：

| 权重 | vs R11（300 局 × 16 种子，配对） |
|---|---|
| 单件（最好的那条臂） | +1.44pp（t=1.60） |
| soup3（3 个成员） | +2.00pp（t=2.32） |
| **soup5（5 个成员）** | **+2.58pp（t=3.33）**，**独立种子集复核 +3.54pp（t=4.44）** |
| soup_all（21 个成员，含 O1 的 13 个轨迹快照） | +2.54pp（t=3.62）⇒ 混进"中毒 ε"的成员反而**拖后腿** |

⇒ 两条结论：① **成员越多越好**（方差缩减的特征，不是选点运气）；
② **别把已知配方有问题的臂的快照混进来**。
⇒ 未来的算力应该花在"**多条独立的短臂 + 平均**"上，而不是"一条长臂"。

用法：

    .venv/Scripts/python.exe -m tools.make_soup                    # 默认成员表
    .venv/Scripts/python.exe -m tools.make_soup --out runs/ab/soups/mine.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

from guandan.console import utf8_stdout
from guandan.rl.ensemble import load_net, soup

#: **确定性成员表**：干净 ε 谱系（快退火 / ε 不重启）+ 上一代 R11。
#: ⚠️ 不包含 `O1_control/pool/snap_*.pt` —— 那条臂是"慢退火中毒"的，实测混进来会变差。
DEFAULT_MEMBERS = [
    "models/best_r11_backup.pt",
    "runs/ab/O2b_fastanneal/best.pt",
    "runs/ab/O2b_fastanneal/last.pt",
    "runs/ab/O3b_loweps/best.pt",
    "runs/ab/O3b_loweps/last.pt",
    "runs/ab/O4_loweps_ext/best.pt",
    "runs/ab/O4_loweps_ext/last.pt",
]


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="多个存档求平均（model soup）")
    ap.add_argument("--members", default=None,
                    help="逗号分隔；默认用内置的确定性成员表")
    ap.add_argument("--glob", default=None, help="额外按通配符加成员（如 'runs/ab/F*/best.pt'）")
    ap.add_argument("--out", default="runs/ab/soups/soup.pt")
    a = ap.parse_args(argv)

    paths = (a.members.split(",") if a.members else list(DEFAULT_MEMBERS))
    if a.glob:
        paths += sorted(glob.glob(a.glob))
    paths = [p for p in dict.fromkeys(paths) if p.strip()]
    have = [p for p in paths if os.path.exists(p)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"⚠️ 跳过不存在的 {len(missing)} 个：" + "，".join(missing))
    if not have:
        sys.exit("一个成员都不存在，别平均空气")

    nets = [load_net(p) for p in have]
    s = soup(nets)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"net": s.state_dict(), "games": None,
                "note": f"soup of {len(have)}: " + ",".join(have)}, a.out)
    print(f"平均 {len(have)} 个成员 -> {a.out}")
    for p in have:
        print("   ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
