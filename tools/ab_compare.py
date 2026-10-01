"""A/B 臂体检：同一把尺子上量 `vs 贪心` / `vs 规则式` / 白炸 / 用炸率。

尺子（各臂**共用同一批牌 / 同一组对手**，否则不可比）：

- `vs 贪心`：`match(400 局, seed=1002)` —— 种子固定，所以各臂评的是**同一批牌**
- `vs 规则式`：同口径。**这是现在的主尺子** —— `vs 贪心` 早已饱和（90%+），
  量不出代差；规则式「像人」（不压队友、留炸、算剩牌），还留着几十个百分点的余量
- `炸弹浪费` / `用炸率`：对**若干代表成员分别量**（`bomb_waste` 一次只吃一个对手），取中位。
  ⚠️ 只对贪心量是不够的 —— 那把尺子已经饱和，对它的数不会动
- `炸弹浪费 · 自对弈`那个数也报：`bomb_waste` 的 docstring 说两个都要看

用法：

    .venv/Scripts/python.exe -m tools.ab_compare runs/A_old runs/B_new

    # 按**同局数快照**比（预登记的配对点就是这么比的）——
    # 默认找 `last.pt`，但**被中断过的臂没有 last.pt**，所以这里得指快照：
    .venv/Scripts/python.exe -m tools.ab_compare runs/A_old runs/B_new --ckpt pool/snap_540000.pt

    # 「对池」那几行默认拿面板在用的 models/best.pt 当对手，要换就 --member（可重复）
    .venv/Scripts/python.exe -m tools.ab_compare runs/B_new --member models/best.pt
"""
import os
import statistics
import sys

import torch

from guandan import paths
from guandan.rl.eval import bomb_rate, bomb_waste, match
from guandan.rl.net import QNet, load_state
from guandan.rl.policies import greedy_policy
from guandan.rl.rule_policy import rule_policy
from guandan.rl.selfplay import net_play
from guandan.console import utf8_stdout

utf8_stdout()

#: 「对池」量浪费率用的对手。**留空 = 用面板在用的那份权重**（`models/best.pt`）；
#: 要指定别的对手就 `--member <路径>`（可重复）。
MEMBERS: list[str] = []
GAMES_WASTE = 60


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"])
    n.eval()
    return net_play(n), d


def waste(pol, opp):
    w, c = bomb_waste(pol, games=GAMES_WASTE, seed=3001, opponent=opp)
    return w, c, (w / c if c else float("nan"))


def rate(pol, opp):
    """**用炸率** —— 必须和炸弹浪费率一起看（spec §5.3）。

    挂代价最典型的失败**不是没效果，是「从此一刀切不炸」**：那不比乱炸好，
    而且用户在面板上会觉得建议变蠢。`bomb_waste` 回答不了这个问题。
    """
    b, g = bomb_rate(pol, games=GAMES_WASTE, seed=3001, opponent=opp)
    return b, g, (b / g if g else float("nan"))


def main(arm, ckpt="last.pt"):
    path = os.path.join(arm, ckpt)
    if not os.path.exists(path):
        print(f"== {arm}: 没有 {ckpt}（训练没跑完？）")
        return None
    pol, d = load(path)
    selfrep = d.get("winrate_greedy")
    selfrep = f"，自报 vs贪心 {selfrep:.1%}" if selfrep is not None else "（快照没存胜率）"
    print(f"== {arm}/{ckpt}  （{d.get('games'):,} 局{selfrep}）")
    wr = match(pol, greedy_policy, games=400, seed=1002)
    print(f"   vs 贪心（400 局，同种子）    {wr:6.1%}")
    # `vs 规则式`：2026-09-28 加的**第二把尺子**。
    # 理由：`vs 贪心` 已经饱和（现役 93.8%），而规则式对手是「像人」的
    # （不压队友、留炸、算剩牌），现役对它只有 **54.8%** —— 多 45pp 余量，
    # ⚠️ 2026-09-29 之前是 88.2%：规则式当时有个「拆自己炸」的 bug（修完它自己
    # 对贪心也从 67.0% 涨到 84.2%）。**修复前后的这个数不可直接比**，见
    # 规则式尺子修正台账（已归档到 `master` 分支）。
    # 而且量的是「对面会像人一样打时你还行不行」。
    # ⚠️ 它比贪心慢（每个决策点要枚举+估风险，纯 Python），一次约 20~40 秒。
    wr_r = match(pol, rule_policy(), games=400, seed=1002)
    print(f"   vs 规则式（400 局，同种子）  {wr_r:6.1%}")

    w, c, r = waste(pol, None)
    b, g, br = rate(pol, None)
    print(f"   炸弹浪费 · 自对弈          {w:3d}/{c:4d} = {r:5.1%}")
    print(f"   用炸率   · 自对弈          {b:3d}/{g:3d} = {br:5.2f} 手/局")

    rs, brs = [], []
    for mp in (MEMBERS or [str(paths.BEST)]):
        mp_pol, _ = load(mp)
        w, c, r = waste(pol, mp_pol)
        b, g, br = rate(pol, mp_pol)
        rs.append(r)
        brs.append(br)
        tag = os.path.basename(os.path.dirname(mp))
        print(f"   炸弹浪费 · vs {tag:16s}  {w:3d}/{c:4d} = {r:5.1%}")
        print(f"   用炸率   · vs {tag:16s}  {b:3d}/{g:3d} = {br:5.2f} 手/局")
    med, bmed = statistics.median(rs), statistics.median(brs)
    print(f"   >>> 对池中位 {med:5.1%}   用炸中位 {bmed:5.2f} 手/局   "
          f"vs 贪心 {wr:5.1%}   vs 规则式 {wr_r:5.1%}")
    return wr, med


if __name__ == "__main__":
    argv = sys.argv[1:]
    ckpt = "last.pt"
    if "--ckpt" in argv:
        i = argv.index("--ckpt")
        ckpt = argv[i + 1]
        del argv[i:i + 2]
    while "--member" in argv:
        i = argv.index("--member")
        MEMBERS.append(argv[i + 1])
        del argv[i:i + 2]
    for a in argv:
        main(a, ckpt)
        print()
