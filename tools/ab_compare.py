"""Task 7 Step 3：三臂在同一把尺子上的对比。

尺子（三臂**共用同一批牌 / 同一组对手**，否则不可比）：

- `vs 贪心`：`match(400 局, seed=1002)` —— 种子固定，所以三臂评的是**同一批牌**，
  也保住了与历史跑（0033 / 0940 / 1407）的可比性
- `炸弹浪费`：对**若干代表成员分别量**（`bomb_waste` 一次只吃一个对手），取中位。
  ⚠️ 只对贪心量是不够的 —— 那把尺子已经饱和（95.2%），对它的数不会动
- `炸弹浪费 · 自对弈`那个数也报：`bomb_waste` 的 docstring 说两个都要看

用法：
    .venv/Scripts/python.exe -m tools.ab_compare runs/ab/A_old runs/ab/C_fix runs/ab/B_pool

    # 按**同局数快照**比（预登记的配对点就是这么比的）——
    # 默认找 `last.pt`，但**被中断过的臂没有 last.pt**，所以这里得指快照：
    .venv/Scripts/python.exe -m tools.ab_compare runs/ab/R8_rule runs/ab/R9_rule --ckpt pool/snap_540000.pt

（原来它躺在 `.superpowers/` 里 —— 那是 gitignore 的临时目录，清理后台账里的数字
  就不可复现了。评审的 M8 提的这件事，搬进 `tools/` 解决。）
⚠️ **2026-09-30**：示例原来指的是 `R4_b10` / `R4_b05` —— 那两个臂（连同 `R6_pg`）已按
「结论都进台账了」清掉，**这里不能指不存在的路径**，换成现存的两臂。
它们当年的数字仍在 `plans/2026-09-28-nstep-bootstrap-delivery.md` 与
`plans/2026-09-29-policy-gradient.md` 里。
"""
import os
import statistics
import sys

import torch

from train.eval import bomb_rate, bomb_waste, match
from train.net import QNet
from train.policies import greedy_policy
from train.rule_policy import rule_policy
from train.selfplay import net_play
from tools.accept_meld import _utf8_stdout

_utf8_stdout()

#: 对池量炸弹浪费用的代表成员 —— **三臂共用这一批**。
#: ⚠️ 括号里的数是**本脚本这把尺子**上的（400 局、`seed=1002`），不是存档自报的 ——
#: 两者口径不同（0033 自报 91.0%，同尺子 87.8%）。引用时别混。
MEMBERS = [
    "runs/rl/20260926-1407/best.pt",   # 热启动的起点（同尺子 95.2%）
    "runs/rl/20260926-0033/best.pt",   # 上一代纯自对弈（同尺子 87.8%）
]
GAMES_WASTE = 60


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    n = QNet()
    n.load_state_dict(d["net"])
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
    # `docs/superpowers/plans/2026-09-29-rule-ruler-fix.md`。
    # 而且量的是「对面会像人一样打时你还行不行」。
    # ⚠️ 它比贪心慢（每个决策点要枚举+估风险，纯 Python），一次约 20~40 秒。
    wr_r = match(pol, rule_policy(), games=400, seed=1002)
    print(f"   vs 规则式（400 局，同种子）  {wr_r:6.1%}")

    w, c, r = waste(pol, None)
    b, g, br = rate(pol, None)
    print(f"   炸弹浪费 · 自对弈          {w:3d}/{c:4d} = {r:5.1%}")
    print(f"   用炸率   · 自对弈          {b:3d}/{g:3d} = {br:5.2f} 手/局")

    rs, brs = [], []
    for mp in MEMBERS:
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
    for a in argv:
        main(a, ckpt)
        print()
