"""**擦掉"浪费型"选择**：在"能不能压过桌面"完全不变的前提下，改掉用户抱怨的那两类选择。

用户 2026-10-01 实机反馈（原话）：

1. 「有牌可以压制的情况下会使用炸弹进行压制」⇒ **白炸**（实测学生 7.5%，规则式 **0%**）；
2. 「实际炸个 8888 就行了，但他会用万能牌构造成五个 8 去炸」⇒ **用万能牌放大炸弹**。

设计原则（三条，缺一不可）：

- **只在浪费时介入**：不浪费的决策一律不动 ⇒ 改动面小、可归因。
- **不降低"压得过"的能力**：候选本来就已经按 `beats` 过滤过，所以任何候选都能压过桌面；
  擦掉的是"用了更贵的资源"，不是"放弃了压制"。
- **单一实现**：判定复用 `rl/eval.py` 的 `is_wasted_bomb` / `is_wasted_wild`
  （与战报、尺子同一份口径），绝不在这里再写一份。

⚠️ 它**不是**"替模型做决策"，而是**资源使用的下限约束**：
真到必须炸（没有普通牌能压）时一动不动 —— 这也是规则式自己 0% 白炸的原因。
"""
from __future__ import annotations

import numpy as np

from guandan.rl.net import q_values


def tidy_index(q, acts, chosen_i: int, *, has_table: bool, bombs: bool = True,
               wilds: bool = True) -> int:
    """在候选里挑"最省资源"的一手（**同级之内**仍按网络自己的 Q 挑）。没有更省的就不动。

    ⚠️ **只有"桌上有牌要压"时才谈得上浪费**（`has_table`）—— 与 `rl/eval.py` 的
    `bomb_opportunity` / `wild_opportunity` **同一口径**。我 2026-10-01 改字典序时
    顺手删了 `table` 参数、把这条前提弄丢了，结果**主动出炸（领出）也被降级**
    ⇒ 300 局里**一次炸都不出、万能牌也不用**。别再弄丢。

    资源成本按 **字典序** 排（这是 2026-10-01 用户两条原话的直接编码）：

    1. **不用炸弹** > 用炸弹（「有牌可以压制的情况下会使用炸弹进行压制」）；
    2. **不用万能牌** > 用万能牌（「炸个 8888 就行了，却用万能牌凑成五个 8」）。

    ⇒ 顺序是：普通牌 > 用万能牌的普通牌 > 天然炸弹 > 万能牌放大的炸弹。
    `bombs=False` / `wilds=False` 可以把对应那一维**从字典序里去掉**（只在同级里按 Q 挑）。

    ⚠️ 为什么必须是字典序而不是"两条 if 顺序判断"：先判"白炸"会让
    「天然 8888 就在候选里、却用万能牌凑了 5 张炸」这类局面**永远轮不到**第二条规则
    （它先被第一条降级成普通牌；而那个普通牌可能还是用万能牌的）。字典序一次说清。
    """
    if len(acts) <= 1 or not has_table:
        return chosen_i

    # ⚠️ **「过」不算"更省资源"**：它有它自己的代价（把出牌权让出去）。
    # 我 2026-10-01 把它算成了最省（`None` 既不是炸弹也不用万能牌）⇒
    # 该炸的局面被判成"浪费"、然后改成**过牌** —— 实测 300 局**一次炸都不出**。
    # 这条 bug 由 `tests/test_tidy.py::test_pass_is_never_a_tidy_alternative` 钉住。
    cands = [i for i in range(len(acts)) if acts[i] is not None]
    if chosen_i not in cands or not cands:
        return chosen_i

    def cost(i):
        m = acts[i]
        return (1 if (bombs and m.is_bomb) else 0,
                1 if (wilds and m.wild_used) else 0)

    best = min(cost(i) for i in cands)
    if cost(chosen_i) == best:                     # 已经是最省的 ⇒ **一动不动**
        return chosen_i
    tier = [i for i in cands if cost(i) == best]
    return max(tier, key=lambda i: float(q[i]))


def tidy_net_policy(net, *, bombs: bool = True, wilds: bool = True):
    """把网络策略包一层"擦浪费"。**Q 只算一次**，不浪费时与原来逐位一致。"""
    def pol(obs, acts, hist=None):
        q = q_values(net, obs, acts, hist)
        i = int(np.argmax(q))
        return tidy_index(q, acts, i, has_table=obs.table is not None,
                          bombs=bombs, wilds=wilds)
    return pol
