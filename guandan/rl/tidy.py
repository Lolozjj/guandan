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
               wilds: bool = True, margin: float = 0.0) -> int:
    """在候选里挑"最省资源"的一手（**同级之内**仍按网络自己的 Q 挑）。没有更省的就不动。

    `margin`：**只在网络没有强烈偏好时才介入** —— 若"首选 − 最省候选的 Q 最高者"
    超过 `margin`，就认为网络是**有理由**这么打的（例如必须靠这颗炸抢回出牌权），
    一动不如一静。`margin=0`（默认）= 老行为（只要存在更省的候选就换）。

    ⚠️ 为什么要这个旋钮：硬替换的代价是实测 **−2.11pp（t=−3.33，normal）**，
    而它擦掉的白炸只有 7.4 个百分点里的 7.4 个 —— 用户要的是"别乱炸"，
    不是"别炸"。先用阈值把"网络自己都犹豫"的那些擦掉，看看性价比能不能好得多。
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
    alt = max(tier, key=lambda i: float(q[i]))     # 最省档里 Q 最高的那个
    if margin > 0 and float(q[chosen_i]) - float(q[alt]) > margin:
        # 网络**强烈**偏好这一手（差值超过阈值）⇒ 认为它有理由（例如靠这颗炸抢回出牌权）
        return chosen_i
    return alt


def tidy_net_policy(net, *, bombs: bool = True, wilds: bool = True, margin: float = 0.0):
    """把网络策略包一层"擦浪费"。**Q 只算一次**，不浪费时与原来逐位一致。"""
    def pol(obs, acts, hist=None):
        q = q_values(net, obs, acts, hist)
        i = int(np.argmax(q))
        return tidy_index(q, acts, i, has_table=obs.table is not None,
                          bombs=bombs, wilds=wilds, margin=margin)
    return pol
