"""**配合护栏**：队友快走完时，给他一个**接得住的出口**。

为什么（本夜量出来的具体短板，`plans/2026-10-03-human-play.md` §2.4）：
`tools/coop_profile.py` 量到 —— **队友剩 2 张时，模型只有 16.9% 的时候领出对子**（规则式 47.9%）✗；
剩 1 张时给单张 63%（规则式 87.7%）✗。人类队友会直接抱怨"你不带我"。

**为什么模型学不会**：状态里本来就有 `left[mate]`，但**脚本对手从不惩罚坏配合** ✓
（与"不惩罚乱炸""不惩罚信息劣势"同源）。所以这里先给一个**能立刻验证**的护栏：
只在「我领出 + 队友剩 1~2 张 + 首选不是他需要的牌型 + 存在该牌型的候选」时介入 ✓
—— 别的场合**一动不动**（改动面小、可归因）。

⚠️ 与"擦浪费"是两件事、可叠加：那条管**资源**，这条管**配合**。
⚠️ 这不是"教模型打牌"，而是**把一条人类共识（喂队友）先接上**，
   真正的修法是训练侧的 `--mate-mix`（队友多样性），护栏只是**先拿到收益 + 先有判据** ✓
"""
from __future__ import annotations

import numpy as np

from guandan.rl.net import q_values
from guandan.sim import meld, rules

#: 队友剩 ≤ 这么多张时，给他"能接的牌型"
FEED_LEFT = 2


def coop_index(q, acts, left_mate: int, table, chosen_i: int) -> int:
    """领出时把首选换成"队友需要的牌型"（单张 / 对子）；**没有就一动不动**。

    `table`：桌面（领出时为空 ⇒ 只有领出才谈得上"喂"）；
    `left_mate`：队友剩几张。
    """
    if table:                      # ⚠️ 空元组 = 我领出（这一条本仓库踩过三次，别再写成 is not None）
        return chosen_i
    if not (0 < left_mate <= FEED_LEFT):
        return chosen_i
    need = meld.SINGLE if left_mate == 1 else meld.PAIR
    m = acts[chosen_i]
    if m is not None and m.kind == need:
        return chosen_i             # 首选已经是那个牌型 ⇒ 不动
    cand = [i for i, x in enumerate(acts) if x is not None and x.kind == need]
    if not cand:
        return chosen_i             # 手里没有那个牌型 ⇒ 帮不了，不硬凑
    return max(cand, key=lambda i: float(q[i]))     # 该牌型里挑 Q 最高的


def coop_net_policy(net, *, feed_left: int = FEED_LEFT):
    """把网络策略包一层"喂队友"护栏（Q 只算一次）。"""
    def pol(obs, acts, hist=None):
        q = q_values(net, obs, acts, hist)
        i = int(np.argmax(q))
        mate = rules.PARTNER[obs.seat]
        return coop_index(q, acts, obs.left[mate], obs.table, i)
    return pol

def combined_net_policy(net, *, margin: float = 0.25, leads: bool = True, feed_left: int = 2):
    """**配合护栏 + 擦浪费**：先按"喂队友"改，再按"省资源"收 —— 两个索引级变换串起来。

    顺序有意如此：`coop_index` 先选出"队友接得住"的候选（如果首选不对），
    `tidy_index` 再在该候选上做资源检查（对子/单张通常不涉及炸弹，所以几乎不会互相抵消）✓
    """
    import numpy as np
    from guandan.rl.net import q_values as _qv
    from guandan.rl.tidy import tidy_index
    from guandan.sim import rules as _rules

    def pol(obs, acts, hist=None):
        q = _qv(net, obs, acts, hist)
        i = int(np.argmax(q))
        mate = _rules.PARTNER[obs.seat]
        i = coop_index(q, acts, obs.left[mate], obs.table, i)
        return tidy_index(q, acts, i, has_table=bool(obs.table), margin=margin, leads=leads)

    return pol
