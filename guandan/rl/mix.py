"""**混合策略**：在 Q 近似并列时随机化，而不是永远取 argmax。

为什么它值得做（理论 + 我们的处境）：
在**不完全信息**博弈里，**确定性策略是可被利用的** —— 对手只要摸清"你总是出最小的能压牌"，
就能围绕它做计划。人类会这么干（记牌 + 猜你的习惯），而我们的脚本对手不会 ✗
⇒ 这正是"对人有较好水平"该补的一维，而且它**零训练**、可被联赛尺子度量。

实现上要克制：**只在近似并列时随机**（默认 `margin=0.15`，与"擦浪费"用的是同一个思路 ——
Q 差在噪声量级时，选择本来就是抛硬币 ✓）。差得明显的地方照旧取 argmax，
否则会把自己变成"乱出牌的模型" ✗。

⚠️ 与"擦浪费"（`rl/tidy.py`）**可以叠加**：先按资源字典序挑出候选档，再在档内按 Q 混合。
"""
from __future__ import annotations

import random

import numpy as np

from guandan.rl.net import q_values

#: 默认只在"与首选差 ≤ 0.15"的候选里混合；温度控制混合的均匀程度
MIX_MARGIN = 0.15
MIX_TEMP = 0.05


def mixed_index(q, margin: float = MIX_MARGIN, temp: float = MIX_TEMP, rng=None):
    """在 `q` 的近似并列集合里按 softmax(q/temp) 抽一个下标；差得明显 ⇒ 就是 argmax。"""
    q = np.asarray(q, dtype=np.float64)
    if q.size <= 1:
        return 0
    rng = rng or random
    top = np.flatnonzero(q >= q.max() - margin)
    if top.size <= 1:
        return int(np.argmax(q))
    z = q[top] / max(temp, 1e-6)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choices(top.tolist(), weights=p.tolist(), k=1)[0])


def mixed_net_policy(net, *, margin: float = MIX_MARGIN, temp: float = MIX_TEMP, seed: int = 0):
    """把网络策略包一层混合。**Q 只算一次**；并列不明显时与 argmax 逐位一致。"""
    rng = random.Random(seed)

    def pol(obs, acts, hist=None):
        q = q_values(net, obs, acts, hist)
        return mixed_index(q, margin=margin, temp=temp, rng=rng)

    return pol
