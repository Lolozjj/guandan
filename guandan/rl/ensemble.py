"""**集成**：多个权重一起投票（零训练）。

为什么值得试：这是唯一"几乎不可能更差"的路 —— 它跟已经证否的**推理时搜索**完全不同类：
搜索是**跨视角搬值**（把对手回合的 Q 取负），失败机制是"把一个状态的 Q 搬到另一个状态"；
集成只是把**同一个量**的多个估计平均 ⇒ 那个失败机制不适用。

两条实现：

- `ensemble_policy(nets)`：**输出集成** —— 每个网络各算一遍候选 Q，取（加权）平均后 argmax。
  面板能接受（建议场景多算两三次前向无所谓）。
- `soup(nets)`：**权重集成**（model soup）—— 直接把 state_dict 平均。更便宜（推理不变慢），
  但只在各成员处在"可线性插值"的同一盆地时才有意义。

⚠️ 成员的形状可以不同（(700,143) / (727,143) / (727,146)）：统一走 `net.load_state`，
**缺的维度零填充** ⇒ 老成员的行为逐位不变（新维度乘 0）。
"""
from __future__ import annotations

import numpy as np
import torch

from guandan.rl.net import QNet, load_state, q_values


def load_net(path: str) -> QNet:
    """装一份存档（形状不同也能装：零填充）。"""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def ensemble_policy(nets, weights=None):
    """输出集成：`(obs, acts, hist) -> 下标`（与 `greedy_policy` 同形状）。"""
    ws = list(weights) if weights is not None else [1.0] * len(nets)
    if len(ws) != len(nets):
        raise ValueError(f"权重个数 {len(ws)} 与网络个数 {len(nets)} 不一致")
    tot = float(sum(ws))
    if tot <= 0:
        raise ValueError("权重之和必须为正")

    def choose(obs, acts, hist=None) -> int:
        acc = None
        for w, n in zip(ws, nets):
            v = np.asarray(q_values(n, obs, acts, hist), dtype=np.float64) * w
            acc = v if acc is None else acc + v
        return int(np.argmax(acc / tot))

    return choose


def soup(nets) -> QNet:
    """权重集成：把成员的 state_dict 逐参数平均（成员必须已装在**同一架构**上）。"""
    if not nets:
        raise ValueError("至少给一个网络")
    avg = None
    for n in nets:
        sd = n.state_dict()
        if avg is None:
            avg = {k: v.detach().float().clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg[k] += v.detach().float()
    for k in avg:
        avg[k] /= len(nets)
    out = QNet()
    out.load_state_dict(avg)
    out.eval()
    return out
