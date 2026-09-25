"""网络与「给一批候选打分」的工具 —— `train/smoke.py` 与 `train/selfplay.py` 共用。

**不是复制品**：这两个脚本原本各要一份 `QNet` 与 `_q`，而两份实现会漂
（本仓库为「副本会漂」吃过亏）。所以只留这一份。

网络形状照 spec §4.3（对齐 DouZero）：
    历史(15×147) → LSTM(128) ─┐
                              ├→ 拼接 → 6 层 MLP(512) → 一个 Q 值
    状态(700) + 单个候选动作(143) ┘
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from net.sim import env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLP_LAYERS = 6
MLP_HIDDEN = 512
LSTM_HIDDEN = 128


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(env.HISTORY_DIM, LSTM_HIDDEN, batch_first=True)
        layers, d = [], env.STATE_DIM + env.ACTION_DIM + LSTM_HIDDEN
        for _ in range(MLP_LAYERS):
            layers += [nn.Linear(d, MLP_HIDDEN), nn.ReLU()]
            d = MLP_HIDDEN
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, state, action, hist):
        _out, (h, _c) = self.lstm(hist)
        return self.mlp(torch.cat([state, action, h[-1]], dim=-1)).squeeze(-1)


def _q(net, obs, acts, hist):
    """一次前向算出一批候选的 Q。`obs`/`hist` 是单个局面的。"""
    st = torch.from_numpy(env.encode_state(obs)).unsqueeze(0).to(DEVICE)
    ac = torch.from_numpy(np.stack([env.encode_action(a, obs.level)
                                    for a in acts])).to(DEVICE)
    hi = torch.from_numpy(hist).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return net(st.expand(len(acts), -1), ac, hi.expand(len(acts), -1, -1))


def q_values(net, obs, acts, hist):
    """一次前向算出一批候选的 Q。`obs`/`hist` 是**单个局面**的，`acts` 是它的候选。

    ⚠️ **设备从 `net` 的参数上取，不用模块级的 `DEVICE`。**
    写死 `DEVICE` 的话，任何在 CPU 上造网络的调用方（测试、诊断脚本最自然就这么写）
    会撞上 `RuntimeError: Input and parameter tensors are not at the same device`
    —— 那个报错看不出是「网络在 CPU、输入被送去了 GPU」。
    """
    dev = next(net.parameters()).device
    st = torch.from_numpy(env.encode_state(obs)).unsqueeze(0).to(dev)
    ac = torch.from_numpy(np.stack([env.encode_action(a, obs.level)
                                    for a in acts])).to(dev)
    hi = torch.from_numpy(hist).unsqueeze(0).to(dev)
    with torch.no_grad():
        return net(st.expand(len(acts), -1), ac, hi.expand(len(acts), -1, -1))


def q_argmax_batch(net, pending):
    """`pending = [(obs, acts, hist), ...]` —— **一次前向**给所有决策点的候选打分，
    返回每个决策点的 argmax 下标。

    **为什么必须批量**：自对弈里每个决策点只问一次网络，一次前向的候选常常不到 10 个，
    GPU 的批处理完全浪费。剖面（2026-09-25）显示这条路径被调用 37,689 次、
    累计 **70.8 秒**，是训练步里最大的一块 —— 比规则引擎大一个数量级。
    把同一时刻的所有决策点拼成一批之后，同样次数的前向变成 1/N。

    不等长的候选**不补 padding**：直接首尾相接，用每组的下标区间取 argmax。
    补 padding 会白白多算一截，而且要把「无效候选」屏蔽掉，多一处出错的机会。
    """
    dev = next(net.parameters()).device
    counts = [len(acts) for _o, acts, _h in pending]
    st = np.stack([env.encode_state(o) for o, _a, _h in pending])
    hi = np.stack([h for _o, _a, h in pending])
    ac = np.stack([env.encode_action(a, o.level)
                   for o, acts, _h in pending for a in acts])
    idx = np.repeat(np.arange(len(pending)), counts)
    with torch.no_grad():
        q = net(torch.from_numpy(st[idx]).to(dev),
                torch.from_numpy(ac).to(dev),
                torch.from_numpy(hi[idx]).to(dev))
    out, off = [], 0
    for c in counts:
        out.append(int(q[off:off + c].argmax()))
        off += c
    return out
