"""网络与「给一批候选打分」的工具 —— `train/selfplay.py` 用。

**不是复制品**：这两个脚本原本各要一份 `QNet` 与 `_q`，而两份实现会漂
（本仓库为「副本会漂」吃过亏）。所以只留这一份。

网络形状照 spec §4.3（对齐 DouZero）：
    历史(15×147) → LSTM(128) ─┐
                              ├→ 拼接 → 6 层 MLP(512) → 一个 Q 值
    状态(700) + 单个候选动作(143) ┘
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from net.sim import env

#: 用哪个设备训练/推理。**默认「有 CUDA 就用」**，但允许环境变量覆盖 ——
#: 因为实测（2026-09-26）这个训练循环**在 GPU 上反而更慢**：
#:     CPU 14~15 局/秒（Plan 3 那 5 万局也是这个量级）  vs  CUDA 11.5 局/秒
#: 瓶颈不是前向的算力，而是「一次前向只算一个局面的候选」的调用开销
#: （spec §14.3 量到的那 37,689 次调用），GPU 摊不平这个开销。
#: 所以夜里的长跑用 `GUANDAN_DEVICE=cpu`（还能与已有曲线直接比）。
DEVICE = os.environ.get("GUANDAN_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
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


def _flat_scores(net, pending):
    """`pending = [(obs, acts, hist), ...]` -> `(q, counts)`：所有候选的 Q 首尾相接。

    不等长的候选**不补 padding**：直接首尾相接，用每组的下标区间取 max / argmax。
    补 padding 会白白多算一截，而且要把「无效候选」屏蔽掉，多一处出错的机会。
    状态与历史按**局面**存一次（靠 `idx` 重复），只有动作是「每个候选一行」——
    这是内存上的关键一手：`hi` 若是按候选行铺开，一个训练步就要多几百 MB。
    """
    dev = next(net.parameters()).device
    counts = [len(acts) for _o, acts, _h in pending]
    st = np.stack([env.encode_state(o) for o, _a, _h in pending])
    hi = np.stack([h for _o, _a, h in pending])
    ac = np.stack([env.encode_action(a, o.level)
                   for o, acts, _h in pending for a in acts])
    idx = np.repeat(np.arange(len(pending)), counts)
    with torch.no_grad():
        return net(torch.from_numpy(st[idx]).to(dev),
                   torch.from_numpy(ac).to(dev),
                   torch.from_numpy(hi[idx]).to(dev)), counts


def q_argmax_batch(net, pending):
    """`pending = [(obs, acts, hist), ...]` —— **一次前向**给所有决策点的候选打分，
    返回每个决策点的 argmax 下标。

    **为什么必须批量**：自对弈里每个决策点只问一次网络，一次前向的候选常常不到 10 个，
    GPU 的批处理完全浪费。剖面（2026-09-25）显示这条路径被调用 37,689 次、
    累计 **70.8 秒**，是训练步里最大的一块 —— 比规则引擎大一个数量级。
    把同一时刻的所有决策点拼成一批之后，同样次数的前向变成 1/N。
    """
    q, counts = _flat_scores(net, pending)
    out, off = [], 0
    for c in counts:
        out.append(int(q[off:off + c].argmax()))
        off += c
    return out


def q_max_batch(net, pending):
    """`pending = [(obs, acts, hist) | None, ...]` -> `list[float | None]`。

    每个局面的 `V(s) = max_a Q(s, a)` —— 自举项（spec §3.1）。
    `None` 的位置（越过终局的点 / 本轮不算自举的）原样返回 `None`，**位置不许挪**：
    挪一格就是「拿别人的未来当自己的标签」，而那种错在 loss 曲线上看不出来。

    ⚠️ 返回的是 **Python float，不是 tensor** —— 这是故意的：
    目标项绝不能带梯度，而 float 根本没法 `backward`。类型上就断掉了，不靠注释
    （本仓库对「明牌泄漏」用的也是同一招：靠类型，不靠纪律）。
    """
    idxs = [i for i, p in enumerate(pending) if p is not None]
    out = [None] * len(pending)
    if not idxs:                       # 全 None 时 `np.stack([])` 会炸，先挡掉
        return out
    q, counts = _flat_scores(net, [pending[i] for i in idxs])
    off = 0
    for i, c in zip(idxs, counts):
        out[i] = float(q[off:off + c].max())
        off += c
    return out


#: 发散守门（spec §6）：|Q| 超过这个数就**响亮地炸**。
#: 标签尺度是 ±3，一个数量级以上的偏离只可能是自举发散了。
Q_ABS_MAX = 30.0


def check_q_scale(q_abs_max: float, games: int, loss: float, what: str = "Q",
                  limit: float = Q_ABS_MAX) -> None:
    """`|Q|` 超过 `limit` 就 raise（本项目纪律：失败必须响，不许静默地训下去）。

    ⚠️ **必须用 `not (x <= limit)` 写。** NaN 与任何数比较都是 False，
    写成 `x > limit` 会让 NaN 悄悄溜过去 —— 而 NaN 正是发散最典型的形态。
    """
    if not (q_abs_max <= limit):
        raise RuntimeError(
            f"{what} 的量级炸了：{q_abs_max:.4g} > {limit:g}"
            f"（第 {games:,} 局，loss={loss:.3f}）—— 自举发散，停在这里。"
            f"（标签尺度是 ±3；确认要放宽就动 train/net.py::Q_ABS_MAX）")
