"""网络与「给一批候选打分」的工具 —— `guandan/rl/selfplay.py` 用。

**不是复制品**：这两个脚本原本各要一份 `QNet` 与 `_q`，而两份实现会漂
（本仓库为「副本会漂」吃过亏）。所以只留这一份。

网络形状照 spec §4.3（对齐 DouZero）：
    历史(15×147) → LSTM(128) ─┐
                              ├→ 拼接 → 6 层 MLP(512) → 一个 Q 值
    状态(700) + 单个候选动作(143) ┘
"""
from __future__ import annotations

import os

import statistics

import numpy as np
import torch
import torch.nn as nn

from guandan.sim import env

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


def _flat_scores(net, pending, grad: bool = False):
    """`pending = [(obs, acts, hist), ...]` -> `(q, counts)`：所有候选的 Q 首尾相接。

    `grad=True` 时**不套 `no_grad`** —— 策略梯度要穿过 logits（见
    `log_prob_and_entropy`）。⚠️ 默认 `False` 是**故意的**：老的调用方
    （`q_argmax_batch` / `q_max_batch`）要的是"算出来的值"，带梯度进去只会白建图。

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
    args = (torch.from_numpy(st[idx]).to(dev), torch.from_numpy(ac).to(dev),
            torch.from_numpy(hi[idx]).to(dev))
    if grad:
        return net(*args), counts
    with torch.no_grad():
        return net(*args), counts


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


def policy_sample_batch(net, pending, rng) -> list:
    """从 `π = softmax(该局面全部候选的 logits)` **采样**，返回每个决策点的下标。

    这是 PG 的**行为策略** —— on-policy 的定义就落在这一个函数上：
    数据必须来自 π 自己，否则策略梯度估的是另一个分布的梯度（静默学歪）。

    ⚠️ 用**调用方的 `rng`**（`random.Random`）而不是 torch 的生成器：
    整条链的可复现性都挂在同一个 `rng` 上（`generate_batch` 连「哪一队当对手」
    都用它抽）。走逆累积分布，不用 `torch.multinomial`。
    """
    q, counts = _flat_scores(net, pending)      # 采样不需要梯度
    out, off = [], 0
    for c in counts:
        p = torch.softmax(q[off:off + c], dim=0)
        off += c
        r, acc, pick = rng.random(), 0.0, c - 1     # 兜底取最后一个，防浮点累积误差
        for j in range(c):
            acc += float(p[j])
            if r < acc:
                pick = j
                break
        out.append(pick)
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


#: **logits 的溢出兜底（不是"发散守门"）**。
#:
#: ⚠️ **别把 Q 的那个阈值（`Q_ABS_MAX = 30`）搬过来** —— 两者是不同量纲的东西：
#: `Q` 是**值**，标签尺度 ±3，所以 30 就是发散；而 **logits 是「对数几率」，没有
#: 有界尺度** —— `softmax` 把任何有限的 logits 映射成合法分布，`|z|` 大只说明
#: 「这一手很确定」，**不是发散**。
#:
#: 2026-09-29 实测（这个坑真踩了）：照搬 30 之后，PG 跑到 **8,256 局被误杀**
#: ——当时 `loss=0.152`、熵 0.84，**一切正常**。所以这里只留一个纯溢出兜底（1e4），
#: 真正管「策略退化」的是 `check_entropy`。
LOGIT_ABS_MAX = 1e4


#: 熵的下限：低于「候选数对应的均匀熵」的这个比例，就认为策略塌成了 argmax。
#: 0.05 = 均匀熵的二十分之一 —— 那是"几乎确定"的意思，不是"有一点偏好"。
ENT_FLOOR_FRAC = 0.05


def log_prob_and_entropy(net, samples):
    """`samples = [(obs, acts, 选中下标, 出牌人, hist), ...]`（`replay.expand` 的产出形状）

    返回 `(lp, ent, zmax)`：

    - `lp[i] = log π(a_i | s_i)`，`π = softmax(该局面全部候选的 logits)`
    - `ent[i] = π 的香农熵`（自然对数）
    - `zmax` = 这一批 logits 的 `|z|` 最大值（float，给发散守门用）

    `lp`/`ent` 是 **1-D tensor 而不是 float** —— 与 `q_max_batch` 正好相反：
    那里返回 float 是**故意的**（目标项不许带梯度），这里要的就是梯度。

    ⚠️ 内部走 `_flat_scores(..., grad=True)`。**忘了 `grad=True` 的话，loss 会变成
    常数、梯度为 `None`、训练一步都不动，而日志上只看到 loss 平着不动**
    —— `tests/test_pg_logprob.py` 里那条「训一步参数必须变」就是钉这个的。
    """
    q, counts = _flat_scores(net, [(o, a, h) for o, a, _i, _s, h in samples],
                             grad=True)
    lps, ents, off = [], [], 0
    for (_o, _a, i, _s, _h), c in zip(samples, counts):
        seg = torch.log_softmax(q[off:off + c], dim=0)
        off += c
        lps.append(seg[i])
        ents.append(-(seg.exp() * seg).sum())
    return torch.stack(lps), torch.stack(ents), float(q.detach().abs().max())


def check_logits(zmax: float, games: int, loss: float,
                 limit: float = LOGIT_ABS_MAX) -> None:
    """logits 的**溢出兜底**（不是"发散守门"）。见 `LOGIT_ABS_MAX` 的注释。

    ⚠️ 用 `not (x <= limit)` 写 ⇒ **NaN / inf 一起拦住**（与 `check_q_scale` 同一个坑）。
    """
    if not (zmax <= limit):
        raise RuntimeError(
            f"logits 溢出：|z| = {zmax:.4g} > {limit:g}（第 {games:,} 局，loss={loss:.3f}）"
            f" —— 数值炸了。（注意：**几十的 logits 是正常的**，见 LOGIT_ABS_MAX 的注释）")


def check_entropy(hs, log_ks, frac: float = ENT_FLOOR_FRAC) -> None:
    """逐个决策点算「熵 / 均匀熵」，**中位数**低于 `frac` 就 raise。

    熵塌 = 策略退化成确定性的 argmax = **白换框架**，所以它必须响，不许静默训完。

    ⚠️ **为什么是逐样本的中位数，而不是两个批均值相除**（评审 2026-09-29 的 I2）：
    批均值会被「候选多的局面」抬过去 —— 90% 的决策点已经 argmax（H=0）、
    10% 还有 20 个候选且均匀（H=3.0）⇒ 两个均值一比仍然"健康" ✗，
    可实际上绝大部分决策已经退化了 ✗。中位数对这种情况不会瞎。

    `k=1` 的点（只有一手可出）**跳过** —— 那里的熵恒为 0，没有信息（也没有除零）。
    ⚠️ 用 `not (x >= frac)` 写 —— NaN 会顺着中位数传上来，写成 `<` 会让它溜过去。
    """
    ratios = [h / lk for h, lk in zip(hs, log_ks) if lk > 0]
    if not ratios:
        return
    med = statistics.median(ratios)
    if not (med >= frac):
        raise RuntimeError(
            f"策略熵塌了：逐点「熵/均匀熵」的**中位**={med:.4g} < {frac:.4g}"
            f"（{len(ratios)} 个决策点）—— 已经退化成 argmax，等于白换框架。"
            f"调 `--beta-ent`（当前默认见 BETA_ENT）")


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
            f"（标签尺度是 ±3；确认要放宽就动 guandan/rl/net.py::Q_ABS_MAX）")
