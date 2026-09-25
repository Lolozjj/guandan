"""出牌建议的推理链（spec §8.1）：`GameState → Observation → 候选 → 打分 → 建议`。

**不加载模拟器**：这里不建 `GuandanEnv`、不建 `rules.Hand`、不跑对局循环。
但要**复用**它的编码器与裁判（`net/sim/env.py` 的 `encode_*`、`net/sim/meld.py` 的
`legal_moves/as_meld`、`train/net.py` 的 `q_values`）—— 本仓库为「副本会漂」吃过亏，
推理侧另写一份编码器，训练与上线就会悄悄不一致。

⚠️ **`import train.net` 是刻意的**：模型定义只能有一份。要改方向就把 `QNet`
挪到共用模块，**别复制**。

这个模块是**明牌与策略之间唯一的窄口**（和 `env.observe` 同一个角色）：
进来的是面板的状态机（里面有四家的出牌记录），出去的 `Observation` 里
只有公开信息 + 我的手牌 —— 那个类型的字段表里就没有「对手手牌」。
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from net.sim import env, meld, rules
from net.state import GameState

# 「算不了」的原因。**每一个都要能出现在影子日志里** —— 静默跳过等于数据有偏。
SKIP_SEAT = "座位未确认"
SKIP_LEVEL = "级别未知或越界"
SKIP_TABLE = "桌面牌解不出牌型"
SKIP_HIST = "流水里有解不出的历史着法"
SKIP_HAND = "手牌为空"
SKIP_NOCAND = "枚举不出任何着法"
SKIP_NOTURN = "没轮到我"

#: 一个座位起手 27 张（两副牌 108 / 4）。`remaining` 没记到时用它兜底。
DEAL_EACH = 27


@dataclass
class Skip:
    reason: str


@dataclass
class Built:
    obs: Optional[env.Observation] = None
    hist: Optional[np.ndarray] = None
    table_meld: Optional[meld.Meld] = None
    reason: str = ""


@dataclass
class Advice:
    obs: env.Observation
    hist: np.ndarray
    cands: list                 # 候选，含 None（过）
    q: List[float]              # 与 cands 等长
    order: List[int]            # 按 Q 降序的下标


class _HandLike:
    """`env.encode_history` 只要两样东西：`.steps`（每步有 `.meld` / `.seat`）与 `.level`。

    **故意复用它，不另写一份编码器** —— 训练与推理的历史必须逐位一致。
    """

    def __init__(self, steps, level):
        self.steps = steps
        self.level = level


def _left_of(st: GameState, seat: int) -> int:
    """该座位还剩几张。`remaining` 是服务器给的真值（每条出牌消息都带）。"""
    if seat in st.remaining:
        return st.remaining[seat]
    played = sum(len(p.cards) for p in (st.history.get(seat) or []))
    return max(0, DEAL_EACH - played)


def history_from_steps(steps, level: int, seat: int) -> Optional[np.ndarray]:
    """动作流水 -> `(15, 147)` 历史。**解不出的历史着法直接返回 None**（不猜）。

    ⚠️ 传进来的必须是**决策点当时**的流水（`st.steps` 的当前快照）——
    用打完之后的流水会把后面才发生的牌塞进历史，那是未来信息泄漏。
    """
    out = []
    for s, cs in steps[-env.HISTORY_LEN:]:
        if cs is None:
            out.append(rules.Step(s, None, 0))
            continue
        m = meld.as_meld(list(cs), level)
        if m is None:
            return None
        out.append(rules.Step(s, m, 0))
    return env.encode_history(_HandLike(out, level), seat)


def build(st: GameState) -> Built:
    """`GameState` -> 可观测状态 + 历史。算不了就返回 `reason`（**不抛异常**）。"""
    if not st.me_confirmed:
        return Built(reason=SKIP_SEAT)
    if st.turn != st.me:
        return Built(reason=SKIP_NOTURN)
    if not st.hand:
        return Built(reason=SKIP_HAND)
    if st.level is None or not 1 <= st.level <= 13:
        return Built(reason=SKIP_LEVEL)

    table, table_meld = (), None
    kind, rank = 0, -1
    if st.table and st.table.cards:
        # 桌面那串牌 ID 必须先解释成带 rank 的 Meld，`beats` 才用得上 ——
        # 这正是 `meld.as_meld` 存在的理由（推理链是它的生产调用方）。
        table_meld = meld.as_meld(list(st.table.cards), st.level)
        if table_meld is None:
            # **不能当成「桌上无牌」**：那会让候选变成「随便领出」，全错。
            return Built(reason=SKIP_TABLE)
        table, kind, rank = tuple(table_meld.cards), table_meld.kind, table_meld.rank

    hist = history_from_steps(st.steps, st.level, st.me)
    if hist is None:
        return Built(reason=SKIP_HIST)

    obs = env.Observation(
        seat=st.me,
        hand=frozenset(st.hand),
        played=tuple(frozenset(c for p in (st.history.get(s) or []) for c in p.cards)
                     for s in rules.SEATS),
        left=tuple(_left_of(st, s) for s in rules.SEATS),
        table=table, table_kind=kind, table_rank=rank,
        passed=tuple(s in st.passes for s in rules.SEATS),
        turn=st.turn,
        level=st.level,
    )
    return Built(obs=obs, hist=hist, table_meld=table_meld)


def candidates(b: Built) -> list:
    """当前候选。口径与 `rules.Hand.actions` **完全一致**（那是在 55 局真牌上验过的）：
    `melds_from(sorted(hand))` 再按 `beats` 过滤；跟牌时 `None` 永远在候选里。
    """
    moves = meld.legal_moves(sorted(b.obs.hand), b.table_meld, b.obs.level)
    if b.table_meld is None:
        return list(moves)
    return list(moves) + [None]


def advise(st: GameState, net, topk: int = 3) -> "Advice | Skip":
    """算一次建议。算不了返回 `Skip(原因)` —— 调用方负责计数落盘。"""
    from train.net import q_values

    b = build(st)
    if b.reason:
        return Skip(b.reason)
    cands = candidates(b)
    if not cands:
        return Skip(SKIP_NOCAND)
    q = [float(x) for x in q_values(net, b.obs, cands, b.hist)]
    order = sorted(range(len(cands)), key=lambda i: -q[i])
    return Advice(obs=b.obs, hist=b.hist, cands=cands, q=q, order=order)


def newest_weights(root: str = "runs/rl") -> Optional[str]:
    """最新的一版 `best.pt`（按修改时间）。没有就返回 None。"""
    found = glob.glob(os.path.join(root, "*", "best.pt"))
    return max(found, key=os.path.getmtime) if found else None


def resolve_weights(path: str = None) -> Optional[str]:
    """**这一版实际会加载哪份权重 —— 只有这一处口径。**

    `load_net` 与影子日志的溯源字段都走它。分开写会漂：面板原来一边
    `load_net()`（吃 `GUANDAN_WEIGHTS`）、一边把 `newest_weights()` 记进日志，
    设了环境变量之后 session 行说的就是另一个文件。
    """
    return path or os.environ.get("GUANDAN_WEIGHTS") or newest_weights()


def load_net(path: str = None, device: str = "cpu") -> Tuple[Optional[object], str]:
    """加载权重。返回 `(net, 错误说明)`；**加载失败不抛异常**（面板不许因为这个崩）。

    默认放 **CPU**：一次决策点只做一次前向、候选不到 20 个，几十毫秒的量级；
    CPU 不跟游戏抢显存，结果也可复现（验收要拿它做逐位比对）。
    """
    import torch

    from train.net import QNet

    p = resolve_weights(path)
    if not p:
        return None, f"找不到权重（{os.environ.get('GUANDAN_WEIGHTS') or 'runs/rl/*/best.pt'}）"
    if not os.path.exists(p):
        return None, f"权重文件不存在：{p}"
    try:
        ck = torch.load(p, map_location="cpu")
        net = QNet()
        net.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)
        net.to(device).eval()
    except Exception as exc:                      # noqa: BLE001 - 面板不许崩
        return None, f"权重加载失败（{type(exc).__name__}: {exc}）：{p}"
    return net, ""


def weights_info(path: str) -> dict:
    """权重文件里的元信息（训练时存了局数与胜率）——写进影子日志的会话行做溯源。"""
    import torch
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception:                             # noqa: BLE001
        return {}
    if not isinstance(ck, dict):
        return {}
    return {k: ck[k] for k in ("games", "winrate_greedy", "winrate_random") if k in ck}
