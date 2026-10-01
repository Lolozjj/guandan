"""**推理时的搜索**（独立于信用分配那条线）：一步前瞻，**不改任何权重**。

为什么值得试：训练是算力穷的（~31 局/秒），而**面板给建议可以有秒级预算** ——
这是 10~50 倍的不对称，一直没利用。诊断说 Q 的动作分辨率只有 0.100，
那么"多算一次前向"是唯一能在**不重训**的前提下把分辨率放大的办法。

**做法（1 步前瞻 = policy improvement operator）**：对每个候选 `a` 走一步，然后

    V(a) = 这一手直接终局 ? 终局分（精确）
                         : sign × max_{a'} Q(s'(a), a')

`sign = +1` 若后继该出手的是**我队友**，`-1` 若是对手 —— 奖励是零和的
（`rules.reward` 按队伍给分），所以"对手的最优值"就是我方的负值。
⚠️ 这是**近似**：Q 是相对出手人编码的，跨座位只能靠零和这条性质搬过去。

**成本控制**：只在前 `top_k` 个候选上做（先算一次 Q 取短名单）⇒ 每个决策点
**两次前向**（一次短名单、一次后继批量），而不是"每候选一次"。
`top_k=1` 时它**必须退化成 argmax**（测试钉着这条）。

⚠️ 潜在坑：Q 的差只有 0.1 量级 ⇒ 一步前瞻完全可能只是**放大了噪声**。
所以先筛（`tools/lookahead_screen.py`，多种子配对），只当筛子过了才接进 `advise`。
"""
from __future__ import annotations

import copy

from guandan.rl.net import q_max_batch, q_values
from guandan.sim import env as envmod
from guandan.sim import rules


def one_step_backup(net, items, top_k: int = 4, signed: bool = True) -> list:
    """`items = [(env, (obs, acts, hist)), ...]` -> 每个挑一个下标。

    一次调用把这一时刻**所有局**的决策点一起算 —— 前向是批量的，别逐局调。

    ⚠️ `signed=False` 只是**诊断用**：它去掉"对手回合取负"那一步，
    用来分辨"是符号规则错"还是"值本身不可比"（见 `rollout_backup` 的 docstring）。
    """
    if top_k < 1:
        raise ValueError(f"top_k 至少 1（收到 {top_k}）—— 0 的话无候选可选")

    orders: list = []
    rows: list = []          # 后继的 (obs, acts, hist)
    slot: dict = {}          # (j, k) -> rows 下标
    term: dict = {}          # (j, k) -> 精确终局分
    sign: dict = {}          # (j, k) -> ±1

    for j, (e, (obs, acts, hist)) in enumerate(items):
        q = [float(x) for x in q_values(net, obs, acts, hist)]
        # 短名单按当前 Q 排；**平手时保持原 argmax 在前**（`sorted` 稳定）
        order = sorted(range(len(acts)), key=lambda i: -q[i])[:top_k]
        orders.append(order)
        for k in order:
            e2 = copy.deepcopy(e)
            e2.step(k)
            if e2.done:
                # 终局分是精确值，与 Q 同尺度（同一 reward），可以直接比
                term[(j, k)] = float(rules.reward(e2.ranks, obs.seat))
            else:
                nxt = e2.hand.turn
                same = rules.TEAM[nxt] == rules.TEAM[obs.seat]
                sign[(j, k)] = 1.0 if (same or not signed) else -1.0
                slot[(j, k)] = len(rows)
                rows.append((e2.observe(), e2.legal(),
                             envmod.encode_history(e2.hand, nxt)))

    vals = q_max_batch(net, rows) if rows else []

    picks = []
    for j, order in enumerate(orders):
        best_v, best_k = None, order[0]        # 平手 / 全靠终局时退化到原 argmax
        for k in order:
            if (j, k) in term:
                v = term[(j, k)]
            else:
                v = sign[(j, k)] * float(vals[slot[(j, k)]])
            if best_v is None or v > best_v:
                best_v, best_k = v, k
        picks.append(best_k)
    return picks


def rollout_backup(net, items, top_k: int = 4, max_steps: int = 12,
                   sign_same_team_only: bool = False) -> list:
    """**更讲道理的一步前瞻**：走到"我队下一次出手"再估值。

    为什么不用 `one_step_backup`：它把**下一个出手人**的 Q 直接搬过来，
    而 Q 的量级（≈±2.5，"这局大概率赢"）**远大于**动作差（≈0.1）⇒
    一旦按"对手回合取负"，比较就被**符号**支配，策略退化成
    "尽量把出牌权交给队友"（实测 **−17.8pp**，t=−30.2 —— 见计划 §二）。

    这里改成：候选走了之后，用**规则式当别人的模型**往前推到"轮到我队"，
    在那个状态上取 `max Q`（**同队视角，不需要符号翻转**）；中途终局就用精确终局分。

    `sign_same_team_only`：推不到我队时（超过 `max_steps`）是否仍按符号规则近似 ——
    默认 False（直接按"当前出手方是不是对手"取符号，等价于老口径，只是少见）。
    """
    if top_k < 1:
        raise ValueError(f"top_k 至少 1（收到 {top_k}）")
    from guandan.rl.rule_policy import rule_choose

    orders: list = []
    rows: list = []
    slot: dict = {}
    term: dict = {}
    sign: dict = {}

    for j, (e, (obs, acts, hist)) in enumerate(items):
        my_team = rules.TEAM[obs.seat]
        q = [float(x) for x in q_values(net, obs, acts, hist)]
        order = sorted(range(len(acts)), key=lambda i: -q[i])[:top_k]
        orders.append(order)
        for k in order:
            e2 = copy.deepcopy(e)
            e2.step(k)
            n = 0
            while (not e2.done and rules.TEAM[e2.hand.turn] != my_team
                   and n < max_steps):
                e2.step(rule_choose(e2.observe(), e2.legal()))
                n += 1
            if e2.done:
                term[(j, k)] = float(rules.reward(e2.ranks, obs.seat))
                continue
            same = rules.TEAM[e2.hand.turn] == my_team
            if not same and not sign_same_team_only:
                sign[(j, k)] = -1.0        # 推不到我队：退回老口径（少见）
            else:
                sign[(j, k)] = 1.0
            slot[(j, k)] = len(rows)
            rows.append((e2.observe(), e2.legal(),
                         envmod.encode_history(e2.hand, e2.hand.turn)))

    vals = q_max_batch(net, rows) if rows else []
    picks = []
    for j, order in enumerate(orders):
        best_v, best_k = None, order[0]
        for k in order:
            v = (term[(j, k)] if (j, k) in term
                 else sign[(j, k)] * float(vals[slot[(j, k)]]))
            if best_v is None or v > best_v:
                best_v, best_k = v, k
        picks.append(best_k)
    return picks
