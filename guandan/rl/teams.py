"""**按座位分派策略**：让"我"和"队友"用不同的策略 —— 这正是**面板的真实场景**。

为什么要它（本夜的方向）：
面板给的是**建议**，牌桌上的队友是**人**，不是另一个模型 ✗。
所以"我的模型 + 一个外来队友"能不能赢，才是"对人有较好水平"最接近的可测代理 ✓。
`rl/eval.py::match(a, b)` 只支持"一队一个策略"，这里补上"一队两个策略"。

约定（两个朝向都要成立）：**同队里座位号小的那个坐"我"、大的坐"队友"**。
`PARTNER` 把 0↔2、1↔3 对应起来，所以这条约定在换边后依然一致 ✓。
"""
from __future__ import annotations

from guandan.sim import rules


def by_seat(seat_policies: dict):
    """`{座位: 策略}` —— 每个座位各自用它的策略（缺的座位会 KeyError，**不许静默兜底**）。"""
    def pol(obs, acts, hist=None):
        return seat_policies[obs.seat](obs, acts, hist)
    return pol


def pair_of(mine, mate):
    """返回一个策略：**我队**里小的座位用 `mine`、大的用 `mate`（对手队不经过这里）。"""
    def pol(obs, acts, hist=None):
        seat = obs.seat
        mate_seat = rules.PARTNER[seat]
        lower = min(seat, mate_seat)
        return (mine if seat == lower else mate)(obs, acts, hist)
    return pol
