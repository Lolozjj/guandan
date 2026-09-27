"""对手池的记账与采样（spec §3.1/§3.2）。

**只管两件事**：记住「每个成员打了多少局、学习者赢了多少」，按 PFSP 算出采样权重。
不碰进程、不碰 torch、不碰文件 —— 所以它能在任何地方单测。

## 为什么权重是 `p·(1−p)`

`p` 是**学习者对该成员的胜率**。

- `p → 1`（已经打穿）：赢它不带来任何信息，白花对局。
- `p → 0`（完全打不过）：全是输，标签里没有对比度，也没梯度。
- `p ≈ 0.5`：**只有这里学得到东西**。

所以两头都趋 0，再加一个**均匀下限**（防饿死 + 兜住预热期与陈旧的胜率估计）。

## 为什么贪心要留一份固定份额

取 0.8。它是判据的尺子（`vs 贪心` 要和历史跑可比），它便宜（走 Python 捷径、
不占前向），而且它是**防池子退化的锚** —— 池子全塌成一个成员时，至少还有它在变。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: 局数少于这个数的成员，`p` 还是噪声（0/0）—— 按均匀算。
PFSP_MIN_GAMES = 20
#: 胜率滑窗。学习者在变强，老胜率会过期。
PFSP_WINDOW = 200
#: 均匀下限：占采样权重的这个比例。防饿死 + 兜底。
PFSP_UNIFORM = 0.2
#: 混合局里贪心的固定份额。它是判据的尺子，也是防池子退化的锚。
GREEDY_SHARE = 0.2
#: Beta 平滑的伪计数（`(wins+a)/(games+a+b)`）。
PRIOR = 2.0
#: 有效成员数低于这个值就报「池子塌了」。
COLLAPSE_BELOW = 2.0


@dataclass(frozen=True)
class PoolMember:
    """池子里的一个成员：一个 id + 一份权重文件。"""

    mid: int
    path: str


class WinRates:
    """每个成员一个**滑窗**的胜负计数（**学习者视角**：True = 学习者赢）。

    为什么是滑窗而不是累计：学习者在变强。一个成员在 50 万局前的胜率
    说明不了现在的事 —— 累计计数会让「早就打穿的成员」永远挂着高胜率。
    """

    def __init__(self, window: int = PFSP_WINDOW):
        self.window = window
        self._w = {}

    def record(self, mid: int, learner_won: bool) -> None:
        self._w.setdefault(mid, deque(maxlen=self.window)).append(bool(learner_won))

    def games(self, mid: int) -> int:
        return len(self._w.get(mid, ()))

    def rate(self, mid: int) -> float:
        """Beta 平滑后的胜率。没打过 -> 0.5（中性）。"""
        d = self._w.get(mid)
        if not d:
            return 0.5
        return (sum(d) + PRIOR) / (len(d) + 2 * PRIOR)


def pfsp_weights(rates: dict, games: dict = None, uniform: float = PFSP_UNIFORM,
                 min_games: int = PFSP_MIN_GAMES) -> dict:
    """`{mid: p}` -> `{mid: 采样权重}`（已归一化，和恒为 1）。

    局数 < `min_games` 的成员，**把它的 `p` 当成中性的 0.5**（它的胜率还是噪声）。

    ⚠️ **不能给预热成员一个「特殊的大常数」**：`p(1−p)` 的最大值只有 **0.25**，
    给 `1.0` 就等于让它比健康成员重 4 倍 —— 池子会一直扑向没测过的成员。
    中性的 0.5 落在 PFSP 量程顶端，语义也对：「不知道，就先当它可能是最合适的那个」。
    """
    ids = sorted(rates)
    k = len(ids)
    if k == 0:
        return {}
    games = games or {}
    raw = {}
    for i in ids:
        p = 0.5 if games.get(i, 0) < min_games else rates[i]
        raw[i] = p * (1.0 - p)
    tot = sum(raw.values())
    if tot <= 0:                             # 全塌成 0（理论上不会，防一手）
        return {i: 1.0 / k for i in ids}
    return {i: (1.0 - uniform) * raw[i] / tot + uniform / k for i in ids}


def pick_opponent(rng, member_ids, weights: dict,
                  greedy_share: float = GREEDY_SHARE):
    """这一局的固定对手是谁。

    先按 `greedy_share` 决定「是不是贪心」，否则按 `weights` 在池成员里抽一个。

    ⚠️ **权重还没到（第一次 PFSP 广播之前）时按均匀抽一个成员，不退回贪心** ——
    退回贪心会让「池子刚起步那一段」偷偷变成纯贪心局，而日志上一概看不出来。
    """
    if rng.random() < greedy_share or not member_ids:
        return ("greedy",)
    ids = sorted(member_ids)
    tot = sum(weights.get(i, 0.0) for i in ids)
    if tot <= 0:
        return ("member", ids[rng.randrange(len(ids))])
    r = rng.random()
    acc = 0.0
    for i in ids:
        acc += weights.get(i, 0.0)
        if r <= acc:
            return ("member", i)
    return ("member", ids[-1])               # 浮点兜底


def effective_members(weights: dict) -> float:
    """有效成员数 `1 / Σw²` —— 池子塌成一个成员时会趋近 1（spec §1.4）。"""
    s = sum(w * w for w in weights.values())
    return 1.0 / s if s > 0 else 0.0
