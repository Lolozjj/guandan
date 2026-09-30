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

import glob
import os
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

#: 每这么多局存一个池子快照。**与「有没有刷新最好」解耦** ——
#: 只存 `best.pt` 的话 144 万局只落几个点，池子原料不够（spec §3.1）。
SNAP_EVERY_GAMES = 20_000
#: 池子留多少个成员（约 156 MB/worker）。
POOL_SIZE = 20


@dataclass(frozen=True)
class PoolMember:
    """池子里的一个成员：一个 id + 一份权重文件。"""

    mid: int        # 成员 id（池内唯一；`WinRates` 按它记账）
    path: str       # 权重文件路径（`<run>/pool/snap_<局数>.pt`）


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

    预热成员（局数 < `min_games`）拿**均匀那一份** `1/K`，不参与 PFSP 的分配。
    ⚠️ **不能给它「PFSP 的最大项」**（也就是把 p 当 0.5）：`p(1−p)` 在 0.5 取最大，
    那样没测过的成员会比健康成员重 —— 实测一个 0 局的新成员吃掉了 **38.4%**
    的采样权重（`runs/B_pool.log`），池子会一直扑向刚加进来的那个。
    剩下的 `1 − 预热份额` 由预热完的成员按 `p(1−p)` 分，再混均匀下限。
    """
    ids = sorted(rates)
    k = len(ids)
    if k == 0:
        return {}
    games = games or {}
    warm = [i for i in ids if games.get(i, 0) < min_games]
    hot = [i for i in ids if games.get(i, 0) >= min_games]
    out = {i: 1.0 / k for i in warm}         # 预热：均匀那一份
    if not hot:
        return out
    raw = {i: rates[i] * (1.0 - rates[i]) for i in hot}
    tot = sum(raw.values())
    base = ({i: raw[i] / tot for i in hot} if tot > 0
            else {i: 1.0 / len(hot) for i in hot})
    hot_share = 1.0 - len(warm) / k          # 预热吃掉的那部分不再参与分配
    for i in hot:
        out[i] = (1.0 - uniform) * base[i] * hot_share + uniform / k
    return out


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


# ---------------------------------------------------------------- 快照与装载

def snapshot_path(out_dir: str, games: int) -> str:
    """快照路径：`<out_dir>/pool/snap_<games>.pt`。

    ⚠️ **故意嵌套一层、且不叫 `best.pt`** —— 面板只认 `models/best.pt`
    （`guandan/advice/advise.py::resolve_weights`），快照落在 `runs/<臂>/pool/` 下，
    两层都命不中。命中就等于**静默换源**：用户面板上的建议会换成另一个模型，
    而日志上一概看不出来。
    """
    return os.path.join(out_dir, "pool", f"snap_{games}.pt")


def prune_snapshots(out_dir: str, keep: int = POOL_SIZE) -> list:
    """只留**最新**的 `keep` 个（按修改时间），返回被删掉的路径。"""
    found = sorted(glob.glob(os.path.join(out_dir, "pool", "snap_*.pt")),
                   key=os.path.getmtime)
    gone = found[:-keep] if keep > 0 else found
    for p in gone:
        os.remove(p)
    return gone


def load_members(paths) -> list:
    """磁盘上的一批权重文件 -> `PoolMember`（`mid` 按路径排序，稳定可复现）。"""
    return [PoolMember(mid=i, path=p) for i, p in enumerate(sorted(paths))]
