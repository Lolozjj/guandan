"""对局中三方（我们的网络 / 贪心 / 随机）的策略接口。

三者同一个形状：

    policy(obs, acts, hist) -> index

`acts` 是 `env.legal()` 给的候选（**可能含 `None`** 表示「过」）；
`hist` 是 `env.encode_history(...)` 给的最近 15 手。评测器与训练循环共用这一套，
spec §7 的「vs 随机」「vs 贪心」两行就只用写一份评测器。

⚠️ **`hist` 由调用方算好递进来，策略自己拿不到 `env.hand`（明牌）。**
让策略收 `env` 会更省事，但那等于把四家手牌塞进策略手里 —— 这个项目对明牌泄漏的
态度是「靠类型堵，不靠自觉」（见 `net/sim/env.py`）。所以策略只吃可观测的东西。

**贪心基线的定义**（2026-09-25 与用户确认，它就是「Plan 3 过没过」那根尺子）：
> 能出就出；要出就出**最小的**那一手。表上有牌时，能压就压**最小的能压的那手**，
> 压不过才过。排序键 = `(张数, 主点数, 牌型)`。
"""
from __future__ import annotations

import random


def _sort_key(m):
    """`(张数, 主点数, 牌型)` —— 贪心眼里的「大小」。

    ⚠️ 这是**启发式的全序**，不是牌力序：`rank` 在序列类（顺子/连板，取自然值）
    与其余（`point_value`）之间口径不同，跨牌型比 `rank` 严格说是苹果比橘子。
    基线只需要「确定、单调、不瞎炸」—— **张数优先** 正好保证它不会主动砸炸弹
    （炸弹最少 4 张），这正是我们想要的「新手打法」。
    """
    return (m.size, m.rank, m.kind)


def greedy_policy(obs, acts, hist=None) -> int:
    """「有牌就出最小」。**能压就压，从不主动过** —— 这是它的弱点，也正是基线该有的样子。"""
    playable = [(i, m) for i, m in enumerate(acts) if m is not None]
    if not playable:
        return acts.index(None)              # 只能过
    return min(playable, key=lambda im: _sort_key(im[1]))[0]


def random_policy(rng: random.Random = None):
    """均匀随机（含「过」）。`rng` 传进来是为了评测可复现。"""
    r = rng or random.Random(0)

    def policy(obs, acts, hist=None) -> int:
        return r.randrange(len(acts))

    return policy


def net_policy(score_candidates):
    """把网络包成策略：对每个候选打分取 argmax。

    `score_candidates(obs, acts, hist) -> 张量` 由调用方给（`train/net.py` 的 `q_values`），
    这一层因此不需要知道张量在哪个设备上。
    """
    def policy(obs, acts, hist=None) -> int:
        return int(score_candidates(obs, acts, hist).argmax())

    return policy


def batch_net_policy(q_argmax_batch, net):
    """网络策略，**额外带一个 `batch_choose`** —— 评测器/训练循环可以一次把
    同一时刻的所有决策点送进网络（见 `train/net.py` 的 `q_argmax_batch`）。

    带这个属性的策略会被批量调用；不带的（随机 / 贪心）就逐决策点调用 ——
    它们便宜，没必要批。
    """
    from train.net import q_values

    def policy(obs, acts, hist=None) -> int:
        return int(q_values(net, obs, acts, hist).argmax())

    def batch_choose(pending):
        """`pending = [(obs, acts, hist), ...]` -> 每个决策点的选中下标。"""
        return q_argmax_batch(net, pending)

    policy.batch_choose = batch_choose
    return policy
