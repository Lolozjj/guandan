"""为**残局**生成一批"专家标签"：`(状态, 动作) -> 确定性 rollout 的均值回报`。

依据（2026-10-01 夜的干预实验，`tools/endgame_probe.py`）：
在 ~22% 的残局决策上，网络的选择**比 rollout-best 差 ~0.36 点**（同一副真实暗牌配对，
sd 0.50，t=2.39）⇒ 残局里有**稀疏但真实**的可提升空间。

做法：打若干局（我方 = 网络、对手 = 规则式）→ 在所有手牌 ≤ `--endgame-n` 的时刻抓快照 →
对网络 Q 的前 K 个候选各做 M 次确定性 rollout（暗牌随机分给三家）→ 把
`encode_state(obs)` / `encode_action_now(动作, level, 我的手牌)` / 均值回报 存成 `.npz`。

⚠️ 这是**专家蒸馏**，不是自对弈：标签的"对错"完全取决于 rollout 的质量
（M 次确定性采样 + 规则式当对手模型）。所以样本量要给足（默认 2 万条），
而且**必须**先看 `tools/endgame_probe.py` 的干预读数是对的方向，再花这个算力。
"""
from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys

import numpy as np

from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, features, rules
from tools.endgame_probe import _determinize, _unseen_ids

DEFAULT_ENDGAME_N = 8


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _configs(e_snap, m: int, rng) -> list:
    """M 组「暗牌分配」——**同一个局面的所有候选共用这一批**。

    这是**公共随机数**（common random numbers）：候选之间比的是差值，
    而"这副牌本身好打不好打"这部分的噪声在配对相减时**互相抵消**
    ⇒ 同样的 M 次采样能换到低得多的差值方差。

    为什么必须要（2026-10-01 夜量的）：M=24 时每个候选的 rollout 值 SE ≈ 0.4 点，
    而候选之间真正的差只有 ~0.36 点 ⇒ 不配对的话，**排序标签本身就有一半是噪声**，
    网络学到的成对准确率只能到 0.63（贴着噪声天花板）。
    """
    obs0 = e_snap.observe()
    out = []
    for _ in range(m):
        pool = _unseen_ids(obs0)
        rng.shuffle(pool)
        k, cfg = 0, []
        for s in [x for x in rules.SEATS if x != obs0.seat]:
            n = obs0.left[s]
            cfg.append(set(pool[k:k + n]))
            k += n
        out.append(cfg)
    return out


def _rollout(net, e_snap, k_act, rng, my_team, max_steps: int = 200, cfg=None) -> float:
    """走一步 `k_act` 之后：我方用网络、对手用规则式，返回我方终局回报。

    `cfg` 给定则用这一组暗牌（**配对**）；不给就自己抽一组（独立）。
    """
    e = copy.deepcopy(e_snap)
    if cfg is None:
        _determinize(e, e.observe(), rng)
    else:
        for s, cards in zip([x for x in rules.SEATS if x != e.hand.turn], cfg):
            e.hand.hands[s] = set(cards)
    e.step(k_act)
    steps = 0
    while not e.done and steps < max_steps:
        seat = e.hand.turn
        obs, acts = e.observe(), e.legal()
        if rules.TEAM[seat] == my_team:
            i = int(np.argmax(q_values(net, obs, acts, env.encode_history(e.hand, seat))))
        else:
            i = rule_choose(obs, acts)
        e.step(i)
        steps += 1
    return float(rules.reward(e.ranks, e_snap.hand.turn)) if e.done else 0.0


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="生成残局专家标签")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--positions", type=int, default=400, help="最多抓多少个残局快照")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--endgame-n", type=int, default=DEFAULT_ENDGAME_N)
    ap.add_argument("--out", default="runs/ab/endgame_set.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    net = load(a.weights)
    rng = random.Random(a.seed)
    snaps, games_done = [], 0
    while len(snaps) < a.positions and games_done < a.games:
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        team0 = games_done % 2
        games_done += 1
        taken = False
        while not e.done:
            seat = e.hand.turn
            mine = rules.TEAM[seat] == team0
            if (not taken and mine
                    and max(len(e.hand.hands[s]) for s in rules.SEATS) <= a.endgame_n):
                snaps.append((copy.deepcopy(e), team0))
                taken = True
            obs, acts = e.observe(), e.legal()
            i = (int(np.argmax(q_values(net, obs, acts, env.encode_history(e.hand, seat))))
                 if mine else rule_choose(obs, acts))
            e.step(i)
    print(f"残局快照 {len(snaps)} 个（打了 {games_done} 局）")

    states, actions, hists, values, groups = [], [], [], [], []
    for gi, (e_snap, team0) in enumerate(snaps):
        obs, acts = e_snap.observe(), e_snap.legal()
        seat = e_snap.hand.turn
        q = [float(x) for x in q_values(net, obs, acts, env.encode_history(e_snap.hand, seat))]
        order = sorted(range(len(acts)), key=lambda i: -q[i])[:a.top_k]
        st = env.encode_state(obs)
        # ⚠️ **必须存真实历史**：网络的前向是 (state, action, 15x147 历史)，
        # 用零历史微调 = 在一个推理时不会出现的输入分布上训练（口径漂）。
        hist = env.encode_history(e_snap.hand, seat)
        cfgs = _configs(e_snap, a.samples, rng)          # **公共随机数**：候选共用
        for k in order:
            v = statistics.mean([_rollout(net, e_snap, k, rng, team0, cfg=c)
                                 for c in cfgs])
            states.append(st)
            actions.append(env.encode_action_now(acts[k], obs.level, obs.hand))
            hists.append(hist)
            values.append(v)
            groups.append(gi)
    if not states:
        sys.exit("一条样本都没生成")
    np.savez_compressed(a.out, states=np.stack(states).astype(np.float32),
                        actions=np.stack(actions).astype(np.float32),
                        hists=np.stack(hists).astype(np.float32),
                        values=np.array(values, dtype=np.float32),
                        groups=np.array(groups, dtype=np.int32))
    print(f"写出 {len(values)} 条样本 -> {a.out}"
          f"（状态 {np.stack(states).shape}，动作 {np.stack(actions).shape}）")
    print(f"标签均值 {np.mean(values):+.3f} 点，sd {np.std(values):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
