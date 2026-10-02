"""**坏习惯画像**：把用户实机反馈的那几类毛病，一次量清楚（带配对/对照）。

用户 2026-10-01 实机反馈：

1. **白炸** —— 「有牌可以压制的情况下会使用炸弹进行压制」；
2. **白用万能牌 / 过度升级** —— 「实际炸个 8888 就行了，但他会用万能牌构造成五个 8 去炸」。

两条都同构：**桌上有牌要压**（⇒ 每个合法候选都压得过）+ **存在"不浪费"的候选**
⇒ 选了浪费的那一手就不是必需的。判定与"白炸"共用同等口径（`rl/eval.py` 里只有一份）。

同时报**规则式自己**的同名比率 —— 它就是用户心里的"正常打法"参照。

    .venv/Scripts/python.exe -m tools.mistake_profile --weights models/best.pt --games 300
"""
from __future__ import annotations

import argparse
import sys

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_policy
from guandan.sim import env


def load_net(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def net_play(net):
    import numpy as np

    def pol(obs, acts, hist=None):
        return int(np.argmax(q_values(net, obs, acts, hist)))
    return pol


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="坏习惯画像（白炸 / 白用万能牌 / 过度升级）")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1002)
    a = ap.parse_args(argv)

    net = load_net(a.weights)
    pol = net_play(net)
    rule = rule_policy()

    print(f"权重 {a.weights}；{a.games} 局，**一次走局量全部**（换边口径与 eval.py 一致）\n")

    # ① 学生 vs 规则式（用户看到的就是这个组合）
    w = ev._bomb_stats(pol, a.games, a.seed, opponent=rule)
    ww = ev._wild_stats(pol, a.games, a.seed, opponent=rule)
    print("对方 = 规则式（≈ 用户实机里的对手）：")
    print(f"  学生  白炸 {w[0]:4d}/{w[1]:5d} = {w[0] / max(w[1], 1):6.1%}"
          f"   白用万能牌 {ww[0]:3d}/{ww[2]:5d} = {ww[0] / max(ww[2], 1):6.1%}"
          f"   其中『过度升级』{ww[1]:3d} = {ww[1] / max(ww[2], 1):5.1%}"
          f"   （{a.games} 局里用炸 {w[2]} 手、用万能牌 {ww[3]} 手）")
    # ② 规则式 vs 学生（参照）
    w2 = ev._bomb_stats(rule, a.games, a.seed, opponent=pol)
    ww2 = ev._wild_stats(rule, a.games, a.seed, opponent=pol)
    print(f"  规则式 白炸 {w2[0]:4d}/{w2[1]:5d} = {w2[0] / max(w2[1], 1):6.1%}"
          f"   白用万能牌 {ww2[0]:3d}/{ww2[2]:5d} = {ww2[0] / max(ww2[2], 1):6.1%}"
          f"   其中『过度升级』{ww2[1]:3d} = {ww2[1] / max(ww2[2], 1):5.1%}"
          f"   （{a.games} 局里用炸 {w2[2]} 手、用万能牌 {ww2[3]} 手）")
    # ③ 学生自对弈（历史教训：自对弈里不炸就被炸，浪费从未被罚）
    w3 = ev._bomb_stats(pol, a.games, a.seed)
    ww3 = ev._wild_stats(pol, a.games, a.seed)
    print(f"  学生自对弈（诊断用） 白炸 {w3[0]:4d}/{w3[1]:5d} = {w3[0] / max(w3[1], 1):6.1%}"
          f"   白用万能牌 {ww3[0]:3d}/{ww3[2]:5d} = {ww3[0] / max(ww3[2], 1):6.1%}")
    # ④ 用 tile 之后的率（同样一次走局；直接量"擦浪费"能压到多少）
    from guandan.rl.tidy import tidy_net_policy
    tp = tidy_net_policy(net)
    wt = ev._bomb_stats(tp, a.games, a.seed, opponent=rule)
    wwt = ev._wild_stats(tp, a.games, a.seed, opponent=rule)
    print(f"  **擦浪费后** 白炸 {wt[0]:4d}/{wt[1]:5d} = {wt[0] / max(wt[1], 1):6.1%}"
          f"   白用万能牌 {wwt[0]:3d}/{wwt[2]:5d} = {wwt[0] / max(wwt[2], 1):6.1%}"
          f"   （{a.games} 局里用炸 {wt[2]} 手、用万能牌 {wwt[3]} 手）")
    print("\n判读：白炸率/白用万能牌率**越低越好**；规则式那一行是「正常打法」的参照。"
          "\n  「擦浪费后」是**不改权重、只在推理时擦掉浪费**的效果（面板可以开这个）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
