"""**配合诊断**：队友快走完时，我这一手**喂不喂得到他**？

为什么这是"打人"最该看的一项：
人类牌友最先抱怨的从来不是"你某手算错了"，而是**"你都不喂我"** ✗。
面板给的是建议，队友是**人**（而且水平未知）⇒ 会喂队友是"带得动人"的关键 ✓。

口径（只在**领出**时算，因为跟牌时你没有选择"喂谁"的自由）：

- 触发：轮到我**领出**（桌面空），且**队友剩 ≤2 张**；
- **喂到**：我领出的牌型/张数**队友跟得上**（队友手里确实有同牌型能压的牌）
  —— 这是"喂"的定义：**给他一个能接的出口** ✓
  （只用公开信息判不了的，这里用模拟器里的真实手牌算 —— 这是**诊断**，不是策略）；
- 同时报"队友剩 1 张时喂单张 / 剩 2 张时喂对子"的比例（规则式有明确规则做这件事 ✓）。

对照：同一个局面定义下量**规则式自己**的"喂到率" ⇒ 我们的模型差多少一目了然 ✓

    .venv/Scripts/python.exe -m tools.coop_profile --weights models/best.pt --games 200
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

import numpy as np
import torch

from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, meld, rules

#: 由 `--policy` 设定（模块级：`_model_pick` 要用）
_POLICY = "model"


def load_net(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _mate_can_follow(e, mate: int, m) -> bool:
    """队友手里有没有**能压过这一手**的牌（用真实手牌判 —— 只为诊断）。"""
    if m is None:
        return False
    hand = e.hand.hands[mate]
    for cand in meld.melds_from(set(hand), e.hand.level):
        if meld.beats(cand, m):
            return True
    return False


def _rate(pick, net, games, seed, tag):
    """`pick`：`决定这一手用什么` 的函数（模型 argmax / 规则式）⇒ 返回喂到率等。"""
    rng = random.Random(seed)
    lead_n = feed_n = single_need_ok = pair_need_ok = need1 = need2 = 0
    for g in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        mine_team = g % 2
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            mate = rules.PARTNER[seat]
            is_mine = rules.TEAM[seat] == mine_team
            if is_mine:
                i = pick(net, e, obs, acts)
                m = acts[i]
                if not obs.table and 0 < obs.left[mate] <= 2:          # 我领出（空元组！）+ 队友快走完
                    lead_n += 1
                    if obs.left[mate] == 1:
                        need1 += 1
                        single_need_ok += int(m is not None and m.kind == meld.SINGLE)
                    else:
                        need2 += 1
                        pair_need_ok += int(m is not None and m.kind == meld.PAIR)
                    feed_n += int(m is not None and _mate_can_follow(e, mate, m))
            else:
                i = (rule_choose(obs, acts) if tag == "model" else pick(net, e, obs, acts))
            e.step(i)
    return dict(lead=lead_n, feed=feed_n, need1=need1, need2=need2,
                single_ok=single_need_ok, pair_ok=pair_need_ok)


def _model_pick(net, e, obs, acts):
    from guandan.rl.coop import coop_index
    from guandan.sim import rules as _r
    q = q_values(net, obs, acts, env.encode_history(e.hand, e.hand.turn))
    i = int(np.argmax(q))
    if _POLICY == "coop":
        mate = _r.PARTNER[obs.seat]
        i = coop_index(q, acts, obs.left[mate], obs.table, i)
    return i


def _rule_pick(net, e, obs, acts):
    return rule_choose(obs, acts)


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="配合诊断：喂队友")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1002)
    ap.add_argument("--policy", choices=("model", "coop"), default="model",
                    help="model = 裸 Q-argmax；coop = 带「喂队友」护栏")
    a = ap.parse_args(argv)

    net = load_net(a.weights)
    print(f"{a.games} 局；「喂到」= 我领出的牌型队友**确实跟得上**（用真实手牌判，仅诊断）\n")
    global _POLICY
    _POLICY = a.policy
    for tag, pick, who in (("model", _model_pick, f"模型（{a.policy}）"), ("rule", _model_pick, "规则式（对照）")):
        # 对照跑法：我方也用规则式（对手仍是规则式）⇒ 同一局面定义下的规则式喂到率
        r = _rate(_rule_pick if tag == "rule" else pick, net, a.games, a.seed, tag)
        if not r["lead"]:
            print(f"  {who}: 没有触发局面"); continue
        print(f"  {who:14s} 触发 {r['lead']:4d} 次；**喂到率 {r['feed'] / r['lead']:6.1%}**"
              f"；队友剩1张时出单张 {r['single_ok']}/{r['need1']}"
              f"（{r['single_ok'] / max(r['need1'], 1):5.1%}）"
              f"；剩2张时出对子 {r['pair_ok']}/{r['need2']}"
              f"（{r['pair_ok'] / max(r['need2'], 1):5.1%}）")
    print("\n判读：喂到率低 ⇒ 队友（人）会觉得「你不带我」⇒ 这是「打人」最直接的一项短板。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
