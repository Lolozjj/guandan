"""量**标签噪声**：同一个残局局面，把 M 次确定性采样劈成两半，
两次得到的"哪个候选更好"是否一致？不一致率 = 排序标签的噪声。

为什么必须有这个量（2026-10-01 夜）：M=24 时每个候选的 rollout 值 SE ≈ 0.4 点，
而候选之间真正的差只有 ~0.36 点 ⇒ **不降噪的话排序标签本身一半是噪声**，
网络学到的成对准确率只能卡在 0.63 那种"贴着噪声天花板"的水平。

对照：**独立采样**（旧做法，每个候选各抽各的）vs **公共随机数**（新做法，候选共用同一批暗牌）。
后者在配对相减时把"这副牌好打不好打"的噪声抵消掉 ⇒ 差值方差应当显著更小。

    .venv/Scripts/python.exe -m tools.endgame_label_noise --positions 40 --samples 24 --top-k 4
"""
from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys

from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, rules
from tools.build_endgame_set import _configs, _rollout
from tools.endgame_probe import _determinize


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="残局标签噪声：公共随机数 vs 独立采样")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--samples", type=int, default=24, help="必须是偶数（要劈两半）")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--endgame-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.samples % 2:
        sys.exit("--samples 必须是偶数（要劈成两半比较）")

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
            i = (int(abs(q_values(net, obs, acts, env.encode_history(e.hand, seat)).argmax()))
                 if mine else rule_choose(obs, acts))
            e.step(i)
    print(f"残局快照 {len(snaps)} 个（打了 {games_done} 局），每局面劈两半各 {a.samples // 2} 次采样")

    m = a.samples // 2
    agree_crn, agree_ind, sd_crn, sd_ind = [], [], [], []
    for e_snap, team0 in snaps:
        obs, acts = e_snap.observe(), e_snap.legal()
        seat = e_snap.hand.turn
        q = [float(x) for x in q_values(net, obs, acts, env.encode_history(e_snap.hand, seat))]
        order = sorted(range(len(acts)), key=lambda i: -q[i])[:a.top_k]

        # ① 公共随机数：先抽 2m 组配置，前半/后半各用一半
        cfgs = _configs(e_snap, a.samples, rng)
        halves = []
        for part in (cfgs[:m], cfgs[m:]):
            halves.append([statistics.mean([_rollout(net, e_snap, k, rng, team0, cfg=c)
                                            for c in part]) for k in order])
        # ② 独立采样：每个候选自己抽（旧做法）
        ind = []
        for part in range(2):
            ind.append([statistics.mean([_rollout(net, e_snap, k, rng, team0)
                                         for _ in range(m)]) for k in order])
        agree_crn.append(int(halves[0].index(max(halves[0])) == halves[1].index(max(halves[1]))))
        agree_ind.append(int(ind[0].index(max(ind[0])) == ind[1].index(max(ind[1]))))
        if len(order) > 1:
            sd_crn.append(statistics.pstdev([x - y for x, y in zip(*halves)]))
            sd_ind.append(statistics.pstdev([x - y for x, y in zip(*ind)]))

    n = len(snaps)
    print(f"\n两半「谁更好」一致率：")
    print(f"  **公共随机数**（候选共用暗牌）：{sum(agree_crn)}/{n} = {sum(agree_crn) / n:.1%}")
    print(f"  独立采样（旧做法）        ：{sum(agree_ind)}/{n} = {sum(agree_ind) / n:.1%}")
    if sd_crn:
        print(f"\n两半估计之差的标准差（越小越稳）：")
        print(f"  **公共随机数** {statistics.mean(sd_crn):.3f} 点  "
              f"独立采样 {statistics.mean(sd_ind):.3f} 点  "
              f"⇒ 降噪 {1 - statistics.mean(sd_crn) / max(statistics.mean(sd_ind), 1e-9):.0%}")
    print("\n判读：一致率越高 ⇒ 排序标签越可信。若两者都只有 ~60%，"
          "说明 M 还不够，得继续加采样（或换更聪明的估计量）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
