"""**表示探针**：网络的内部表示里，有没有"这张牌在谁手上"的信息？

问题（`plans/2026-10-02-hand-probe.md`）：状态里有推牌所需的输入（已出牌 / 剩张 / passed / 桌面），
但**网络有没有真的把它变成对手手牌的推断**从未测过 ——
`belief-search.md` 否掉的是我**手工写**的信念，不是这件事。

四档对照，**只有输入不同**（同一个 2 层小 MLP、同样训练预算）：

| 档 | 输入 | 含义 |
|---|---|---|
| ① 计数基线 | 各家剩张数（猜剩得最多的那家） | 完全不作推理的下限 |
| ② 公开特征 | 只用公开信息派生的成对特征 | "把公开信息直接交给小网络" |
| ③ ② + 网络激活 | 再加 LSTM 隐层(128) / 首层 MLP 隐层(512) | **③−② 就是网络自己学出来的那部分** |

⚠️ **防泄漏三条**（不然整个实验就是自欺）：
1. 特征里**绝不许**出现真实持有者或任何上帝视角的量；
2. 训练/测试**按局切分**（同一局内的样本高度相关）；
3. 激活取自**当前局面**（同一局面的所有牌共用同一个向量），不许把标签编码进去。

    .venv/Scripts/python.exe -m tools.hand_probe --games 60 --positions 400
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from guandan.capture import cards as cardmod
from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, meld, rules


def load(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


#: 已出的牌按点数缓存（避免每对样本都扫 108 张）
def _rank_of(c: int) -> int:
    return cardmod.parts(c)[0]


def pair_features(obs, card: int, seat: int, mate: int) -> list:
    """`(这张牌, 这家)` 的**公开**特征 —— 逐条都能从公开信息算出来。

    特征顺序（12 条）：
        0 是不是我队友 / 1 这家已出几张 / 2 这家剩几张 / 3 本轮是否要不起 /
        4 这家同点数已出几张 / 5 该点数还有几张没露面 / 6 点数 /
        7 是否级牌 / 8 是否王 / 9 我剩几张 / 10 是否同队 / 11 桌面几张
    """
    rank = _rank_of(card)
    played_rank = sum(1 for c in obs.played[seat] if _rank_of(c) == rank)
    seen = set(obs.hand)
    for s in rules.SEATS:
        seen |= set(obs.played[s])
    unseen_rank = 0
    for cid in range(1, cardmod.MAX_ID + 1):
        try:
            if _rank_of(cid) == rank and cid not in seen:
                unseen_rank += 1
        except Exception:
            pass
    return [
        1.0 if seat == mate else 0.0,
        len(obs.played[seat]) / 27.0,
        obs.left[seat] / 27.0,
        1.0 if obs.passed[seat] else 0.0,
        played_rank / 2.0,
        unseen_rank / 8.0,
        rank / 15.0,
        1.0 if rank == obs.level else 0.0,
        1.0 if rank >= meld.POINT_SMALL else 0.0,
        len(obs.hand) / 27.0,
        1.0 if (seat % 2) == (obs.seat % 2) else 0.0,
        len(obs.table) / 8.0,
    ]


class Probe(nn.Module):
    """②③ 共用的同一个 2 层小 MLP —— 差别**只能**来自输入。"""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit(Xtr, ytr, Xte, *, epochs, seed, hidden=128, lr=1e-3):
    torch.manual_seed(seed)
    m = Probe(Xtr.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    x = torch.from_numpy(Xtr).float()
    t = torch.from_numpy(ytr).float()
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lf(m(x), t).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(Xte).float()).numpy()


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="手牌信息表示探针")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1002)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--splits", type=int, default=5,
                    help="重复几次随机的**按局**切分，用来给配对差算误差棒")
    a = ap.parse_args(argv)

    net = load(a.weights)
    rng = random.Random(a.seed)
    P, L, M, y, gid = [], [], [], [], []
    n_pos, game = 0, 0
    while n_pos < a.positions and game < a.games:
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        team0 = game % 2
        game += 1
        step = 0
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            mine = rules.TEAM[seat] == team0
            hist = env.encode_history(e.hand, seat)
            if mine and step % a.every == 0 and n_pos < a.positions:
                with torch.no_grad():
                    st = torch.from_numpy(env.encode_state(obs)).unsqueeze(0)
                    hi = torch.from_numpy(hist).unsqueeze(0)
                    _o, (h, _c) = net.lstm(hi)
                    lstm128 = h[-1][0].numpy()
                    # 首层 MLP 在 **pass 动作**（全 0）上的隐层：与动作无关，同一局面一个向量
                    mlp512 = net.mlp[0](torch.cat([st, torch.zeros(1, env.ACTION_DIM), h[-1]],
                                                  dim=-1))[0].numpy()
                mate = rules.PARTNER[seat]
                others = [s for s in rules.SEATS if s != seat]
                truth = {c: s for s in others for c in e.hand.hands[s]}   # 标签（唯一用途）
                for c, holder in truth.items():
                    for s in others:
                        P.append(pair_features(obs, c, s, mate))
                        L.append(lstm128)
                        M.append(mlp512)
                        y.append(1.0 if s == holder else 0.0)
                        gid.append(game)
                n_pos += 1
            if mine:
                i = int(np.argmax(q_values(net, obs, acts, hist)))
            else:
                i = rule_choose(obs, acts)
            e.step(i)
            step += 1

    if not P:
        sys.exit("没采到样本")
    P = np.stack(P).astype(np.float32)
    L = np.stack(L).astype(np.float32)
    M = np.stack(M).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    g = np.array(gid)
    games = np.unique(g)
    print(f"样本 {len(P)}（{n_pos} 个局面 / {game} 局）；"
          f"**重复 {a.splits} 次随机的按局切分**（配对：同一批切分训 ②③ 两档）")

    # 分组：每张牌在该局面有 3 行（三个候选座位），采样顺序 ⇒ 每 3 行一组
    assert len(P) % 3 == 0
    groups = [list(range(i, i + 3)) for i in range(0, len(P), 3)]

    def acc(logits, te_set, te_local):
        """⚠️ `logits` 只在**测试子集**上算过 ⇒ 组内下标要先映射到子集坐标。
        （按局切分保证同一组的 3 行一定在同一侧 ✓）"""
        hit = tot = 0
        for idx in groups:
            if idx[0] not in te_set:
                continue
            loc = [te_local[i] for i in idx]
            k = max(range(len(loc)), key=lambda j: logits[loc[j]])   # 组内 logit 最大的候选
            hit += int(y[idx[k]] == 1.0)                             # 它就是预测持有者
            tot += 1
        return hit / tot if tot else float("nan")

    variants = (("② 公开特征", P),
                ("③ +激活128(LSTM)", np.hstack([P, L])),
                ("③ +激活512(首层)", np.hstack([P, M])),
                ("③ +两者", np.hstack([P, L, M])))
    per = {name: [] for name, _ in variants}
    base_accs = []
    for s in range(a.splits):
        rng_np = np.random.default_rng(a.seed + 1000 * s)
        gs = games.copy()
        rng_np.shuffle(gs)
        n_val = max(1, int(len(gs) * 0.25))
        val_g = set(gs[:n_val].tolist())
        tr = np.where(~np.isin(g, list(val_g)))[0]
        te = np.where(np.isin(g, list(val_g)))[0]
        te_set = set(te.tolist())
        te_local = {int(gidx): k for k, gidx in enumerate(te)}
        # ① 计数基线：猜"剩得最多"的那家（特征第 2 列）
        rel = P[:, 2].reshape(-1, 3)
        base = np.zeros(len(P), dtype=np.float32)
        base[np.arange(0, len(P), 3) + rel.argmax(axis=1)] = 1.0   # 组起点 + 组内最大
        base_accs.append(acc(base, te_set, te_local))
        for name, X in variants:
            logits = _fit(X[tr], y[tr], X[te], epochs=a.epochs, seed=a.seed + s)
            per[name].append(acc(logits, te_set, te_local))
    print(f"  ① 计数基线（猜剩最多的那家）   命中率 {np.mean(base_accs):6.2%}")
    for name, _ in variants:
        v = np.array(per[name])
        print(f"  {name:22s} 命中率 {v.mean():6.2%}  "
              f"（{a.splits} 次切分的 sd {v.std(ddof=1) * 100:.2f}pp）")

    a2 = np.array(per["② 公开特征"])
    print(f"\n判据（预登记）：③ − ② ≥ **2.0pp** ⇒ 有隐式推牌能力；< 1.0pp ⇒ 没有；"
          f"1~2pp ⇒ 弱信号只记录。\n（配对：每次切分里 ③ 与 ② 用同一批训练/测试局）")
    for name in ("③ +激活128(LSTM)", "③ +激活512(首层)", "③ +两者"):
        d = np.array(per[name]) - a2
        se = d.std(ddof=1) / max(1, len(d)) ** 0.5
        print(f"  {name:22s} 差 = {d.mean() * 100:+.2f}pp  ± {se * 100:.2f}（配对 SE）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
