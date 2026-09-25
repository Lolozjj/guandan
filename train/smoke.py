"""最小 DMC 冒烟（spec §7 第一行）—— **只证明 env 能训，不是训练实验**。

跑法：
    .venv/Scripts/python.exe -m train.smoke                 # 默认 30 分钟
    .venv/Scripts/python.exe -m train.smoke 600             # 10 分钟

判据（写死在退出码里）：
    末次评测的胜率 **≥ 60%**（随机基线实测 51.1%）且最后几次评测**没有掉下去**。
    spec §7 的「很快到 90%+」是最终目标；本任务要回答的是「env 学不学得起来」。

网络形状照 spec §4.3（对齐 DouZero）：
    历史(15×147) → LSTM(128) ─┐
                              ├→ 拼接 → 6 层 MLP(512) → 一个 Q 值
    状态(700) + 单个候选动作(143) ┘

**刻意不做的东西**：replay buffer（spec §5.3 的 50 万局）。那是在线训练循环的组件，
写进来就要连采样/淘汰逻辑一起写，而那套逻辑本身不该在冒烟里被信任。
冒烟用在线 batch 就够了 —— 它的目的不是训出好模型，是证明「学得起来」。

## 两个对照（2026-09-25 实测，跑训练之前先要有这两个数）

    随机 vs 随机（两队都随机）        51.1%   <- 评测无偏、赛制对称
    「座位 0/2 能过就过」这个确定性坏策略  0.0%   <- 评测能识别出坏策略

有了这两个数，「训练出来的胜率」才有参照：**低于 50% 不代表环境坏了**
（未训练的网络是个确定性策略，确定性坏策略可以差到 0%），
但**一直停在 50% 附近不动、或者曲线不涨**，才是「学不起来」。

## 判据为什么是 60% 而不是 spec §7 的 90%

spec §7 的「vs 随机很快到 90%+」是**最终目标**，不是几分钟冒烟的门槛。
本机实测训练速度约 **13 局/秒**（含网络前向反向；`tools/bench_sim.py` 量的是
不含网络的规则引擎吞吐 = 31.9 局/秒）。3 分钟 ≈ 2300 局，在 DMC 的量级上
连热身都算不上。所以冒烟的门槛定为
**「明显高于 50% 的随机基线（≥60%）且最后几次评测没掉下去」** ——
它区分的是「学得起来」与「学不起来」，这正是 spec §7 第一行要做的事。
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from net.sim import env, rules
# `_utf8_stdout` 复用 accept_meld 那份，**不复制**（同 tools/accept_sim.py 的理由）。
# 2026-09-25：正是漏了它，1800 秒训练跑完、末次胜率 92.5%，却在打印
# 「通过 ✓」那一行抛 UnicodeEncodeError —— **训练成功、退出码丢掉**。
from tools.accept_meld import _utf8_stdout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_GAMES = 32              # spec §5.3：batch 32 局
LR = 1e-4
EPS_START, EPS_END = 1.0, 0.1
MLP_LAYERS = 6
MLP_HIDDEN = 512
LSTM_HIDDEN = 128
EVAL_EVERY_GAMES = BATCH_GAMES * 20
EVAL_GAMES = 200
PASS_WINRATE = 0.60      # 见 docstring：60% + 曲线不降 = 学得起来


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(env.HISTORY_DIM, LSTM_HIDDEN, batch_first=True)
        layers, d = [], env.STATE_DIM + env.ACTION_DIM + LSTM_HIDDEN
        for _ in range(MLP_LAYERS):
            layers += [nn.Linear(d, MLP_HIDDEN), nn.ReLU()]
            d = MLP_HIDDEN
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, state, action, hist):
        _out, (h, _c) = self.lstm(hist)
        return self.mlp(torch.cat([state, action, h[-1]], dim=-1)).squeeze(-1)


def _q(net, obs, acts, hist):
    """一次前向算出一批候选的 Q。`obs`/`hist` 是单个局面的。"""
    st = torch.from_numpy(env.encode_state(obs)).unsqueeze(0).to(DEVICE)
    ac = torch.from_numpy(np.stack([env.encode_action(a, obs.level)
                                    for a in acts])).to(DEVICE)
    hi = torch.from_numpy(hist).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return net(st.expand(len(acts), -1), ac, hi.expand(len(acts), -1, -1))


def self_play_batch(net, rng, eps):
    """跑 `BATCH_GAMES` 局 ε-greedy 自对弈，返回一批 `(state, action, hist, y)`。

    **DMC 的关键**：中间步没有即时反馈，所以每个决策点的回归目标是
    「这一局打完，他所在队拿了多少」—— 整局结束再回填，不做 bootstrap。
    """
    st, ac, hi, y = [], [], [], []
    for _ in range(BATCH_GAMES):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        points = []
        obs = e.observe()
        while not e.done:
            acts = e.legal()
            seat = e.hand.turn
            # ⚠️ **历史必须在这里就取下来**（`hist` 是这一步之前的最近 15 手）。
            # 整局结束之后再 `encode_history(e.hand, seat)` 会把**后面才发生的牌**
            # 塞进历史的最后几行 —— 那是另一种泄漏（未来信息），而且同样静默。
            # `evaluate()` 里也是在循环内取的，别把两处写成不一样。
            hist = env.encode_history(e.hand, seat)
            if rng.random() < eps:
                i = rng.randrange(len(acts))
            else:
                i = int(_q(net, obs, acts, hist).argmax())
            points.append((obs, acts, i, seat, hist))
            obs, _r, _d, _info = e.step(i)
        ranks = e.ranks
        for o, a, i, seat, hist in points:
            st.append(env.encode_state(o))
            ac.append(env.encode_action(a[i], o.level))
            hi.append(hist)
            y.append(rules.reward(ranks, seat))
    return (torch.from_numpy(np.stack(st)).to(DEVICE),
            torch.from_numpy(np.stack(ac)).to(DEVICE),
            torch.from_numpy(np.stack(hi)).to(DEVICE),
            torch.tensor(y, dtype=torch.float32).to(DEVICE))


def evaluate(net, games=EVAL_GAMES, seed=999):
    """座位 0、2 用网络（贪心），座位 1、3 随机 —— 报「我们的胜率」。

    随机基线实测 **51.1%**（零点五、两队对称；见模块 docstring 的对照）。
    """
    rng = random.Random(seed)
    wins = 0
    for _ in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        obs = e.observe()
        while not e.done:
            acts = e.legal()
            seat = e.hand.turn
            if rules.TEAM[seat] == 0:
                hist = env.encode_history(e.hand, seat)
                i = int(_q(net, obs, acts, hist).argmax())
            else:
                i = rng.randrange(len(acts))
            obs, _r, _d, _info = e.step(i)
        if rules.winner_team(e.ranks) == 0:
            wins += 1
    return wins / games


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    budget = float(argv[0]) if argv else 1800.0
    _utf8_stdout()          # 不调这个，GBK 控制台下最后那行「通过 ✓」会抛异常

    torch.manual_seed(0)
    rng = random.Random(0)
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)          # spec §5.3：Adam / 1e-4
    print(f"device={DEVICE}  预算 {budget:.0f} 秒  batch={BATCH_GAMES} 局  "
          f"判据 胜率 ≥ {PASS_WINRATE:.0%}\n")

    t0 = time.perf_counter()
    games = 0
    history = []
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        eps = EPS_START + (EPS_END - EPS_START) * frac         # ε 线性退火
        st, ac, hi, y = self_play_batch(net, rng, eps)
        loss = nn.functional.mse_loss(net(st, ac, hi), y)
        opt.zero_grad(); loss.backward(); opt.step()
        games += BATCH_GAMES
        el = time.perf_counter() - t0
        print(f"  {el:6.1f}s  局数 {games:6d}  ε={eps:.2f}  loss={loss.item():.3f}  "
              f"{games / el:.1f} 局/秒")
        if games % EVAL_EVERY_GAMES == 0:
            wr = evaluate(net)
            history.append(wr)
            print(f"     >> 胜率 vs 随机 = {wr:.1%}")

    wr = evaluate(net)
    history.append(wr)
    print(f"\n末次胜率 vs 随机：{wr:.1%}")
    print(f"曲线：{['%.0f%%' % (h * 100) for h in history]}")

    rising = len(history) < 3 or all(b >= a - 0.02
                                     for a, b in zip(history[-4:-1], history[-3:]))
    ok = wr >= PASS_WINRATE and rising
    print("冒烟" + ("通过 ✓ —— env 能训" if ok else
                    "**未通过** ✗ —— 先查 env 的状态/动作编码，别加大训练量"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
