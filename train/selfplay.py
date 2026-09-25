"""Plan 3 的 DMC 训练循环（spec §5）。

跑法：
    .venv/Scripts/python.exe -m train.selfplay               # 默认 3600 秒（1 小时）
    .venv/Scripts/python.exe -m train.selfplay 600           # 10 分钟

与 `train/smoke.py` 的区别（冒烟只证明「env 能训」）：

| | smoke | selfplay |
|---|---|---|
| 样本 | 只用刚打完的那一批（在线） | 从 **replay buffer** 里采（spec §5.2/5.3） |
| 生成 | 一局一局地打，每个决策点问一次网络 | **32 局同步推进**，一步一批前向 |
| 判据 | 打得过**随机** | 打得过**贪心**（spec §7 的分水岭）+ 随机 |
| 产出 | 一个退出码 | `runs/rl/<版本>/best.pt` + 两条评测曲线 |

**DMC 的关键**（spec §5.1/5.2）：中间步没有即时反馈，所以每个决策点的回归目标是
「这一局打完，他所在队拿了多少」—— 整局结束再回填，不做 bootstrap。

**四个座位共享一套权重、状态按出牌人相对化**，所以一局里的每一步都是同一条策略的样本。

## 为什么 32 局要同步推进（不是一局一局打）

剖面（2026-09-25）显示训练步里最大的一块不是规则引擎，而是**网络前向**：
`q_values` 被调用 37,689 次、累计 **70.8 秒**。原因很朴素 —— 一次前向只算
**一个**局面（候选常常不到 10 个），GPU 的批处理完全浪费。

改成「32 局各走一步、攒成一批再前向」之后，同样次数的前向变成 1/32。
代价是代码复杂一点（要维护 32 个并发局面），换来的是**一个数量级**。
"""
from __future__ import annotations

import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

from net.sim import env, rules
from train import replay
from train.eval import match
from train.net import QNet, q_argmax_batch
from train.policies import greedy_policy, random_policy

BATCH_GAMES = 32              # spec §5.3
LR = 1e-4                     # spec §5.3
EPS_START, EPS_END = 1.0, 0.1
#: buffer 容量（局）。**spec §5.3 写的是 50 万局** —— 按紧凑记录算是 923 MB
#: （见 `train/replay.py` 的算术），存张量则是 747 GB。
#: 默认取 5 万局（约 92 MB）是**保守**：一台还要同时跑微信/掼蛋的机器上，
#: 先别默认占掉近 1 GB。要按 spec 的原数跑就传 `--buffer 500000`。
BUFFER_GAMES = 50_000
EVAL_EVERY_GAMES = 10_000     # spec §5.3「每 N 局存一次权重 + 跑一次基线评测」
#: 评测局数。**别往下调**：实测同一个网络换 6 个种子各测 200 局，
#: 结果是 4.8% ± 1.4%（二项理论值 ≈1.5%）—— 也就是 200 局时噪声约 ±2%，
#: 判据「≥55%」不会被噪声骗过。100 局时会飘到 ±5%，不够用。
EVAL_GAMES = 200
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "rl")


def generate_batch(net, rng, eps, n_games, capture=True):
    """同步推进 `n_games` 局，返回 `[(GameRecord, points, y), ...]`。

    `points` 是每步的 `(obs, acts, 选中下标, 出牌人, hist)`（`capture=False` 时为空）——
    刚打完的这批**马上要拿来训练**，所以现场抓住，免得再从记录重放一遍（重放一局约 27 ms）。
    """
    envs, hands0, log, caps = [], [], [], []
    for _ in range(n_games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        envs.append(e)
        hands0.append([set(e.hand.hands[s]) for s in rules.SEATS])   # 发牌快照
        log.append([])
        caps.append([])

    alive = list(range(n_games))
    while alive:
        pending = []
        for k in alive:
            e = envs[k]
            pending.append((e.observe(), e.legal(),
                            env.encode_history(e.hand, e.hand.turn)))
        if eps >= 1.0:
            # 纯随机阶段不必前向 —— 早期是 ε=1.0，省掉这一大截
            picks = [rng.randrange(len(a)) for _o, a, _h in pending]
        else:
            picks = q_argmax_batch(net, pending)
            picks = [rng.randrange(len(a)) if rng.random() < eps else p
                     for p, (_o, a, _h) in zip(picks, pending)]

        nxt = []
        for k, (obs, acts, hist), i in zip(alive, pending, picks):
            e = envs[k]
            log[k].append(i)
            if capture:
                caps[k].append((obs, acts, i, e.hand.turn, hist))
            e.step(i)
            if not e.done:
                nxt.append(k)
        alive = nxt

    out = []
    for k in range(n_games):
        e = envs[k]
        rec = replay.GameRecord.of(e, log[k], hands0[k])
        if capture:
            ranks = e.ranks
            out.append((rec, caps[k],
                        [rules.reward(ranks, s) for (_o, _a, _i, s, _h) in caps[k]]))
        else:
            out.append((rec, [], []))
    return out


def _tensors(samples):
    """把 `(obs, acts, i, seat, hist)` 铺成张量。"""
    st = torch.from_numpy(np.stack([env.encode_state(o) for o, _a, _i, _s, _h in samples]))
    ac = torch.from_numpy(np.stack([env.encode_action(a[i], o.level)
                                    for o, a, i, _s, _h in samples]))
    hi = torch.from_numpy(np.stack([h for _o, _a, _i, _s, h in samples]))
    return st, ac, hi


def net_play(net):
    """把网络包成策略（`hist` 由评测器算好递进来）。"""
    from train.net import q_values
    return lambda obs, acts, hist: int(q_values(net, obs, acts, hist).argmax())


def train(seconds: float = 3600.0, seed: int = 0, buffer_games: int = BUFFER_GAMES,
          eval_games: int = EVAL_GAMES, eval_every: int = EVAL_EVERY_GAMES,
          out_dir: str = None, log=print):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    log(f"device={next(net.parameters()).device}  预算 {seconds:.0f}s  "
        f"batch={BATCH_GAMES} 局（同步推进）  buffer={buffer_games:,} 局  "
        f"评测每 {eval_every:,} 局\n权重 -> {out_dir}")

    t0 = time.perf_counter()
    games = steps = 0
    curve = []
    best_greedy = -1.0

    while time.perf_counter() - t0 < seconds:
        frac = min(1.0, (time.perf_counter() - t0) / seconds)
        eps = EPS_START + (EPS_END - EPS_START) * frac

        # 1) 同步打一批，记进 buffer（现场抓好决策点，省一次重放）
        fresh = {}
        for rec, pts, y in generate_batch(net, rng, eps, BATCH_GAMES):
            buf.add(rec)
            fresh[id(rec)] = (pts, y)
            games += 1

        # 2) 从 buffer 采一批（spec §5.2：整局的终局 reward 当回归目标）
        samples, targets = [], []
        for rec in buf.sample(BATCH_GAMES, rng):
            pts, y = fresh.get(id(rec)) or replay.expand(rec)
            samples += pts
            targets += y
        st, ac, hi = _tensors(samples)
        dev = next(net.parameters()).device
        y = torch.tensor(targets, dtype=torch.float32).to(dev)

        # 3) 训一步
        loss = torch.nn.functional.mse_loss(
            net(st.to(dev), ac.to(dev), hi.to(dev)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        steps += 1

        el = time.perf_counter() - t0
        if steps % 10 == 0:
            log(f"  {el:6.0f}s  局数 {games:7d}  ε={eps:.2f}  loss={loss.item():.3f}  "
                f"{games / el:.1f} 局/秒  buffer {len(buf):,}")

        # 4) 每 N 局：存权重 + 评测（spec §5.3）
        if games % eval_every < BATCH_GAMES:
            wr_r = match(net_play(net), random_policy(random.Random(101)),
                         games=eval_games, seed=1001)
            wr_g = match(net_play(net), greedy_policy, games=eval_games, seed=1002)
            curve.append((games, wr_r, wr_g))
            log(f"     >> 评测 @ {games:7d} 局：vs 随机 {wr_r:.1%}   vs 贪心 {wr_g:.1%}")
            if wr_g > best_greedy:
                best_greedy = wr_g
                torch.save({"net": net.state_dict(), "games": games,
                            "winrate_random": wr_r, "winrate_greedy": wr_g},
                           os.path.join(out_dir, "best.pt"))

    # 收尾：两次评测看方差（spec §7 最后一行）
    wr_g1 = match(net_play(net), greedy_policy, games=eval_games, seed=2001)
    wr_g2 = match(net_play(net), greedy_policy, games=eval_games, seed=2002)
    wr_r = match(net_play(net), random_policy(random.Random(202)),
                 games=eval_games, seed=2003)
    log("")
    log(f"总共 {games:,} 局 / {steps:,} 步 / {time.perf_counter() - t0:.0f} 秒")
    log(f"末次：vs 随机 {wr_r:.1%}   vs 贪心 {wr_g1:.1%} / {wr_g2:.1%}"
        f"（两次，差 {abs(wr_g1 - wr_g2):.1%} —— 200 局的噪声约 ±2%）")
    if curve:
        log("曲线（局数:vs随机/vs贪心）：" + "  ".join(
            f"{g // 1000}k:{r:.0%}/{k:.0%}" for g, r, k in curve))
    torch.save({"net": net.state_dict(), "games": games,
                "winrate_random": wr_r, "winrate_greedy": wr_g1},
               os.path.join(out_dir, "last.pt"))
    return {"games": games, "curve": curve, "best_greedy": best_greedy,
            "wr_random": wr_r, "wr_greedy": (wr_g1 + wr_g2) / 2, "out_dir": out_dir}


def main(argv=None) -> int:
    from tools.accept_meld import _utf8_stdout
    _utf8_stdout()          # 不调这个，GBK 控制台下打不出 ✓ 会丢退出码（踩过）
    argv = sys.argv[1:] if argv is None else argv
    seconds = float(argv[0]) if argv else 3600.0
    buf = BUFFER_GAMES
    if "--buffer" in argv:
        buf = int(argv[argv.index("--buffer") + 1])
    r = train(seconds=seconds, buffer_games=buf)
    print(f"\n权重：{r['out_dir']}")
    # spec §7 的分水岭：**明确打过**贪心。50% 只是「五五开」，不算打过。
    ok = r["wr_greedy"] >= 0.55
    print("判据（spec §7 分水岭：明确打得过贪心）：" + (
        "**过了** ✓" if ok else f"**没过** ✗（vs 贪心 {r['wr_greedy']:.1%}，要 ≥55%）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
