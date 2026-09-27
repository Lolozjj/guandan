"""Plan 3 的 DMC 训练循环（spec §5）。

跑法：
    .venv/Scripts/python.exe -m train.selfplay               # 默认 3600 秒（1 小时）
    .venv/Scripts/python.exe -m train.selfplay 600           # 10 分钟

**为什么长这样**（对照那个已删的最小冒烟脚本 —— 它只证明「env 能训」）：

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
from train import pool, replay
from train.eval import match
from train.net import DEVICE, QNet, q_argmax_batch
from train.policies import greedy_policy, random_policy

BATCH_GAMES = 32              # spec §5.3（同步推进的局数；`--batch` 可调大）
LR = 1e-4                     # spec §5.3
EPS_START, EPS_END = 1.0, 0.1
#: buffer 容量（局）。**spec §5.3 写的是 50 万局** —— 按紧凑记录算是 923 MB
#: （见 `train/replay.py` 的算术），存张量则是 747 GB。
#: 默认取 5 万局（约 92 MB）是**保守**：一台还要同时跑微信/掼蛋的机器上，
#: 先别默认占掉近 1 GB。要按 spec 的原数跑就传 `--buffer 500000`。
BUFFER_GAMES = 50_000
EVAL_EVERY_GAMES = 10_000     # spec §5.3「每 N 局存一次权重 + 跑一次基线评测」
#: ε 退火到 `EPS_END` 所需的**局数**（不是秒）。
#:
#: 按**时间**退火的毛病（2026-09-26 用户实测后改的）：9 小时的预算下跑到第 25 分钟
#: ε 还是 **0.96** —— 大半时间在做近乎随机的探索，真正的学习挤在最后两三小时。
#: 上一轮「加时长收益越来越小」**不是撞墙，是探索没退下去**（老的一小时跑法
#: ε 一小时就退完，所以 5 万局就到 80%）。按局数退，出数快的机器自然学得快。
#: 25 万局 ≈ 9 小时跑到 70% 处退完，最后一段留给纯利用。
EPS_GAMES = 250_000


def eps_for(games: int, eps_games: int = EPS_GAMES) -> float:
    """第 `games` 局时的 ε（线性退火，退到底就不再变）。"""
    frac = min(1.0, games / max(1, eps_games))
    return EPS_START + (EPS_END - EPS_START) * frac
#: 评测局数。**别往下调**：实测同一个网络换 6 个种子各测 200 局，
#: 结果是 4.8% ± 1.4%（二项理论值 ≈1.5%）—— 也就是 200 局时噪声约 ±2%，
#: 判据「≥55%」不会被噪声骗过。100 局时会飘到 ±5%，不够用。
EVAL_GAMES = 200
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "rl")


def _fixed_pick(kind, pending, rng):
    """固定对手出一手。`kind = ("greedy"|"random", 队号)`。"""
    obs, acts, hist = pending
    if kind[0] == "random":
        return rng.randrange(len(acts))
    return greedy_policy(obs, acts, hist)      # 复用评测那套贪心，不另写一份


def load_init(net, path: str = None) -> None:
    """把 checkpoint 装进 `net`。`path=None` 时什么都不做（默认行为不变）。

    只认 `{"net": state_dict}` 这一种（与 `best.pt` / `last.pt` 同格式）。
    装不上**必须炸** —— 静默从随机开始会让「热启动」变成假的，
    而这件事在日志和曲线上一概看不出来（`tests/test_train_init.py` 钉着）。
    """
    if not path:
        return
    ck = torch.load(path, map_location="cpu")
    net.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)


def plan_step(learn_seats, turn, fixed) -> tuple:
    """这一步由谁出手 → `("learner", None)` / `("member", mid)` / `("fixed", kind)`。

    **纯函数**，故意抽出来 —— 分组错了会**静默用错权重**（池子里全变成一个模型），
    而那种错在日志上完全看不出来。

    「不属学习队、又没有固定对手」是不该出现的状态（`learn` 与 `fixed` 是一起定的），
    所以**炸掉**而不是猜一个 —— 猜的后果是静默退回贪心（本仓库纪律：失败必须响）。
    """
    if turn in learn_seats:
        return ("learner", None)
    if fixed is None:
        raise ValueError(
            f"座位{turn} 不属于学习队 {tuple(learn_seats)}、又没有固定对手 —— "
            f"分组的前提被破坏了")
    if fixed[0] == "member":
        return ("member", fixed[1])
    return ("fixed", fixed[0])


def generate_batch(net, rng, eps, n_games, capture=True, opp_mix=0.0,
                   greedy_share=0.8, learn_all_seats=False, members=None,
                   pick_fixed=None):
    """同步推进 `n_games` 局，返回 `[(GameRecord, points, y), ...]`。

    `points` 是每步的 `(obs, acts, 选中下标, 出牌人, hist)`（`capture=False` 时为空）——
    刚打完的这批**马上要拿来训练**，所以现场抓住，免得再从记录重放一遍（重放一局约 27 ms）。

    **`opp_mix`：混入固定对手的比例**（用户 2026-09-26 定，治「有普通牌可压却出炸」）。
    纯自对弈里对手也爱炸，「不炸就被炸」成了均衡，而 `vs 贪心` 那把尺子看不见浪费
    （贪心从不主动炸）。所以每局以 `opp_mix` 的概率把**一个队**换成固定策略
    （`greedy_share` 的比例用贪心、其余用随机），学习那一队照旧打网络。
    ⚠️ 固定对手的决策点**不进训练目标**（那不是网络选的，记进去等于拿它当老师）。
    这件事靠 `learn` 这一个变量承载 —— 现场抓取与写进记录都用它，
    所以不存在「只改了一半」的可能（`tests/test_expand_learn.py` 两条路都钉着）。

    `learn_all_seats`：**A/B 的对照臂**。开了之后对手照旧换，但四家照旧都学 ——
    也就是改动前的行为（连固定对手的着法也进训练目标）。量「修 `expand` 值多少」用。

    `members`：`{mid: 网络}` —— 池子成员（spec §3.3）。
    `pick_fixed(rng) -> ("greedy",) | ("random",) | ("member", mid)`：这一局的固定
    对手是谁，由调用方给（PFSP 在 learner 侧算）。不给就沿用老的 `greedy_share` 二分。
    **哪一队当固定对手统一在这里抽**（池成员也一样）—— `learn` 由它推出来。
    """
    envs, hands0, log, caps = [], [], [], []
    learn, fixed = [], []          # 每局：学习那一队的座位 / 固定对手（None = 纯自对弈）
    members = members if members is not None else {}
    for _ in range(n_games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        envs.append(e)
        hands0.append([set(e.hand.hands[s]) for s in rules.SEATS])   # 发牌快照
        log.append([])
        caps.append([])
        if rng.random() < opp_mix:
            opp = rng.randrange(2)      # 哪一队当固定对手 —— **统一在这里抽**（池成员也一样）
            if pick_fixed is not None:
                kind = pick_fixed(rng)  # ("greedy",) | ("random",) | ("member", mid)
                # 形状统一成 (kind, x)：member 的 x 是**成员 id**，其余是**队号**
                fixed.append(("member", kind[1]) if kind[0] == "member"
                             else (kind[0], opp))
            else:
                kind = "greedy" if rng.random() < greedy_share else "random"
                fixed.append((kind, opp))
            learn.append(tuple(s for s in rules.SEATS if rules.TEAM[s] != opp))
        else:
            fixed.append(None)
            learn.append(tuple(rules.SEATS))
        if learn_all_seats:
            # 对照臂：**一处改**，现场抓取（caps）与写进记录（GameRecord.learn）一起跟 ——
            # 只改一处的话两臂差的就不止「expand 有没有过滤」这一个变量了
            learn[-1] = tuple(rules.SEATS)

    alive = list(range(n_games))
    while alive:
        pending = []
        for k in alive:
            e = envs[k]
            pending.append((e.observe(), e.legal(),
                            env.encode_history(e.hand, e.hand.turn)))
        # 按「这一步谁在出手」分组，**每组各做一次批量前向**（spec §3.3）——
        # 「当前权重」只是众多组里的一组。
        # 顺序显式定死（fixed → learner → 成员按 id）：rng 的消费顺序别随 dict 插入序漂，
        # 否则同一个种子在不同局面下打出来的东西会变。
        picks = [None] * len(pending)
        groups = {}
        for j, k in enumerate(alive):
            who = plan_step(learn[k], envs[k].hand.turn, fixed[k])
            groups.setdefault(who, []).append(j)

        def _order(item):
            (who, x), _js = item
            rank = 0 if who == "fixed" else (1 if who == "learner" else 2)
            return (rank, x if isinstance(x, int) else 0)

        for (who, mid), js in sorted(groups.items(), key=_order):
            if who == "fixed":
                # 贪心 / 随机：便宜的 Python 路径，**不占前向**（占混合局的 20%）
                for j in js:
                    picks[j] = _fixed_pick(fixed[alive[j]], pending[j], rng)
                continue
            if who == "member":
                if mid not in members:
                    raise KeyError(
                        f"对手池里没有成员 {mid}（有 {sorted(members)}）—— "
                        f"不许静默换个对手，那会让池子悄悄少一个成员")
                neti = members[mid]
            else:
                neti = net
            if who == "learner" and eps >= 1.0:
                # 纯随机阶段不必前向 —— 早期是 ε=1.0，省掉这一大截
                for j in js:
                    picks[j] = rng.randrange(len(pending[j][1]))
                continue
            got = q_argmax_batch(neti, [pending[j] for j in js])
            for j, p in zip(js, got):
                # ⚠️ ε **只属于学习者**；对手一律 argmax（spec §4）——
                # 手滑把 ε 也施加到池成员上，池子就成了一群会随机出牌的对手
                picks[j] = (rng.randrange(len(pending[j][1]))
                            if who == "learner" and rng.random() < eps else p)

        nxt = []
        for k, (obs, acts, hist), i in zip(alive, pending, picks):
            e = envs[k]
            log[k].append(i)
            if capture and e.hand.turn in learn[k]:
                # 固定对手的着法**不进训练目标**（不是网络选的）
                caps[k].append((obs, acts, i, e.hand.turn, hist))
            e.step(i)
            if not e.done:
                nxt.append(k)
        alive = nxt

    out = []
    for k in range(n_games):
        e = envs[k]
        rec = replay.GameRecord.of(e, log[k], hands0[k], learn=learn[k],
                                   opp=fixed[k])
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
    """把网络包成策略。**带 `batch_choose`** —— 评测器会一次把同一时刻的
    所有决策点送进网络（`train/eval.py` 的 `match` 认这个属性）。"""
    from train.policies import batch_net_policy
    from train.net import q_argmax_batch
    return batch_net_policy(q_argmax_batch, net)


WEIGHT_SYNC_GAMES = 1000     # 多进程：每这么多局把权重广播给 worker


def _maybe_eval(net, games, curve, best_greedy, eval_games, eval_every,
                batch_games, out_dir, log, snap_every=pool.SNAP_EVERY_GAMES,
                pool_size=pool.POOL_SIZE):
    """每 `eval_every` 局评测一次并（可能）存 `best.pt`。**两条训练路线共用这一份。**

    顺带按 `snap_every` 存池子快照 —— **与「有没有刷新最好」无关**：
    只存 `best.pt` 的话池子原料不够（144 万局只落几个点）。
    """
    if games % eval_every < batch_games:
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
    if snap_every and games and games % snap_every < batch_games:
        p = pool.snapshot_path(out_dir, games)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"net": net.state_dict(), "games": games}, p)
        gone = pool.prune_snapshots(out_dir, pool_size)
        log(f"     >> 池子快照 {os.path.basename(p)}"
            + (f"（挤掉 {len(gone)} 个）" if gone else ""))
    return best_greedy


def _finish(net, games, steps, t0, curve, best_greedy, eval_games, out_dir, log,
            bomb_games: int = 25):
    """收尾：两次评测看方差 + 报告 + 存 last.pt + 返回结果。**两条路线共用。**

    `elapsed` 是**训练阶段**的秒数（在收尾评测之前取）—— 吞吐要用它算，
    别把固定的评测开销算进去（否则 bench 的短跑会被评测淹没）。
    `bomb_games=0` 跳过炸弹浪费率那一步（bench 用，省 30 秒）。
    """
    elapsed = time.perf_counter() - t0
    wr_g1 = match(net_play(net), greedy_policy, games=eval_games, seed=2001)
    wr_g2 = match(net_play(net), greedy_policy, games=eval_games, seed=2002)
    wr_r = match(net_play(net), random_policy(random.Random(202)),
                 games=eval_games, seed=2003)
    log("")
    log(f"总共 {games:,} 局 / {steps:,} 步 / {elapsed:.0f} 秒")
    log(f"末次：vs 随机 {wr_r:.1%}   vs 贪心 {wr_g1:.1%} / {wr_g2:.1%}"
        f"（两次，差 {abs(wr_g1 - wr_g2):.1%} —— 200 局的噪声约 ±2%）")
    if curve:
        log("曲线（局数:vs随机/vs贪心）：" + "  ".join(
            f"{g // 1000}k:{r:.0%}/{k:.0%}" for g, r, k in curve))
    # 炸弹浪费率（用户 2026-09-26 报的毛病）——**只报告，不当判据**：
    # 判据仍是「vs 贪心 ≥55%」（与老几次跑可比），这个数是给你看「有没有变好」的。
    if bomb_games:
        try:
            from train.eval import bomb_waste
            w, c = bomb_waste(net_play(net), games=bomb_games, seed=3001)
            log(f"  炸弹浪费率（能用普通牌压却出炸）：{w}/{c} = {w / max(1, c):.1%}"
                f"   [对照：贪心恒为 0%]")
        except Exception as exc:                  # noqa: BLE001 - 报告失败不该带崩训练
            log(f"  炸弹浪费率：没量成（{type(exc).__name__}: {exc}）")
    torch.save({"net": net.state_dict(), "games": games,
                "winrate_random": wr_r, "winrate_greedy": wr_g1},
               os.path.join(out_dir, "last.pt"))
    return {"games": games, "curve": curve, "best_greedy": best_greedy,
            "wr_random": wr_r, "wr_greedy": (wr_g1 + wr_g2) / 2,
            "out_dir": out_dir, "elapsed": elapsed}


def train(seconds: float = 3600.0, seed: int = 0, buffer_games: int = BUFFER_GAMES,
          eval_games: int = EVAL_GAMES, eval_every: int = EVAL_EVERY_GAMES,
          out_dir: str = None, log=print, opp_mix: float = 0.5,
          greedy_share: float = 0.8, batch_games: int = BATCH_GAMES,
          eps_games: int = EPS_GAMES, learn_all_seats: bool = False,
          init: str = None, snap_every: int = pool.SNAP_EVERY_GAMES,
          pool_size: int = pool.POOL_SIZE):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    # ⚠️ **`.to(DEVICE)` 不能省。** 漏了它的后果是静默的：日志第一行写着 `device=cpu`，
    # 训练照跑、只是慢 —— Plan 3 那 5 万局与 2026-09-26 那次 9 小时跑都是这么过去的
    # （GPU 从没被用上）。训练步的瓶颈就是网络前向（spec §14.3）。
    # `tests/test_train_device.py` 用 1 秒预算真跑一次钉住这件事。
    net = QNet().to(DEVICE)
    load_init(net, init)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    log(f"device={next(net.parameters()).device}  预算 {seconds:.0f}s  "
        f"batch={batch_games} 局（同步推进）  buffer={buffer_games:,} 局  "
        f"评测每 {eval_every:,} 局  ε 按 {eps_games:,} 局退火  对手混合 {opp_mix:.0%}"
        f"（其中贪心 {greedy_share:.0%}）"
        + (f"  热启动 {init}" if init else "")
        + f"\n权重 -> {out_dir}")

    t0 = time.perf_counter()
    games = steps = 0
    curve = []
    best_greedy = -1.0

    while time.perf_counter() - t0 < seconds:
        eps = eps_for(games, eps_games)          # **按局数退火**，不按时间（见 EPS_GAMES）

        # 1) 同步打一批，记进 buffer（现场抓好决策点，省一次重放）
        fresh = {}
        for rec, pts, y in generate_batch(net, rng, eps, batch_games,
                                          opp_mix=opp_mix,
                                          greedy_share=greedy_share,
                                          learn_all_seats=learn_all_seats):
            buf.add(rec)
            fresh[id(rec)] = (pts, y)
            games += 1

        # 2) 从 buffer 采一批（spec §5.2：整局的终局 reward 当回归目标）
        samples, targets = [], []
        for rec in buf.sample(batch_games, rng):
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

        # 4) 每 N 局：存权重 + 评测（spec §5.3）—— 公共件，两条训练路线共用
        best_greedy = _maybe_eval(net, games, curve, best_greedy, eval_games,
                                  eval_every, batch_games, out_dir, log,
                                  snap_every=snap_every, pool_size=pool_size)

    return _finish(net, games, steps, t0, curve, best_greedy, eval_games,
                   out_dir, log)


def train_parallel(seconds: float = 3600.0, workers: int = 1, seed: int = 0,
                   buffer_games: int = BUFFER_GAMES, eval_games: int = EVAL_GAMES,
                   eval_every: int = EVAL_EVERY_GAMES, out_dir: str = None,
                   log=print, opp_mix: float = 0.5, greedy_share: float = 0.8,
                   batch_games: int = BATCH_GAMES, eps_games: int = EPS_GAMES,
                   learn_all_seats: bool = False, init: str = None,
                   snap_every: int = pool.SNAP_EVERY_GAMES,
                   pool_size: int = pool.POOL_SIZE,
                   _kill_worker_after: float = None):
    """多进程：`workers` 个进程打牌、本进程学习。

    设计见 `docs/superpowers/specs/2026-09-26-multiprocess-selfplay-design.md`。
    三条纪律（都在测试里钉着）：worker 死了**必须炸**；队列**有界**（背压）；
    ε 由本进程按**全局局数**算完广播下去（各 worker 自己算会「每个都以为自己是全部」）。

    `_kill_worker_after` **只给测试用**（到点杀一个 worker，验死亡检测）。
    """
    import multiprocessing as mp

    from train import worker as worker_mod

    ctx = mp.get_context("spawn")           # Windows 只有 spawn；入口必须是模块级函数
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = QNet().to(DEVICE)
    load_init(net, init)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    send_q = ctx.Queue(maxsize=max(2, workers * 2))   # 有界 = 背压（不堆内存）
    ctrls = [ctx.Queue() for _ in range(workers)]
    procs = [ctx.Process(target=worker_mod.run_worker, daemon=True,
                         args=(send_q, ctrls[k],
                               worker_mod.worker_cfg(seed + 1 + k, EPS_START,
                                                     opp_mix, greedy_share,
                                                     batch_games,
                                                     learn_all_seats=learn_all_seats,
                                                     init=init)))
             for k in range(workers)]
    log(f"device={next(net.parameters()).device}  预算 {seconds:.0f}s  "
        f"worker {workers} 个（各自 CPU）+ 主进程学习  batch={batch_games} 局  "
        f"buffer={buffer_games:,} 局  评测每 {eval_every:,} 局  "
        f"ε 按 {eps_games:,} 局退火  对手混合 {opp_mix:.0%}"
        f"（其中贪心 {greedy_share:.0%}）"
        + (f"  热启动 {init}" if init else "")
        + f"\n权重 -> {out_dir}")

    t0 = time.perf_counter()
    games = steps = 0
    curve = []
    best_greedy = -1.0
    last_sync = 0
    qmax = 0                                # 队列积压峰值（背压有没有生效，看这个）
    t_kill = (time.perf_counter() + _kill_worker_after) if _kill_worker_after else None
    try:
        for p in procs:
            p.start()
        sd0 = {k: v.cpu() for k, v in net.state_dict().items()}
        for q in ctrls:                     # 开局先广播一次（worker 种子相同，但更稳）
            q.put(("weights", (sd0, EPS_START)))
        while time.perf_counter() - t0 < seconds:
            dead = [k for k, p in enumerate(procs) if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    f"worker {dead} 挂了（exitcode="
                    f"{[procs[k].exitcode for k in dead]}）—— 不许静默变慢")
            if t_kill is not None and time.perf_counter() >= t_kill:
                procs[0].terminate()
                t_kill = None
            try:
                recs = send_q.get(timeout=300)
            except Exception as exc:        # noqa: BLE001
                raise RuntimeError(
                    f"等 worker 的记录超时（{type(exc).__name__}）—— 它可能卡住了") from exc
            qmax = max(qmax, send_q.qsize())
            for rec in recs:
                buf.add(rec)
                games += 1
            # 采样 + 训一步（与单进程那条路逐字相同）
            samples, targets = [], []
            for rec in buf.sample(batch_games, rng):
                pts, y = replay.expand(rec)
                samples += pts
                targets += y
            st, ac, hi = _tensors(samples)
            dev = next(net.parameters()).device
            y = torch.tensor(targets, dtype=torch.float32).to(dev)
            loss = torch.nn.functional.mse_loss(
                net(st.to(dev), ac.to(dev), hi.to(dev)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            el = time.perf_counter() - t0
            if steps % 10 == 0:
                log(f"  {el:6.0f}s  局数 {games:7d}  ε={eps_for(games, eps_games):.2f}  "
                    f"loss={loss.item():.3f}  {games / el:.1f} 局/秒  "
                    f"buffer {len(buf):,}")
            if games - last_sync >= WEIGHT_SYNC_GAMES:
                sd = {k: v.cpu() for k, v in net.state_dict().items()}
                for q in ctrls:
                    q.put(("weights", (sd, eps_for(games, eps_games))))
                last_sync = games
            best_greedy = _maybe_eval(net, games, curve, best_greedy, eval_games,
                                      eval_every, batch_games, out_dir, log,
                                      snap_every=snap_every, pool_size=pool_size)
    finally:
        for q in ctrls:                     # worker 不许变成孤儿（review focus 5）
            try:
                q.put(("stop", None))
            except Exception:               # noqa: BLE001
                pass
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    r = _finish(net, games, steps, t0, curve, best_greedy, eval_games,
                out_dir, log)
    r["qmax"] = qmax
    return r


def main(argv=None) -> int:
    from tools.accept_meld import _utf8_stdout
    _utf8_stdout()          # 不调这个，GBK 控制台下打不出 ✓ 会丢退出码（踩过）
    argv = sys.argv[1:] if argv is None else argv
    seconds = float(argv[0]) if argv else 3600.0
    buf = BUFFER_GAMES
    if "--buffer" in argv:
        buf = int(argv[argv.index("--buffer") + 1])
    opp = 0.5
    if "--opp-mix" in argv:
        opp = float(argv[argv.index("--opp-mix") + 1])
    kw = {}
    if "--batch" in argv:
        kw["batch_games"] = int(argv[argv.index("--batch") + 1])
    if "--eps-games" in argv:
        kw["eps_games"] = int(argv[argv.index("--eps-games") + 1])
    if "--learn-all-seats" in argv:
        # A/B 的对照臂：恢复改动前的行为（固定对手的着法也进训练目标）
        kw["learn_all_seats"] = True
    if "--init" in argv:
        kw["init"] = argv[argv.index("--init") + 1]
    if "--out-dir" in argv:
        # 两臂必须落在不同目录，否则后跑的会覆盖 best.pt / last.pt ——
        # 而面板按修改时间挑权重（net/advise.py::newest_weights），会**静默换源**
        kw["out_dir"] = argv[argv.index("--out-dir") + 1]
    if "--snap-every" in argv:
        # 1 小时的 A/B 默认只会长出 8 个成员；调小才有「有强度谱」的池子
        kw["snap_every"] = int(argv[argv.index("--snap-every") + 1])
    if "--pool-size" in argv:
        kw["pool_size"] = int(argv[argv.index("--pool-size") + 1])
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1
    if workers > 1:
        r = train_parallel(seconds=seconds, workers=workers, buffer_games=buf,
                           opp_mix=opp, **kw)
    else:
        r = train(seconds=seconds, buffer_games=buf, opp_mix=opp, **kw)
    print(f"\n权重：{r['out_dir']}")
    # spec §7 的分水岭：**明确打过**贪心。50% 只是「五五开」，不算打过。
    ok = r["wr_greedy"] >= 0.55
    print("判据（spec §7 分水岭：明确打得过贪心）：" + (
        "**过了** ✓" if ok else f"**没过** ✗（vs 贪心 {r['wr_greedy']:.1%}，要 ≥55%）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
