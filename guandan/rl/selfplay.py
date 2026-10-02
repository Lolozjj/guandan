"""Plan 3 的 DMC 训练循环（spec §5）。

跑法：
    .venv/Scripts/python.exe -m guandan.rl.selfplay               # 默认 3600 秒（1 小时）
    .venv/Scripts/python.exe -m guandan.rl.selfplay 600           # 10 分钟

**为什么长这样**（对照那个已删的最小冒烟脚本 —— 它只证明「env 能训」）：

| | smoke | selfplay |
|---|---|---|
| 样本 | 只用刚打完的那一批（在线） | 从 **replay buffer** 里采（spec §5.2/5.3） |
| 生成 | 一局一局地打，每个决策点问一次网络 | **32 局同步推进**，一步一批前向 |
| 判据 | 打得过**随机** | 打得过**贪心**（spec §7 的分水岭）+ 随机 |
| 产出 | 一个退出码 | `runs/<版本>/best.pt` + 两条评测曲线 |

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

import copy
import glob
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

import numpy as np
import torch

from guandan import paths
from guandan.sim import env, rules
from guandan.rl import pool, replay
from guandan.rl.eval import match
from guandan.rl.net import (DEVICE, QNet, check_entropy, check_logits, check_q_scale,
                            load_state,
                            log_prob_and_entropy, policy_sample_batch,
                            q_argmax_batch, q_max_batch)
from guandan.rl.policies import greedy_policy, random_policy
from guandan.rl.rule_policy import rule_choose, rule_policy

BATCH_GAMES = 32              # spec §5.3（同步推进的局数；`--batch` 可调大）
LR = 1e-4                     # spec §5.3
EPS_START, EPS_END = 1.0, 0.1
#: buffer 容量（局）。**spec §5.3 写的是 50 万局** —— 按紧凑记录算是 923 MB
#: （见 `guandan/rl/replay.py` 的算术），存张量则是 747 GB。
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


def eps_for(games: int, eps_games: int = EPS_GAMES,
            start: float = EPS_START) -> float:
    """第 `games` 局时的 ε（线性退火，退到底就不再变）。

    `start` 是**这一轮的起点** —— 热启动时它不是 1.0（见 `EPS_START_WARM`）。
    """
    frac = min(1.0, games / max(1, eps_games))
    return start + (EPS_END - start) * frac


#: 热启动时的 ε 起点（初值，**按实测调**）。
#:
#: 为什么不是 `EPS_START`（1.0）：2026-09-27 的三臂 A/B/C 里**每一臂**都比起点差
#: （1407 的 95.2% → 90~94.5%），共同原因就是热启动之后 ε 仍从 1.0 起 ——
#: 头一万多局近乎随机，把热启动整个冲掉了。**不修它，任何「从某个存档出发」的
#: 实验都会被这个效应盖住**（池子那一轮就是这么被盖住的）。
#: 为什么不是 `EPS_END`（0.1）：那样就完全没有退火，探索只剩一成，也学不动。
#: 0.3 = 保留三成探索、又不至于把已有策略冲散。**要动就动这一个数。**
EPS_START_WARM = 0.3


def resolve_eps_start(init: str = None, explicit: float = None) -> float:
    """这一轮的 ε 起点。

    显式给了就用显式的；**热启动**用 `EPS_START_WARM`（低起点）；否则 1.0（老行为）。
    起点会打进日志头 —— 上一轮三臂实验花了 3 小时才看出「热启动被冲掉了」，
    就是因为这件事在日志上查不到。
    """
    if explicit is not None:
        return explicit
    return EPS_START_WARM if init else EPS_START


#: 评测局数。**别往下调**：实测同一个网络换 6 个种子各测 200 局，
#: 结果是 4.8% ± 1.4%（二项理论值 ≈1.5%）—— 也就是 200 局时噪声约 ±2%，
#: 判据「≥55%」不会被噪声骗过。100 局时会飘到 ±5%，不够用。
EVAL_GAMES = 200
RUNS_DIR = str(paths.RUNS)


#: 训练算法。`dmc` = 回归标量 Q + argmax（默认、老行为）；`pg` = 策略梯度（REINFORCE）。
ALGOS = ("dmc", "pg", "ppo")


def _check_algo(name: str) -> str:
    """`--algo` 的唯一校验处。**不认识的必须炸** —— 静默退回 DMC 会让
    「PG 臂」其实是对照臂，而且日志上一个字都不提（评审 M1）。"""
    if name not in ALGOS:
        raise ValueError(f"认不出的算法：{name!r}（只有 {ALGOS}）")
    return name


#: 固定对手有哪几种「强」可选。`greedy` = 老基线；`rule` = 规则式（像人）。
OPP_KINDS = ("greedy", "rule")
OPP_KIND_CN = {"greedy": "贪心", "rule": "规则式"}


def _fixed_pick(kind, pending, rng):
    """固定对手出一手。`kind = ("greedy"|"rule"|"random", 队号)`。

    `rule` 走 `guandan/rl/rule_policy.py`（不压队友 / 留炸 / 算剩牌卡对手）——
    2026-09-28 加的，因为贪心**不像人**：它 100% 压自己队友、从不主动炸、不算剩牌。
    """
    obs, acts, hist = pending
    if kind[0] == "random":
        return rng.randrange(len(acts))
    if kind[0] == "rule":
        return rule_choose(obs, acts)
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
    load_state(net, ck["net"] if isinstance(ck, dict) else ck)


def plan_step(learn_seats, turn, fixed, mates=None) -> tuple:
    """这一步由谁出手 → `("learner", None)` / `("member", mid)` / `("fixed", kind)`。

    **纯函数**，故意抽出来 —— 分组错了会**静默用错权重**（池子里全变成一个模型），
    而那种错在日志上完全看不出来。

    「不属学习队、又没有固定对手」是不该出现的状态（`learn` 与 `fixed` 是一起定的），
    所以**炸掉**而不是猜一个 —— 猜的后果是静默退回贪心（本仓库纪律：失败必须响）。

    `mates`（A3）：`{座位: kind}` —— **学习队里被换成固定策略的那个队友**。
    ⚠️ 它必须在 `fixed is None` 那道检查**之前**判：队友多样性可以用在纯自对弈局里
    （那种局的 `fixed` 就是 None），晚一步就会炸在"没有固定对手"上。
    """
    if turn in learn_seats:
        return ("learner", None)
    if mates and turn in mates:
        return ("fixed", mates[turn])
    if fixed is None:
        raise ValueError(
            f"座位{turn} 不属于学习队 {tuple(learn_seats)}、又没有固定对手 —— "
            f"分组的前提被破坏了")
    if fixed[0] == "member":
        return ("member", fixed[1])
    return ("fixed", fixed[0])


def generate_batch(net, rng, eps, n_games, capture=True, opp_mix=0.0,
                   greedy_share=0.8, learn_all_seats=False, members=None,
                   pick_fixed=None, bomb_cost: float = 0.0,
                   opp_kind: str = "greedy", sample: bool = False,
                   mate_mix: float = 0.0, shaping: float = 0.0):
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

    **`mate_mix`（A3）：队友多样性。** 每局以此概率把**学习队的搭档**也换成固定策略
    （`opp_kind`），于是那一局**只剩一个座位学**。理由：真实搭档不是自己
    （面板/现实里配合的是人），而纯自对弈里两个座位永远同源 ⇒ 网络只在"和自己打配合"上
    被训练。固定队友的着法与固定对手一样**不进训练目标** —— 靠的还是 `learn` 这一个
    变量（`caps` 现场抓取与 `GameRecord.learn` 都用它）⇒ 不存在只改一半的可能。
    ⚠️ **只在「有固定对手」的局里生效**（纯自对弈局里"学习队的搭档"没有定义，
    硬换会留下半个效果）⇒ **A3 的实际剂量 = `opp_mix × mate_mix`**。
    ⚠️ 默认 0.0：关掉时 `mates` 恒为空，行为与加这个开关之前**完全一样**（连 rng 都不多抽）。
    """
    envs, hands0, log, caps, seq = [], [], [], [], []
    learn, fixed = [], []          # 每局：学习那一队的座位 / 固定对手（None = 纯自对弈）
    mates: list = []               # 每局：{座位: kind} —— 被换成固定策略的**队友**（A3）
    members = members if members is not None else {}
    for _ in range(n_games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        envs.append(e)
        hands0.append([set(e.hand.hands[s]) for s in rules.SEATS])   # 发牌快照
        log.append([])
        caps.append([])
        seq.append([])          # 每步的 (座位, Meld)：标签的唯一产地 `mc_targets` 要它
        if rng.random() < opp_mix:
            opp = rng.randrange(2)      # 哪一队当固定对手 —— **统一在这里抽**（池成员也一样）
            if pick_fixed is not None:
                kind = pick_fixed(rng)  # ("greedy",) | ("random",) | ("member", mid)
                # 形状统一成 (kind, x)：member 的 x 是**成员 id**，其余是**队号**
                fixed.append(("member", kind[1]) if kind[0] == "member"
                             else (kind[0], opp))
            else:
                kind = (opp_kind if rng.random() < greedy_share else "random")
                if kind not in OPP_KINDS and kind != "random":
                    raise ValueError(f"认不出的固定对手类型：{kind!r}（只有 {OPP_KINDS} / random）")
                fixed.append((kind, opp))
            learn.append(tuple(s for s in rules.SEATS if rules.TEAM[s] != opp))
            # A3：把**学习队的搭档**也换成固定策略（**队友多样性**）。
            # ⚠️ 只在「有固定对手」的局里做 —— 纯自对弈局里"学习队的搭档"没有定义
            # （四家都学，两个座位互为搭档），硬换一个座位会留下"另一对还在自己配自己"
            # 的半个效果。所以 A3 的剂量就是 `opp_mix × mate_mix`，日志里两个都打。
            # ⚠️ `mate_mix == 0.0` 时**一个 rng 都不许消费** —— 否则同一个种子打出来的牌
            # 会与加这个开关之前不同，老臂就不再可比（本文件那条「rng 消费顺序别漂」）。
            # `and` 的短路就是保证这件事的地方，别改写成先抽再判。
            mates.append({})
            if mate_mix > 0.0 and rng.random() < mate_mix:
                if opp_kind not in OPP_KINDS:
                    raise ValueError(f"认不出的队友类型：{opp_kind!r}（只有 {OPP_KINDS}）")
                cand = learn[-1]
                seat = cand[rng.randrange(len(cand))]
                mates[-1][seat] = opp_kind
                learn[-1] = tuple(s for s in cand if s != seat)
        else:
            fixed.append(None)
            learn.append(tuple(rules.SEATS))
            mates.append({})
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
            who = plan_step(learn[k], envs[k].hand.turn, fixed[k], mates[k])
            groups.setdefault(who, []).append(j)

        def _order(item):
            (who, x), _js = item
            rank = 0 if who == "fixed" else (1 if who == "learner" else 2)
            return (rank, x if isinstance(x, int) else 0)

        for (who, mid), js in sorted(groups.items(), key=_order):
            if who == "fixed":
                # 贪心 / 随机：便宜的 Python 路径，**不占前向**（占混合局的 20%）
                # ⚠️ 用的必须是**这一组的 kind**（`mid`），不是 `fixed[k]` 的：
                # A3 之后同一个"fixed"组里可能有队友（另一种 kind）与对手两种来源。
                for j in js:
                    picks[j] = _fixed_pick((mid,), pending[j], rng)
                continue
            if who == "member":
                if mid not in members:
                    raise KeyError(
                        f"对手池里没有成员 {mid}（有 {sorted(members)}）—— "
                        f"不许静默换个对手，那会让池子悄悄少一个成员")
                neti = members[mid]
            else:
                neti = net
            if who == "learner" and sample:
                # PG 的**行为策略**：从 π 采样，**完全忽略 ε**（spec §3.3）。
                # 放在 `eps >= 1.0` 那个分支**之前**是故意的 —— ε 在 PG 模式下必须
                # 一点都不起作用，否则数据不是 π 的，策略梯度就估错了分布。
                for j, pick in zip(js, policy_sample_batch(
                        neti, [pending[j] for j in js], rng)):
                    picks[j] = pick
                continue
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
            # ⚠️ 座位要取**动手之前**的（`step` 之后就换人了）
            seq[k].append((e.hand.turn, acts[i]))
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
        # 固定对手那一队的胜负 —— PFSP 的归因靠它。
        # ⚠️ **不重放**：实测重放一局 18.3 ms，每秒几十局池对局就是几十个百分点的开销。
        # （对照臂 `learn_all_seats` 下 learn 是四家，这里算出来的「我方」没有意义 ——
        #   但对照臂没有池子，没人看这个字段。）
        won = None
        if fixed[k] is not None:
            won = rules.winner_team(e.ranks) == rules.TEAM[learn[k][0]]
        rec = replay.GameRecord.of(e, log[k], hands0[k], learn=learn[k],
                                   opp=fixed[k], won=won)
        if capture:
            labels = replay.mc_targets(seq[k], e.ranks, learn=learn[k],
                                       bomb_cost=bomb_cost,
                                       hands0=hands0[k], shaping=shaping)
            assert len(labels) == len(caps[k]), (
                f"标签条数 {len(labels)} 与决策点数 {len(caps[k])} 对不上")
            out.append((rec, caps[k], labels))
        else:
            out.append((rec, [], []))
    return out


#: 自举的步数（spec §8 的初值 3 已被 2026-09-28 的评审改掉，见下）。
#: ⚠️ **必须是偶数**：`V(s_{t+n})` 是「**那一刻出手的人**那一队」的值，而出手顺序是
#: `0→3→2→1`，所以**奇数 n 的自举源落在对家** —— 符号反、标签被往 0 拉。
#: 实测（`1407`）：n=1/3 的自举命中率只有 5%/16%（corr 与标签是 -0.35/-0.15），
#: n=2/4 是 89%/88%（corr +0.27）。详见 `guandan/rl/replay.py::_boot_source_ok`。
#: 一局每座位约 33 个决策点；n 越小偏差越大、n→∞ 就是 MC。
N_STEP = 2
#: β —— MC 与自举的混合比。**1.0 = 完全就是现在的 DMC（默认，行为不变）**。
MC_MIX = 1.0
#: 目标网络同步的间隔（局）—— 与权重广播同拍（那个节奏已经验过不拖死 learner）。
TGT_SYNC_GAMES = 1000


def sync_target(net, net_tgt, games, last_sync, every=TGT_SYNC_GAMES) -> int:
    """到点就把 `net` 复制进目标网络，返回新的 `last_sync`。

    ⚠️ 目标网络**永远不参与优化**，只用来算 `V`（spec §3.3）。它是为自举而存在的：
    没有它，目标就是「追自己的尾巴」，发散风险大增。
    `net_tgt=None`（β=1）或 `every<=0` 时是空操作。
    """
    if every <= 0:
        raise ValueError(
            f"tgt_sync 必须为正，给的是 {every} —— 0 会让目标网络**永不同步**，"
            f"靶子停在热启动权重上（那是 bug，不是配置）。配置错了必须响。")
    if net_tgt is None:            # β=1：压根没建目标网络，空操作
        return last_sync
    if games - last_sync < every:
        return last_sync
    net_tgt.load_state_dict(net.state_dict())
    return games


def build_samples(buf, rng, batch_games, *, fresh=None, bomb_cost=0.0, n_step=0,
                  shaping: float = 0.0):
    """从 buffer 采一批、重放成张量原料。返回 `(samples, y_mc, boot)`，**等长同序**。

    `fresh` 是单进程那条路的「刚打完的这一批」快捷缓存（省一次重放，约 18ms/局）。
    ⚠️ **要自举时它必须让路**（`n_step > 0`）—— 缓存里没有 `boot`，
    用它就等于让这一部分样本悄悄退回纯 MC，而 loss 曲线上一概看不出来。
    （多进程那条路本来就没有这个缓存，不受影响。）

    ⚠️ 长度必须对齐：`samples` / `y_mc` / `boot` 三者错开一格就是
    「拿别人的未来当自己的标签」，而 `blend` 只能挡住后两者的错位。
    """
    if n_step:
        fresh = None
    samples, y_mc, boot = [], [], []
    for rec in buf.sample(batch_games, rng):
        got = fresh.get(id(rec)) if fresh else None
        if got is None:
            pts, y, b = replay.expand(rec, bomb_cost=bomb_cost, n=n_step, shaping=shaping)
        else:
            pts, y = got
            b = []
        samples += pts
        y_mc += y
        boot += b
    return samples, y_mc, boot


def _check_boot_args(mc_mix, n_step, tgt_sync) -> None:
    """自举三个参数的**唯一一处校验**（两条训练路线共用 —— 各写一份就会漂）。"""
    if not 0.0 <= mc_mix <= 1.0:
        raise ValueError(f"mc_mix(β) 必须在 [0, 1]，给的是 {mc_mix}")
    if mc_mix < 1.0 and n_step <= 0:
        raise ValueError(
            f"β={mc_mix:g} < 1 却把 n_step 设成 {n_step} —— 要自举就得往后看至少 1 步，"
            f"这两个参数是矛盾的：日志会写「自举 β={mc_mix:g}」，实际一个自举项都不算。")
    if tgt_sync <= 0:
        raise ValueError(f"tgt_sync 必须为正，给的是 {tgt_sync}（0 = 目标网络永不同步）")


def _pg_log(algo: str, beta_ent: float, weight_sync_games: int = None) -> str:
    """日志头里的算法那一段。**两条路线共用一份**（各写一份就会漂）。

    ⚠️ PG 模式下**必须写清 ε 不适用**：否则日志上写着 `ε 起点 0.30` 而实际没用，
    就是「换源不可见」（本仓库纪律）。同理要写明不用 buffer。

    `weight_sync_games=None`（单进程那条路）**不打广播那一行** —— 它没有 worker，
    也就没有广播这件事；打出来就是在日志里说一件不会发生的事（反过来也是骗人）。
    """
    if algo != "pg":
        return ""
    out = ("\n  算法 pg（REINFORCE）  "
           f"β_ent {beta_ent:g}  **ε 不适用**（采样本身就是探索）"
           "\n  **不用 replay buffer**（on-policy，只用刚打完的那一批）")
    if weight_sync_games:
        out += f"\n  权重广播每 {weight_sync_games} 局（PG 的 staleness 靠它压小）"
    return out


def _boot_log(mc_mix, n_step, tgt_sync) -> str:
    """日志头里的自举那一段。**两条路线共用一份**（各写一份就会漂）。"""
    if mc_mix >= 1.0:
        return "  自举关（β=1）"
    odd = ("  ⚠️ n 是**奇数** —— 自举源落在对家、命中率很低"
           "（见 replay._boot_source_ok 的实测表），请用偶数" if n_step % 2 else "")
    return f"  自举 β={mc_mix:g} n={n_step}（目标网络每 {tgt_sync} 局同步）{odd}"


#: 熵系数（spec §8 的初值）。**这是 PG 实验里第一个要动的数**：
#: 太小 ⇒ 策略提前塌成 argmax（有守门会炸）；太大 ⇒ 一直乱出、学不动。
BETA_ENT = 0.01


class _RunningMean:
    """`R` 的滑动均值 —— PG 的基线。

    为什么不学一个 V 网络（spec §3.2）：现在 95% 的局都赢 ⇒ `R` 的方差本来就小
    ⇒ 滑动均值够用，而学 V 要多一处能出错的地方。**这是刻意的简化。**
    """

    def __init__(self, window: int = 1000):
        if window <= 0:
            raise ValueError("window 必须为正")
        self.window, self._buf, self._sum = window, [], 0.0

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        self._sum += float(x)
        while len(self._buf) > self.window:
            self._sum -= self._buf.pop(0)
        return self.value

    @property
    def value(self) -> float:
        return self._sum / len(self._buf) if self._buf else 0.0


def _pg_step(net, opt, samples, rewards, base, games, beta_ent: float = BETA_ENT):
    """一步策略梯度。**更新只作用在实际出的那一手**上 ——
    这就是它绕开「一局 132 个决策点共享同一个标签」的全部理由。

        L = − mean_i [ log π(a_i|s_i) · (R_i − b) ] − β_ent · mean_i H(π(·|s_i))

    `rewards` 是每个决策点的 `R`（同一局里每个点都一样 —— 标签仍由 `mc_targets` 产出）。
    返回 `{"loss", "entropy", "adv"}`（float，给日志用）。
    """
    if len(samples) != len(rewards):
        raise ValueError(f"样本数 {len(samples)} 与标签数 {len(rewards)} 对不上")
    lp, ent, zmax = log_prob_and_entropy(net, samples)
    # ⚠️ **设备要跟 `lp` 走**：网络在 `DEVICE`（可能是 cuda）上，而标签是新造的 cpu 张量
    # ⇒ 不搬就 `Expected all tensors to be on the same device`。
    # （单测里网络在 cpu，所以只有走 `train()` 的集成路径才会撞上 —— 它抓到了。）
    #
    # ⚠️ 顺序是**先用旧基线、再把它喂进去**（不是反过来）：
    # 用更新后的基线会让这一批自己出现在自己的基线里 ⇒ adv 被自我抵消一部分。
    # （`tests/test_pg_step.py::test_pg_step_reports_the_advantage_after_the_baseline` 钉着）
    adv = torch.tensor(rewards, dtype=torch.float32, device=lp.device) - base.value
    # ⚠️ **这一行不能少**：少了就 `= R − 0 ≡ R`，而这个任务 ~95% 的局都赢
    # ⇒ `R` 几乎恒正 ⇒ 更新退化成「把采样到的那一手无条件往上抬」✗
    # ⇒ 那本身就足以把策略磨塌，**塌陷就不能干净地归因到信任域** ✗
    # （评审 2026-09-29 抓到的 Critical：`update` 全仓库只有测试在调。）
    base.update(sum(rewards) / len(rewards))
    loss = -(lp * adv).mean() - beta_ent * ent.mean()
    # 两处守门（「失败必须响」）：logits 溢出 / 熵塌
    # ⚠️ 这里**不能用 `check_q_scale`** —— 它的阈值是按"值"定的（标签尺度 ±3），
    # 搬到 logits 上会误杀（2026-09-29 真踩过：8,256 局、一切正常却炸了）。
    check_logits(zmax, games, float(loss.detach()))
    check_entropy([float(x) for x in ent.detach()],
                  [math.log(len(a)) for _o, a, _i, _s, _h in samples])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {"loss": float(loss.detach()), "entropy": float(ent.mean().detach()),
            "adv": float(adv.mean())}


PPO_CLIP = 0.2        # PPO 的裁剪半径（`--ppo-clip`）
PPO_EPOCHS = 4       # 每批数据重复更新几轮（PG 是 1 轮 —— 那正是它塌陷的原因）
PPO_MINIBATCH = 256


def _ppo_step(net, opt, samples, rewards, base, games, beta_ent, *, clip: float = PPO_CLIP,
              epochs: int = PPO_EPOCHS, minibatch: int = PPO_MINIBATCH,
              normalize_adv: bool = True, rng=None):
    """**PPO**：一次采样、**多轮小批裁剪更新**。修的是 `--algo pg` 的塌陷。

    为什么 PG 会塌（台账里的诊断）：`_pg_step` 只用一批数据更新**一次** ——
    重要性比恒等于 1，所以"信任域"这件事在它那里**根本不存在**；
    熵奖励是软约束，β 从 0.01 扫到 0.1 都治不了。PPO 的做法是：
    同一批数据走 `epochs` 轮小批，每轮把 `π_new/π_old` **裁到 [1−ε, 1+ε]**，
    超过半径的梯度直接归零 ⇒ 一步最多把策略推这么远。

    ⚠️ **行为策略的 log-prob 在这里当场算**（`lp_old`）：PG 是 on-policy、
       采样与更新之间没有别的更新 ⇒ 更新前那一刻的网络**就是**行为策略。
       （所以不需要改采样路径去传 log-prob —— 那会让 worker 的记录格式变复杂。）

    ⚠️ `epochs=1` 且 `clip` 很大、`normalize_adv=False` 时，它与 `_pg_step` **等价**
       —— 这条被测试钉着（否则"改了什么"说不清）。
    """
    import numpy as _np

    lp_old, _e0, zmax0 = log_prob_and_entropy(net, samples)
    lp_old = lp_old.detach()
    adv = torch.tensor(rewards, dtype=torch.float32, device=lp_old.device) - base.value
    base.update(sum(rewards) / len(rewards))          # 顺序与 `_pg_step` 一致：先用旧基线
    if normalize_adv and len(samples) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    check_logits(zmax0, games, 0.0)

    order = list(range(len(samples)))
    if rng is not None:
        rng.shuffle(order)
    out = {"loss": float("nan"), "entropy": float("nan"), "adv": float(adv.mean()),
           "ratio": 1.0}
    for _ep in range(epochs):
        ents, max_ents = [], []
        for i0 in range(0, len(order), minibatch):
            ix = order[i0:i0 + minibatch]
            sub = [samples[i] for i in ix]
            lp, ent, zmax = log_prob_and_entropy(net, sub)
            ratio = torch.exp(lp - lp_old[ix])
            a = adv[ix]
            loss = -(torch.min(ratio * a,
                               torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a).mean()
                     ) - beta_ent * ent.mean()
            check_logits(zmax, games, float(loss.detach()))
            opt.zero_grad(); loss.backward(); opt.step()
            ents += [float(x) for x in ent.detach()]
            max_ents += [math.log(len(s[1])) for s in sub]
            out = {"loss": float(loss.detach()), "entropy": float(ent.mean().detach()),
                   "adv": float(a.mean()), "ratio": float(ratio.mean().detach())}
        check_entropy(ents, max_ents)                  # 熵塌了要**响**（PG 就是这样被逮到的）
    return out


def _targets(net_tgt, buf, rng, batch_games, bomb_cost, mc_mix, n_step,
             fresh=None, shaping: float = 0.0):
    """这一批训练样本的目标值。**自举在这里、且只在这里进入标签。**

    `mc_mix >= 1` 是纯 MC（默认）—— 那时连 `V` 都不算，`boot` 也不产出。
    `boot` 里越界的那些点是 `None`，`blend` 会把它们整项退回 `y_mc`。

    ⚠️ **只收目标网络**（在线网络根本不传进来）—— 老签名里挂着一个没用到的
    `net` 参数，会让人以为「在线网络也参与算目标」，顺手写成
    `q_max_batch(net, ...)`（= 追自己的尾巴）时**全部测试照样绿**（评审 I2）。
    现在这件事由**签名**排除，不靠注释。
    """
    n_boot = 0 if mc_mix >= 1.0 else n_step
    samples, y_mc, boot = build_samples(buf, rng, batch_games, fresh=fresh,
                                        bomb_cost=bomb_cost, n_step=n_boot,
                                        shaping=shaping)
    # `boot` 的第 4 位是出手人（判视角用的）—— 打分只吃前三位
    vals = (q_max_batch(net_tgt, [b[:3] if b is not None else None for b in boot])
            if n_boot else None)
    return samples, replay.blend(y_mc, vals, mc_mix)


def _close_weights(targets, close_weight: float, dev):
    """**按胶着度**给样本加权（信用分配 spec 的设计 2）。

    每个决策点的标签**本身就带着胶着度**：`rules.reward` 按胜方**最差名次**给分，
    `|y| = 3` ⟺ 双上 / 被双上（一边倒，所有决策拿到同一个 ±3，对"该不该炸"零区分度），
    `|y| ∈ {1,2}` ⟺ 1-3 / 1-4 / 2-3 / 2-4（胶着局）。

    ⚠️ 归一化交给调用方用 `/ w.sum()` —— 否则改权重会顺手改掉**有效学习率**。
    """
    w = torch.ones(len(targets), dtype=torch.float32, device=dev)
    for i, t in enumerate(targets):
        if abs(float(t)) < 3.0:
            w[i] = close_weight
    return w


def _learn_step(net, net_tgt, buf, rng, opt, games, *, batch_games, bomb_cost,
                mc_mix, n_step, fresh=None, close_weight: float = 1.0,
                shaping: float = 0.0):
    """从 buffer 采一批 → 重放 → 拼张量 → 一步 MSE。**两条训练路线共用这一份。**

    `mc_mix < 1` 时多一次前向算自举项（用**目标网络**、`no_grad`、返回 float）。
    `|Q|` 或 `|标签|` 超限会**在这里 raise**（发散必须响，不许静默地训下去）。

    `close_weight`（默认 1.0 = 关）：胶着局样本的权重。**1.0 时走原来那一行**
    `F.mse_loss`（逐位不变），不开加权分支 —— 免得"默认没变"只是近似成立。
    """
    samples, targets = _targets(net_tgt, buf, rng, batch_games, bomb_cost,
                                mc_mix, n_step, fresh=fresh, shaping=shaping)
    st, ac, hi = _tensors(samples)
    dev = next(net.parameters()).device
    y_hat = net(st.to(dev), ac.to(dev), hi.to(dev))
    y = torch.tensor(targets, dtype=torch.float32).to(dev)
    if close_weight == 1.0:
        loss = torch.nn.functional.mse_loss(y_hat, y)
    else:
        w = _close_weights(targets, close_weight, dev)
        loss = (w * (y_hat - y) ** 2).sum() / w.sum()
    # 发散守门：**预测与标签都查**（自举跑飞时，标签通常先炸）
    # `.detach()` 不能省：`float()` 直接作用在带梯度的张量上，PyTorch 会告警
    # （"Converting a tensor with requires_grad=True to a scalar"）—— 守门是纯读，
    # 不该把预测卷进任何图里。
    check_q_scale(float(y_hat.detach().abs().max()), games, loss.item())
    check_q_scale(float(y.detach().abs().max()), games, loss.item(), what="标签")
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def _tensors(samples):
    """把 `(obs, acts, i, seat, hist)` 铺成张量。"""
    st = torch.from_numpy(np.stack([env.encode_state(o) for o, _a, _i, _s, _h in samples]))
    ac = torch.from_numpy(np.stack([env.encode_action_now(a[i], o.level, o.hand)
                                    for o, a, i, _s, _h in samples]))
    hi = torch.from_numpy(np.stack([h for _o, _a, _i, _s, h in samples]))
    return st, ac, hi


def net_play(net):
    """把网络包成策略。**带 `batch_choose`** —— 评测器会一次把同一时刻的
    所有决策点送进网络（`guandan/rl/eval.py` 的 `match` 认这个属性）。"""
    from guandan.rl.policies import batch_net_policy
    from guandan.rl.net import q_argmax_batch
    return batch_net_policy(q_argmax_batch, net)


WEIGHT_SYNC_GAMES = 1000     # 多进程：每这么多局把权重广播给 worker


def _maybe_eval(net, games, curve, best, eval_games, eval_every,
                batch_games, out_dir, log, snap_every=pool.SNAP_EVERY_GAMES,
                pool_size=pool.POOL_SIZE, rule_games=None, rule_seeds: int = 1):
    """每 `eval_every` 局评测一次并（可能）存 `best.pt`。**两条训练路线共用这一份。**

    三把尺子：`vs 随机` / `vs 贪心` / **`vs 规则式`**（`rule_policy()`）。
    `vs 贪心` 早在 90%+ 饱和，量不出代差 —— **`vs 规则式` 才是现在的主尺子**
    （`tools/ruler.py` 就是拿它做多臂配对）。三个数**共用同一个 `eval_games`**，
    所以互相可比；但要跟 `tools/ruler.py` 那套（多种子配对 × 400 局）**分开口径**，
    别把这里的数直接当台账上的数引用。

    ## `best.pt` 按 `vs 规则式` 挑（2026-09-30 改，`best` 就是它）

    老口径是按训练内 200 局 `vs 贪心` 挑。但那把尺子有两个毛病叠加：
    **噪声 ±2pp**（200 局）而**一代的真实进步只有 ~3pp** —— 噪声与信号同量级，
    于是挑出来的是「运气最好的那一次评测」（实测 R8 的 `best.pt` 比它自己的末尾差 2.0pp）。
    规则式那把还留着几十个百分点余量，挑出来更接近真实强度。

    ⚠️ `rule_games=0`（规则式没跑）时**退回按 `vs 贪心` 挑**，并且把口径写进日志与
    权重元信息 —— 不许静默换口径（本仓库的「换源必须可见」）。

    `rule_games`：规则式那把跑多少局。`None` = 与 `eval_games` 相同；`0` = 不跑
    —— 规则式是纯 Python 启发式，比贪心慢（200 局约 10 秒 vs 6.6 秒），
    所以量吞吐的短跑（`tools/bench_train.py`）必须能把它关掉。

    顺带按 `snap_every` 存池子快照 —— **与「有没有刷新最好」无关**：
    只存 `best.pt` 的话池子原料不够（144 万局只落几个点）。
    """
    if games % eval_every < batch_games:
        wr_r = match(net_play(net), random_policy(random.Random(101)),
                     games=eval_games, seed=1001)
        wr_g = match(net_play(net), greedy_policy, games=eval_games, seed=1002)
        rg = eval_games if rule_games is None else rule_games
        # ⚠️ **多种子取均值**（点③，2026-10-01）：单种子 400 局的 sd ≈ 2.5pp，
        # 而"一代的进步"只有 1~3pp ⇒ 单种子挑出来的 `best.pt` 自带**选点偏差**
        # （实测：训练内挑出 +2.02pp 的件，扩大种子后缩到 +1.45pp）。
        # 多跑 `rule_seeds` 个种子取均值把这份偏差压掉 —— 代价是每次评测多几十秒。
        wr_rule = (statistics.mean(
            [match(net_play(net), rule_policy(), games=rg, seed=1004 + i)
             for i in range(rule_seeds)]) if rg else None)
        curve.append((games, wr_r, wr_g, wr_rule))
        log(f"     >> 评测 @ {games:7d} 局：vs 随机 {wr_r:.1%}   vs 贪心 {wr_g:.1%}"
            + (f"   vs 规则式 {wr_rule:.1%}" if wr_rule is not None else ""))
        # 挑选口径 = 主尺子；规则式没跑时退回贪心，并在下面那行日志里说明
        metric = "vs 规则式" if wr_rule is not None else "vs 贪心（规则式未跑）"
        score = wr_rule if wr_rule is not None else wr_g
        if score > best:
            best = score
            torch.save({"net": net.state_dict(), "games": games,
                        "winrate_random": wr_r, "winrate_greedy": wr_g,
                        "winrate_rule": wr_rule, "best_metric": metric},
                       os.path.join(out_dir, "best.pt"))
            log(f"        ↑ 刷新最好（{metric} {score:.1%}）-> best.pt")
    if snap_every and games and games % snap_every < batch_games:
        p = pool.snapshot_path(out_dir, games)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"net": net.state_dict(), "games": games}, p)
        gone = pool.prune_snapshots(out_dir, pool_size)
        log(f"     >> 池子快照 {os.path.basename(p)}"
            + (f"（挤掉 {len(gone)} 个）" if gone else ""))
    return best


def _finish(net, games, steps, t0, curve, best, eval_games, out_dir, log,
            bomb_games: int = 25, rule_games=None):
    """收尾：两次评测看方差 + 报告 + 存 last.pt + 返回结果。**两条路线共用。**

    `best` 是 `_maybe_eval` 一路带上来的**挑选分数**（默认口径 = `vs 规则式`，
    规则式没跑时退回 `vs 贪心`；见那个函数的 docstring）。

    `elapsed` 是**训练阶段**的秒数（在收尾评测之前取）—— 吞吐要用它算，
    别把固定的评测开销算进去（否则 bench 的短跑会被评测淹没）。
    `bomb_games=0` 跳过炸弹浪费率那一步（bench 用，省 30 秒）；
    `rule_games=0` 跳过规则式那把尺子（同理，bench 用）。
    """
    elapsed = time.perf_counter() - t0
    wr_g1 = match(net_play(net), greedy_policy, games=eval_games, seed=2001)
    wr_g2 = match(net_play(net), greedy_policy, games=eval_games, seed=2002)
    wr_r = match(net_play(net), random_policy(random.Random(202)),
                 games=eval_games, seed=2003)
    # 规则式用**同样的两个 seed**（2001/2002）—— 也就是同一副牌，
    # 所以「vs 贪心」与「vs 规则式」这两组数可以直接对着看。
    rg = eval_games if rule_games is None else rule_games
    wr_u1 = match(net_play(net), rule_policy(), games=rg, seed=2001) if rg else None
    wr_u2 = match(net_play(net), rule_policy(), games=rg, seed=2002) if rg else None
    log("")
    log(f"总共 {games:,} 局 / {steps:,} 步 / {elapsed:.0f} 秒")
    log(f"末次：vs 随机 {wr_r:.1%}   vs 贪心 {wr_g1:.1%} / {wr_g2:.1%}"
        f"（两次，差 {abs(wr_g1 - wr_g2):.1%} —— 200 局的噪声约 ±2%）")
    if wr_u1 is not None:
        log(f"      vs 规则式 {wr_u1:.1%} / {wr_u2:.1%}"
            f"（两次，差 {abs(wr_u1 - wr_u2):.1%}，**与上面同一副牌**）"
            f" ← **主尺子**；口径与 `tools/ruler.py`（多种子配对 ×400 局）不同，别混")
    if best >= 0:
        log(f"  best.pt 的挑选分数 {best:.1%}"
            + ("（规则式未跑，退回 vs 贪心）" if rule_games == 0 else "（按 vs 规则式）"))
    if curve:
        log("曲线（局数:vs随机/vs贪心/vs规则式）：" + "  ".join(
            f"{g // 1000}k:{r:.0%}/{k:.0%}" + (f"/{u:.0%}" if u is not None else "")
            for g, r, k, u in curve))
    # 炸弹浪费率（用户 2026-09-26 报的毛病）——**只报告，不当判据**：
    # 判据仍是「vs 贪心 ≥55%」（与老几次跑可比），这个数是给你看「有没有变好」的。
    if bomb_games:
        try:
            from guandan.rl.eval import bomb_waste
            w, c = bomb_waste(net_play(net), games=bomb_games, seed=3001)
            log(f"  炸弹浪费率（能用普通牌压却出炸）：{w}/{c} = {w / max(1, c):.1%}"
                f"   [对照：贪心恒为 0%]")
        except Exception as exc:                  # noqa: BLE001 - 报告失败不该带崩训练
            log(f"  炸弹浪费率：没量成（{type(exc).__name__}: {exc}）")
    wr_rule = (wr_u1 + wr_u2) / 2 if wr_u1 is not None else None
    torch.save({"net": net.state_dict(), "games": games,
                "winrate_random": wr_r, "winrate_greedy": wr_g1,
                "winrate_rule": wr_rule},
               os.path.join(out_dir, "last.pt"))
    return {"games": games, "curve": curve, "best_score": best,
            "wr_random": wr_r, "wr_greedy": (wr_g1 + wr_g2) / 2,
            "wr_rule": wr_rule,
            "out_dir": out_dir, "elapsed": elapsed}


def pool_report(wr, weights, log) -> bool:
    """打印池子健康度表，返回**是否塌陷**（spec §1.4）。

    塌了是「失败」，不是一行日志 —— 返回值就是那个「响」，调用方据此报告
    （本仓库纪律：失败必须响）。
    """
    eff = pool.effective_members(weights)
    log(f"  池子（有效成员数 {eff:.2f}，共 {len(weights)} 个）:")
    for mid in sorted(weights):
        log(f"    #{mid:<3d} 打了 {wr.games(mid):5d} 局  "
            f"学习者胜率 {wr.rate(mid):5.1%}  采样权重 {weights[mid]:5.1%}")
    if eff < pool.COLLAPSE_BELOW:
        log(f"  ⚠️ **池子塌了**（有效成员数 {eff:.2f} < {pool.COLLAPSE_BELOW}）"
            f"—— 对手退化成同一个模型，训练会绕圈")
        return True
    return False


def load_seed_pool(paths, pool_size: int = pool.POOL_SIZE):
    """磁盘上的一批 `best.pt` -> `(pool_sds, pool_order)`。

    **按 `pool_size` 截断**（取修改时间最新的那几个）。⚠️ 不截断的后果：
    `--pool-size` 就只约束运行时新增的成员，种子那 5 个一路全留 ——
    想按文档（现用方案 §五「池子留最近 `--pool-size` 个」）调小内存的人调不动。

    读不出来**必须炸** —— 静默少一个成员，A/B 的结果就没法解释。
    """
    sds, order = {}, []
    if pool_size <= 0:
        return sds, order
    newest = sorted(paths, key=os.path.getmtime, reverse=True)[:pool_size]
    for mid, p in enumerate(newest):
        try:
            sds[mid] = torch.load(p, map_location="cpu")["net"]
        except Exception as exc:            # noqa: BLE001
            raise RuntimeError(
                f"种子池成员读不出来：{p}（{type(exc).__name__}: {exc}）") from exc
        order.append(mid)
    return sds, order


def train(seconds: float = 3600.0, seed: int = 0, buffer_games: int = BUFFER_GAMES,
          eval_games: int = EVAL_GAMES, eval_every: int = EVAL_EVERY_GAMES,
          out_dir: str = None, log=print, opp_mix: float = 0.5,
          greedy_share: float = 0.8, batch_games: int = BATCH_GAMES,
          eps_games: int = EPS_GAMES, learn_all_seats: bool = False,
          init: str = None, snap_every: int = pool.SNAP_EVERY_GAMES,
          pool_size: int = pool.POOL_SIZE, eps_start: float = None,
          bomb_cost: float = 0.0, mc_mix: float = MC_MIX,
          n_step: int = N_STEP, tgt_sync: int = TGT_SYNC_GAMES,
          opp_kind: str = "greedy", algo: str = "dmc",
          beta_ent: float = BETA_ENT, weight_sync_games: int = WEIGHT_SYNC_GAMES,
          eval_rule_games: int = None, mate_mix: float = 0.0,
          close_weight: float = 1.0, shaping: float = 0.0,
          rule_eval_seeds: int = 1, ppo_clip: float = PPO_CLIP, ppo_epochs: int = PPO_EPOCHS):
    _check_boot_args(mc_mix, n_step, tgt_sync)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    # ⚠️ **`.to(DEVICE)` 不能省。** 漏了它的后果是静默的：日志第一行写着 `device=cpu`，
    # 训练照跑、只是慢 —— Plan 3 那 5 万局与 2026-09-26 那次 9 小时跑都是这么过去的
    # （GPU 从没被用上）。训练步的瓶颈就是网络前向（spec §14.3）。
    # `tests/test_train_device.py` 用 1 秒预算真跑一次钉住这件事。
    if algo == "pg" and close_weight != 1.0:
        raise ValueError(
            "`--close-weight` 只接在 DMC 那条路上（PG 的损失是策略梯度，没有样本权重）"
            "—— 别让它静默失效（日志说改了、其实没改）")
    net = QNet().to(DEVICE)
    load_init(net, init)
    eps_start = resolve_eps_start(init, eps_start)   # 热启动 -> 低起点（EPS_START_WARM）

    # 目标网络：**deepcopy 而不是再 `QNet()`** —— 后者会多消耗一份 RNG，
    # 于是同一个 `--seed` 下两臂的网络初始化就不一样了（那是看不见的变量）。
    # β=1（默认）时根本不建：不多算一次前向，A/B 的对照臂跑的就是**老代码**
    # 外加一个恒假的 `if`。
    net_tgt = copy.deepcopy(net).requires_grad_(False) if mc_mix < 1.0 else None
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    log(f"device={next(net.parameters()).device}  预算 {seconds:.0f}s  "
        f"batch={batch_games} 局（同步推进）  buffer={buffer_games:,} 局  "
        f"评测每 {eval_every:,} 局  ε 起点 {eps_start:.2f} 按 {eps_games:,} 局退火  "
        f"对手混合 {opp_mix:.0%}"
        f"（其中{OPP_KIND_CN.get(opp_kind, opp_kind)} {greedy_share:.0%}）"
        + (f"  **规则尺子 {rule_eval_seeds} 种子**" if rule_eval_seeds > 1 else "")
        + (f"  **队友混合 {mate_mix:.0%}**（队友={OPP_KIND_CN.get(opp_kind, opp_kind)}）"
           if mate_mix > 0 else "")
        + (f"  炸弹代价 λ={bomb_cost:g}" if bomb_cost else "")
        + (f"  **胶着加权 ×{close_weight:g}**" if close_weight != 1.0 else "")
        + (f"  **势函数 shaping β={shaping:g}**" if shaping else "")
        + ("" if algo == "pg" else _boot_log(mc_mix, n_step, tgt_sync))
        + (f"  热启动 {init}" if init else "")
        + _pg_log(algo, beta_ent)      # 单进程没有权重广播 ⇒ 不打那一行
        + (f"  **PPO clip={ppo_clip:g} × {ppo_epochs} 轮**" if algo == "ppo" else "")
        + f"\n权重 -> {out_dir}")

    t0 = time.perf_counter()
    games = steps = 0
    last_tgt = 0                # 目标网络上次同步的局数
    base = _RunningMean()       # PG 的基线（DMC 用不到，留着不占什么）
    curve = []
    best = -1.0                 # `best.pt` 的挑选分数（默认 = vs 规则式，见 _maybe_eval）

    while time.perf_counter() - t0 < seconds:
        # **按局数退火**，不按时间（见 EPS_GAMES）；起点见 eps_start（热启动会压低）
        eps = eps_for(games, eps_games, start=eps_start)

        # 1) 同步打一批（PG 模式下**只吃这一批**；DMC 模式记进 buffer 待采）
        batch = generate_batch(net, rng, eps, batch_games,
                               opp_mix=opp_mix, greedy_share=greedy_share,
                               learn_all_seats=learn_all_seats,
                               bomb_cost=bomb_cost, opp_kind=opp_kind,
                               mate_mix=mate_mix,
                               sample=(algo in ("pg", "ppo")), shaping=shaping)

        if algo in ("pg", "ppo"):
            # **on-policy：不写 buffer**（写了就是 off-policy，要重要性采样）。
            # 标签仍由 `mc_targets` 产出 —— `generate_batch` 内部走的就是它。
            samples = [p for _rec, pts, _y in batch for p in pts]
            rewards = [r for _rec, _pts, y in batch for r in y]
            if algo == "ppo":
                out = _ppo_step(net, opt, samples, rewards, base, games, beta_ent,
                                clip=ppo_clip, epochs=ppo_epochs, rng=rng)
            else:
                out = _pg_step(net, opt, samples, rewards, base, games, beta_ent)
            loss = out["loss"]
            games += len(batch)
        else:
            fresh = {}
            for rec, pts, y in batch:
                buf.add(rec)
                fresh[id(rec)] = (pts, y)
                games += 1
            # 从 buffer 采一批 + 训一步（spec §5.2）—— **共享实现**
            loss = _learn_step(net, net_tgt, buf, rng, opt, games,
                               batch_games=batch_games, bomb_cost=bomb_cost,
                               mc_mix=mc_mix, n_step=n_step, fresh=fresh,
                               close_weight=close_weight, shaping=shaping)
            last_tgt = sync_target(net, net_tgt, games, last_tgt, tgt_sync)
        steps += 1

        el = time.perf_counter() - t0
        if steps % 10 == 0:
            if algo == "pg":
                log(f"  {el:6.0f}s  局数 {games:7d}  loss={loss:.3f}  "
                    f"H={out['entropy']:.3f}  {games / el:.1f} 局/秒")
            else:
                log(f"  {el:6.0f}s  局数 {games:7d}  ε={eps:.2f}  loss={loss:.3f}  "
                    f"{games / el:.1f} 局/秒  buffer {len(buf):,}")

        # 4) 每 N 局：存权重 + 评测（spec §5.3）—— 公共件，两条训练路线共用
        best = _maybe_eval(net, games, curve, best, eval_games,
                           eval_every, batch_games, out_dir, log,
                           snap_every=snap_every, pool_size=pool_size,
                           rule_games=eval_rule_games,
                           rule_seeds=rule_eval_seeds)

    return _finish(net, games, steps, t0, curve, best, eval_games,
                   out_dir, log, rule_games=eval_rule_games)


def train_parallel(seconds: float = 3600.0, workers: int = 1, seed: int = 0,
                   buffer_games: int = BUFFER_GAMES, eval_games: int = EVAL_GAMES,
                   eval_every: int = EVAL_EVERY_GAMES, out_dir: str = None,
                   log=print, opp_mix: float = 0.5, greedy_share: float = 0.8,
                   batch_games: int = BATCH_GAMES, eps_games: int = EPS_GAMES,
                   learn_all_seats: bool = False, init: str = None,
                   snap_every: int = pool.SNAP_EVERY_GAMES,
                   pool_size: int = pool.POOL_SIZE, pfsp: bool = False,
                   pool_greedy_share: float = pool.GREEDY_SHARE,
                   eps_start: float = None, bomb_cost: float = 0.0,
                   mc_mix: float = MC_MIX, n_step: int = N_STEP,
                   tgt_sync: int = TGT_SYNC_GAMES, opp_kind: str = "greedy",
                   algo: str = "dmc", mate_mix: float = 0.0, beta_ent: float = BETA_ENT,
                   close_weight: float = 1.0, shaping: float = 0.0,
                   rule_eval_seeds: int = 1,
                   ppo_clip: float = PPO_CLIP, ppo_epochs: int = PPO_EPOCHS,
                   weight_sync_games: int = WEIGHT_SYNC_GAMES,
                   eval_rule_games: int = None,
                   _kill_worker_after: float = None):
    """多进程：`workers` 个进程打牌、本进程学习。

    设计见 多进程自对弈设计（已归档到 `master` 分支）。
    三条纪律（都在测试里钉着）：worker 死了**必须炸**；队列**有界**（背压）；
    ε 由本进程按**全局局数**算完广播下去（各 worker 自己算会「每个都以为自己是全部」）。

    `_kill_worker_after` **只给测试用**（到点杀一个 worker，验死亡检测）。
    """
    import multiprocessing as mp

    from guandan.rl import worker as worker_mod

    _check_boot_args(mc_mix, n_step, tgt_sync)
    ctx = mp.get_context("spawn")           # Windows 只有 spawn；入口必须是模块级函数
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if algo == "pg" and close_weight != 1.0:
        raise ValueError(
            "`--close-weight` 只接在 DMC 那条路上（PG 的损失是策略梯度，没有样本权重）"
            "—— 别让它静默失效（日志说改了、其实没改）")
    net = QNet().to(DEVICE)
    load_init(net, init)
    eps_start = resolve_eps_start(init, eps_start)   # 热启动 -> 低起点（EPS_START_WARM）

    # 目标网络：**deepcopy 而不是再 `QNet()`** —— 后者会多消耗一份 RNG，
    # 于是同一个 `--seed` 下两臂的网络初始化就不一样了（那是看不见的变量）。
    # β=1（默认）时根本不建：不多算一次前向，A/B 的对照臂跑的就是**老代码**
    # 外加一个恒假的 `if`。
    net_tgt = copy.deepcopy(net).requires_grad_(False) if mc_mix < 1.0 else None
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    send_q = ctx.Queue(maxsize=max(2, workers * 2))   # 有界 = 背压（不堆内存）
    ctrls = [ctx.Queue() for _ in range(workers)]

    # ---- 对手池（spec §3.1）：**只有 --pfsp 才开**
    # 默认关闭 = 老行为。这是 A/B 的前提：两臂只能差「有没有池子」这一个变量，
    # 否则赢了也不知道是谁的功劳（spec §5.7 的写法就是一臂 --pfsp、一臂纯贪心）。
    pool_sds, pool_order, next_mid = {}, [], 0
    wr = pool.WinRates()
    if pfsp:
        seeds = glob.glob(os.path.join(RUNS_DIR, "*", "best.pt"))
        pool_sds, pool_order = load_seed_pool(seeds, pool_size)
        if not pool_sds:
            # ⚠️ 空种子池**必须炸**：`pick_opponent` 对空池返回 ("greedy",)，
            # 于是 `--pfsp` 会**静默**退化成一个纯贪心臂（连老二分那 20% 随机都没了），
            # 而唯一那句话还把「空」误报成「塌陷」—— 一整夜的跑其实是个标错的对照臂。
            raise RuntimeError(
                f"种子池是空的（{os.path.join(RUNS_DIR, '*', 'best.pt')} 一个都没有）"
                f"—— `--pfsp` 会静默退化成纯贪心臂。先跑一次不带 --pfsp 的训练，"
                f"或把 --pool-size 调大（当前 {pool_size}）。")
        next_mid = max(pool_order) + 1

    def current_pfsp():
        """当前该按什么权重抽对手。**只在这里算** —— worker 不自己算（一处口径）。"""
        if not pfsp:
            return {}
        return pool.pfsp_weights({i: wr.rate(i) for i in pool_sds},
                                 {i: wr.games(i) for i in pool_sds})
    procs = [ctx.Process(target=worker_mod.run_worker, daemon=True,
                         args=(send_q, ctrls[k],
                               worker_mod.worker_cfg(seed + 1 + k, eps_start,
                                                     opp_mix, greedy_share,
                                                     batch_games,
                                                     learn_all_seats=learn_all_seats,
                                                     init=init, use_pool=pfsp,
                                                     pool_greedy_share=pool_greedy_share,
                                                     bomb_cost=bomb_cost,
                                                     opp_kind=opp_kind,
                                                     mate_mix=mate_mix,
                                                     shaping=shaping,
                                                     sample=(algo in ("pg", "ppo")))))
             for k in range(workers)]
    log(f"device={next(net.parameters()).device}  预算 {seconds:.0f}s  "
        f"worker {workers} 个（各自 CPU）+ 主进程学习  batch={batch_games} 局  "
        f"buffer={buffer_games:,} 局  评测每 {eval_every:,} 局  "
        f"ε 起点 {eps_start:.2f} 按 {eps_games:,} 局退火"
        + (f"（规则尺子 {rule_eval_seeds} 种子）" if rule_eval_seeds > 1 else "")
        + f"  对手混合 {opp_mix:.0%}"
        # ⚠️ **两个份额要分开展示**：池子开着时，「其中贪心 80%」说的是**老二分**
        # 那一路（跟池子无关），而池子的份额是 `pool_greedy_share`。
        # 上一轮就是这一行把人骗过去的：日志写着「其中贪心 80%」，我读着它
        # 写下了结论，却没看出池子其实只拿到 20%（全局 10%）。别再合并成一句。
        + (f"（其中池成员 {1 - pool_greedy_share:.0%}、贪心 {pool_greedy_share:.0%}）"
           f"  池子 {len(pool_sds)} 个种子成员" if pfsp
           else f"（其中{OPP_KIND_CN.get(opp_kind, opp_kind)} {greedy_share:.0%}、"
                f"随机 {1 - greedy_share:.0%}）"
                f"  **池子关**")
        + (f"  炸弹代价 λ={bomb_cost:g}" if bomb_cost else "")
        + (f"  **胶着加权 ×{close_weight:g}**" if close_weight != 1.0 else "")
        + (f"  **势函数 shaping β={shaping:g}**" if shaping else "")
        + (f"  **队友混合 {mate_mix:.0%}**（队友={OPP_KIND_CN.get(opp_kind, opp_kind)}）"
           if mate_mix > 0 else "")
        + ("" if algo == "pg" else _boot_log(mc_mix, n_step, tgt_sync))
        + (f"  热启动 {init}" if init else "")
        + _pg_log(algo, beta_ent, weight_sync_games)
        + (f"  **PPO clip={ppo_clip:g} × {ppo_epochs} 轮**" if algo == "ppo" else "")
        + f"\n权重 -> {out_dir}")

    t0 = time.perf_counter()
    games = steps = 0
    curve = []
    best = -1.0                 # `best.pt` 的挑选分数（默认 = vs 规则式，见 _maybe_eval）
    last_sync = 0
    last_tgt = 0                # 目标网络上次同步的局数
    base = _RunningMean()       # PG 的基线（DMC 用不到）
    qmax = 0                                # 队列积压峰值（背压有没有生效，看这个）
    t_kill = (time.perf_counter() + _kill_worker_after) if _kill_worker_after else None
    try:
        for p in procs:
            p.start()
        sd0 = {k: v.cpu() for k, v in net.state_dict().items()}
        for q in ctrls:                     # 开局先广播一次（worker 种子相同，但更稳）
            for mid in sorted(pool_sds):
                q.put(("member", (mid, pool_sds[mid])))
            q.put(("weights", (sd0, eps_start, current_pfsp())))
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
                games += 1
                # PFSP 的归因：这一局打的是谁、谁赢了。**不重放** ——
                # worker 顺手把胜负写在记录里了（实测重放一局 18.3 ms，太贵）
                if rec.opp and rec.opp[0] == "member" and rec.won is not None:
                    wr.record(rec.opp[1], rec.won)
                if algo != "pg":
                    buf.add(rec)        # PG 是 on-policy ⇒ **不写 buffer**
            if algo == "pg":
                # 数据就是刚收到的这一批（worker 发的是紧凑记录 ⇒ learner 侧重放）
                samples, rewards = [], []
                for rec in recs:
                    # ⚠️ `bomb_cost` 必须传 —— 漏了它，多进程 PG 的标签与日志头写的
                    # 「炸弹代价 λ=0.2」不符（评审 I1；单进程那条路是传的）。
                    pts, y, _b = replay.expand(rec, bomb_cost=bomb_cost, shaping=shaping)
                    samples += pts
                    rewards += y
                out = _pg_step(net, opt, samples, rewards, base, games, beta_ent)
                loss = out["loss"]
            else:
                # 采样 + 训一步（**与单进程那条路同一份实现**）
                loss = _learn_step(net, net_tgt, buf, rng, opt, games,
                                   batch_games=batch_games, bomb_cost=bomb_cost,
                                   mc_mix=mc_mix, n_step=n_step,
                                   close_weight=close_weight, shaping=shaping)
            steps += 1
            el = time.perf_counter() - t0
            if steps % 10 == 0:
                if algo == "pg":
                    log(f"  {el:6.0f}s  局数 {games:7d}  loss={loss:.3f}  "
                        f"H={out['entropy']:.3f}  {games / el:.1f} 局/秒")
                else:
                    log(f"  {el:6.0f}s  局数 {games:7d}  ε={eps_for(games, eps_games, start=eps_start):.2f}  "
                        f"loss={loss:.3f}  {games / el:.1f} 局/秒  "
                        f"buffer {len(buf):,}")
            if games - last_sync >= weight_sync_games:
                sd = {k: v.cpu() for k, v in net.state_dict().items()}
                w = current_pfsp()
                for q in ctrls:
                    q.put(("weights", (sd, eps_for(games, eps_games, start=eps_start), w)))
                last_sync = games
            # ⚠️ **不能放进上面那个 `if` 里**（评审 I1）：那样 `--tgt-sync` 会被
            # 静默夹到 `WEIGHT_SYNC_GAMES`（1000）那一拍上 —— 传 500 实际是 1000，
            # 而日志头照写用户给的值（参数说谎）。放在外面才真的按参数走。
            last_tgt = sync_target(net, net_tgt, games, last_tgt, tgt_sync)
            best = _maybe_eval(net, games, curve, best, eval_games,
                               eval_every, batch_games, out_dir, log,
                               snap_every=snap_every, pool_size=pool_size,
                               rule_games=eval_rule_games,
                               rule_seeds=rule_eval_seeds)
            # 新快照进池 + **增量**广播（只发这一个成员，7.8MB，每 snap_every 局一次）。
            # 用「文件在不在」判、不重算取模 —— 免得与 _maybe_eval 里的条件漂开。
            if pfsp and os.path.exists(pool.snapshot_path(out_dir, games)):
                sd = {k: v.cpu() for k, v in net.state_dict().items()}
                pool_sds[next_mid] = sd
                pool_order.append(next_mid)
                for q in ctrls:
                    q.put(("member", (next_mid, sd)))
                log(f"     >> 池子 +1 个成员（#{next_mid}）")
                next_mid += 1
                while len(pool_order) > pool_size:
                    # 挤掉最老的；worker 侧靠下一次 pfsp 广播里少了这个 id 自己删
                    pool_sds.pop(pool_order.pop(0), None)
                pool_report(wr, current_pfsp(), log)
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
    collapsed = pool_report(wr, current_pfsp(), log) if pfsp else False
    r = _finish(net, games, steps, t0, curve, best, eval_games,
                out_dir, log, rule_games=eval_rule_games)
    r["qmax"] = qmax
    r["pool_collapsed"] = collapsed
    return r


def main(argv=None) -> int:
    from guandan.console import utf8_stdout
    utf8_stdout()          # 不调这个，GBK 控制台下打不出 ✓ 会丢退出码（踩过）
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
    if "--bomb-cost" in argv:
        # λ：每用一手炸弹，从**那一步起**的标签就少这么多（spec §3.3）。
        # 默认 0 = 老行为；A/B 的处理臂用 0.2。
        kw["bomb_cost"] = float(argv[argv.index("--bomb-cost") + 1])
    if "--algo" in argv:
        # 训练算法：dmc（默认，老行为）或 pg（策略梯度 / REINFORCE）
        # ⚠️ **不认识的取值必须炸**：`--algo ppo` 会静默走 DMC，
        # 而 PG 那段日志对非 pg 返回空串 ⇒ 一次手滑的「PG 臂」其实是对照臂，
        # 而且**没有任何东西会响**（评审 M1）。
        kw["algo"] = _check_algo(argv[argv.index("--algo") + 1])
    if "--beta-ent" in argv:
        kw["beta_ent"] = float(argv[argv.index("--beta-ent") + 1])
    if "--weight-sync-games" in argv:
        n = int(argv[argv.index("--weight-sync-games") + 1])
        if n <= 0:
            # 0 会让 `games - last_sync >= 0` 恒真 ⇒ 每步广播 7.8 MB（评审 M7）
            raise ValueError(f"--weight-sync-games 必须为正：{n}")
        kw["weight_sync_games"] = n
    if "--opp-kind" in argv:
        # 固定对手用哪种「强」：greedy（默认，老行为）或 rule（规则式，像人）。
        kw["opp_kind"] = argv[argv.index("--opp-kind") + 1]
    if "--mc-mix" in argv:
        # β：MC 与自举的混合比。1.0 = 现在的 DMC（默认）；处理臂用 0.5。
        kw["mc_mix"] = float(argv[argv.index("--mc-mix") + 1])
    if "--n-step" in argv:
        # 自举往后看几步（默认 3，spec §8）。β=1 时它不起作用。
        kw["n_step"] = int(argv[argv.index("--n-step") + 1])
    if "--tgt-sync" in argv:
        # 目标网络每多少局同步一次（默认 1000，与权重广播同拍）。
        kw["tgt_sync"] = int(argv[argv.index("--tgt-sync") + 1])
    if "--eps-start" in argv:
        # ε 的**起点**。默认：热启动 -> 0.3（EPS_START_WARM），否则 1.0。
        # 想强制一个值（比如确认「起点低到底有多重要」）就传它。
        kw["eps_start"] = float(argv[argv.index("--eps-start") + 1])
    if "--learn-all-seats" in argv:
        # A/B 的对照臂：恢复改动前的行为（固定对手的着法也进训练目标）
        kw["learn_all_seats"] = True
    if "--init" in argv:
        kw["init"] = argv[argv.index("--init") + 1]
    if "--out-dir" in argv:
        # 两臂必须落在不同目录，否则后跑的会覆盖 best.pt / last.pt（结果就丢了）。
        # 面板只认 `models/best.pt`，所以**不会**再出现「静默换源」那类事故。
        kw["out_dir"] = argv[argv.index("--out-dir") + 1]
    if "--snap-every" in argv:
        # 1 小时的 A/B 默认只会长出 8 个成员；调小才有「有强度谱」的池子
        kw["snap_every"] = int(argv[argv.index("--snap-every") + 1])
    if "--pool-size" in argv:
        kw["pool_size"] = int(argv[argv.index("--pool-size") + 1])
    if "--ppo-clip" in argv:
        # PPO 的裁剪半径（默认 0.2）。只有 `--algo ppo` 用得到。
        kw["ppo_clip"] = float(argv[argv.index("--ppo-clip") + 1])
    if "--ppo-epochs" in argv:
        # 同一批数据重复更新几轮（默认 4）。`--algo pg` 等价于 1 轮 —— 那是它塌陷的原因。
        kw["ppo_epochs"] = int(argv[argv.index("--ppo-epochs") + 1])
    if "--rule-eval-seeds" in argv:
        # 点③：训练内**规则式那把尺子跑几个种子取均值**（默认 1 = 老行为）。
        # 调大能压掉 best.pt 的选点偏差（实测偏差量级 ~0.6pp），代价是每次评测慢一点。
        kw["rule_eval_seeds"] = int(argv[argv.index("--rule-eval-seeds") + 1])
    if "--eval-rule-games" in argv:
        # 规则式那把尺子跑多少局：留空 = 与 `--eval-games` 相同；0 = 不跑。
        kw["eval_rule_games"] = int(argv[argv.index("--eval-rule-games") + 1])
    if "--eval-games" in argv:
        # B4：训练内评测多少局（默认 200）。**`best.pt` 就是按这个读数挑的** ——
        # 200 局的 sd ≈ 2pp，与一代的进步（~3pp）同量级，等于在抽签。
        # 想让它挑得稳就调大（代价：每次评测多花时间，规则式 400 局约 20 秒）。
        kw["eval_games"] = int(argv[argv.index("--eval-games") + 1])
    if "--eval-every" in argv:
        # B4：每多少局评测一次（默认 10000）。**它同时决定快照/评测点的密度**：
        # 短臂要调小才看得到曲线。
        kw["eval_every"] = int(argv[argv.index("--eval-every") + 1])
    if "--pfsp" in argv:
        kw["pfsp"] = True
    if "--pool-greedy-share" in argv:
        # ⚠️ **池子自己的**贪心份额，与老二分的 `greedy_share` 是两回事。
        # 混用会让池子只拿到设计的 1/4 剂量（2026-09-27 评审抓到过）。
        kw["pool_greedy_share"] = float(argv[argv.index("--pool-greedy-share") + 1])
    if "--greedy-share" in argv:
        # A5：老二分里「贪心」的份额（默认 0.8 ⇒ 20% 是**随机**对手）。
        # 1.0 = 把随机那一路关掉 —— 那 20% 的局给的梯度基本是噪声，去掉是白送的。
        kw["greedy_share"] = float(argv[argv.index("--greedy-share") + 1])
    if "--shaping" in argv:
        # 信用分配 spec 的设计 1：势函数 shaping 的 β。默认 0.0 = 关。
        kw["shaping"] = float(argv[argv.index("--shaping") + 1])
    if "--close-weight" in argv:
        # 信用分配 spec 的设计 2：胶着局（|标签| < 3）样本的权重。默认 1.0 = 关。
        kw["close_weight"] = float(argv[argv.index("--close-weight") + 1])
    if "--mate-mix" in argv:
        # A3：**队友多样性** —— 以此概率把学习队的一个座位也换成 `--opp-kind` 的固定策略，
        # 只对另一个座位产标签。默认 0.0（关）= 与加开关之前逐位相同。
        kw["mate_mix"] = float(argv[argv.index("--mate-mix") + 1])
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
    if r.get("wr_rule") is not None:
        # 只是报出来：**退出码仍按 vs 贪心**（与老几次跑可比）。
        # 规则式才是主尺子，但它慢、且训练内只有 200 局，不适合当退出码。
        print(f"  参考：vs 规则式 {r['wr_rule']:.1%}（**主尺子**；要按它挑权重请用 tools.ruler）")
    if r.get("pool_collapsed"):
        # 池子塌了 = 失败，不是一行日志（spec §1.4 / 本仓库「失败必须响」）。
        # 原来只写进返回值，退出码照样 0、照样印「判据过了」——
        # 无人值守的跑会在一个退化成单一对手的池子上烧几个小时。
        print("⚠️ **池子塌了** —— 这一轮不算数（对手退化成同一个模型，训练会绕圈）")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
