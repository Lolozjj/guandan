"""Replay buffer —— **存紧凑记录，采样时重放**。

## 为什么不是「存编码好的张量」

spec §5.3 说 buffer 存「最近 50 万局」。按**存张量**算一下就知道那句话做不到：

    每个决策点 state(700) + action(143) + history(15×147) = 3048 个 float32 ≈ 11.9 KB
    每局约 132 个决策点 -> **每局 1.53 MB**
    10,000 局 ->  14.9 GB      100,000 局 -> 149 GB      500,000 局 -> **747 GB**

所以这里换一种存法：**只存「这一局是怎么打出来的」**——发牌（108 个牌 ID）、
级别、谁先出、以及每一步选中的候选下标。采样时把这些**重放**一遍，
再把张量算出来。

    每局约 1.9 KB      10,000 局 -> 18 MB      100,000 局 -> 185 MB      500,000 局 -> 923 MB

**代价是 CPU 不是内存**：重放一局要重新枚举着法，实测约 40 ms。而网络很小
（LSTM 128 + 6×512 MLP），训练那一步本来就是毫秒级 —— 所以拿 CPU 换内存是划算的，
而且它把 spec §5.3 那个数字从「做不到」变成「做得到」。

## 重放为什么是可靠的

同一副牌 + 同一个级别 + 同样的候选下标，`env` 的行为是**确定性**的
（`new_hand(hands=...)` 不洗牌、枚举不依赖随机）。所以重放一定复现原局面 ——
`tests/test_train_replay.py` 里有一条测试逐字段比对「现场产出」与「重放产出」。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from guandan.sim import env, rules


@dataclass(frozen=True)
class GameRecord:
    """一局的紧凑记录 —— 足以重放出全部决策点。

    `actions` 是每一步在 `env.legal()` 候选里的**下标**（含「过」那条），
    不是牌张 —— 重放时会重新枚举，下标对得上就行。

    `learn`：**哪几个座位是学习的**。`None` = 四家都学（纯自对弈 / 老记录）。
    混入固定对手时，对手那一队不该进训练目标（记进去等于拿它当老师），
    而重放侧**没有别的办法**知道这件事 —— 所以它必须跟记录一起走。
    """
    level: int                      # 本局的级别（打几）
    first: int                      # 本局谁先领出（绝对座位号）
    hands: tuple                    # 4 个 tuple（排序后的牌 ID）
    actions: tuple                  # 每一步选中的候选下标
    learn: tuple = None             # 学习座位；None = 四家都学
    #: 这一局的固定对手：`None` = 纯自对弈。
    #: ⚠️ **第二个元素口径不统一**：对 `("member", mid)` 是**成员 id**，
    #: 对 `("greedy", 队号)` / `("random", 队号)` 是**队号** —— 判类型先看 `[0]`。
    #: PFSP 靠它归因胜负（胜者不用记：`expand` 的 `y` 符号就是哪队赢）。
    opp: tuple = None
    #: **固定对手那一队输了没有**（`None` = 纯自对弈，无从谈起）。
    #: PFSP 靠它记分。⚠️ **故意存下来，不用 `expand` 反推**：实测重放一局 **18.3 ms**
    #: （2026-09-27），而学习进程每秒要处理几十局池对局 —— 光为了读 `y` 的符号就多花
    #: 几十个百分点。一个 bool 换掉这件事，值。
    won: bool = None

    @staticmethod
    def of(e: "env.GuandanEnv", actions, hands0, learn=None, opp=None,
           won=None) -> "GameRecord":
        """`hands0` 必须是**发牌时**的四家手牌。

        ⚠️ **不能在局末从 `e.hand.hands` 里取** —— 那时手里只剩「没出完的那几家
        剩下的牌」（`test_record_is_small` 抓到了这个：记录出来四家有三家是空的）。
        调用方要么在 `reset()` 之后立刻快照，要么用 `play_and_record`（它替你做了）。
        """
        return GameRecord(level=e.hand.level, first=e.hand.steps[0].seat,
                          hands=tuple(tuple(sorted(h)) for h in hands0),
                          actions=tuple(actions),
                          learn=tuple(learn) if learn is not None else None,
                          opp=tuple(opp) if opp is not None else None,
                          won=won)


def _phi(base, played, seat) -> float:
    """势函数（**出牌人视角**）：`(对方剩牌 − 我方剩牌) / 本局发出去的牌数` ∈ [−1, 1]。

    只用"发牌数 − 已出数"就能算 —— **不需要给记录新增字段**（紧凑记录是要跨进程传的）。

    ⚠️ **必须归一化**：标签尺度是 ±3，而没归一化的 `Φ` 能到 ±54（张数）
    ⇒ `β=0.3` 会把标签推到 ±16，直接撞上 `check_q_scale` 那道守门。
    除以"本局发出去的牌数"（= 108）之后，`Φ` 与标签同量级。
    """
    me = rules.TEAM[seat]
    mine = sum(base[s] - played[s] for s in rules.SEATS if rules.TEAM[s] == me)
    theirs = sum(base[s] - played[s] for s in rules.SEATS if rules.TEAM[s] != me)
    total = sum(base.values()) or 1
    return (theirs - mine) / total


def mc_targets(seq, ranks, learn=None, bomb_cost: float = 0.0, hands0=None,
               shaping: float = 0.0) -> list:
    """被保留的那些决策点的 DMC 标签 —— **现场与重放共用的唯一实现**。

    `y_t = R − λ · B_t`，其中 `B_t` = **从第 t 步起、该步座位自己**用掉的炸弹数
    （**含第 t 步本身** —— reward-to-go 的口径：逐步代价 `−λ·[a 是炸弹]` 求和，
    标签 = 未来所有代价之和 + 终局回报）。

    `seq` 必须是**一局全部步骤**：`B_t` 要沿着一局往后数，少一步就数错。
    过滤（`learn`）**在函数里面做** —— 让两个调用方各写一份过滤，就是
    「副本会漂」的入口（本仓库为此反复吃过亏）。

    ⚠️ **只数该座位自己的炸弹**：奖励本来就是「出牌人视角」的零和量，
    代价用同一视角才自洽。`is_bomb` 也把**同花顺**算作炸（掼蛋里它本来就是）。

    **`shaping`（信用分配 spec 的设计 1，默认 0.0 = 关）**：
    `y_t += β · (Φ_T − Φ_t)`，`Φ` 见 `_phi`。要 `hands0` 一起给才生效。

    ⚠️ **先看清它的机制，别指望错的东西**：势函数形式（Ng et al. 1999）在 γ=1 下
    **望远镜化** ⇒ 整局的 shaping 之和只差一个与动作无关的常数 ⇒
    **它对"同一个局面里哪个动作更好"毫无影响**（推理的 argmax 口径不变）。
    它改的是**回归目标本身**：把 ±1/±2/±3 那种粗糙离散的标签，变成"终局分 + 本局到现在的进度"，
    跨状态的方差更小、**更好拟合**。
    ⇒ 所以判据第一位是**动作边际**（若边际不变或变窄，这条就没用），胜率放第二位。
    """
    if bomb_cost < 0:
        raise ValueError(f"bomb_cost 不能为负：{bomb_cost}")
    if shaping and hands0 is None:
        raise ValueError("给了 shaping 就必须给 hands0 —— 否则势函数没法算，会**静默**变成没开")
    keep = set(learn) if learn else None
    tail, suffix = {}, [0] * len(seq)
    for i in range(len(seq) - 1, -1, -1):     # 倒着扫一遍就得到全部后缀计数
        seat, m = seq[i]
        tail[seat] = tail.get(seat, 0) + (1 if (m is not None and m.is_bomb) else 0)
        suffix[i] = tail[seat]

    if shaping:
        played = {s: 0 for s in rules.SEATS}
        base = {s: len(hands0[s]) for s in rules.SEATS}
        phi_t = []
        for seat, m in seq:
            phi_t.append(_phi(base, played, seat))        # **决策前**的势（与状态同一时刻）
            played[seat] += 0 if m is None else len(m.cards)
        phi_end = [_phi(base, played, seat) for seat, _m in seq]   # 终局的势（各按自己的视角）
    else:
        phi_t = phi_end = None

    out = []
    for i, (seat, _m) in enumerate(seq):
        if keep is not None and seat not in keep:
            continue
        y = rules.reward(ranks, seat) - bomb_cost * suffix[i]
        if shaping:
            y += shaping * (phi_end[i] - phi_t[i])
        out.append(y)
    return out


def blend(y_mc, boot, beta: float = 1.0) -> list:
    """把 MC 标签与自举值按 β 混合 —— **自举进入标签的唯一一处**（spec §3.1）。

        y_t = (1 - beta) * V(s_{t+n})  +  beta * y_mc

    - `beta = 1.0` 时**原样返回 `y_mc`**（默认 = 现在的 DMC，逐点相等）。
      这里刻意提前返回、而不是算 `(1-β)*v + β*y` —— 结构上相等，而且
      顺带不会去碰 `boot`（那条路上 `boot` 可能是空表或全是 None）。
    - `boot` 里某一项是 `None`（越过终局 / 本轮不算自举）→ 那一项整项退回 `y_mc`。
      **不许拿 0 或上一项顶上** —— 那是凭空造一个未来。
    - `boot` 整条为 `None` 表示「这一批根本没算自举」（`n=0`），也退回 `y_mc`。
    - 长度必须对齐：错开一格就是「拿别人的未来当自己的标签」，
      而这种错在 loss 曲线上完全看不出来（本仓库纪律：失败必须响）。

    ⚠️ **不 import torch、也不碰张量** —— 与 `mc_targets` 一样是纯标量运算，
    所以 `beta = 1` 的逐点相等是**算术上的**相等，不依赖浮点运气。
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"β 必须在 [0, 1]：{beta}")
    if beta >= 1.0 or boot is None:
        return list(y_mc)
    if len(boot) != len(y_mc):
        raise ValueError(f"自举值与标签长度不一致：{len(boot)} vs {len(y_mc)}")
    return [y if v is None else (1.0 - beta) * v + beta * y
            for y, v in zip(y_mc, boot)]


def _boot_source_ok(learn, tgt_seat: int, src_seat: int) -> bool:
    """这个自举源能不能用 —— **视角**与「不拿固定对手当老师」两件事一起判。

    ⚠️ **视角（2026-09-28 评审抓到的 Critical）**：`Q(s, a)` 学的是
    「**出手人那一队**」的收益（`encode_state` 一切以出手人为原点、
    标签又是 `rules.reward(ranks, seat)`），而 `V(s_{t+n}) = max_a Q(s_{t+n}, a)`
    取的是 **`s_{t+n}` 处出手人**的视角。出手顺序是 `0→3→2→1`（`rules.NEXT`），
    所以 **n 是奇数时那个位置在对家** —— 直接用会把符号弄反，
    `(1-β)·V + β·R` 两项互相抵消、标签被往 0 拉。

    实测（`1407`，`greedy_policy` 打出来的局）：

        n      自举源与标签同队     corr(V, 标签)    β=0.5 后保留的方差
        1          5.4%              -0.353              0.27
        2         89.2%              +0.271              0.51
        3         16.0%              -0.145              0.35
        4         88.3%              +0.273              0.52

    ⇒ **n 要取偶数**（默认已改成 2）。奇数不算错、只是命中率很低。

    异队的点**整项退回 MC**，不做取负：取负等于把「对家按最大打」的价值当成我们的，
    而行为策略并不是最大 —— 那是另一种偏差，本轮不引入。

    `learn` 那一半是老纪律：固定对手的着法不进训练目标（记进去等于拿它当老师），
    它的状态自然也不该当自举源。`learn=None`（四家都学）时不设这道闸。
    """
    if learn is not None and src_seat not in learn:
        return False
    return rules.TEAM[src_seat] == rules.TEAM[tgt_seat]


def expand(rec: GameRecord, bomb_cost: float = 0.0, n: int = 0,
           shaping: float = 0.0):
    """把记录重放成 `(决策点, MC 标签, 自举源)`。

    决策点是 `(obs, acts, 选中下标, 出牌人, hist)` —— 与 `env.rollout` 同形状，
    训练循环因此**不需要区分**「刚打的」和「从 buffer 里取的」。

    `rec.learn` 里的座位才产出决策点（`None` = 四家都产出，老行为）。
    ⚠️ **局面必须每一步都往前走** —— 过滤只发生在 `points.append` 那一行。
    提前 `continue` 会让重放错位（这是这个函数最容易被写错的地方）。

    `n > 0` 时额外产出 `boot[i]`：第 i 个决策点**往后数 n 步**那个局面的
    `(obs, acts, hist, 出手人)` —— 自举要的 `s_{t+n}`（spec §3.2）。
    **第 4 位是必需的**：`V` 是「出手人那一队」的值，不带上出手人就没法判视角。
    **这三种点整项退回 MC（`None`）**：越过终局、自举源在对家、自举源是固定对手
    （见 `_boot_source_ok`）。`n = 0` 时 `boot` 是空表，与老行为逐点相等。

    ⚠️ **`boot` 与 `points` 必须等长同序** —— 错开一格就是拿别人的未来当自己的标签。
    对齐靠一个**定长环**（`deque(maxlen=n+1)`）：走到第 t 步时环首正好是
    第 `t-n` 步，此刻的局面就是它要的自举源。历史**必须现在取**
    （`encode_history`）—— 整局打完再取会把后面的牌塞进去（未来信息泄漏）。
    """
    if n < 0:
        raise ValueError(f"n 不能为负：{n}")
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    seq, points, boot = [], [], []
    ring = deque(maxlen=n + 1) if n else None     # 定长环：环首 = 第 t-n 步
    obs = e.observe()
    for i in rec.actions:
        acts = e.legal()
        seat = e.hand.turn
        seq.append((seat, acts[i]))       # **全量步骤**：B_t 要沿着一局往后数
        tag = None                        # 这一步在 points 里的下标（没保留就是 None）
        if learn is None or seat in learn:
            tag = len(points)
            # 历史必须**在这一步当时**取 —— 整局打完再取会把后面的牌塞进历史
            # （那是另一种泄漏：未来信息）。selfplay 那边也是这么取的。
            points.append((obs, acts, i, seat, env.encode_history(e.hand, seat)))
            if ring is not None:
                boot.append(None)         # 自举源在 t+n 步，那时才补得上
                # （`n = 0` 时 boot 保持**空表** —— 「这一批根本没算自举」比
                #   一列 None 更能让调用方一眼看出区别）
        if ring is not None:
            ring.append((tag, seat))      # 环里要带上「那一步谁在出手」——判视角要用
            # 环满（t >= n）时环首才是「第 t-n 步」；不满时那些点的源不存在
            if len(ring) == n + 1:
                j, tgt_seat = ring[0]
                if j is not None and _boot_source_ok(learn, tgt_seat, seat):
                    boot[j] = (obs, acts, env.encode_history(e.hand, seat), seat)
        obs, _r, _done, _info = e.step(i)
    # 标签**只有一个产地**（`mc_targets`）—— 过滤也在它里面做
    return (points, mc_targets(seq, e.ranks, learn=rec.learn, bomb_cost=bomb_cost,
                             hands0=rec.hands, shaping=shaping),
            boot)


def play_capturing(policy, rng, level=None, capture=False, bomb_cost: float = 0.0, shaping: float = 0.0):
    """打一局。返回 `(记录, 决策点, 终局 reward)`；`capture=False` 时决策点是空表。

    **为什么要 `capture`**：刚打完的一批局，每一步的决策点在生成时**本来就算过**了
    （`actions()` + `encode_history()`），把它们直接留下来训练，就不必再从记录重放一遍
    —— 重放一局约 27 ms，白花。`capture=True` 会多占内存（每局约 1.2 MB 的历史数组），
    所以只在「这一批马上要拿来训练」时开。从 buffer 里取出来的老局没有这个待遇，
    只能重放（那就是 `expand`）。
    """
    e = env.GuandanEnv(seed=rng.randrange(1 << 30))
    e.reset(level=level)
    hands0 = [set(e.hand.hands[s]) for s in rules.SEATS]   # 发牌快照（见 GameRecord.of）
    actions, points, seq = [], [], []
    obs = e.observe()
    while not e.done:
        acts = e.legal()
        seat = e.hand.turn
        hist = env.encode_history(e.hand, seat)
        i = policy(obs, acts, hist)
        seq.append((seat, acts[i]))
        if capture:
            points.append((obs, acts, i, seat, hist))
        actions.append(i)
        obs, _r, _done, _info = e.step(i)
    rec = GameRecord.of(e, actions, hands0)
    if not capture:
        return rec, [], []
    # 标签走**同一个产地**（`mc_targets`）—— 三条路都收在一处，改一处就是改三处
    return rec, points, mc_targets(seq, e.ranks, bomb_cost=bomb_cost,
                                     hands0=hands0, shaping=shaping)


def play_and_record(policy, rng, level=None, shaping: float = 0.0):
    """只记紧凑记录（不抓决策点）。返回 `(GameRecord, 决策点数)`。"""
    rec, _pts, _y = play_capturing(policy, rng, level=level, capture=False,
                                  shaping=shaping)
    return rec, len(rec.actions)


class ReplayBuffer:
    """按**局数**限容的环形缓冲（内存估算见模块 docstring）。"""

    def __init__(self, capacity_games: int = 50_000):
        if capacity_games <= 0:
            raise ValueError("capacity_games 必须为正")
        self.capacity_games = capacity_games
        self._games = deque(maxlen=capacity_games)
        self.added = 0              # 累计进过多少局（看进度用，不是当前存量）

    def add(self, rec: GameRecord) -> None:
        self._games.append(rec)
        self.added += 1

    def sample(self, n: int, rng) -> list:
        """取 `n` 局。**允许重复**（有放回）—— DouZero 式 DMC 就是这么采的。"""
        if not self._games:
            raise ValueError("buffer 是空的，先 add 几局")
        return [self._games[rng.randrange(len(self._games))] for _ in range(n)]

    def decision_points(self) -> int:
        """当前存量里大约有多少个决策点（内存/吞吐的粗略尺子）。"""
        return sum(len(r.actions) for r in self._games)

    def __len__(self) -> int:
        return len(self._games)

    def __bool__(self) -> bool:
        return bool(self._games)
