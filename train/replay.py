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

from net.sim import env, rules


@dataclass(frozen=True)
class GameRecord:
    """一局的紧凑记录 —— 足以重放出全部决策点。

    `actions` 是每一步在 `env.legal()` 候选里的**下标**（含「过」那条），
    不是牌张 —— 重放时会重新枚举，下标对得上就行。

    `learn`：**哪几个座位是学习的**。`None` = 四家都学（纯自对弈 / 老记录）。
    混入固定对手时，对手那一队不该进训练目标（记进去等于拿它当老师），
    而重放侧**没有别的办法**知道这件事 —— 所以它必须跟记录一起走。
    """
    level: int
    first: int
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


def mc_targets(seq, ranks, learn=None, bomb_cost: float = 0.0) -> list:
    """被保留的那些决策点的 DMC 标签 —— **现场与重放共用的唯一实现**。

    `y_t = R − λ · B_t`，其中 `B_t` = **从第 t 步起、该步座位自己**用掉的炸弹数
    （**含第 t 步本身** —— reward-to-go 的口径：逐步代价 `−λ·[a 是炸弹]` 求和，
    标签 = 未来所有代价之和 + 终局回报）。

    `seq` 必须是**一局全部步骤**：`B_t` 要沿着一局往后数，少一步就数错。
    过滤（`learn`）**在函数里面做** —— 让两个调用方各写一份过滤，就是
    「副本会漂」的入口（本仓库为此反复吃过亏）。

    ⚠️ **只数该座位自己的炸弹**：奖励本来就是「出牌人视角」的零和量，
    代价用同一视角才自洽。`is_bomb` 也把**同花顺**算作炸（掼蛋里它本来就是）。
    """
    if bomb_cost < 0:
        raise ValueError(f"bomb_cost 不能为负：{bomb_cost}")
    keep = set(learn) if learn else None
    tail, suffix = {}, [0] * len(seq)
    for i in range(len(seq) - 1, -1, -1):     # 倒着扫一遍就得到全部后缀计数
        seat, m = seq[i]
        tail[seat] = tail.get(seat, 0) + (1 if (m is not None and m.is_bomb) else 0)
        suffix[i] = tail[seat]
    return [rules.reward(ranks, seat) - bomb_cost * suffix[i]
            for i, (seat, _m) in enumerate(seq)
            if keep is None or seat in keep]


def expand(rec: GameRecord):
    """把记录重放成 `(决策点, 终局 reward)`。

    决策点是 `(obs, acts, 选中下标, 出牌人, hist)` —— 与 `env.rollout` 同形状，
    训练循环因此**不需要区分**「刚打的」和「从 buffer 里取的」。

    `rec.learn` 里的座位才产出决策点（`None` = 四家都产出，老行为）。
    ⚠️ **局面必须每一步都往前走** —— 过滤只发生在 `points.append` 那一行。
    提前 `continue` 会让重放错位（这是这个函数最容易被写错的地方）。
    """
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    points = []
    obs = e.observe()
    for i in rec.actions:
        acts = e.legal()
        seat = e.hand.turn
        # 历史必须**在这一步当时**取 —— 整局打完再取会把后面的牌塞进历史
        # （那是另一种泄漏：未来信息）。selfplay 那边也是这么取的。
        if learn is None or seat in learn:
            points.append((obs, acts, i, seat, env.encode_history(e.hand, seat)))
        obs, _r, _done, _info = e.step(i)
    ranks = e.ranks
    y = [rules.reward(ranks, seat) for (_o, _a, _i, seat, _h) in points]
    return points, y


def play_capturing(policy, rng, level=None, capture=False):
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
    actions, points = [], []
    obs = e.observe()
    while not e.done:
        acts = e.legal()
        seat = e.hand.turn
        hist = env.encode_history(e.hand, seat)
        i = policy(obs, acts, hist)
        if capture:
            points.append((obs, acts, i, seat, hist))
        actions.append(i)
        obs, _r, _done, _info = e.step(i)
    rec = GameRecord.of(e, actions, hands0)
    if not capture:
        return rec, [], []
    ranks = e.ranks
    return rec, points, [rules.reward(ranks, s) for (_o, _a, _i, s, _h) in points]


def play_and_record(policy, rng, level=None):
    """只记紧凑记录（不抓决策点）。返回 `(GameRecord, 决策点数)`。"""
    rec, _pts, _y = play_capturing(policy, rng, level=level, capture=False)
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
