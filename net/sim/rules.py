"""掼蛋**一手牌**的规则引擎 —— 发牌 / 进贡还贡 / 出牌(含接风) / 结算名次。

**一手牌一个 episode**（spec §13.1）：名次出来即终止。**不实现升级 / 过 A**，
级别（打几）由调用方给，本模块不推 —— 一手一个 episode 时升级没有消费者。

内部是**明牌**（四家手牌都知道，否则判不了输赢）。**喂给策略的状态由
`env.py` 负责裁剪**（spec §3「不许明牌泄漏」）。本模块不知道策略存在，也不该知道。

## 两条由 55 局实测定下来的轮转规则（别按直觉改）

**①「下家」是 `NEXT[s] = (s - 1) % 4`，出牌顺序是 `0 → 3 → 2 → 1 → 0`。**
报文的座位号不是按顺时针递增给的。2026-09-25 拿 55 局真实对局逐手回放才定下来：
按 `(s+1)%4` 写，**55 局一局都走不通**；按 `NEXT` 写，55/55 全过。
（`tools/decision_points.py` 一直没暴露这条，因为它只需要 `(table_seat + 2) % 4`
找队友，方向无关。）

**② 每出一手，`passed` 清空 —— 一手牌打出来等于重新开一轮。**
其余三家重新获得机会，包括之前已经「要不起」过的。不清空的话 55 局里只有 4 局能走通。

**接风**：一圈扫不到人接手时清桌；清桌后如果桌面主人**已经出完**，领出权给他的队友
（队友也出完了就顺着 `NEXT` 找下一个还有牌的）。**不要用服务器的 `NextTurnSeatID` 推**
—— 它会跳过出完的座位，实测按它判会误清 34 手、漏清 27 手。

**局终条件**（2026-09-25 在 55 局上核出，spec §13.5）：
    同一队包了前两名（双上） -> **只出完 2 家就终局**   25/25 局
    否则                     -> 出完 3 家才终局        30/30 局
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from net import cards
from net.sim import meld

SEATS = (0, 1, 2, 3)
TEAM = (0, 1, 0, 1)          # TEAM[s]：座位 0、2 一队，1、3 一队
PARTNER = (2, 3, 0, 1)       # 队友（在 NEXT 环上隔一个：0↔2、1↔3）

#: **下家**。出牌顺序 0 → 3 → 2 → 1（55 局实测，见模块 docstring ①）。
NEXT = (3, 0, 1, 2)


class IllegalPlay(Exception):
    """不合法的着法。**明着炸**，不静默吞（spec §6⑥）。"""


@dataclass
class Step:
    seat: int
    meld: Optional[meld.Meld]      # None = 要不起（过）
    left: int                      # 这一步之后该家还剩几张


@dataclass
class Hand:
    hands: List[Set[int]]
    level: Optional[int] = None
    turn: int = 0
    table: Optional[meld.Meld] = None
    table_seat: Optional[int] = None
    passed: Set[int] = field(default_factory=set)
    order: List[int] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    over: bool = False

    # ------------------------------------------------------------ 查询

    def actions(self, seat: int) -> list:
        """该座位的可行动作。**`None` 表示「过」。**

        - **领出**（桌上无牌）：只有着法，**不含 `None`** —— 领出必须出牌
        - **跟牌**：压得过的着法 **∪ {None}**。
          `None` 在跟牌时**永远**在集合里（能压也可以过）；
          压不过时集合是 `[None]` 而不是空 —— 这是 HANDOFF「Plan 2 开工前第 2 条」要钉的语义。
        """
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        moves = meld.melds_from(sorted(self.hands[seat]), self.level)
        if self.table is None:
            if not moves:
                raise IllegalPlay(
                    f"座位{seat} 手上还有 {len(self.hands[seat])} 张牌，却枚举不出任何着法 —— "
                    f"枚举漏了（spec §6③「领出时合法着法集合非空」）")
            return moves
        return [m for m in moves if meld.beats(m, self.table)] + [None]

    # ------------------------------------------------------------ 行动

    def play(self, seat: int, m: Optional[meld.Meld]) -> Step:
        if m is None:
            return self.pass_turn(seat)
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        # 从牌面重新判一次，**不用调用方给的 Meld** —— 牌型必须由牌面自己说了算。
        # `as_meld` 取最强解释：一手同花连续的牌同时是顺子与同花顺，游戏叫它同花顺
        # （`meld.strongest` 的 docstring 记了这条为什么只能有一份实现）。
        judged = meld.as_meld(list(m.cards), self.level)
        if judged is None:
            raise IllegalPlay(f"不是合法牌型：{cards.decode_all(list(m.cards))}")
        cs = set(judged.cards)
        if not cs <= self.hands[seat]:
            raise IllegalPlay("座位%d 打出了手上没有的牌 %s"
                              % (seat, cards.decode_all(sorted(cs - self.hands[seat]))))
        if self.table is not None and not meld.beats(judged, self.table):
            raise IllegalPlay("压不过桌面：%s vs %s"
                              % (meld.describe_meld(judged), meld.describe_meld(self.table)))

        self.hands[seat] -= cs
        if not self.hands[seat] and seat not in self.order:
            self.order.append(seat)
        # **一手打出来 = 重新开一轮**：其余三家重新获得机会（含之前「要不起」过的）。
        # 不清空的话 55 局真实对局里只有 4 局能走通 —— 见模块 docstring ②。
        self.passed = set()
        self.table = judged
        self.table_seat = seat
        st = Step(seat, judged, len(self.hands[seat]))
        self.steps.append(st)
        self._advance()
        return st

    def pass_turn(self, seat: int) -> Step:
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        if self.table is None:
            raise IllegalPlay(f"座位{seat} 是领出，不能过（桌上没牌，必须出牌）")
        self.passed.add(seat)
        st = Step(seat, None, len(self.hands[seat]))
        self.steps.append(st)
        self._advance()
        return st

    # ------------------------------------------------------------ 轮转

    def _advance(self) -> None:
        """一步之后定下一个该谁。清桌与接风都在这里发生。"""
        if self.is_over():
            self.over = True
            return
        nxt = None
        s = NEXT[self.table_seat]
        for _k in range(4):
            if self.hands[s] and s not in self.passed:
                nxt = s
                break
            s = NEXT[s]
        if nxt is not None and nxt != self.table_seat:
            self.turn = nxt
            return
        # 一圈扫不到人接手（或扫回主人自己）-> 清桌、重新领出
        self.table = None
        self.passed = set()
        if self.hands[self.table_seat]:
            self.turn = self.table_seat                       # 他自己重新领出
        else:
            self.turn = self._lead_after(self.table_seat)     # 接风
        self.table_seat = None

    def _lead_after(self, seat: int) -> int:
        """`seat` 出完了，领出权给谁：**队友优先（接风）**，队友也出完了就顺着 `NEXT` 找。"""
        p = PARTNER[seat]
        if self.hands[p]:
            return p
        s = NEXT[seat]
        for _k in range(4):
            if self.hands[s]:
                return s
            s = NEXT[s]
        return -1   # 全出完了；is_over() 会先拦下，走到这里说明局终判错了

    def is_over(self) -> bool:
        """**局终条件（spec §13.5）**：同一队包了前两名立即终局，否则等 3 家出完。"""
        if len(self.order) == 2 and TEAM[self.order[0]] == TEAM[self.order[1]]:
            return True
        return len(self.order) == 3

    # ------------------------------------------------------------ 结算

    def ranks(self) -> List[int]:
        """`ranks[seat] = 1..4`。出完的按出完顺序 1..k；没出完的按**剩牌少者排前**，
        同数按座位小者排前（只为可复现）。

        双上局里败方那两个人的 3/4 名是**服务器说了算的**：实测 25 局里 23 局
        「剩牌少者第 3」、2 局相反（spec §13.5）。**这两种定法对 reward 无影响**
        —— 双上时败方谁 3 谁 4 不改变形态，所以不为此加规则。
        """
        if not self.is_over():
            raise IllegalPlay("局还没终，名次还没定 —— 不要中途算名次")
        rest = [s for s in SEATS if s not in self.order]
        rest.sort(key=lambda s: (len(self.hands[s]), s))
        ranks = [0] * 4
        for i, s in enumerate(list(self.order) + rest, 1):
            ranks[s] = i
        return ranks


# ---------------------------------------------------------------- 结算与奖励

#: 赢家那一队**较差的名次** -> 升级点。双上 = 前两名都被包 = 较差名次是 2。
#:
#: ⚠️ 「双上 -> 3」只在数据上对了 13/24 局，另 11 局实测是 **4**（spec §13.4）。
#:    同一 `Rank`、同一战前 `trump` 都能出 3 和 4，说明还依赖升级/过A 的字段 ——
#:    按 §13.1 不在 Plan 2 内。这里取 3（用户口述 + spec §5.4），
#:    **由 `tools/accept_sim.py` 把 3/4 的分布单独报出来**，不当成验收失败。
POINTS_BY_WORST_RANK = {2: 3, 3: 2, 4: 1}

#: reward 的倍数开关。**默认 1.0，即不带倍数。**
#:
#: spec §5.4 说「倍数必须带上 —— 打炸弹会抬倍数」，但 54 局实测**不支持这个前提**：
#: `TotalBombRatio` 在 49/54 局是 1（其中 36 局手上有 5~16 个炸弹/同花顺），
#: 真正的倍数是 `FinalDoubleRatio ∈ {1, 1.5, 2, 2.5}`，那是**发牌前选的加倍**，
#: 一手之内不变 ⇒ 对最优策略零影响（只有一手内变化的倍数才会改变打法）。
#: 详见 spec §13.4。Plan 3 想按 `FinalDoubleRatio` 分布采样时改这一个值即可。
MULTIPLIER = 1.0


def winner_team(ranks) -> int:
    """赢家队：两名队员里**较好的名次**更靠前的那一队。"""
    a = min(ranks[0], ranks[2])
    b = min(ranks[1], ranks[3])
    if a == b:
        raise IllegalPlay(f"两队最好名次相同（ranks={list(ranks)}），不可能是合法结算")
    return 0 if a < b else 1


def points(ranks) -> int:
    """赢家这一手的升级点：双上 3 / 1、3 名 2 / 1、4 名 1。"""
    t = winner_team(ranks)
    worst = max(ranks[0], ranks[2]) if t == 0 else max(ranks[1], ranks[3])
    return POINTS_BY_WORST_RANK[worst]


def reward(ranks, seat: int, multiplier: float = None) -> float:
    """**从 `seat` 视角**的零点五奖励 —— 四个座位共享一套策略，必须零和。

    ⚠️ 视角是**出牌人**，不是「我方固定座位」。喂给网络的状态必须先相对化
    （自己 / 下家 / 对家 / 上家），否则这一条会被悄悄用错。
    """
    m = MULTIPLIER if multiplier is None else multiplier
    p = points(ranks) * m
    return p if TEAM[seat] == winner_team(ranks) else -p


# ---------------------------------------------------------------- 牌堆与发牌

#: 完整牌堆：两副牌 108 张的牌 ID。**顺序固定**（`range` 升序），
#: 这样给定 seed 的发牌结果永远一样 —— 训练要可复现。
FULL_DECK = tuple(c for c in range(0, 334) if cards.is_card(c))

_DEAL_EACH = 27                     # 两副牌 108 / 4 家


def _check_level(level) -> None:
    """**级别 14 不许顺着接口流进来**（spec Global Constraints）。

    日志里 A 有时写 14。`cards.parts(14)` 会把 14 当成**小王** —— 于是级牌判定、
    逢人配判定全错，而且一声不响。归一只有 `meld.norm_level` 一处，接口上拦住
    比在内部到处归一安全。
    """
    if level is None:
        return
    if not isinstance(level, int) or not 1 <= level <= 13:
        raise ValueError(
            f"级别必须在 1..13（A=1）。收到 {level!r} —— "
            f"14 是日志里 A 的另一种写法，请先过 meld.norm_level()")


def deal(rng, level=None, first=None) -> "Hand":
    """洗牌发牌，每家 27 张。`first` 不给就随机定领出者。"""
    _check_level(level)
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    hands = [set(deck[i * _DEAL_EACH:(i + 1) * _DEAL_EACH]) for i in SEATS]
    return Hand(hands=hands, level=level,
                turn=rng.randrange(4) if first is None else first)


def new_hand(rng, level=None, hands=None, first=None) -> "Hand":
    """建一手牌。**这是唯一的入口** —— `Hand(...)` 直接构造只允许出现在测试里。

    - `hands` 给定：直接用（**不洗牌**），供回放真实对局
    - `hands=None`：洗牌发牌

    **进贡不走这里。** 进贡只有 `apply_tribute(hand, prev_ranks)` 一个入口
    （见本模块的进贡一节）—— 曾经这里也想收一个 `prev_ranks`，那是两条路做
    同一件事，本仓库为「副本会漂」吃过亏（`melds_from` 的 docstring 记过一条）。
    """
    _check_level(level)
    if hands is None:
        h = deal(rng, level=level, first=first)
    else:
        h = Hand(hands=[set(x) for x in hands], level=level,
                 turn=rng.randrange(4) if first is None else first)
    return h
