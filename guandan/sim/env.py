"""掼蛋自对弈环境（spec §4）。

**这个模块唯一的、不可动摇的纪律：喂给策略的状态里不许有对手手牌。**

做法不是靠注释提醒，是靠**类型**：`Observation` 这个 dataclass 里根本没有
「四家手牌」这个字段 —— 它只有可观测的东西（我的手牌、各家出过的牌、各家剩几张、
桌面、轮到谁、级别）。编码器 `encode_state` 只吃 `Observation`，
所以**它在结构上就无法泄漏**。`tests/test_env_leak.py` 再用自动化测试钉死一遍。

泄漏的后果（spec §3）：会训出靠偷看才成立的打法 —— 训练分数漂亮、真机全废，
**而且静默失效**（同本项目「合成 val 骗过一次」的教训）。

座位一律**相对化**。索引按**出牌顺序**排：0 = 自己、
1 = 先于我的那家、2 = 对家、3 = 我的下家（`rules.NEXT[自己]`）。
⚠️ 出牌顺序是 `0 → 3 → 2 → 1`（见 `rules.py`），所以「下家」落在索引 3 ——
**别再按「1 就是下家」写面板**（这里曾经把 1/3 写反过）。
四个座位共享一套权重，所以相对化不是可选项。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from guandan.capture import cards
from guandan.sim import features, meld, rules

#: spec §4.1 的状态维度，逐项对齐：
#:     我的手牌 108 + 已出过的牌 4×108 + 各家剩几张 4 + 桌面待压 108
#:     + 桌面牌型 10 + 桌面主点数 15 + 谁要不起 4 + 轮到谁 4 + 级别 15
#: **再加 A2 的显式特征**（`sim/features.py` 的 `EXTRA_DIM`，2026-09-30）——
#: 老权重的第一层可以按列零填充后热启动，见 `rl/net.py::load_state`。
STATE_DIM = 108 + 4 * 108 + 4 + 108 + 10 + 15 + 4 + 4 + 15 + features.EXTRA_DIM
assert STATE_DIM == 727

_OFF_HAND = 0
_OFF_PLAYED = _OFF_HAND + 108
_OFF_LEFT = _OFF_PLAYED + 4 * 108
_OFF_TABLE = _OFF_LEFT + 4
_OFF_KIND = _OFF_TABLE + 108
_OFF_RANK = _OFF_KIND + 10
_OFF_PASSED = _OFF_RANK + 15
_OFF_TURN = _OFF_PASSED + 4
_OFF_LEVEL = _OFF_TURN + 4
#: A2 的显式特征（`sim/features.py`）**接在最后** —— 老权重零填充热启动靠这个位置。
_OFF_EXTRA = _OFF_LEVEL + 15

#: 桌面/动作里的「主点数」编码：点数 1..13 -> 0..12；
#: **级牌 -> 13**（`meld.POINT_LEVEL`）；王 -> 14（`POINT_SMALL` / `POINT_BIG` 合到一格，
#: 因为王当桌面牌型时只有「一对王」这一种，大小王不会再区分）。
_RANK_SLOTS = 15


def _rank_slot(point: int) -> int:
    """主点数 -> 0..14 的槽位：普通点数 1..13 -> 0..12、**级牌 -> 13**、王 -> 14。

    ⚠️ 别写成 `point >= POINT_LEVEL` 一档到底 —— 那样**级牌会掉进王的槽位**
    （POINT_LEVEL=14 而王是 15/16），于是「桌面是级牌」与「桌面是一对王」
    在编码里分不开，而它们是两个完全不同的层级。
    """
    if point >= meld.POINT_SMALL:          # 15 / 16 = 王
        return _RANK_SLOTS - 1             # 14
    if point >= meld.POINT_LEVEL:          # 14 = 级牌
        return _RANK_SLOTS - 2             # 13
    return min(max(point, 1), _RANK_SLOTS - 2) - 1


@dataclass(frozen=True)
class Observation:
    """策略能看见的东西。

    **故意没有「四家手牌」这个字段。** 这不是省略，是设计 ——
    只要它不在这个类型里，编码器就没有办法把它编进去。
    """
    seat: int                       # 出牌人（相对化的原点，**绝对**座位号）
    hand: frozenset                 # 我的手牌
    played: Tuple[frozenset, ...]   # 4 家各自出过的牌（公开信息）
    left: Tuple[int, ...]           # 4 家各剩几张（公开信息）
    table: Tuple[int, ...]          # 桌面待压的牌；空元组 = 我领出
    table_kind: int                 # 桌面牌型（0 = 无）
    table_rank: int                 # 桌面主点数（`meld.point_value` 口径；-1 = 无）
    passed: Tuple[bool, ...]        # 本轮谁「要不起」（公开信息）
    turn: int                       # 轮到谁（绝对座位）
    level: int                      # 本局级别（打几）—— 编码进状态的那 15 维 one-hot


def _rel(seq, seat: int):
    """把按绝对座位排的序列转成相对座位。

    索引按**出牌顺序**排：0 = 自己、1 = 先于我的那家、2 = 对家、
    3 = 我的下家（`rules.NEXT[seat]`）。出牌顺序是 `0 → 3 → 2 → 1`，
    所以下家在索引 3 而不是 1。
    """
    return tuple(seq[(seat + i) % 4] for i in range(4))


def encode_action_now(m, level: int, hand) -> np.ndarray:
    """**当前动作的唯一入口**：`ACTION_BASE_DIM` + 动作侧后果特征 = `ACTION_DIM`。

    为什么要动作侧特征：A2 加的是状态侧（同一局面里对每个候选都一样）⇒ 帮不上 argmax；
    这三个量逐个候选不同，是"哪一手更好"真正缺的信息。

    ⚠️ **历史行不要用这个**（`encode_history` 用 `encode_action`）：
    历史里拿不到当时的手牌，两边宽度因此不同 —— 这是**故意**的，不是疏漏。

    ⚠️ `features.CONSEQUENCE_ENABLED=False` 时那 3 维**仍然在**（保持 `ACTION_DIM` 不变），
    只是恒为 0 ⇒ 信息层面等价于"没有这个特征"，可当对照臂用。
    """
    base = encode_action(m, level)
    if not features.CONSEQUENCE_ENABLED:
        return np.concatenate([base, np.zeros(features.CONSEQUENCE_DIM,
                                              dtype=np.float32)])
    return np.concatenate([
        base,
        np.asarray(features.consequence_features(hand, m, level), dtype=np.float32)])


def encode_state(obs: Observation) -> np.ndarray:
    """`Observation` -> `STATE_DIM` 维 float32（700 + A2 的显式特征）。**只吃 `Observation`**。"""
    if not isinstance(obs.level, int) or not 1 <= obs.level <= 13:
        raise ValueError(
            f"级别必须在 1..13（A=1）。收到 {obs.level!r} —— "
            f"14 是日志里 A 的另一种写法（先过 meld.norm_level()），"
            f"None 说明这一局还没定级。"
            f"**别让它静默写进预留槽位**：14 会落到第 14 格、None 会直接 TypeError。")
    v = np.zeros(STATE_DIM, dtype=np.float32)

    for c in obs.hand:
        v[_OFF_HAND + cards.slot(c)] = 1.0

    for i, cards_i in enumerate(_rel(obs.played, obs.seat)):
        for c in cards_i:
            v[_OFF_PLAYED + i * 108 + cards.slot(c)] = 1.0

    for i, n in enumerate(_rel(obs.left, obs.seat)):
        v[_OFF_LEFT + i] = n / 27.0            # 归一，别让 27 这种量级和 0/1 混在一起

    for c in obs.table:
        v[_OFF_TABLE + cards.slot(c)] = 1.0

    if obs.table_kind:
        v[_OFF_KIND + obs.table_kind - 1] = 1.0
    if obs.table_rank >= 0:
        v[_OFF_RANK + _rank_slot(obs.table_rank)] = 1.0

    for i, p in enumerate(_rel(obs.passed, obs.seat)):
        v[_OFF_PASSED + i] = 1.0 if p else 0.0

    v[_OFF_TURN + (obs.turn - obs.seat) % 4] = 1.0
    v[_OFF_LEVEL + obs.level] = 1.0            # 1..13 用第 1..13 格；0 与 14 恒为 0

    # A2：显式特征。**默认关**（`features.ENABLED`，理由见那个常量的注释）——
    # 关着的时候这一块恒为 0：架构不变、开销为零，老权重与 A2 权重都装得上。
    if features.ENABLED:
        v[_OFF_EXTRA:] = features.extra_features(obs)

    return v


def observe(hand: "rules.Hand", seat: int, played, table_meld: Optional["meld.Meld"]):
    """从**明牌**的 `Hand` 里切出 `seat` 视角的可观测状态。

    **这是明牌与策略之间唯一的窄口**：进来的是整个 `Hand`（四家都看得见），
    出去的 `Observation` 里只有公开信息 + 自己的手牌。

    `played` 由调用方维护（`{座位: 出过的牌}`），因为 `Hand` 只记 `steps`，
    不按座位分桶 —— 分桶是面板/记牌器的口径，不是规则的口径。
    """
    table = ()
    kind, rank = 0, -1
    if table_meld is not None:
        table = tuple(table_meld.cards)
        kind = table_meld.kind
        rank = table_meld.rank
    return Observation(
        seat=seat,
        hand=frozenset(hand.hands[seat]),
        played=tuple(frozenset(played.get(s, ())) for s in rules.SEATS),
        left=tuple(len(hand.hands[s]) for s in rules.SEATS),
        table=table, table_kind=kind, table_rank=rank,
        passed=tuple(s in hand.passed for s in rules.SEATS),
        turn=hand.turn,
        level=hand.level,
    )


# ---------------------------------------------------------------- 动作编码

#: spec §4.2：牌型 10 + 主点数 15 + 张数 9 + 逢人配 1 + 牌 108
#: **历史行**里的动作宽度（`encode_history`）：那里**拿不到当时的手牌**，
#: 所以不能带动作侧后果特征。历史行必须用这个宽度。
ACTION_BASE_DIM = 10 + 15 + 9 + 1 + 108
assert ACTION_BASE_DIM == 143

#: **当前动作**的宽度 = 基础 143 + `features.CONSEQUENCE_DIM`（动作侧后果特征）。
#: ⚠️ 与 `ACTION_BASE_DIM` 是两个宽度，**别混用**：混了就是"位置含义漂了"，
#: 而网络那边只会静默学歪。当前动作只有一个入口：`encode_action_now()`。
ACTION_DIM = ACTION_BASE_DIM + features.CONSEQUENCE_DIM
assert ACTION_DIM == 146

_A_OFF_KIND = 0
_A_OFF_RANK = 10
_A_OFF_SIZE = 25
_A_OFF_WILD = 34
_A_OFF_CARDS = 35

#: 张数那一格：1..8 张 -> 0..7；**≥9 张或天王炸 -> 8**。
#: spec §4.2 写的是「1~8 张 + 天王炸」，但掼蛋里还有 9 炸 / 10 炸（8 张同点 + 逢人配），
#: 9 格装不下。这里把它们并进第 8 格 —— **不丢信息**，因为牌 multi-hot(108)
#: 已经把这手牌是什么完全写清楚了，张数格只是给网络的一个提示。
_SIZE_COMPOSITE = 8


def encode_action(m, level: int) -> np.ndarray:
    """一个候选着法 -> **`ACTION_BASE_DIM`（143）** 维 float32。**`None`（过）编成全 0。**

    ⚠️ 这是**基础**编码：`encode_history`（历史行）用它 —— 那里**拿不到当时的手牌**，
    所以不能带动作侧后果特征。**当前动作请用 `encode_action_now()`（146 维）。**
    两个宽度混用就是"位置含义漂了"，而网络只会静默学歪。

    ⚠️ 主点数那一格的口径与 `Meld.rank` 一致，而 `rank` 在两种口径之间：
    序列类（顺子/连顺/钢板）存的是**自然值**（A 可作 1 或 14），
    其余存的是 `meld.point_value`（级牌 14、王 15/16）。
    天王炸的 `rank` 是 0。**所以这一格跨牌型不是单射** ——
    真正的判别力在牌 multi-hot 上，这一格是给网络的便捷特征。
    """
    v = np.zeros(ACTION_BASE_DIM, dtype=np.float32)
    if m is None:
        return v                          # 「过」= 全 0（没有牌型、没有牌）
    v[_A_OFF_KIND + m.kind - 1] = 1.0
    v[_A_OFF_RANK + _rank_slot(m.rank)] = 1.0
    if m.size >= 9 or meld.bomb_class(m) == meld.CLASS_JOKER_BOMB:
        v[_A_OFF_SIZE + _SIZE_COMPOSITE] = 1.0
    else:
        v[_A_OFF_SIZE + m.size - 1] = 1.0
    if m.wild_used:
        v[_A_OFF_WILD] = 1.0
    for c in m.cards:
        v[_A_OFF_CARDS + cards.slot(c)] = 1.0
    return v


# ---------------------------------------------------------------- 环境

@dataclass
class EnvConfig:
    """一手牌怎么开局。

    `tribute=False`（默认）**不走进贡**：发牌后直接由 `first` 或随机座位领出。
    理由见 spec §13.1 —— 一手一个 episode 时，进贡要依赖**上一手的名次**，
    而训练时那个名次是采样的；`tribute=True` 配合 `prev_ranks` 才用得上。
    **进贡规则本身已经实现并验过（Task 4 / 6），这里只是训练时开不开。**
    另外进贡那条基线本身还不稳（spec §13.8：贡最大的牌只对了 18/25），
    所以默认关着也顺带把那份不确定性挡在训练之外。
    """
    tribute: bool = False             # 开不开进贡（默认关，理由见上面的类说明）
    level: Optional[int] = None       # 打几；None = 每局从 1..13 随机采一个
    seed: int = 0                     # 牌堆/级别的随机种子（想复现同一局就固定它；0 就是种子 0，不是「随机」）


class GuandanEnv:
    """一手牌的自对弈环境。

    **`self.hand` 是明牌的 `rules.Hand`，绝不能交给策略。** 策略只能拿到
    `observe()` 出来的 `Observation` 和 `encode_*` 的向量。
    """

    def __init__(self, cfg: EnvConfig = None, seed: int = None):
        self.cfg = cfg or EnvConfig()
        self.rng = random.Random(self.cfg.seed if seed is None else seed)
        self.hand: Optional[rules.Hand] = None
        self._played = {s: set() for s in rules.SEATS}
        self._last_actor = None
        #: 这些座位的候选带**花色变体**（A1b 实验，默认空 = 老行为逐位不变）。
        #: 只有实验对象那一侧开，对手保持老候选 ⇒ 尺子本身不变。
        self.expand_seats: frozenset = frozenset()

    # ------------------------------------------------------------ 开局

    def reset(self, level=None, hands=None, first=None, prev_ranks=None) -> Observation:
        lv = level if level is not None else self.cfg.level
        if lv is None:
            lv = self.rng.randint(1, 13)
        h = rules.new_hand(self.rng, level=lv, hands=hands, first=first)
        if self.cfg.tribute or prev_ranks is not None:
            rules.apply_tribute(h, prev_ranks)      # 会顺手把 turn 设成先出者
        self.hand = h
        self._played = {s: set() for s in rules.SEATS}
        self._last_actor = None
        return self.observe()

    # ------------------------------------------------------------ 查询

    @property
    def done(self) -> bool:
        return self.hand.over

    @property
    def ranks(self) -> list:
        return self.hand.ranks() if self.hand.over else None

    def legal(self) -> list:
        """当前该谁出，他的候选（含 `None` = 过）。

        `expand_seats` 里的座位会拿到**带花色变体**的候选（A1b 实验）。
        `step(i)` 内部调的就是这个函数 ⇒ 下标天然一致，不会错位。
        """
        return self.hand.actions(self.hand.turn,
                                 variants=self.hand.turn in self.expand_seats)

    def observe(self, seat: int = None) -> Observation:
        a = self.hand.turn if seat is None else seat
        return observe(self.hand, a, self._played, self.hand.table)

    # ------------------------------------------------------------ 一步

    def step(self, index: int):
        """走第 `index` 个候选。返回 `(obs, reward, done, info)`。

        `reward` 只在**这一手结束的那一步**非零，且是**出牌人所在队**的收益
        （spec §5.4 的零点五口径）。中间步恒为 0 —— 牌类游戏中间没有即时反馈，
        DMC 的做法是拿终局 reward 当所有决策点的回归目标，那在训练循环里做。
        """
        seat = self.hand.turn
        acts = self.legal()
        if not 0 <= index < len(acts):
            raise rules.IllegalPlay(f"动作下标 {index} 越界（候选 {len(acts)} 个）")
        chosen = acts[index]
        self.hand.play(seat, chosen)
        if chosen is not None:
            self._played[seat] |= set(chosen.cards)
        self._last_actor = seat

        r = 0.0
        if self.hand.over:
            r = rules.reward(self.hand.ranks(), seat)
        info = {"seat": seat, "meld": chosen}
        return self.observe(), r, self.hand.over, info

    # ------------------------------------------------------------ 自对弈

    def rollout(self, policy) -> list:
        """跑完一手牌，返回每个决策点 `(obs, 候选, 选中下标, 出牌人)`。

        `policy(obs, actions) -> index`。`rules.reward` 的终局值由调用方回填 ——
        这里只负责把决策点如实记下来。
        """
        out = []
        obs = self.observe()
        while not self.done:
            acts = self.legal()
            i = policy(obs, acts)
            out.append((obs, acts, i, self.hand.turn))
            obs, _r, _done, _info = self.step(i)
        return out


# ---------------------------------------------------------------- 动作历史

#: spec §4.1「最近 15 手的动作序列 → LSTM」。
HISTORY_LEN = 15

#: 每一行 = `encode_action(m)` ⊕ **出牌人的相对座位** one-hot(4)。
#:
#: ⚠️ 那 4 维是**在 spec §4.2 的动作编码之外加的**（§4.2 只说了动作怎么编）。
#: 加的理由：同一个 9♠ 是下家出的还是对家出的，对判断局面完全不同 ——
#: 不记「谁出的」，这 15 行能提供的信息会少一大半。动作编码本身仍严格按 §4.2（143 维）。
HISTORY_DIM = ACTION_BASE_DIM + 4     # 历史行不带动作侧后果特征（见上）
assert HISTORY_DIM == 147


def encode_history(hand: "rules.Hand", seat: int) -> np.ndarray:
    """最近 `HISTORY_LEN` 步，**右对齐**（最近的落在最后一行），新局前面补 0。

    历史里全是**公开信息**（谁出了什么、谁过了）—— 不构成明牌泄漏。
    传进来的虽然是明牌的 `Hand`，但这里**只读 `hand.steps` 与 `hand.level`**
    （`tests/test_env_history.py` 里有一条对照测试钉住这一点）。
    """
    v = np.zeros((HISTORY_LEN, HISTORY_DIM), dtype=np.float32)
    steps = hand.steps[-HISTORY_LEN:]
    for i, st in enumerate(steps):
        row = v[HISTORY_LEN - len(steps) + i]
        row[:ACTION_BASE_DIM] = encode_action(st.meld, hand.level)   # 过 = 全 0
        row[ACTION_BASE_DIM + (st.seat - seat) % 4] = 1.0
    return v
