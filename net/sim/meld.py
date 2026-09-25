"""掼蛋牌型真源 —— 合法着法枚举与大小比较。

按**牌 ID**（int）工作，复用 net/cards.py 的编码：
    parts(cid) -> (idx, suit, deck)
    idx 口径 A=1、2..10、J=11、Q=12、K=13、小王=14、大王=15

为什么不用牌名字符串：live/rules.py 的 6 个 bug 有一半来自字符串处理
（"10" vs "T"），见 spec §2.3。这里一律用整数。

牌型表从游戏协议的 card_type 字段统计得到（465 手真牌，spec §2.2）。
炸弹顺序由用户口述 + 24 对真实证据交叉验证（0 条矛盾）：
    4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 9炸 < 10炸 < 天王炸
其中 4炸~8炸 有实证；**9炸/10炸（8 张同点数 + 逢人配）是自然延伸**，
只有一手真实「存在」证据、没有「对压」证据（见 `_BOMB_CLASS_BY_SIZE` 旁注）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from net import cards

SINGLE, PAIR, TRIPLE = 1, 2, 3
STRAIGHT, TRIPLE_PAIR = 4, 5
PAIR_RUN, PLATE = 6, 7
BOMB, STRAIGHT_FLUSH, BOMB6 = 8, 9, 10

JOKER_SMALL, JOKER_BIG = 14, 15

# 非序列牌型的点数比较值：级牌 > A > K > ... > 2
_POINT = {**{i: i - 1 for i in range(2, 11)}, 11: 10, 12: 11, 13: 12, 1: 13}
POINT_LEVEL, POINT_SMALL, POINT_BIG = 14, 15, 16

_MIN_BOMB = 4
_MAX_BOMB = 10         # 两副牌一个点数最多 8 张，**再加最多 2 张逢人配 = 10**

# 4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 有用户口述 + 24 对真实证据；
# 9炸 / 10炸（8 张同点数 + 逢人配）是**自然延伸** —— 真数据里有一手 9 张炸
# （J♠J♠(二副)J♥J♥(二副)J♣J♣(二副)J♦J♦(二副) + ♥3，打 3，card_type=10），
# 但只有「存在」证据、没有「对压」证据，位置需用户确认。
_BOMB_CLASS_BY_SIZE = {4: 1, 5: 2, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
CLASS_FLUSH = 3
CLASS_JOKER_BOMB = 9   # 天王炸仍居顶（比 10 张炸还高一层）

# --- 序列类牌型（顺子 / 连对 / 钢板）--------------------------------------
# 规模是定死的：数据里 4 张只有炸弹（没有二连对），顺子也只出现 5 张的。
_SEQ_LEN = 5
_PAIR_RUN_LEN = 3
_PLATE_LEN = 2
_NAT_MAX = 14          # A 当大牌时的自然值；也是序列能给到的最大值

# 同花判断用的花色字符。顺序固定 —— 枚举结果要可复现（不要用 set 迭代序）。
_SUITS = "♠♥♣♦"


def norm_level(level: Optional[int]) -> Optional[int]:
    """级别归一。**日志里偶尔用 14 表示 A**（net/cards.py 的 sort_key 也处理过这条），
    不归一的话 14 会被当成小王，级牌判定与顺子权重全错。
    """
    if level == 14:
        return 1
    return level


def point_value(idx: int, level: Optional[int]) -> int:
    level = norm_level(level)
    if idx == JOKER_BIG:
        return POINT_BIG
    if idx == JOKER_SMALL:
        return POINT_SMALL
    if level is not None and idx == level:
        return POINT_LEVEL
    return _POINT[idx]


def is_wild(cid: int, level: Optional[int]) -> bool:
    """级牌红桃 = 逢人配（万能牌）。"""
    level = norm_level(level)
    if level is None:
        return False
    idx, suit, _ = cards.parts(cid)
    return idx == level and suit == "♥"


def nat_values(idx: int) -> tuple:
    """这个点数在序列里的自然值。**A 可作最小也可作最大**，王不参与序列。

    A2345 与 10JQKA 都合法，所以 A 同时给 1 和 14；这比「比较时特判 A」干净，
    因为枚举端（_seq_lookup）和比较端（rank 取顶端自然值）只有这一个真源。

    王返回空元组 —— 顺子/连对/钢板的「王不能进序列」就落在这一条上，
    枚举端不需要再写一遍 `idx < JOKER_SMALL` 判断。
    """
    if idx == 1:
        return (1, _NAT_MAX)
    if 2 <= idx <= 13:
        return (idx,)
    return ()


@dataclass(frozen=True)
class Meld:
    kind: int
    size: int
    rank: int                       # 比较主键（同 kind 内可比）
    cards: tuple
    wild_used: int = 0

    @property
    def is_bomb(self) -> bool:
        return bomb_class(self) is not None


def _all_jokers(ids) -> bool:
    """四张牌全是王（大小王各两张，即天王炸）。

    先过 `cards.is_card`：单测里的 Meld 允许用占位整数当 cards，比较关系
    不该为此炸掉 KeyError。真实牌局里 cards 全是合法牌 ID，这一层零影响。
    """
    if len(ids) != 4:
        return False
    for c in ids:
        if not cards.is_card(c):
            return False
        if cards.parts(c)[0] not in (JOKER_SMALL, JOKER_BIG):
            return False
    return True


def bomb_class(m: Meld) -> Optional[int]:
    """炸弹层级；不是炸弹返回 None。天王炸最高。"""
    if m.kind == STRAIGHT_FLUSH:
        return CLASS_FLUSH
    if _all_jokers(m.cards):
        return CLASS_JOKER_BOMB
    if m.kind in (BOMB, BOMB6):
        cls = _BOMB_CLASS_BY_SIZE.get(m.size)
        if cls is None:
            raise ValueError(
                f"不认识的炸弹张数 {m.size}（合法 {_MIN_BOMB}~{_MAX_BOMB}；"
                f"牌 {m.cards}）")
        return cls
    return None


def beats(a: Meld, b: Meld) -> bool:
    """a 能不能压过 b。"""
    ca, cb = bomb_class(a), bomb_class(b)
    if ca is not None and cb is not None:
        return ca > cb if ca != cb else a.rank > b.rank
    if ca is not None:
        return True                 # 炸弹压普通牌型
    if cb is not None:
        return False                # 普通牌型压不了炸弹
    if a.kind != b.kind or a.size != b.size:
        return False                # 牌型不同不能压
    return a.rank > b.rank


def _by_idx(hand: Sequence[int]) -> dict:
    g = {}
    for c in hand:
        g.setdefault(cards.parts(c)[0], []).append(c)
    return g


def _melds_basic(hand: Sequence[int], level: Optional[int]) -> list:
    out = []
    for idx, ids in _by_idx(hand).items():
        pv = point_value(idx, level)
        out.append(Meld(SINGLE, 1, pv, (ids[0],)))
        if idx >= JOKER_SMALL:
            # 王不能当普通点数用：两个小王是一对，但不能凑三张/炸弹
            if len(ids) >= 2:
                out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
            continue
        if len(ids) >= 2:
            out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
        if len(ids) >= 3:
            out.append(Meld(TRIPLE, 3, pv, tuple(ids[:3])))
        for n in range(_MIN_BOMB, min(len(ids), _MAX_BOMB) + 1):
            out.append(Meld(BOMB, n, pv, tuple(ids[:n])))
    return out


def _melds_triple_pair(level, triples, pairs) -> list:
    out = []
    for t_idx, t_cards in triples:
        for p_idx, p_cards in pairs:
            if p_idx == t_idx:
                continue
            out.append(Meld(TRIPLE_PAIR, 5, point_value(t_idx, level),
                            tuple(t_cards) + tuple(p_cards)))
    return out


def _melds_joker_bomb(hand: Sequence[int]) -> list:
    """四大天王（大小王各两张）。最高层级，bomb_class 靠 _all_jokers 认。

    王在 _melds_basic 里是 continue 掉的（不能凑三张/普通炸弹），
    所以这里必须单独补一条 —— 漏了它，用户永远拿不到「出天王炸」的建议。
    """
    js = [c for c in hand if cards.parts(c)[0] in (JOKER_SMALL, JOKER_BIG)]
    if len(js) >= 4:
        return [Meld(BOMB, 4, 0, tuple(sorted(js[:4])))]
    return []


def _seq_lookup(g: dict) -> dict:
    """自然值 -> 能占这个位置的牌（都按 `_by_idx` 的顺序，取 [0] 就是最小的那组）。

    A 同时落进 1 和 14 两格，王两格都不落 —— 全部由 `nat_values` 决定。
    同一个窗口内 1 与 14 不可能同时出现（窗口最长 5 个连续自然值，
    取值域 1..14），所以 A 不会被同一个顺子用两次。
    """
    nat = {}
    for idx, ids in g.items():
        for nv in nat_values(idx):
            nat.setdefault(nv, []).extend(ids)
    return nat


def _window(nat: dict, top: int, span: int):
    """[top-span+1, top] 这一段连续自然值；只要有一格没牌就返回 None。

    张数够不够（连对要 2、钢板要 3）由调用方自己判 —— 顺子一格一张就够。
    """
    nats = list(range(top - span + 1, top + 1))
    if any(not nat.get(n) for n in nats):
        return None
    return nats


def _melds_straights(g: dict, level) -> list:
    """顺子（恰好 5 张连续）与同花顺。比较主键 = 顶端自然值。

    ⚠️ **同花顺必须逐花色试，不能只看每格第一张牌。** 两副牌下每个点数通常
    有两张不同花色，所以「代表牌是杂色」是常态而非边角：

        5♥ 5♠ 6♠ 7♠ 8♠ 9♠  -> 只看代表牌（5♥）会漏，5♠6♠7♠8♠9♠ 明明在手上
        5♠ 5♥ 6♠ 7♠ 8♠ 9♠  -> 代表牌正好取到 5♠ 才命中

    漏掉的后果不只是少一条建议：tools/accept_meld.py 的验收① 是按**牌张集合**
    比对的（`sorted(m.cards) == sorted(s.actual)`），会把真实打出的同花顺
    报成「枚举不出」。

    同顶端的花色**全都要出，不 break** —— 同理，只留一个代表的话，玩家实际
    打的是另一种花色时验收① 依然会红。
    """
    nat = _seq_lookup(g)
    out = []
    for top in range(_SEQ_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _SEQ_LEN)
        if nats is None:
            continue
        # 顺子每格取一个代表即可（大小只跟顶端有关）；王的 nat 为空，进不来
        out.append(Meld(STRAIGHT, _SEQ_LEN, top,
                        tuple(nat[n][0] for n in nats)))
        for suit in _SUITS:
            pick = [next((c for c in nat[n] if cards.parts(c)[1] == suit), None)
                    for n in nats]
            if all(c is not None for c in pick):
                out.append(Meld(STRAIGHT_FLUSH, _SEQ_LEN, top, tuple(pick)))
    return out


def _melds_pair_run(g: dict, level) -> list:
    """连对（恰好 3 个连续对子）。4 张只有炸弹，所以没有二连对。"""
    nat = _seq_lookup(g)
    out = []
    for top in range(_PAIR_RUN_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _PAIR_RUN_LEN)
        if nats is None:
            continue
        if any(len(nat[n]) < 2 for n in nats):
            continue
        ids = [c for n in nats for c in nat[n][:2]]
        out.append(Meld(PAIR_RUN, _PAIR_RUN_LEN * 2, top, tuple(ids)))
    return out


def _melds_plate(g: dict, level) -> list:
    """钢板（恰好 2 个连续三张）。"""
    nat = _seq_lookup(g)
    out = []
    for top in range(_PLATE_LEN, _NAT_MAX + 1):
        nats = _window(nat, top, _PLATE_LEN)
        if nats is None:
            continue
        if any(len(nat[n]) < 3 for n in nats):
            continue
        ids = [c for n in nats for c in nat[n][:3]]
        out.append(Meld(PLATE, _PLATE_LEN * 3, top, tuple(ids)))
    return out


# --- 逢人配（万能牌）替牌 ------------------------------------------------
# 级牌红桃能当**任意普通牌**用，但不能当王。枚举分两段：
#   _melds_natural  天然牌型（逢人配只当它自己那张级牌，wild_used 恒为 0）
#   _melds_wild     只有天然凑不成时才补逢人配
# 补出来的 Meld **必须把逢人配那张牌算进 cards** —— tools/accept_meld.py 的
# as_meld() 是按精确牌组比对的（sorted(m.cards) == sorted(实际出的牌)），
# 只放天然那几张的话，真实打出的每一手带逢人配的牌都会被报成「判不出牌型」。


def _split_wild(hand: Sequence[int], level: Optional[int]) -> tuple:
    """把逢人配拆出来。返回 (其余牌, 逢人配列表)。"""
    wilds = [c for c in hand if is_wild(c, level)]
    rest = [c for c in hand if not is_wild(c, level)]
    return rest, wilds


def _missing(nat: dict, nats, per: int) -> int:
    """要凑出 nats 这些自然值、每个 per 张，还缺几张。

    `nat` 是 `_seq_lookup(g)` 的结果（自然值 -> 牌），A 同时落在 1 与 14 两格。
    不要用 idx 直接查 `g` —— A 的两面性只在 `_seq_lookup` 里处理一次。
    """
    d = 0
    for n in nats:
        have = len(nat.get(n, []))
        if have < per:
            d += per - have
    return d


def _take(nat: dict, nats, per: int) -> tuple:
    """窗口里每格取 per 张牌（不够就少取，缺口由调用方补逢人配）。"""
    out = []
    for n in nats:
        out.extend(nat.get(n, [])[:per])
    return tuple(out)


def _with_wild(cards, wilds, d: int) -> tuple:
    """天然那几张 + 补上的 d 张逢人配（`cards` 是具体牌 ID）。"""
    return tuple(cards) + tuple(wilds[:d])


def _melds_wild(g: dict, level, wilds: Sequence[int]) -> list:
    """用逢人配补出来的牌型。g 是**不含逢人配**的牌分组，wilds 是逢人配本身。

    只产出「需要补」的牌型（`0 < 缺口 <= 拿得出的逢人配张数`）—— 天然的那些由
    `_melds_natural` 负责，重复由 `melds_from` 统一收口。

    王一律不参与：`ranks` 掐掉王，序列的自然值表（`_seq_lookup`/`nat_values`）
    本来就不收王，所以逢人配当不成王。
    """
    n_wild = len(wilds)
    if n_wild <= 0:
        return []
    nat = _seq_lookup(g)          # 自然值 -> 牌（A 同时落 1 与 14）
    out = []
    # **按点数降序**，不用手牌顺序：三带二里 t/p 是对称的（手里 10♣10♥ + 2♠2♦ +
    # 逢人配，既可以说成「三个 10 带一对 2」也可以说成「三个 2 带一对 10」），
    # 两种说法牌组相同、只能留一个。手牌顺序会让「留哪个」随发牌顺序漂 ——
    # 于是同一组牌在「真实那一手」与「整手牌」里解释成不同的 rank，验收①会假红。
    # 取点数大的当三张：这是**真值**，不是随手定的 —— 真实数据
    # （net/events.jsonl 2026-09-23 14:14，打 J）那一手 10♥(二副)10♣(二副)+
    # ♥J(二副) + 2♠(二副)2♦(二副) 压在「三个 9 带一对 4」上，只有三张是 10
    # （rank 9）才压得过，三张是 2（rank 1）压不过 —— 游戏自己读的是前者。
    ranks = sorted((i for i in g if i < JOKER_SMALL), reverse=True)

    # 对子 / 三张 / 炸弹：同一个点数的牌不够，就拿逢人配顶上
    for i in ranks:
        have = len(g[i])
        pv = point_value(i, level)
        for need, kind in ((2, PAIR), (3, TRIPLE)):
            d = need - have
            if 0 < d <= n_wild:
                out.append(Meld(kind, need, pv, _with_wild(g[i], wilds, d),
                                wild_used=d))
        for n in range(_MIN_BOMB, _MAX_BOMB + 1):
            d = n - have
            if 0 < d <= n_wild:
                out.append(Meld(BOMB, n, pv, _with_wild(g[i], wilds, d),
                                wild_used=d))

    # 三带二：三张与对子各自可能都不齐（真实数据两种都有）——
    # 「三张齐、只缺对子」（3+1）与「两边都只有两张」（2+2）都必须出。
    # 缺口按**取代表之后**各点数还剩几张算（`[:3]` / `[:2]`，与 `_melds_triple_pair`
    # 同口径）：手里同一张数有 5 张时，仍然可以挑 3 张当三张，不能因为「这个点数
    # 有 4 张以上」就整条不算 —— 那会把真实打出的三带二报成枚举不出。
    for t in ranks:
        for p in ranks:
            if p == t:
                continue
            d = max(0, 3 - len(g[t])) + max(0, 2 - len(g[p]))
            if 0 < d <= n_wild:
                out.append(Meld(TRIPLE_PAIR, 5, point_value(t, level),
                                _with_wild(tuple(g[t][:3]) + tuple(g[p][:2]),
                                           wilds, d),
                                wild_used=d))

    for top in range(_SEQ_LEN, _NAT_MAX + 1):
        nats = list(range(top - _SEQ_LEN + 1, top + 1))
        d = _missing(nat, nats, 1)
        if 0 < d <= n_wild:
            out.append(Meld(STRAIGHT, _SEQ_LEN, top,
                            _with_wild(_take(nat, nats, 1), wilds, d),
                            wild_used=d))
        # 同花顺单独试，且**要在 `if 0 < d` 之外** —— 每个自然值都有牌、
        # 但都不是同一花色时，d == 0 而缺的全靠逢人配补。
        # 同样必须逐花色试（见 `_melds_straights` 的说明），也不能 break：
        # 手里同时有 ♠/♥ 两套同顶端的同花顺时，只留一个代表会让真实打出
        # 另一套的玩家被报「枚举不出」。
        for suit in _SUITS:
            pick, miss = [], 0
            for n in nats:
                c = next((x for x in nat.get(n, [])
                          if cards.parts(x)[1] == suit), None)
                if c is None:
                    miss += 1
                else:
                    pick.append(c)
            if 0 < miss <= n_wild:      # miss == 0 是天然的，由 _melds_straights 负责
                out.append(Meld(STRAIGHT_FLUSH, _SEQ_LEN, top,
                                _with_wild(pick, wilds, miss),
                                wild_used=miss))

    for start in range(1, _NAT_MAX - _PAIR_RUN_LEN + 2):
        nats = list(range(start, start + _PAIR_RUN_LEN))
        d = _missing(nat, nats, 2)
        if 0 < d <= n_wild:
            out.append(Meld(PAIR_RUN, _PAIR_RUN_LEN * 2, nats[-1],
                            _with_wild(_take(nat, nats, 2), wilds, d),
                            wild_used=d))

    for start in range(1, _NAT_MAX - _PLATE_LEN + 2):
        nats = list(range(start, start + _PLATE_LEN))
        d = _missing(nat, nats, 3)
        if 0 < d <= n_wild:
            out.append(Meld(PLATE, _PLATE_LEN * 3, nats[-1],
                            _with_wild(_take(nat, nats, 3), wilds, d),
                            wild_used=d))
    return out


def _melds_natural(hand: Sequence[int], level: Optional[int]) -> list:
    """枚举手牌能组成的**天然**牌型（逢人配只当作它自己那张级牌）。

    单/对/三/三带二/炸弹/天王炸 + 顺子/同花顺/连对/钢板。
    这里是 Task 4/5 原有的那份 melds_from，**整体改名、函数体一行没动** ——
    三带二不排除王（真实数据有一手 2♦2♦2♣+小王小王）、天王炸单独补一条，
    都还在下面。Task 6 的逢人配替牌在 `_melds_wild` 里另写，本函数不掺和。
    """
    level = norm_level(level)
    g = _by_idx(hand)
    out = _melds_basic(hand, level)
    # 三张必须排除王：每种王只有两张，凑不出三张。
    triples = [(i, v[:3]) for i, v in g.items()
               if i < JOKER_SMALL and len(v) >= 3]
    # 对子**不排除王**：游戏允许王当三带二里的对子
    # （真实数据：三个 2 带两张小王）。_by_idx 按 idx 分组，
    # 所以小王只跟小王成对、大王只跟大王成对，不会跨 idx 凑一对。
    pairs = [(i, v[:2]) for i, v in g.items() if len(v) >= 2]
    out += _melds_triple_pair(level, triples, pairs)
    out += _melds_joker_bomb(hand)      # 天王炸（王在 _melds_basic 里是 continue 掉的）
    out += _melds_straights(g, level)
    out += _melds_pair_run(g, level)
    out += _melds_plate(g, level)
    return out


def melds_from(hand: Sequence[int], level: Optional[int] = None) -> list:
    """枚举手牌能组成的牌型，逢人配当万能牌，但**先试天然的**。

    两段拼起来：天然牌型（`_melds_natural`，`wild_used == 0`）+ 补牌牌型
    （`_melds_wild`，只在天然凑不成时才出）。逢人配一张都没有、只有一张、
    两张齐全都能正常工作（`_split_wild` 不假设张数）。

    ⚠️ **契约：每个 `(kind, size, rank)` 只返回一个「代表」，不保证牌张级完整。**

    这里曾被 docstring 写成「所有牌型」—— 那是**过度承诺**，实际产出的是
    「每个形状一条代表」：

      - 顺子 / 同花顺：每格取 `nat[n][0]`（同花顺按 `_SUITS` 逐花色各给一条）
      - 炸弹：取 `ids[:n]` —— 同一点数手里有 8 张时，不会再给「换 4 张」的组合
      - 连对 / 钢板：每格取前 2 / 前 3 张
      - 三带二：三张那一半与对子那一半**两边都枚举**（`_melds_triple_pair`）

    Plan 2 拿它当 RL 的动作空间时按这个契约用：同一个 `(kind, size, rank)` 只会
    出现一次，但**不要**指望它把同一点数里「选哪几张」的所有组合都列出来。
    验收脚本的 `shape()`（`(kind, size, rank)` 键）依赖这条 —— 所以**不要**
    为了「补全」而去改枚举本身，那会让验收①的比对口径当场失效。
    """
    rest, wilds = _split_wild(hand, level)
    out = _melds_natural(hand, level)
    if wilds:
        out += _melds_wild(_by_idx(rest), level, wilds)
    # 收口去重。键带 kind：同一组牌可以同时是顺子与同花顺（含天然那两条），
    # 那是两种解释，都要留下 —— tools/accept_meld.py 的 as_meld() 靠它取最强解释。
    seen, uniq = set(), []
    for m in out:
        key = (m.kind, tuple(sorted(m.cards)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq


def legal_moves(hand: Sequence[int], table: Optional[Meld],
                level: Optional[int] = None) -> list:
    """现在能出的所有牌。table 为 None 表示我领出（此时不产生「过」）。"""
    moves = melds_from(hand, level)
    if table is None:
        return moves
    return [m for m in moves if beats(m, table)]


def _stronger(a: Meld, b: Meld) -> bool:
    """同一组牌的两个解释里，a 是不是更强的那条（`strongest` 的比较口径）。"""
    ca, cb = bomb_class(a), bomb_class(b)
    if (ca is None) != (cb is None):
        return ca is not None                 # 炸弹类优先
    if ca is not None and cb is not None and ca != cb:
        return ca > cb
    return a.kind > b.kind


def strongest(melds: Sequence[Meld]) -> Optional[Meld]:
    """从一组候选里挑最强的一个；没有候选返回 None。

    **「同一组牌符合多个牌型时取最强」是一条规则，不是顺手写下的比较。**
    一手 5 张同花连续的牌**同时**是顺子(4)与同花顺(9)，`melds_from` 两条都产出；
    取第一条（顺子在枚举顺序里靠前）会把
      - 真实打出的同花顺当成顺子（面板上显示错）
      - 桌面上的同花顺低估成顺子 -> `legal_moves` 放进本该压不过的顺子
    游戏自己把它叫同花顺（card_type 9）。

    这条规则在 Plan 1 里被**两处副本各修过一次**（`tools/accept_meld.py` 的
    `as_meld` 与 `live/rules.py` 的适配层）—— 副本会漂，所以只留这一份。
    """
    best: Optional[Meld] = None
    for m in melds:
        if best is None or _stronger(m, best):
            best = m
    return best


def as_meld(ids: Sequence[int], level: Optional[int] = None) -> Optional[Meld]:
    """把一组**具体**的牌判成一个 Meld；判不出返回 None。

    参数就是那一手**确切的牌**（不是「代表牌」），所以按精确集合比对：
    `sorted(m.cards) == sorted(ids)`；同一组牌若符合多个牌型，取最强的那个
    （见 `strongest`：一手 5 张同花连续的牌必须是同花顺，不能降级成顺子）。

    **为什么这个原语在引擎里、而不是留在验收脚本里**：生产推理链
    （spec §3 / §8.1：`net/advise.py -> legal_moves(hand, table=…)`）拿到的桌面是
    `net/state.py` 的 `Play.cards`（一串牌 ID）+ `card_type`，喂给 `beats()` 之前
    必须先把这串牌解释成**带 rank 的 Meld** —— 用的正是这一个函数。留在
    `tools/`（离线验收目录）里会让 Plan 3/4 走错方向的 import，或者被抄出第三份。
    """
    best: Optional[Meld] = None
    for m in melds_from(list(ids), level):
        if sorted(m.cards) != sorted(ids):
            continue
        if best is None or _stronger(m, best):
            best = m
    return best


# ------------------------------------------------- 适配层用的名字转换
# 只在 live/rules.py 这个适配层里用；本模块内部一律用 ID。
# 这是**全项目唯一允许出现牌名字符串**的地方（spec §2.3：旧 live/rules.py 的
# "10" vs "T" 就是字符串处理惹的祸，所以边界只留一处，别处一律走 ID）。
#
# **词表以生产者为准，不是我们编的**：live/ 那条线（live/main.py:229）喂进来的
# 是模型自己的 54 个类名 —— `synth/layout.py` 的 `CLASSES = [f"S{r}" for r in
# "A23456789TJQK"] + …`，即 **十是 `T`**、王是 **`JOKER_S` / `JOKER_B`**。
# 第一版适配层只认 `"S10"` / `"JOKER_SMALL"` —— 那套词表是照着 `net/cards.py`
# 的**显示**习惯编的，生产根本不产它 —— 结果真实着法里 29% 直接抛错。
# 而牌子那时已经打出去了，live/main.py 的 render_lines 又不在 tick 的 try 里，
# 抛错会**永久打断面板刷新链**（那条路径现已由 live/main.py 的
# `safe_render_lines` 兜住 —— 链不再断、错也不再吞，但词表对不齐照样会
# 让面板每次都显示「渲染出错」，所以这条词表要求不因那次修复而放松）。
# 所以：`T` 与 `10` 都收，`JOKER_S/B` 与 `JOKER_SMALL/BIG` 都收，
# 但**测试必须从 `synth.layout.CLASSES` 取材**（见 tests/test_rules_adapter.py）。
#
# 一条铁律：**认不出的名字一律抛 ValueError**。悄悄当成某张牌 = 把识别错误
# 洗成合法结果，比崩掉危险得多（live/ 那条线的输入是模型识别出来的）。

_SUIT_LETTER = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}
_SUIT_BASE = {"♠": 16, "♥": 32, "♣": 48, "♦": 64}
# `T` 是生产写法（模型类名），`"10"` 由下面的数字分支接住。级别同理。
_RANK_LETTER = {"A": 1, "T": 10, "J": 11, "Q": 12, "K": 13}
_JOKER_BASE = {"JOKER_S": JOKER_SMALL, "JOKER_B": JOKER_BIG,        # 生产
               "JOKER_SMALL": JOKER_SMALL, "JOKER_BIG": JOKER_BIG}  # 长写法
_DECKS = (1, 2)
_DECK_SHIFT = 256          # 第二副 = 第一副 + 256（net/cards.py 的编码，王也是）


def cid_from_name(name: str, deck: int = 1) -> int:
    """牌面名 -> 牌 ID。生产写法 `'S3'` / `'ST'` / `'JOKER_S'` / `'JOKER_B'`。

    也收 `'S10'` / `'JOKER_SMALL'` 这类长写法（本模块自己的测试与
    `net/cards.py` 的读者习惯），但**生产的 54 类不带副数后缀**：模型分不出
    一副/二副，`(二副)` 这种写法在这里会抛 ValueError（见 live/rules.py 的 `_to_ids`）。
    """
    if deck not in _DECKS:
        raise ValueError(f"认不出的副数：{deck!r}（只有 1 / 2）")
    shift = _DECK_SHIFT if deck == 2 else 0
    if name.startswith("JOKER"):
        base = _JOKER_BASE.get(name)
        if base is None:
            raise ValueError(
                f"认不出的王：{name!r}（只有 JOKER_S / JOKER_B / JOKER_SMALL / JOKER_BIG）")
        return base + shift
    if len(name) < 2:
        raise ValueError(f"认不出的牌名：{name!r}（形如 'S3' / 'ST' / 'JOKER_S'）")
    letter, rank = name[0], name[1:]
    suit = _SUIT_LETTER.get(letter)
    if suit is None:
        raise ValueError(f"认不出的花色：{name!r}（只有 S/H/C/D 打头）")
    idx = _RANK_LETTER.get(rank)
    if idx is None:
        if not rank.isdigit() or not 2 <= int(rank) <= 10:
            raise ValueError(f"认不出的点数：{name!r}（2~10 / T / J / Q / K / A）")
        idx = int(rank)
    return _SUIT_BASE[suit] + idx + shift


def name_from_cid(cid: int) -> str:
    """牌 ID -> **人读显示**牌面（同 net/cards.py：77 -> 'K♦'，332 -> 'Q♦(二副)'）。

    ⚠️ 显示格式（点数在前），与 `cid_from_name` 的入参格式（花色字母在前）**不同**，
    所以 `cid_from_name(name_from_cid(c))` 会抛错。要往返请直接用 ID。
    """
    return cards.decode(cid)


_KIND_NAMES = {
    SINGLE: "单张", PAIR: "对子", TRIPLE: "三张", STRAIGHT: "顺子",
    TRIPLE_PAIR: "三带二", PAIR_RUN: "三连对", PLATE: "钢板",
    STRAIGHT_FLUSH: "同花顺",
}
# PAIR_RUN 叫「三连对」而不是「连对」，是**故意的**：live/main.py:230 把这个
# 字符串原样打在面板上，而老实现对 3 个连续对子返回的就是「三连对」——
# 改成「连对」会让面板文字在用户眼皮底下变（本项目把面板当交付物）。
# 顺带：老实现那个 4 张的「连对」（二连对）是真的不存在，已按 bug #4 去掉。


def describe_meld(m: Meld) -> str:
    """牌型 -> 中文名。口径与老 live/rules.py 逐字对齐：炸弹带张数、天王炸单列、
    3 个连续对子叫「三连对」（面板原样显示，见 `_KIND_NAMES` 的说明）。"""
    if _all_jokers(m.cards):
        return "四大天王"
    if m.kind in (BOMB, BOMB6):
        return f"{m.size} 张炸"
    return _KIND_NAMES[m.kind]


def level_idx(level) -> Optional[int]:
    """级别 -> 点数索引。接受 **1~13** 或 'A'/'T'/'J'/'Q'/'K'/'2'..'10'。

    旧接口给的 level 是**字符串**，词表与牌名同一套（live/level.py 打十返回 `'T'`，
    不是 `'10'`）：'2'..'9' / 'T' / '10' / 'J' / 'Q' / 'K' / 'A'。
    引擎要的是**整数 idx**，转换只在这里做（A=1，与日志的 Trump 字段同口径；
    `norm_level` 会再把 14 折回 1）。`None` 只表示「没有级牌」这一个合法含义。

    认不出的级别抛 ValueError —— 不返回 None 蒙混过去。**数值范围严格限 1~13**：
    `'0'` / `'99'` 这种「是数字、却不是一个级别」的输入必须炸（旧版 `isdigit()`
    会原样放过去 —— docstring 写着「不返回 None 蒙混」，实现却对它们蒙混了）。
    这不是理论问题：**live/level.py 的字形表里有 '0'**（打十时面板上是「1」+「0」
    两个字形），而它的 `_normalize` 只把 `'1'`/`'10'` 折成 `'T'` —— 打十被切坏、
    只剩「0」时会读出 `'0'`，放过去就会把**合法着法判成「不合法」**，
    正是 spec §6⑥ 要防的那种「把错结论端到用户脸上」。
    """
    if level is None:
        return None
    if isinstance(level, int):
        idx = level
    else:
        text = str(level).strip().upper()
        if text in _RANK_LETTER:
            return _RANK_LETTER[text]
        if not text.isdigit():
            raise ValueError(
                f"认不出的级别：{level!r}"
                f"（'2'~'10' / 'T' / 'J' / 'Q' / 'K' / 'A'）")
        idx = int(text)
    if not 1 <= idx <= 13:
        raise ValueError(
            f"认不出的级别：{level!r}（数值必须是 1~13；'0' / '99' 这类不是级别）")
    return idx
