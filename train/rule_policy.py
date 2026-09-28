"""**规则式对手** —— 「像人」的固定对手，替掉那个「能压就压最小」的贪心。

## 为什么要它

现在的固定对手是 `policies.greedy_policy`，它的定义（写在它自己的 docstring 里）
是「能压就压，从不主动过」，排序键 `(张数, 主点, 牌型)`。那三条**天然**导致它：

- **会压自己队友的牌** —— 它根本不看现在是谁在赢这一手
- **从不主动炸** —— 张数优先，炸弹最少 4 张，永远排在最后
- **从不主动过**、**完全不算剩牌**

而人类打法恰好是这几条的反面。所以它**惩罚不了浪费**：你白炸一手它抓不住，
因为它的字典里就没有「炸」这件事（这正是用户最初的抱怨）。

## 规则从哪来

主体十条移植自一份开源规则式掼蛋 AI：`github.com/QinlinChen/guandan-ai`
（`client/ai_client.py` + `client/utils.py`，我逐行读过）。**不是**「网上共识」——
中文攻略站（知乎/百科/头条）在**这台机器上抓不到**（走代理，机房 IP 被反爬拒）。

用户 2026-09-28 补了第 11 条、也是最重要的一条：**威胁不是「剩 1~2 张」**——
3 张（三张）、5 张（三带二/顺子）、6 张（三连对/钢板）**都可能一手走完**。
所以要**按剩的张数**估「他一手走完的风险」，再决定要不要压制。

## 纪律

- **只吃 `obs`（公开信息）**：`Observation` 里没有四家手牌，这里也不去拿 ——
  明牌泄漏那条红线靠类型堵，不靠自觉（见 `net/sim/env.py`）。
- 「台面那手是谁出的」**没有单独字段**，从公开信息**推**：`obs.played[座位]`
  包含台面上的每一张牌 ⇒ 就是那家出的（牌 ID 带副牌标记，判断是精确的）。
"""
from __future__ import annotations

from typing import Optional

from net import cards
from net.sim import meld, rules

#: 剩 `n` 张「一手走完」的风险到多少才算「必须压制」。
#: **这是一个启发式门槛**，不是概率 —— `finish_risk` 返回的是刻度，不是概率。
DANGER = 0.5

#: 「够硬」的门槛：主点 ≥ K。按 `point_value` 口径算出来，不写死数字
#: （K → 12、A → 13、级牌 → 14、王 → 15/16）。队友压到这个份上就别添乱了。
K_POINT = meld.point_value(meld.level_idx("K"), None)


def is_fire(m) -> bool:
    """火力牌：炸（含天王炸）/ 同花顺。那份开源 AI 的 `is_fire_card` 口径。"""
    return m is not None and (m.is_bomb or m.kind == meld.STRAIGHT_FLUSH)


def opponents(seat: int) -> tuple:
    """对手 = 另外一队的两个座位（队是 (0,2) / (1,3)，所以就是 ±1）。"""
    return ((seat + 1) % 4, (seat + 3) % 4)


def ally_of(seat: int) -> int:
    """队友 = 对家（±2）。"""
    return (seat + 2) % 4


def table_owner(obs) -> Optional[int]:
    """台面那手**是谁出的** —— 从公开信息推。

    `obs.played[s]` 是「座位 s 出过的所有牌」，台面那几张必然全在里面；
    一张牌只会被打出一次（ID 带副牌标记，精确），所以命中至多一家。
    领出（台面为空）返回 `None`。
    """
    if not obs.table:
        return None
    t = set(obs.table)
    for s in rules.SEATS:
        if t <= set(obs.played[s]):
            return s
    return None


def unseen_pool(obs) -> dict:
    """**还没露面**的牌：点数 -> 张数。整副 108 张减去我的手牌与四家出过的牌。

    另外三家的手牌只能是这个池子里的子集 —— 所以估「他能不能一手走完」
    只需要它，**不需要看明牌**。
    """
    seen = set(obs.hand)
    for s in rules.SEATS:
        seen |= set(obs.played[s])
    pool: dict = {}
    for cid in range(1, cards.MAX_ID + 1):
        if cards.is_card(cid) and cid not in seen:
            idx = cards.parts(cid)[0]
            pool[idx] = pool.get(idx, 0) + 1
    return pool


def _has_run(pool: dict, length: int, per: int) -> bool:
    """池子里有没有 `length` 个连续点数、每个至少 `per` 张（A 可当最小也可最大）。

    借 `meld.nat_values` —— A 的两头、王不参与序列，都由它一处定义。
    """
    have = set()
    for idx, c in pool.items():
        if c >= per:
            have |= set(meld.nat_values(idx))
    return any(all(v in have for v in range(s, s + length))
               for s in range(1, 15 - length + 1))


def finish_risk(n: int, pool: dict) -> float:
    """剩 `n` 张「**一手走完**」的风险 —— **启发式刻度（0~1），不是概率**。

    ⚠️ 数值只用来跟 `DANGER` 比大小，**别当概率读**（那个分布我没有任何依据）。

    口径（`n` 张要一手出掉，分别需要什么牌型）：

        1 张  任意单张（**必然**）
        2 张  对子
        3 张  三张
        4 张  炸（4 张同点）
        5 张  三带二 或 顺子
        6 张  三连对 / 钢板 / 6 张炸
        7~8 张 更大的炸

    能不能凑出来看**未露面的牌池**（`unseen_pool`）—— 他手上的牌只能是池子里的。
    池子里都凑不出，他手上就不可能有，所以那种情况给低值。

    ⚠️ 这条是**用户 2026-09-28 补的**：原来的规则只看「剩 1~2 张」，
    而 3/5/6 张同样可能一手走完（三张 / 三带二 / 三连对）。
    """
    if n <= 0:
        return 1.0
    if n == 1:
        return 1.0
    cnt = sorted(pool.values(), reverse=True)

    def has(k):
        return any(c >= k for c in cnt)

    if n == 2:
        return 1.0 if has(2) else 0.4
    if n == 3:
        return 0.9 if has(3) else 0.4
    if n == 4:
        return 0.6 if has(4) else 0.1
    if n == 5:
        return 0.8 if ((has(3) and has(2)) or _has_run(pool, 5, 1)) else 0.15
    if n == 6:
        return 0.6 if (_has_run(pool, 3, 2) or _has_run(pool, 2, 3) or has(6)) else 0.1
    if n <= 8:
        return 0.35 if has(n) else 0.05
    return 0.05


def hand_partition(hand, level) -> list:
    """把手牌**贪心**拆成尽量少的几手 —— 估「我还要几手才能走完」。

    口径来自那份开源 AI 的 `utils.partition`：三张尽量配一对凑成三带二
    （一次消 5 张）；剩下的对子、单张各算一手；4 张以上同点算炸。

    ⚠️ **不做最优划分**（那是指数级的），**也不认顺子/三连对/钢板** ——
    所以它**高估**手数，只当粗估用。要问「整手是不是一个牌型」（一把走完），
    用 `melds_from` 精确判，别用这个。
    """
    by_idx: dict = {}
    for c in hand:
        by_idx.setdefault(cards.parts(c)[0], []).append(c)
    singles, pairs, triples, bombs = [], [], [], []
    for idx, cs in by_idx.items():
        if not meld.nat_values(idx):          # 王：不参与序列，单独算一张
            singles += cs
        elif len(cs) >= 4:
            bombs.append(cs)
        elif len(cs) == 3:
            triples.append(cs)
        elif len(cs) == 2:
            pairs.append(cs)
        else:
            singles += cs
    groups = list(bombs)
    while triples:
        t = triples.pop(0)
        groups.append(t + pairs.pop(0) if pairs else t)   # 三张优先配一对 -> 三带二
    groups += pairs + [[c] for c in singles]
    return [m for m in (meld.as_meld(g, level) for g in groups) if m is not None]


# ------------------------------------------------------------------ 选牌助手
#
# 全部返回 `acts` 的**下标**；找不到就返回 None，由调用方兜底。
# 排序键与 `policies.greedy_policy` 一致 `(张数, 主点, 牌型)` —— 张数优先，
# 所以「最小」永远不会是炸弹（炸弹最少 4 张）。


def _key(m):
    return (m.size, m.rank, m.kind)


def _cheapest(acts):
    """最便宜的一手（含火力）—— 就是贪心的选择，做兜底用。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else _pass(acts)


def _cheapest_plain(acts):
    """最便宜的一手，**排除火力牌**（留着炸）。没有就 None。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and not is_fire(m)]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else None


def _cheapest_fire(acts):
    """最便宜的一手**火力牌**。没有就 None。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and is_fire(m)]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else None


def _of_kinds(acts, kinds):
    """最小的、属于 `kinds` 的那一手。没有就 None。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and m.kind in kinds]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else None


def _size_not(acts, n, cheat=False):
    """最小的、**张数不等于 `n`** 的一手（`cheat` 时连火力牌也允许）。

    为什么是「张数不等于」：**要跟牌必须同牌型同张数** —— 所以领一张
    `n` 张以外的牌型，剩 `n` 张的对手就**根本接不上**，一手走完的路就断了。
    """
    cand = [(i, m) for i, m in enumerate(acts)
            if m is not None and m.size != n and (cheat or not is_fire(m))]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else None


def _planned_plain(acts, hands):
    """分区里**非火力**的那几手中，最小的一个在 `acts` 里的下标。

    为什么按分区挑、而不是挑「最小的合法牌」：这一步的意图是**执行计划** ——
    手里只剩 2~3 手时，随手甩一张计划外的单张会把原本配好的对子拆掉。
    源规则就是这么写的（`min_strategy(card_type=分区里那个非火力牌型)`）。
    """
    want = {frozenset(h.cards) for h in hands if not is_fire(h)}
    cand = [(i, m) for i, m in enumerate(acts)
            if m is not None and frozenset(m.cards) in want]
    return min(cand, key=lambda im: _key(im[1]))[0] if cand else None


def _pass(acts):
    """「过」的下标；领出时没有「过」就返回 None。"""
    return acts.index(None) if None in acts else None


# ------------------------------------------------------------------ 决策
#
# 两条决策链，**按优先级依次试，第一个给出答案的赢**。每条都标了出处：
# `[源]` = 移植自那份开源规则式 AI；`[用户]` = 用户 2026-09-28 补的。


def _lead(obs, acts, pool) -> Optional[int]:
    """**领出**（台面是空的，我赢着这一手）。"""
    ally = ally_of(obs.seat)

    # [源 1] 整手就是一个牌型 -> 一把走完
    for i, m in enumerate(acts):
        if m is not None and set(m.cards) == set(obs.hand):
            return i

    hands = hand_partition(obs.hand, obs.level)
    # [源 2] 还剩 2~3 手、且既有火力又有非火力 -> 先走非火力那手（留炸）
    if 2 <= len(hands) <= 3 and any(map(is_fire, hands)) and not all(map(is_fire, hands)):
        i = _planned_plain(acts, hands)
        if i is not None:
            return i

    # [源 3] 喂队友：队友剩 1 张 -> 出最小单张；剩 2 张 -> 出最小对子
    if obs.left[ally] == 1:
        i = _of_kinds(acts, (meld.SINGLE,))
        if i is not None:
            return i
    if obs.left[ally] == 2:
        i = _of_kinds(acts, (meld.PAIR,))
        if i is not None:
            return i

    # [源 4] 优先三带二（一次消 5 张）
    i = _of_kinds(acts, (meld.TRIPLE_PAIR,))
    if i is not None:
        return i

    # [源 5 + 用户] 卡对手：领出的**张数不等于他剩的张数**，他就一手走不完。
    # 按风险从高到低挑那个最该卡的对手。⚠️ 这条比原来「剩 1 张出对子、
    # 剩 2 张出单张」更一般 —— 3/5/6 张同样可能一手走完（用户补的那条）。
    for opp in sorted(opponents(obs.seat), key=lambda s: -finish_risk(obs.left[s], pool)):
        if finish_risk(obs.left[opp], pool) >= DANGER:
            i = _size_not(acts, obs.left[opp])
            if i is not None:
                return i
    return None


def _follow(obs, acts, pool) -> Optional[int]:
    """**跟牌**（台面上有牌要压）。

    ⚠️ 跟牌必须**同牌型同张数**（炸弹/同花顺除外），所以「有能压的普通牌」
    就等于「`acts` 里除「过」和火力牌之外还有东西」—— `acts` 已经按 `beats`
    过滤过了。
    """
    ally = ally_of(obs.seat)
    owner = table_owner(obs)
    table = meld.as_meld(list(obs.table), obs.level) if obs.table else None

    # [用户 4] **队友赢着这一手 -> 让给他**（只有「我这一手能直接走完」才抢）。
    #
    # ⚠️ 这里**偏离了源规则**，有实测依据（2026-09-28，各 400 局同种子）：
    #
    #     变体                                      vs 贪心   压队友
    #     greedy（现状）                            基准      100.0%
    #     A 照搬源规则（只在队友那手是火力/≥K 时让）  67.5%     62.3%
    #     B 队友赢着就过（危险时除外）                68.0%     37.9%
    #     C 现在这个（能一把走完除外）                67.0%      1.2%
    #
    # 三条的胜率全在噪声内（±2.5%）⇒ **「不压队友」是免费的**，那就取最像人的那个。
    # 源规则只在「队友那手够硬」时才让，队友出小牌时照压 —— 而那正是用户抱怨的
    # 「人类不会这么打」。**危险时**也不该压队友：他本来就赢着这一手，
    # 压他并不能解决「对手快走完」那个问题。
    if owner == ally:
        for i, m in enumerate(acts):
            if m is not None and set(m.cards) == set(obs.hand):
                return i                      # 能一把走完 -> 抢
        return _pass(acts)

    danger = max((finish_risk(obs.left[o], pool) for o in opponents(obs.seat)),
                 default=0.0)

    # [用户 2] 有对手面临「一手走完」-> **必须拦**：先普通牌，没有就动炸
    if danger >= DANGER:
        i = _cheapest_plain(acts)
        if i is not None:
            return i
        i = _cheapest_fire(acts)
        return i if i is not None else _pass(acts)

    # [源 2] 不危险：压最小的**普通牌**（火力留着）
    i = _cheapest_plain(acts)
    if i is not None:
        return i
    # [源 3] 台面主点 > 10 且我凑得出炸 -> 才动炸
    if obs.table_rank > 10:
        i = _cheapest_fire(acts)
        if i is not None:
            return i
    return _pass(acts)


def rule_choose(obs, acts) -> int:
    """规则式选择，返回 `acts` 的下标。**两条链的唯一实现。**"""
    pool = unseen_pool(obs)
    i = _follow(obs, acts, pool) if obs.table else _lead(obs, acts, pool)
    return _cheapest(acts) if i is None else i


def rule_policy():
    """工厂：返回一个和 `greedy_policy` 同形状的策略 `(obs, acts, hist) -> 下标`。

    只吃 `obs`（公开信息）—— **拿不到明牌**，与那条红线一致。
    """
    def policy(obs, acts, hist=None) -> int:
        return rule_choose(obs, acts)
    return policy
