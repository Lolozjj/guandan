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
  明牌泄漏那条红线靠类型堵，不靠自觉（见 `guandan/sim/env.py`）。
- 「台面那手是谁出的」**没有单独字段**，从公开信息**推**：`obs.played[座位]`
  包含台面上的每一张牌 ⇒ 就是那家出的（牌 ID 带副牌标记，判断是精确的）。

## 用户 2026-09-29 补的第 12 条：**不许拆自己的炸**

「这个规则式，为啥第一手就把炸弹毫无意义的给拆了」—— 查出来是**取舍口径**的
毛病（「优先三带二」那条规则本身没错）：`_key` 看不见三带二里那一对，
`is_fire` 又只看**这一手本身**是不是炸，于是 `222+77`（从四个 7 里抽两张）
既不同分、也没人拦，纯粹**按枚举顺序抽签**抽中了。实测它 4.79% 的决策
**本可不拆却拆了**，波及 39/40 局 ⇒ `vs 规则式 88.2%` 那把尺子**虚高**。
修法见 `breaks_bomb` / `_prefer_intact` / `_spent`（`[源 4]` 那条捷径也加了闸）。

**修完的账**（同一批牌，40 局 / 400 局）：本可不拆 3.96% → **0.36%**；
规则式自己对贪心 67.0% → **84.2%**，用炸率 1.25 → 2.42 手/局（不是不炸了，
是**炸不再被当对子单张糟蹋**）；现役 `1407` 对**新**规则式只有 **54.8%**
（旧 88.2% —— **那个数作废**）。完整记录见
规则式尺子修正台账（已归档到 `master` 分支）。
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Optional

from guandan.capture import cards
from guandan.sim import meld, rules
# A2（2026-09-30）：这几个量从 sim/ 搬下来了（见 guandan/sim/features.py）
from guandan.sim.features import _has_run, hand_partition, unseen_pool   # noqa: F401  # 老名字照旧可用

#: 剩 `n` 张「一手走完」的风险到多少才算「必须压制」。
#: **这是一个启发式门槛**，不是概率 —— `finish_risk` 返回的是刻度，不是概率。
DANGER = 0.5

#: [源 2]「还剩几手就先走非火力那手（留炸）」的门槛。
#: ⚠️ **这是个魔数**：`hand_partition` 是粗估（高估手数），所以门槛定在几，
#: 是「什么时候开始按计划出牌」的开关。实测触发率（领出决策点）：
#: ≤3 → 1.3%、≤5 → 4.0%、≤6 → 7.3%、≤8 → 13.0%。
SOURCE2_MAX_HANDS = 12

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


# ------------------------------------------------------------------ 选牌助手
#
# 全部返回 `acts` 的**下标**；找不到就返回 None，由调用方兜底。
# 排序键与 `policies.greedy_policy` 一致 `(张数, 主点, 牌型)` —— 张数优先，
# 所以「最小」永远不会是炸弹（炸弹最少 4 张）。


def _key(m):
    """主排序键 —— 与 `policies.greedy_policy` 的 `_sort_key` 同口径
    （张数优先，所以「最小」永远不会是炸）。**并列时另有 `_spent` 细化。**"""
    return (m.size, m.rank, m.kind)


def breaks_bomb(hand, m) -> bool:
    """这一手是不是**从自己手里的炸里挖牌**：某个点数我握着 ≥4 张，
    这一手只用了其中 1~3 张 ⇒ 那个炸废了。

    ⚠️ 豁免只有一种：**把整个炸当炸打出去**（`is_bomb` 且四张全用掉）。
    不能只看「四张全用掉」—— 逢人配会骗人：`8♠8♥8♣8♦J♠`（8♥ 当 J 去配对子）
    里真被吃掉的 8 只有 3 张，8 炸没了，但按牌面点数看 4 个 8 全在。

    用户 2026-09-29 报的：「为啥第一手就把炸弹毫无意义的给拆了」。
    查出来两层毛病**都在这一层**（移植来的是「优先三带二」那条规则，
    怎么在三带二之间取舍是我写的）：`_key` 看不见「那一对」，
    而 `is_fire(m)` 判的是**这一手本身**是不是炸 —— `222+77` 不是炸，
    就当普通牌放行了。实测代价：规则式 4.7% 的决策**本可不拆却拆了**。
    """
    if m is None:
        return False
    have = Counter(cards.parts(c)[0] for c in hand)
    used = Counter(cards.parts(c)[0] for c in m.cards)
    for idx, n in have.items():
        if n < 4:
            continue
        k = used.get(idx, 0)
        if k and not (m.is_bomb and k == n):
            return True
    return False


def _prefer_intact(cand, hand):
    """`cand = [(下标, Meld), …]` —— **有没挖炸的就只在那些里挑**，下标原地不动。

    ⚠️ 只在**同一批候选内**做偏好，**不做成「先把 acts 全局过滤一遍」**。
    那样有两处会坏（都实测过）：
    (a) 手里唯一能压的普通牌型就埋在炸里时（跟三张只剩 `777` 可出），
        过滤会把 `777` 剔掉，于是被逼去**开整个炸** —— 比拆炸更蠢；
    (b) 过滤后的列表下标与调用方的 `acts` 不是一套，返回的下标**直接是错的**。
    """
    keep = [(i, m) for i, m in cand if not breaks_bomb(hand, m)]
    return keep or cand          # 没得挑时该拆还得拆 —— 这是偏好，不是禁令


def _spent(m, level):
    """这一手**花掉的最大那张**的点数（级牌按 14 算）—— `_key` 并列时的细化键。

    为什么需要它：`_key` 只看**三张**的点数，于是 `222+33` / `222+77` / `222+KK`
    的键**完全一样**（那 11 个候选同分），`min` 退化成「按枚举顺序抽签」——
    用户报的那一手就是这么抽中 `222+77` 的。
    """
    return max((meld.point_value(cards.parts(c)[0], level) for c in m.cards),
               default=0)


def _extreme(cand, hand, level, strongest: bool = False):
    """所有选牌助手共用的出口：**先别拆自己的炸**，再按 `(_key, _spent)` 取最小/最大。

    `strongest=True` 是给「卡不住形状就卡强度」用的（见 `_lead` [源 5]）——
    那不是把偏好翻过来，是**另一边**：没得选的时候出最大而不是最小。
    """
    cand = _prefer_intact(cand, hand)
    if not cand:
        return None
    key = lambda im: (_key(im[1]), _spent(im[1], level))
    return (max if strongest else min)(cand, key=key)[0]


def _pick(cand, hand, level):
    """最小的那个（正常路线）。"""
    return _extreme(cand, hand, level, strongest=False)


def _cheapest(acts, hand, level):
    """最便宜的一手（含火力）—— 就是贪心的选择，做兜底用。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None]
    i = _pick(cand, hand, level)
    return _pass(acts) if i is None else i


def _cheapest_plain(acts, hand, level):
    """最便宜的一手，**排除火力牌**（留着炸）。没有就 None。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and not is_fire(m)]
    return _pick(cand, hand, level)


def _cheapest_fire(acts, hand, level):
    """最便宜的一手**火力牌**。没有就 None。

    火力牌里也有「挖炸」的情况（同花顺从四张同点里抽两张），所以照样过 `_pick`。
    """
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and is_fire(m)]
    return _pick(cand, hand, level)


def _strongest_plain(acts, hand, level):
    """**最大**的一手，排除火力牌（留炸）—— 只给 [源 5] 的「卡强度」用。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and not is_fire(m)]
    return _extreme(cand, hand, level, strongest=True)


def _strongest(acts, hand, level):
    """**最大**的一手，含火力牌（兜底用）。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None]
    return _extreme(cand, hand, level, strongest=True)


def _of_kinds(acts, kinds, hand, level):
    """最小的、属于 `kinds` 的那一手。没有就 None。"""
    cand = [(i, m) for i, m in enumerate(acts) if m is not None and m.kind in kinds]
    return _pick(cand, hand, level)


def _size_not(acts, n, hand, level, cheat=False):
    """最小的、**张数不等于 `n`** 的一手（`cheat` 时连火力牌也允许）。

    为什么是「张数不等于」：**要跟牌必须同牌型同张数** —— 所以领一张
    `n` 张以外的牌型，剩 `n` 张的对手就**根本接不上**，一手走完的路就断了。
    """
    cand = [(i, m) for i, m in enumerate(acts)
            if m is not None and m.size != n and (cheat or not is_fire(m))]
    return _pick(cand, hand, level)


def _planned_plain(acts, hands, hand, level):
    """分区里**非火力**的那几手中，最小的一个在 `acts` 里的下标。

    为什么按分区挑、而不是挑「最小的合法牌」：这一步的意图是**执行计划** ——
    手里只剩 2~3 手时，随手甩一张计划外的单张会把原本配好的对子拆掉。
    源规则就是这么写的（`min_strategy(card_type=分区里那个非火力牌型)`）。
    """
    want = {frozenset(h.cards) for h in hands if not is_fire(h)}
    cand = [(i, m) for i, m in enumerate(acts)
            if m is not None and frozenset(m.cards) in want]
    return _pick(cand, hand, level)


def _pass(acts):
    """「过」的下标；领出时没有「过」就返回 None。"""
    return acts.index(None) if None in acts else None


# ------------------------------------------------------------------ 决策
#
# 两条决策链，**按优先级依次试，第一个给出答案的赢**。每条都标了出处：
# `[源]` = 移植自那份开源规则式 AI；`[用户]` = 用户 2026-09-28 补的。


@dataclasses.dataclass(frozen=True)
class Style:
    """规则式的**风格旋钮**（路线图 A4）：默认 = 现役规则式，**逐位不变**。

    **为什么要有它**：A6 只量出一种"可避免"（单点翻盘 14.7%），A6b 量出输局最大的一类是
    「2-3」胶着局 —— 两条都指向同一个可能的弱点：**训练里只有单一对手风格**
    （50% 的局打规则式）。而"打不打得过规则式"与"换个风格还打不打得过"是两件事。
    所以这三个旋钮的真正用途是**造新尺子**（`tools/style_profile.py`），
    让"鲁棒性"变成可测量的东西 —— 顺带也让 A4 那条臂有对手可换。

    三个旋钮都只动**门槛**，不动规则本身（不动枚举、不动纪律）。
    """
    name: str = "normal"
    #: 对手「一手走完」的风险到多少才必须拦（越高越不管；`DANGER` 是原值）
    danger: float = DANGER
    #: 台面主点 > 它才动炸（越低越爱炸；原值 10）
    bomb_rank: int = 10
    #: 领出时是否"留炸"（False = 残局计划里也先动炸）
    hold_fire: bool = True


NORMAL = Style()
#: 炸侠：几乎见牌就炸、残局先动炸
BOMB = Style(name="bomb", bomb_rank=4, hold_fire=False)
#: 龟：只在对手**非常**可能一手走完时才拦，几乎不主动动炸
HOLD = Style(name="hold", danger=0.8, bomb_rank=13)

#: 名字 -> 风格。训练与尺子都按名字取，**别在别处再写一份字典**。
STYLES = {s.name: s for s in (NORMAL, BOMB, HOLD)}


def _lead(obs, acts, pool, style: Style = NORMAL) -> Optional[int]:
    """**领出**（台面是空的，我赢着这一手）。"""
    ally = ally_of(obs.seat)

    # [源 1] 整手就是一个牌型 -> 一把走完
    for i, m in enumerate(acts):
        if m is not None and set(m.cards) == set(obs.hand):
            return i

    hands = hand_partition(obs.hand, obs.level)
    # [源 2] 还剩 2~3 手、且既有火力又有非火力 -> 先走非火力那手（留炸）
    # `style.hold_fire=False`（炸侠）反过来：这一手先动炸
    if 2 <= len(hands) <= SOURCE2_MAX_HANDS and any(map(is_fire, hands))             and not all(map(is_fire, hands)):
        i = (_cheapest_fire(acts, obs.hand, obs.level) if not style.hold_fire
             else _planned_plain(acts, hands, obs.hand, obs.level))
        if i is not None:
            return i

    # [源 3] 喂队友：队友剩 1 张 -> 出最小单张；剩 2 张 -> 出最小对子
    if obs.left[ally] == 1:
        i = _of_kinds(acts, (meld.SINGLE,), obs.hand, obs.level)
        if i is not None:
            return i
    if obs.left[ally] == 2:
        i = _of_kinds(acts, (meld.PAIR,), obs.hand, obs.level)
        if i is not None:
            return i

    # [源 4] 优先三带二（一次消 5 张）
    #
    # ⚠️ 但**不拆自己的炸优先于这条**（用户 2026-09-29 报的）。实测剩下那 37 例
    # 「本可不拆却拆了」全是这个形状：为了凑三带二从自己炸里抽 3 张，
    # 而不拆的出路明明有（炸本身、对子、单张）。所以拆炸时不走捷径，
    # 让它落到后面的规则去（后面那几条都过 `_prefer_intact`）。
    i = _of_kinds(acts, (meld.TRIPLE_PAIR,), obs.hand, obs.level)
    if i is not None and not breaks_bomb(obs.hand, acts[i]):
        return i

    # [源 5 + 用户] 卡对手：领出的**张数不等于他剩的张数**，他就一手走不完。
    # 按风险从高到低挑那个最该卡的对手。⚠️ 这条比原来「剩 1 张出对子、
    # 剩 2 张出单张」更一般 —— 3/5/6 张同样可能一手走完（用户补的那条）。
    for opp in sorted(opponents(obs.seat), key=lambda s: -finish_risk(obs.left[s], pool)):
        if finish_risk(obs.left[opp], pool) >= style.danger:
            i = _size_not(acts, obs.left[opp], obs.hand, obs.level)
            if i is not None:
                return i
            # ⚠️ **卡不住形状就卡强度**（2026-09-29 加）：出我**最大**的那手。
            # 不加这条会落到兜底 `_cheapest`（最小的合法牌）—— 而「对手只剩 1 张、
            # 我手里全是单张」时，那正好是**把牌权递给对手**：
            # 实测手牌 `3♠4♥5♦` 会出 `3♠`，而人出 `5♦`（压不过他接不上，牌权还在我手上）。
            # 只有在**形状卡不住**（`_size_not` 返回 None）时才走这里，所以不覆盖 [源 5] 的原意。
            i = _strongest_plain(acts, obs.hand, obs.level)
            if i is None:
                i = _strongest(acts, obs.hand, obs.level)
            return i
    return None


def _follow(obs, acts, pool, style: Style = NORMAL) -> Optional[int]:
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
    # ⚠️ 2026-09-29：这张表的**绝对胜率已过时**（拆炸那个 bug 修掉之后 C 变成
    # 84.2%），而「三条都在噪声内 ⇒ 不压队友免费」这个**相对**结论**修复后没复核**。
    # 「不压队友」的依据本来就不只是胜率（压队友 100%→1.2% 是行为事实），所以先不动。
    # 见 规则式尺子修正台账（已归档到 `master` 分支） §6.3。
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
    if danger >= style.danger:
        i = _cheapest_plain(acts, obs.hand, obs.level)
        if i is not None:
            return i
        i = _cheapest_fire(acts, obs.hand, obs.level)
        return i if i is not None else _pass(acts)

    # [源 2] 不危险：压最小的**普通牌**（火力留着）
    i = _cheapest_plain(acts, obs.hand, obs.level)
    if i is not None:
        return i
    # [源 3] 台面主点 > 门槛 且我凑得出炸 -> 才动炸（`style.bomb_rank` 越低越爱炸）
    if obs.table_rank > style.bomb_rank:
        i = _cheapest_fire(acts, obs.hand, obs.level)
        if i is not None:
            return i
    return _pass(acts)


def rule_choose(obs, acts, style: Style = NORMAL) -> int:
    """规则式选择，返回 `acts` 的下标。**两条链的唯一实现。**

    `style`（A4）：只动几个门槛（见 `Style`）。默认 `NORMAL` ⇒ 与加这个参数之前逐位相同。
    """
    pool = unseen_pool(obs)
    i = _follow(obs, acts, pool, style) if obs.table else _lead(obs, acts, pool, style)
    return _cheapest(acts, obs.hand, obs.level) if i is None else i


def rule_policy(style: Style = NORMAL):
    """工厂：返回一个和 `greedy_policy` 同形状的策略 `(obs, acts, hist) -> 下标`。

    只吃 `obs`（公开信息）—— **拿不到明牌**，与那条红线一致。

    `style`：见 `Style`（A4 的对手风格）；也可以直接传名字（`rule_policy("bomb")`）。
    """
    if isinstance(style, str):
        style = STYLES[style]

    def policy(obs, acts, hist=None) -> int:
        return rule_choose(obs, acts, style)
    return policy
