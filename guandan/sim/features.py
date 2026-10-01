"""`Observation` -> **显式特征**（路线图 A2）：把人也在算的几个量直接喂给网络。

**为什么要有它**（A1b-1 的负结果给的直接理由）：扩候选但不重训**掉了 3.08pp** ——
冻结的 Q 在「用哪几张」上**系统性选错**，说明它在动作之间的**分辨率不够**。
而"最少几手能走完 / 还有几个炸 / 对家剩几张"这些量**人也在算**，
网络却得从 108 维手牌 multi-hot 里自己解组合优化。给网络眼睛，是提高分辨率最直接的办法。

⚠️ **全部只用「公开信息 + 我的手牌」**（`Observation` 里根本没有对手手牌），
所以这一块不碰「不许明牌泄漏」那条纪律 —— 与 `encode_state` 同一个窄口。

⚠️ 本模块里的 `unseen_pool` / `hand_partition` 是**从 `rl/rule_policy.py` 搬下来的**
（2026-09-30）：它们本来就在算 sim 层的量，而 A2 要在 `env.encode_state` 里用 ——
让 `sim/` 反过来 import `rl/` 会把依赖方向弄反。`rule_policy` 现在从本模块 import
（老名字照旧可用，测试与 `tools/` 不用改）。
"""
from __future__ import annotations

import functools
from collections import Counter

from guandan.capture import cards
from guandan.sim import meld, rules

#: 这一块的维度：2（手数）+ 15（未见牌点数分布）+ 4（剩牌对比）+ 3（火力）+ 3（残局）
EXTRA_DIM = 2 + 15 + 4 + 3 + 3          # = 27


#: **A2 的开关（默认关，2026-09-30）。**
#:
#: 为什么默认关：A2 的臂在 3 万局以内**没有任何超过 R11 的迹象**（中期判读见
#: `plans/2026-09-30-beat-70-roadmap.md` §十一），而这一块特征**每局常驻 30 ms**
#: （吞吐 19 → 10.9 局/秒）—— 在"涨分要靠局数"的账本上，这是**双重不利**。
#:
#: ⚠️ 关着的时候 `env.encode_state` 只是**不写**那一块（恒为 0）：
#: `STATE_DIM` 不变、架构不变、老权重与 A2 权重都装得上，而**开销是零**。
#: 打开它（`ENABLED = True`）就完全恢复 A2 —— 单变量、随时可复验。
ENABLED = False


#: **动作侧后果特征**的开关（2026-10-01）。关掉时**宽度不变**（那 3 维恒为 0）——
#: 架构、缓存键、老存档装载全都不受影响，而**信息层面**等价于"没有这个特征"
#: ⇒ 才有资格当"特征有用吗"的对照。
#:
#: 为什么要这个开关：2026-10-01 夜核实发现，昨晚那三条"干净 ε"的臂**本来就带着特征在跑**
#: （存档列数 1001 = (727,146)），所以"特征有没有用"这个问题**当时答不了**；
#: 而它们的动作边际（0.106~0.111）与"特征关"的臂（0.100~0.108）完全重叠 ⇒ 先验很弱。
#: 跑一条把这个开关关掉的臂，才给出干净对照（顺带它还是一条**多样成员**，能进 soup）。
CONSEQUENCE_ENABLED = True


#: `{点数: nat_values(点数)}` 的**全程预算表**（只有 15 个键，import 时算一次）。
#: `fire_counts` / `_take_run` 都在热路径上，别在那儿反复调 `nat_values`。
_NAT: dict = {idx: meld.nat_values(idx) for idx in range(1, 16)}


#: 整副牌的 `(牌 ID, 点数)`（**import 时算一次**）—— `unseen_pool` 在热路径上，
#: 不能每调一次都拿 `range(1, 334)` 去 `is_card` + `parts` 扫一遍。
_DECK: tuple = tuple((cid, cards.parts(cid)[0])
                     for cid in range(1, cards.MAX_ID + 1) if cards.is_card(cid))


def unseen_pool(obs) -> dict:
    """**还没露面**的牌：点数 -> 张数。整副 108 张减去我的手牌与四家出过的牌。

    另外三家的手牌只能是这个池子里的子集 —— 所以估「他能不能一手走完」
    只需要它，**不需要看明牌**。

    ⚠️ 与老实现（`range(1,334)` 逐个查表）的区别只有一处：**全被看见的点数会留下 0**
    （老实现直接不出这个键）。所有消费方都是 `pool.get(点数, 0)` 或 `c >= per`，
    所以等价 —— 但这一条写在注释里，免得下次有人当它是 bug。
    """
    pool: dict = {}
    for _cid, idx in _DECK:
        pool[idx] = pool.get(idx, 0) + 1
    seen = set(obs.hand)
    for s in rules.SEATS:
        seen |= set(obs.played[s])
    for c in seen:
        pool[cards.parts(c)[0]] -= 1
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


def _nat_cards(avail: dict, v: int, natmap: dict = None) -> list:
    """自然值 `v` 现在还有哪些牌可用（A 的两个头由 `meld.nat_values` 一处定义）。

    ⚠️ **四张同点的点数不参与** —— 那是一个炸，不许拿去凑顺子
    （「不许拆自己的炸」这条纪律对分区同样成立，见 `rule_policy.breaks_bomb`）。

    `natmap` 是 `{点数: nat_values(点数)}` 的预算表（A2 加的）：不传就自己算 ——
    `encode_state` 是热路径，**别在这里反复调 `nat_values`**。
    """
    out = []
    for idx, cs in avail.items():
        if len(cs) >= 4:
            continue
        if v in (natmap[idx] if natmap is not None else meld.nat_values(idx)):
            out += cs
    return out


def _by_nat(avail: dict, natmap: dict) -> dict:
    """自然值 -> 可用的牌（**一次建表**，供 `_take_run` 逐窗口查）。"""
    out: dict = {}
    for idx, cs in avail.items():
        if not cs or len(cs) >= 4:
            continue
        for v in natmap[idx]:
            out.setdefault(v, []).extend(cs)
    return out


def _take_run(avail: dict, length: int, per: int, natmap: dict = None):
    """找一个「`length` 个连续自然值、每个至少 `per` 张」的组合并**就地取走**。

    取走是就地改 `avail` —— 同一个牌 ID 因此不可能被两个组合用到。
    从**最小的起点**开始找（同样能成，先花小的）。

    ⚠️ 2026-09-30（A2）**改了实现但没改语义**：原来对每个窗口都重建一遍
    `_nat_cards`（`13 个起点 × length 个值 × 全部点数`），实测 ~150 µs/手，
    而它现在进了 `encode_state` 的热路径。改成**每个窗口只查一次预算表** `_by_nat`。
    等价性用「2155 手（1655 真实 + 500 随机）的 `hand_partition` 输出逐字节相同」钉住
    （见 `tests/test_a2_features.py`）。
    """
    if natmap is None:
        natmap = _NAT
    by = _by_nat(avail, natmap)
    for start in range(1, 15 - length + 1):
        window = range(start, start + length)
        if all(len(by.get(v, ())) >= per for v in window):
            out = [c for v in window for c in by[v][:per]]
            for c in out:
                avail[cards.parts(c)[0]].remove(c)
            return out
    return None


def hand_partition(hand, level) -> list:
    """把手牌**贪心**拆成尽量少的几手 —— 估「我还要几手才能走完」。

    口径：**先抽序列类**（钢板/三连对 6 张、顺子 5 张 —— 一次消得多的先抽），
    剩下的照开源 AI 的 `utils.partition`：三张尽量配一对凑成三带二（一次消 5 张）；
    对子、单张各算一手；4 张以上同点算炸。

    ⚠️ **不做最优划分**（那是指数级的），**也不认逢人配** ——
    所以它**高估**手数，只当粗估用。要问「整手是不是一个牌型」（一把走完），
    用 `melds_from` 精确判，别用这个。

    ⚠️ 2026-09-29：**原来不认顺子/三连对/钢板**，手里有顺子时把 5 张算成 5 张单牌
    ⇒ 系统性高估手数 ⇒ 规则式的 [源 2] 几乎不触发。序列类那一段是补上的。
    """
    avail: dict = {}
    for c in hand:
        avail.setdefault(cards.parts(c)[0], []).append(c)
    natmap = {idx: meld.nat_values(idx) for idx in avail}    # A2：预算一次，别在循环里反复算
    groups: list = []
    for length, per in ((2, 3), (3, 2), (5, 1)):      # 钢板 → 三连对 → 顺子
        while True:
            g = _take_run(avail, length, per, natmap)
            if g is None:
                break
            groups.append(g)
    singles, pairs, triples, bombs = [], [], [], []
    for idx, cs in avail.items():
        if not cs:
            continue                                  # 已被序列类抽空
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
    groups += list(bombs)
    while triples:
        t = triples.pop(0)
        groups.append(t + pairs.pop(0) if pairs else t)   # 三张优先配一对 -> 三带二
    groups += pairs + [[c] for c in singles]
    return [m for m in (meld.as_meld(g, level) for g in groups) if m is not None]


def breaks_bomb(hand, m) -> bool:
    """这一手是不是**从自己手里的炸里挖牌**：某个点数我握着 ≥4 张，
    这一手只用了其中 1~3 张 ⇒ 那个炸废了。

    ⚠️ 豁免只有一种：**把整个炸当炸打出去**（`is_bomb` 且四张全用掉）。
    不能只看「四张全用掉」—— 逢人配会骗人：`8♠8♥8♣8♦J♠`（8♥ 当 J 去配对子）
    里真被吃掉的 8 只有 3 张，8 炸没了，但按牌面点数看 4 个 8 全在。

    （原在 `rl/rule_policy.py`；2026-09-30 搬到 `sim/` —— 动作侧特征要用它，
    让 `sim/` 反向 import `rl/` 会把依赖方向弄反。老名字在 `rule_policy` 里照旧可用。）
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


#: 每个**当前动作**额外带的后果特征维度（见 `consequence_features`）
CONSEQUENCE_DIM = 3


def consequence_features(hand, m, level, groups=None) -> tuple:
    """**动作侧**的后果特征（3 维）：这一手对"我计划好的牌"做了什么。

    为什么必须是动作侧：A2 加的是**状态**特征 —— 同一局面里对**每个候选都一样**，
    所以**不可能**帮 argmax 分辨候选（实测边际 0.110 → 0.104，没动）。
    这三个量**逐个候选不同**，正是比较时缺的那点信息。

        1. 正好打出一手**计划好的牌**（`hand_partition` 的某一组）—— 效率最高的一手
        2. **拆了一手计划好的牌**（部分用了某一组）—— 半途而废
        3. **拆了自己的炸**（某点数 ≥4 张却只用了 1~3 张）—— 纪律问题（「白炸」的来源）

    `groups` 可传入复用：**同一个决策点的所有候选共用一次 `hand_partition`**
    —— 逐个候选都算一次会让编码慢一个数量级（那是热路径）。
    """
    if m is None:
        return (0.0, 0.0, 0.0)
    used = set(m.cards)
    if groups is None:
        groups = hand_partition(sorted(hand), level)
    uses = breaks_plan = 0.0
    for g in groups:
        cs = set(g.cards)
        if used == cs:
            uses = 1.0
            break
        if used & cs:
            breaks_plan = 1.0
    return (uses, breaks_plan, 1.0 if breaks_bomb(hand, m) else 0.0)


def fire_counts(hand, level) -> tuple:
    """我的火力**粗计**：`(自然炸个数, 有没有同花顺潜力, 有没有天王炸)`。

    ⚠️ **故意是粗的**（只看点数 ≥4、只看同花色 5 连）：`encode_state` 是热路径，
    这里**不能**调 `melds_from`（它占自对弈总耗时的 81%）。粗计只当"给网络的提示"。

    ⚠️ 2026-09-30（A2）：**一遍扫完**（原版扫两遍 + 每个点数都调 `nat_values`）。
    """
    ranks: dict = {}
    vals_by_suit: dict = {}
    n_joker = 0
    for c in hand:
        idx, suit, _deck = cards.parts(c)
        ranks[idx] = ranks.get(idx, 0) + 1
        if idx >= meld.JOKER_SMALL:
            n_joker += 1
        else:
            vals_by_suit.setdefault(suit, set()).update(_NAT[idx])
    bombs = sum(1 for idx, n in ranks.items()
                if idx < meld.JOKER_SMALL and n >= 4)
    flush = any(len(vals) >= 5
                and any(all(v in vals for v in range(s, s + 5))
                        for s in range(1, 11))
                for vals in vals_by_suit.values())
    return bombs, flush, n_joker >= 4


@functools.lru_cache(maxsize=4096)
def extra_features(obs) -> tuple:
    """`Observation` -> `EXTRA_DIM` 个 float。布局（`env.encode_state` 接在 700 维之后）：

        0        手数 / 27（`hand_partition` 的粗估 —— "我还要几手"）
        1        手数 <= 3（残局快到了）
        2..16    未见牌的点数分布（点数 1..15，各除以 8）
        17..20   剩牌对比：我−对家 / 我−先于我 / 我−下家 / 对家−对手里最少的那家，各 /27
        21..23   火力：自然炸个数 /4、有同花顺潜力、有天王炸
        24..26   残局信号：我 <= 11 / <= 8 / <= 5 张

    `lru_cache` 的理由：同一个 `Observation` 会被反复编码（打分、评测、影子模式），
    而这些都是**纯函数**；`Observation` 是 frozen dataclass ⇒ 可哈希。

    ⚠️ 相对座位索引按 `env._rel` 的定义：0 = 自己、1 = 先于我、2 = 对家、3 = 我的下家。
    这里用 `rules.PARTNER` / `rules.NEXT` 表达，**不复制那份索引算术**。
    """
    hand = tuple(sorted(obs.hand))
    parts = len(hand_partition(hand, obs.level)) if hand else 0
    pool = unseen_pool(obs)
    seat = obs.seat
    partner = rules.PARTNER[seat]
    prev = (seat + 1) % 4                     # 索引 1 = 先于我的那家
    nxt = rules.NEXT[seat]                    # 索引 3 = 我的下家
    left_me = obs.left[seat]
    left_partner = obs.left[partner]
    opp_min = min(obs.left[prev], obs.left[nxt])
    bombs, flush, joker_bomb = fire_counts(hand, obs.level)

    out = [parts / 27.0,
           1.0 if parts <= 3 else 0.0]
    out += [pool.get(i, 0) / 8.0 for i in range(1, 16)]
    out += [(left_me - left_partner) / 27.0,
            (left_me - obs.left[prev]) / 27.0,
            (left_me - obs.left[nxt]) / 27.0,
            (left_partner - opp_min) / 27.0]
    out += [bombs / 4.0, 1.0 if flush else 0.0, 1.0 if joker_bomb else 0.0]
    out += [1.0 if left_me <= 11 else 0.0,
            1.0 if left_me <= 8 else 0.0,
            1.0 if left_me <= 5 else 0.0]
    return tuple(out)
