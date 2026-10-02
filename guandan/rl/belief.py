"""**对手手牌信念**：当场抽均匀粒子 + 按**历史行为事件**重新加权。

## 为什么不是"粒子滤波"（第一版踩过，记在这儿）

第一版写成时间上的粒子滤波：出牌时把"没把这些牌算在他手上"的粒子**杀掉** ⇒
ESS 立刻掉到 0（K=12 几下就全死）。改成"修复式"（把牌从错误座位换过来）之后 ESS 回到 100%，
**但命中率反而比当场重抽的均匀基线更差**（27.6% vs 44.7%）：因为每次修复都要从该座位
**随机**换一张出去 ⇒ 几十步以后粒子被随机化得比"重新抽一遍"还糟（**扩散**）✗。

⇒ 正确做法是分开：**粒子永远按当前的公开信息均匀抽**（不扩散），
**行为信息全部装在权重里** —— 而历史事件要用的"当时手牌"可以精确重建：
`当时手牌 = 当前手牌 ∪ 该家此后打出的牌`（全是公开信息）。

## 两条行为先验（可解释、不需要另训模型）

- `alpha`：他**过牌**、而重建出的手牌**本来能压** ⇒ ×`alpha`（默认 0.2）
  —— 「能压却过」合法但不常见（软证据，不是硬约束）；
- `beta`：他打出的那一手**不是**手上最便宜的能压选择 ⇒ ×`beta`（默认 0.3）
  —— 规则式自己就偏爱最便宜的那一手（`rule_policy._cheapest`），先验与真实对手对齐。

⚠️ **只此一份口径**：能不能压用 `meld.melds_from` + `meld.beats`，不另写第二份实现。
"""
from __future__ import annotations

import random

from guandan.sim import features, meld, rules

ALPHA_PASS = 0.2            # 能压却过牌
BETA_NOT_CHEAPEST = 0.3     # 有更便宜的选择却没用


def unseen_ids(obs) -> list:
    """没露面的牌（真实 card id）：整副牌 − 我的手牌 − 各家已出的牌。"""
    seen = set(obs.hand)
    for s in rules.SEATS:
        seen |= set(obs.played[s])
    return [cid for cid, _idx in features._DECK if cid not in seen]


def cheapest_beating(hand, table, level):
    """`hand` 里**最便宜**的能压 `table` 的一手（(张数, 点数) 字典序）；压不过返回 None。"""
    if table is None:
        return None
    best = None
    for m in meld.melds_from(set(hand), level):
        if not meld.beats(m, table):
            continue
        key = (len(m.cards), m.rank)
        if best is None or key < best[0]:
            best = (key, m)
    return None if best is None else best[1]


def can_beat(hand, table, level) -> bool:
    return cheapest_beating(hand, table, level) is not None


class Belief:
    """一批**按当前公开信息均匀抽**的粒子 + 一组权重（权重装行为信息）。"""

    def __init__(self, obs, k: int = 24, rng=None, alpha: float = ALPHA_PASS,
                 beta: float = BETA_NOT_CHEAPEST):
        self.level = obs.level
        self.my_seat = obs.seat
        self.k = k
        self.alpha, self.beta = alpha, beta
        self.rng = rng or random.Random(0)
        self.others = [s for s in rules.SEATS if s != obs.seat]
        self.hands = []
        for _ in range(k):
            pool = unseen_ids(obs)
            self.rng.shuffle(pool)
            h, i = {}, 0
            for s in self.others:
                n = obs.left[s]
                h[s] = set(pool[i:i + n])
                i += n
            self.hands.append(h)
        self.w = [1.0] * k
        self.n_events = 0
        self.dead = 0

    # ---------------------------------------------------------------- 加权
    def reweight(self, events, window: int = None) -> None:
        """按历史行为事件重新加权。

        `events`：按时间顺序的 `(seat, table_meld, chosen_meld_or_None, played_after)`，
        其中 `played_after` 是**该事件之后**该家打出的所有牌（用来重建"当时的手牌"）。
        `window`：只看最近这么多条（省时间；默认全看）。
        """
        evs = events[-window:] if window else events
        for p in range(self.k):
            w = 1.0
            for seat, table, chosen, played_after in evs:
                if seat == self.my_seat or table is None:
                    continue
                hand_then = set(self.hands[p].get(seat, ())) | set(played_after)
                if chosen is None:                      # 他过了
                    if self.alpha != 1.0 and can_beat(hand_then, table, self.level):
                        w *= self.alpha
                elif self.beta != 1.0:                  # 他打了，但可能不是最便宜的
                    cheap = cheapest_beating(hand_then, table, self.level)
                    if cheap is not None and len(cheap.cards) < len(chosen.cards):
                        w *= self.beta
                if w == 0.0:
                    break
            self.w[p] = w
        self.n_events += 1
        if sum(self.w) <= 0:
            self.dead += 1
            self.w = [1.0] * self.k

    # ---------------------------------------------------------------- 读数
    def ess(self) -> float:
        tot = sum(self.w)
        if tot <= 0:
            return 0.0
        return (tot * tot) / sum(x * x for x in self.w)

    def holder_probs(self, card: int) -> dict:
        """这张牌在每家手上的概率（权重归一后的边际）。"""
        tot = sum(self.w) or 1.0
        return {s: sum(w for w, h in zip(self.w, self.hands) if card in h.get(s, ())) / tot
                for s in self.others}

    def predict_holder(self, card: int):
        pr = self.holder_probs(card)
        return max(pr, key=pr.get) if pr else None

    def sample(self, n: int = 1) -> list:
        """按权重抽 n 个粒子（搜索用）。"""
        tot = sum(self.w)
        if tot <= 0:
            return [{s: set(c) for s, c in self.hands[0].items()} for _ in range(n)]
        out = []
        for _ in range(n):
            r, acc, pick = self.rng.random() * tot, 0.0, self.k - 1
            for i, w in enumerate(self.w):
                acc += w
                if r <= acc:
                    pick = i
                    break
            out.append({s: set(c) for s, c in self.hands[pick].items()})
        return out
