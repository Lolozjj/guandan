"""动作空间探针（路线图 A1）：`melds_from` 的「每形状一条代表」丢掉了多少「用哪几张」？

`melds_from` 的契约是「**每个 `(kind, size, rank)` 只给一条代表**」。验收① 按**形状**比对，
所以它一直没暴露问题；但学策略时模型**只能在代表之间选**，选不了"这一手用哪几张牌"。

本探针不训练、不加载权重，只读语料 + 枚举，三个读数：

    M1 结构性   手写几组手牌，把「候选」与「手上真实存在的同形状牌组」摆在一起
    M2 真实对局 1655 手真实着法，按**精确牌张**是否出现在候选集合里
                （`tools/accept_meld.py` 的验收① 是**按形状**比的，比这宽松）
    M3 影响面   ① 每个候选有几个「同形状、不同花色组合」的替代
                ② 「逢人配被迫花掉」的决策点比例

判据（预登记在 `plans/2026-09-30-beat-70-roadmap.md` §三）：

    M2 >= 99%                        -> 动作空间**不是**瓶颈，本条关闭
    M2 < 95% 或 M3② > 5%             -> **是**瓶颈：A2（特征）之前先修动作空间

跑法：
    .venv/Scripts/python.exe -m tools.action_space_probe
"""
from __future__ import annotations

import collections
import itertools
import sys

from guandan.capture import cards
from guandan.console import utf8_stdout
from guandan.sim import meld
from tools.decision_points import decision_points
from tools.game_log import load_corpus

#: 只看这几个"同点数"的牌型 —— 序列类的「选哪几张」是另一回事（见文末说明）。
BASIC = (meld.SINGLE, meld.PAIR, meld.TRIPLE, meld.BOMB)

#: 预登记判据
M2_CLOSE = 0.99          # 精确牌张命中率 >= 这个数就关闭本条
M2_BOTTLENECK = 0.95     # < 这个数就是瓶颈
M3_FORCED_MAX = 0.05     # 被迫花逢人配的候选占比 > 这个数也是瓶颈


def _names(ids) -> str:
    return " ".join(cards.decode(c) for c in ids)


def suit_pool(hand, idx) -> list:
    """该点数手上所有牌（按牌 ID 排序）。"""
    return sorted(c for c in hand if cards.parts(c)[0] == idx)


def suit_multisets(hand, idx, k) -> set:
    """该点数手上**所有 k 张组合**的花色多重集（**副数不区分** —— 两副的同名牌等价）。

    这是「同形状、用不同花色组合」的计数口径：一个花色多重集 = 一种"用哪几张"的选择。
    ⚠️ 它是**下界**：不含「逢人配替掉某个点数」那一类跨点数的替换。
    """
    out = set()
    for combo in itertools.combinations(suit_pool(hand, idx), k):
        out.add(tuple(sorted(cards.parts(c)[1] for c in combo)))
    return out


# ------------------------------------------------------------------ M1

def m1_structural() -> None:
    """三组手写手牌，把问题摆出来（含一个**对照组**，证明探针能分辨真缺口与假缺口）。"""
    A = meld.cid_from_name
    cases = [
        ("① 级牌炸（打 5）：手上有 8 张 5，其中 2 张 ♥ = 逢人配",
         [A("S5"), A("S5", deck=2), A("C5"), A("C5", deck=2), A("D5"), A("D5", deck=2),
          A("H5"), A("H5", deck=2), A("S9")], 5),
        ("② 同一点数的三张单张（打 9）：♠5 ♥5 ♦5",
         [A("S5"), A("H5"), A("D5"), A("S9")], 9),
        ("③ 对照组：两套同顶端的同花顺（打 9）—— 枚举**应该**两套都给",
         [A("S5"), A("S6"), A("S7"), A("S8"), A("S9"),
          A("H5"), A("H6"), A("H7"), A("H8"), A("H9")], 9),
    ]
    for title, hand, level in cases:
        hand = sorted(hand)
        print(f"\n--- {title}")
        print(f"    手牌：{_names(hand)}")
        for m in meld.melds_from(hand, level):
            if m.kind not in BASIC:
                continue
            idx = cards.parts(m.cards[0])[0]
            alts = suit_multisets(hand, idx, m.size) if idx < meld.JOKER_SMALL else set()
            extra = ""
            if m.kind == meld.BOMB:
                wilds = [c for c in m.cards if meld.is_wild(c, level)]
                natural = [c for c in suit_pool(hand, idx) if not meld.is_wild(c, level)]
                if wilds:
                    extra = (f"   ⚠️ 含逢人配 {_names(wilds)}"
                             f"（手上非红桃的有 {len(natural)} 张，够 {m.size} 张 ⇒ 本可不花）"
                             if len(natural) >= m.size else
                             f"   （含逢人配 {_names(wilds)}，缺口中必须用它）")
            print(f"    {meld.describe_meld(m):8s} -> {_names(m.cards):22s}"
                  f" 同形状可选 {len(alts):2d} 种{extra}")


# ------------------------------------------------------------------ M2

def _suit_key(ids) -> tuple:
    """牌组的**花色多重集** —— 副数不区分（两副的同名牌完全等价）。

    用它把「只是换了副数」（无害）和「换了花色/点数」（材质差异，真缺口）分开。
    """
    return tuple(sorted(cards.parts(c)[1] for c in ids))


def m2_exact_hit(snaps) -> dict:
    """真实着法按**精确牌张**的命中率（验收① 只按形状比，这里更严）。

    miss 再分两类：
      - **仅副数不同**：候选里有同形状、同花色多重集的牌组 ⇒ 两副同名牌等价，**无害**
      - **材质差异**：连花色组合都对不上 ⇒ 模型学不到/出不了人家那一手，**真缺口**
    """
    r = collections.Counter()
    misses = collections.Counter()
    for s in snaps:
        table = None
        if s.table:
            table = meld.as_meld(s.table, s.level)
            if table is None:
                continue
        real = meld.as_meld(s.actual, s.level)
        if real is None:
            continue
        r["total"] += 1
        moves = meld.legal_moves(sorted(s.hand), table, s.level)
        if any(sorted(m.cards) == sorted(s.actual) for m in moves):
            r["exact"] += 1
            continue
        same_shape = [m for m in moves
                      if (m.kind, m.size, m.rank) == (real.kind, real.size, real.rank)]
        if not same_shape:
            r["neither"] += 1
            misses[(real.kind, real.size, "连形状都没有")] += 1
            continue
        if any(_suit_key(m.cards) == _suit_key(s.actual) for m in same_shape):
            r["deck_only"] += 1                    # 只差副数 —— 无害
            continue
        r["material"] += 1                         # 真缺口
        misses[(real.kind, real.size, "材质差异")] += 1
        if len(r.get("examples", [])) < 4:
            r.setdefault("examples", []).append(
                (f"{s.t:%m-%d %H:%M}", s.level, _names(s.actual),
                 _names(same_shape[0].cards)))
    r["misses"] = misses
    return r


# ------------------------------------------------------------------ M3

def m3_metrics(snaps) -> dict:
    """① 同形状替代数分布；② 逢人配被迫花掉的比例。"""
    alt_hist = collections.Counter()
    cand_total = 0
    forced = 0
    bomb_total = 0
    for s in snaps:
        level = s.level
        for m in meld.melds_from(sorted(s.hand), level):
            if m.kind not in BASIC or m.size > 4:
                continue
            idx = cards.parts(m.cards[0])[0]
            if idx >= meld.JOKER_SMALL:            # 天王炸没有"别的组合"这回事
                continue
            cand_total += 1
            alt_hist[len(suit_multisets(s.hand, idx, m.size))] += 1
            if m.kind == meld.BOMB:
                bomb_total += 1
                wilds = [c for c in m.cards if meld.is_wild(c, level)]
                natural = [c for c in suit_pool(s.hand, idx)
                           if not meld.is_wild(c, level)]
                if wilds and len(natural) >= m.size:
                    forced += 1
    return {"alt_hist": alt_hist, "cand_total": cand_total,
            "forced": forced, "bomb_total": bomb_total}


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    utf8_stdout()
    snaps = [s for g in load_corpus() if g.settle for s in decision_points(g)]
    print(f"语料：{len(snaps)} 个决策点（真实对局）\n")

    print("=" * 78)
    print("M1 结构性：候选 vs 手上真实存在的同形状牌组")
    print("=" * 78)
    m1_structural()

    print("\n" + "=" * 78)
    print("M2 真实对局：真实着法按**精确牌张**是否在候选里（验收① 是按形状比的）")
    print("=" * 78)
    m2 = m2_exact_hit(snaps)
    total = m2["total"]
    print(f"  可判定决策点 {total}")
    print(f"  精确命中       {m2['exact']:5d}  {m2['exact'] / max(1, total):6.2%}")
    print(f"  仅副数不同     {m2['deck_only']:5d}  {m2['deck_only'] / max(1, total):6.2%}"
          f"   （两副同名牌等价 ⇒ **无害**）")
    print(f"  **材质差异**   {m2['material']:5d}  {m2['material'] / max(1, total):6.2%}"
          f"   <- 换了花色组合，模型**表达不出来**")
    print(f"  连形状都没有   {m2['neither']:5d}  {m2['neither'] / max(1, total):6.2%}")
    for (kind, size, why), n in m2["misses"].most_common(8):
        print(f"      {why}：kind={kind} size={size} × {n}")
    for ex in m2.get("examples", [])[:3]:
        print(f"      例：{ex[0]} 打{ex[1]} 实际出 {ex[2]}；候选只有同形状的 {ex[3]}")

    print("\n" + "=" * 78)
    print("M3 影响面")
    print("=" * 78)
    m3 = m3_metrics(snaps)
    ct = m3["cand_total"]
    print(f"  候选总数 {ct}（单/对/三/4 炸）")
    print("  ① 同形状可选的花色组合数分布（1 = 没有任何选择余地）：")
    for k in sorted(m3["alt_hist"]):
        n = m3["alt_hist"][k]
        print(f"       {k:2d} 种：{n:6d} 个候选  {n / max(1, ct):6.2%}")
    solid = sum(n for k, n in m3["alt_hist"].items() if k >= 2)
    print(f"     ⇒ 有 ≥2 种选择的候选占 {solid / max(1, ct):.2%}")
    fr = m3["forced"] / max(1, m3["bomb_total"])
    print(f"  ② 炸弹候选 {m3['bomb_total']} 个，其中**本可不花却带了逢人配**的 "
          f"{m3['forced']} 个 = {fr:.2%}")

    print("\n" + "=" * 78)
    hit = m2["exact"] / max(1, total)
    material = m2["material"] / max(1, total)
    if hit >= M2_CLOSE and fr <= M3_FORCED_MAX:
        print(f"判据：M2 {hit:.2%} >= {M2_CLOSE:.0%} 且 M3② {fr:.2%} <= {M3_FORCED_MAX:.0%}"
              f" ⇒ **动作空间不是瓶颈**，A1 关闭")
    else:
        print(f"判据：M2 {hit:.2%}（**材质差异 {material:.2%}**）／M3② {fr:.2%}"
              f" ⇒ **动作空间是瓶颈**"
              f"（M2 < {M2_BOTTLENECK:.0%} 或 M3② > {M3_FORCED_MAX:.0%}）"
              f"，A2 之前先修它")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
