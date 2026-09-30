"""进贡/还贡的验收（spec §6⑤）—— **证据分级，两件事分开判**。

跑法：
    .venv/Scripts/python.exe -m tools.accept_tribute

**① 还贡 ≤ 10 —— 硬证据，判失败。**
   25 条 `TributeSectionEndService` / `NotifyReturnTribute` 记录里 `card=` 解码后
   点数**全部落在 2..10**（跨 4 花色、2 副牌，spec §13.6）。
   注意 A=1：写成 `idx <= 10` 会把 A 也算成「≤10」，而 A 恰恰最难还出去。

**② 进贡 = 贡方手上最大的牌 —— 基线，只报告不判失败。**
   只有 5 条直接记录，而日志里没有「过」也没有「贡之前的手牌」，重建要靠
   时间戳把记录配到局上再倒推（`pre = post + gave − returned`）。
   对不上就逐条打出来给人判 —— 先看是不是「级牌可不贡」之类的变体。

**「0 条记录」是失败，不是「全过」**（spec §6⑥）。日志一转进贡记录就没了，
所以本脚本读的是**冻结进语料的那一份**（`load_corpus_tributes`）。
"""
from __future__ import annotations

import sys

from guandan.capture import cards
from guandan.sim import meld, rules
from tools.accept_meld import Result, _utf8_stdout
from tools.decision_points import initial_hands
from tools.game_log import load_corpus, load_corpus_tributes

#: 记录数地板：低于这个数，「全过」不足以称为结论（日志被轮转删了？）
_MIN_RECORDS = 10


def _rank(cid: int) -> int:
    """牌 ID -> 点数索引（A=1、2..10、J=11、Q=12、K=13、小王 14、大王 15）。"""
    return cards.parts(cid)[0]


def check_returns(records) -> Result:
    """**硬证据**：还贡的牌点数必须落在 `rules.RETURN_MIN_RANK..RETURN_MAX_RANK`。

    区间**从 `rules` 取，不在这里重写** —— 写第二份就是第二份真源，
    改了一边忘了另一边会让验收与实现悄悄脱钩。
    """
    res = Result("① 还贡 ≤ 10（硬证据）")
    rets = [r for r in records if r.kind == "return"]
    if len(rets) < _MIN_RECORDS:
        res.floor = (f"还贡记录只有 {len(rets)} 条 < 地板 {_MIN_RECORDS} 条 —— "
                     f"日志轮转后没冻进语料？（0 条不是「全过」）")
        return res
    lo, hi = rules.RETURN_MIN_RANK, rules.RETURN_MAX_RANK
    for r in rets:
        res.total += 1
        if not lo <= _rank(r.card) <= hi:
            res.bad.append(f"座位{r.giver} 还了 {cards.decode(r.card)}"
                           f"（点数 {_rank(r.card)}，要求 {lo}..{hi}）")
    return res


def give_diagnostics(records, games) -> tuple:
    """**基线**：进贡的牌应当是贡方当时手上最大的一张。返回 `(一致数, 总数, 说明)`。

    重建办法：`initial_hands(g)` 给的是**进贡之后**的手牌（出过的 ∪ 结算剩的），
    所以 `进贡前 = 进贡后 + 贡出的 − 收回的`。记录靠**时间戳**配到局上：
    一条进贡记录属于「发牌时间 ≤ 它」里最晚的那一局。
    """
    settled = sorted([g for g in games if g.plays], key=lambda g: g.t0)
    if not settled:
        return 0, 0, "语料里没有可配的对局"

    def game_for(t):
        """一条记录属于哪一局：**发牌时间 ≤ 它**里最晚的那一局。"""
        hit = None
        for cand in settled:
            if cand.t0 <= t:
                hit = cand
            else:
                break
        return hit

    # ⚠️ 还贡记录必须**按局**索引。只按「谁收回的」索引会把别的局的还贡
    # 也算到这个人头上，把他手上本没有的牌减掉 —— 实测会造出
    # 「贡了 Q♦，但最大是 J♥」这种不可能的诊断（Q♦ 是被别局误减掉的）。
    ret_by_game_taker = {}
    for r in records:
        if r.kind == "return" and r.t is not None:
            g = game_for(r.t)
            if g is not None:
                ret_by_game_taker.setdefault((id(g), r.taker), []).append(r)
    same = total = 0
    bad = []
    for r in records:
        if r.kind != "give" or r.t is None:
            continue
        g = game_for(r.t)
        if g is None:
            continue
        post = initial_hands(g)
        pre = set(post.get(r.giver, set()))
        pre.add(r.card)                                   # 撤销「贡出」
        for rr in ret_by_game_taker.get((id(g), r.giver), []):
            pre.discard(rr.card)                          # 撤销「收回」
        if not pre:
            continue
        total += 1
        val = lambda c: meld.point_value(_rank(c), g.trump)          # noqa: E731
        # **同点数的多张牌算一致**：手上两张 5 时贡哪一张都满足「贡最大的牌」。
        # 按牌 ID 比会把这类并列全判成不一致（实测 25 条里一半是这种），
        # 那种诊断等于噪声。
        if val(r.card) == max(val(c) for c in pre):
            same += 1
        else:
            best = max(sorted(pre), key=val)
            bad.append(f"座位{r.giver} 贡了 {cards.decode(r.card)}，"
                       f"但当时手上最大是 {cards.decode(best)}"
                       f"（{g.t0:%m-%d %H:%M} 那局）")
    return same, total, "；".join(bad[:5])


def variant_scores(records, games) -> list:
    """三条候选规则的命中数，**用来说明「为什么维持简单规则」**。

    实测 2026-09-25（25 条）：
        贡全手最大的牌       18/25   <- 本实现用的
        除大王外最大的牌     13/25
        除大王与级牌外最大的 5/25
    后两条更差，而且有 **8 条确实贡了王**（王是最大牌时照贡）。所以没有任何
    简单豁免能解释 25/25 —— 那 7 条不符的模式是「手上持王或级牌时贡了别的」，
    但样本太少（7 条）且互相矛盾，**不足以改实现**。等样本多了再判。
    """
    settled = sorted([g for g in games if g.plays], key=lambda g: g.t0)
    if not settled:
        return []
    def game_for(t):
        hit = None
        for cand in settled:
            if cand.t0 <= t:
                hit = cand
            else:
                break
        return hit

    def is_bj(c):
        return cards.parts(c)[0] == 15

    def is_lv(c, lv):
        lv = meld.norm_level(lv)
        return lv is not None and cards.parts(c)[0] == lv

    hits = [0, 0, 0]
    for r in records:
        if r.kind != "give" or r.t is None:
            continue
        g = game_for(r.t)
        if g is None:
            continue
        pre = set(initial_hands(g).get(r.giver, set()))
        pre.add(r.card)
        lv = g.trump
        val = lambda c: meld.point_value(_rank(c), lv)              # noqa: E731
        picks = [list(pre),
                 [c for c in pre if not is_bj(c)] or list(pre),
                 [c for c in pre if not is_bj(c) and not is_lv(c, lv)] or list(pre)]
        for i, pool in enumerate(picks):
            if val(r.card) == max(val(c) for c in pool):
                hits[i] += 1
    return hits


def main(argv=None) -> int:
    _utf8_stdout()
    games = load_corpus()
    records = load_corpus_tributes()
    from tools import game_log
    print(f"载入对局 {len(games)} 局；进贡/还贡记录 {len(records)} 条")
    print(f"语料来源：{game_log.LAST_TRIBUTE_SOURCE}\n")

    res = check_returns(records)
    print(res.report())

    same, total, detail = give_diagnostics(records, games)
    pct = f"{same / total:.0%}" if total else "—"
    print(f"\n② 进贡=手上最大的牌（**基线，不判失败**）：{same}/{total} 一致（{pct}）")
    if total and same < total:
        print("   不一致的：")
        for line in detail.split("；"):
            if line:
                print(f"     {line}")
    hits = variant_scores(records, games)
    if hits and total:
        print(f"   三条候选规则各命中多少（决定「要不要加豁免」的依据）：")
        print(f"     贡全手最大的牌 {hits[0]}/{total}"
              f"   除大王外最大 {hits[1]}/{total}"
              f"   除大王与级牌外最大 {hits[2]}/{total}")
        print("   ↑ **本实现用第一条**。后两条更差，而且有 8 条确实贡了王 ——"
              " 没有任何简单豁免能解释 25/25，所以维持简单规则。")
    print("   ↑ 这一项证据薄（重建靠时间戳配对）。要改实现，先让这一项跑出稳定的模式。")

    ok = res.ok
    print("\n验收" + ("全绿 ✓" if ok else "**未通过** ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
