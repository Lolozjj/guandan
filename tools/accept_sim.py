"""第二层验收（spec §6①③）：拿 55 局真实对局回放**整局**，验模拟器的出牌轮转。

与 Plan 1 的 `tools/accept_meld.py` 的分工：
  - `accept_meld` 验**牌型引擎**（这一手是不是合法牌型、谁压谁）
  - 本脚本验**牌局引擎**（轮转对不对、接风对不对、什么时候终局、名次对不对）

跑法：
    .venv/Scripts/python.exe -m tools.accept_sim

**回放的起点是「进贡之后」**：`decision_points.initial_hands` 用的是「出过的 ∪ 结算剩的」，
而进贡/还贡在**第一手之前**就完成了，所以重建出来的正是进贡后的手牌。
**因此本脚本不验进贡**（那在 `tools/accept_tribute.py` 里单独验，用的另一份证据）。

比对口径是**形状**（`(kind, size, rank)`）而不是牌张集合 —— 因为 `melds_from` 的契约是
「每个形状一条代表」（见 `guandan/sim/meld.py` 的 docstring）。真人打出的可能是 ♥5♥5，
枚举给的代表可能是 ♠5♠5，形状相同就是对的。**这与 `accept_meld` 的验收①同口径。**

**两路独立校验**（不经过 `meld.py` 的枚举）：
  1. 服务器在每条出牌消息里报的 `LeftCardLen`（这一手之后该家还剩几张）——
     拿它对我们算出来的手牌记账。实测 53 局 / 1655 手逐手相符。
  2. 名次：出完顺序与日志的 `Rank` 逐局比。

**这一层的能力边界要说清**：判「这一手能不能出」用到的 rank，两侧都出自同一批函数，
所以**看得见轮转/接风/终局/记账的错，看不见「rank 读弱了」这类错**。
（`rank` 读弱是 `melds_from` 的已知偏差，见交付文档里的 Plan 3 闸门。）
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

from guandan.sim import meld, rules
# `utf8_stdout` 直接复用 accept_meld 那份，**不复制**：它就是为「GBK 控制台下
# 打不出 ✓ 会让全绿的脚本返回 1」写的（Plan 1 踩过）。复制一份就是第二份真源，
# 本仓库为「副本会漂」吃过亏。
from guandan.console import utf8_stdout
from tools.accept_meld import Result
from tools.decision_points import initial_hands
from tools.game_log import load_corpus

#: 语料地板：能回放的对局少于这个数，「全过」不足以称为结论。
#: 与 `tools/accept_meld.py` 的 `_MIN_SETTLED` 同口径。
_MIN_REPLAYABLE = 20

#: 「过」的推断上限。**不是 3。** 日志里没有「过」，所以一手真实出牌前面可能挂着
#: 好几次「过」：三家过完之后清桌，同一个人可能**先过了、又成了领出者**
#: （接风就是这么发生的：主人出完，队友轮到时先过、清桌后再由他领出）。
#: 实测 55 局里最长的连过是 5，这里留到 6 并在超限时报出上下文。
_MAX_INFERRED_PASSES = 6


def shape(m) -> tuple:
    """`(kind, size, rank)` —— 牌型的「形状」。比较候选时用这个口径，不用牌张。"""
    return (m.kind, m.size, m.rank)


@dataclass
class ReplayResult:
    steps: int
    passes_inferred: int
    hand: rules.Hand


def replay(g, record=None) -> ReplayResult:
    """把一局真实对局走一遍。**认不出的局面直接炸**，不跳过。

    `record(kind, hand)` 每**将要**走一步时调一次（`kind` 是 `"pass"` 或 `"play"`），
    传进来的是**动手之前**的 `hand` —— 影子模式的验收要拿它在每个决策点取真值
    （手牌、桌面、谁要不起、各家剩几张）。**回放循环只有这一份**（本仓库为
    「副本会漂」吃过亏），所以真值侧复用它、不另写一份。

    ⚠️ **日志里没有「过」。** 所以不能写成「轮到日志的座位了就出手」——
    真人可能先过。判据必须是：**轮到日志的座位了，而且他这一手确实压得过桌面**
    （或桌面已清）。否则就把他记成「过」，继续往后走。

    这不是小事：**接风就是靠这个走出来的** —— 主人出完，队友轮到时先过（压不过），
    三家过完清桌，**再由队友领出**。只判「轮到谁」的话，接风那一手会被判成
    「压不过桌面」而炸掉（实测：只判轮到的版本在 55 局里挂 34 局）。
    """
    if not g.plays:
        raise ValueError(f"{g.t0} 这一局没有任何出牌记录")
    hands = initial_hands(g)
    hand = rules.Hand(hands=[set(hands[s]) for s in rules.SEATS],
                      level=g.trump, turn=g.plays[0].seat)
    passes = 0
    for i, rec in enumerate(g.plays):
        m = meld.as_meld(list(rec.cards), g.trump)
        if m is None:
            raise rules.IllegalPlay(
                f"{g.t0} 第 {i} 手：座位{rec.seat} 的真实着法判不出牌型 "
                f"card_type={rec.card_type} cards={rec.cards}")
        guard = 0
        while hand.turn != rec.seat or (hand.table is not None
                                        and not meld.beats(m, hand.table)):
            if guard > _MAX_INFERRED_PASSES:
                raise rules.IllegalPlay(
                    f"{g.t0} 第 {i} 手：连推了 {guard} 个「过」仍走不到座位{rec.seat} "
                    f"出手（我们停在 {hand.turn}，桌面"
                    f"{'清' if hand.table is None else meld.describe_meld(hand.table)}）")
            if hand.table is None and hand.turn != rec.seat:
                raise rules.IllegalPlay(
                    f"{g.t0} 第 {i} 手：轮到领出的是座位{hand.turn}，"
                    f"但日志说这手是座位{rec.seat}出的 —— 领出者判错了")
            if record is not None:
                # **动手之前**的快照：这一刻就是「轮到 hand.turn 出牌」的决策点
                record("pass", hand)
            hand.pass_turn(hand.turn)
            passes += 1
            guard += 1

        # ① 真实着法必须能在候选里找到（按形状比）
        cands = {shape(x) for x in hand.actions(rec.seat) if x is not None}
        if shape(m) not in cands:
            raise rules.IllegalPlay(
                f"{g.t0} 第 {i} 手：座位{rec.seat} 真实出了 {meld.describe_meld(m)}，"
                f"但候选里没有这个形状（候选 {len(cands)} 个）—— 枚举漏了")
        if record is not None:
            record("play", hand)
        hand.play(rec.seat, m)

        # ② **独立于牌型枚举的一路校验**：服务器自己在每条出牌消息里报了这一手之后
        # 该家还剩几张（`LeftCardLen`）。拿它对我们算出来的手牌记账 ——
        # 牌解析错、漏扣牌、手牌重建错，都会在这里露出来，而这一路不经过 meld.py。
        # 实测 53 局 / 1655 手逐手相符（2026-09-25）。
        if len(hand.hands[rec.seat]) != rec.left:
            raise rules.IllegalPlay(
                f"{g.t0} 第 {i} 手：座位{rec.seat} 出完后我们算他剩 "
                f"{len(hand.hands[rec.seat])} 张，服务器说 {rec.left} 张 —— "
                f"手牌记账错了")

    if not hand.is_over():
        raise rules.IllegalPlay(
            f"{g.t0} 牌都出完了却没判成终局（出完 {len(hand.order)} 家）")
    return ReplayResult(steps=len(g.plays), passes_inferred=passes, hand=hand)


def check_replay(games) -> Result:
    """跑全部能回放的局：① 轮转能走到尾、② 手牌记账、③ 名次与日志一致。

    顺带把「我们算的升级点 vs 日志的 Upgrade」的分布写进 `note` ——
    **只报告、不判失败**：spec §13.4 实测「双上 -> 3」只对了 13/24 局、
    另 11 局是 4，而那一格依赖升级/过A 的字段，按 §13.1 不在 Plan 2 内。
    报出来是为了让下一层（Plan 3）看得见这个偏差有多大。
    """
    res = Result("① 真实对局回放（轮转 / 接风 / 终局 / 名次 / 手牌记账）")
    playable = [g for g in games if g.settle and g.plays]
    if len(playable) < _MIN_REPLAYABLE:
        res.floor = (f"能回放的对局只有 {len(playable)} 局 < 地板 {_MIN_REPLAYABLE} 局 —— "
                     f"「全过」不足以称为结论（语料被轮转删了？）")
        return res
    pairs = Counter()
    passes_all = 0
    for g in playable:
        res.total += 1
        try:
            r = replay(g)
        except (rules.IllegalPlay, ValueError) as e:
            res.bad.append(str(e))
            continue
        passes_all += r.passes_inferred
        logged = g.settle["Rank"]
        for i, seat in enumerate(r.hand.order):
            if logged[seat] != i + 1:
                res.bad.append(
                    f"{g.t0} 出完顺序 {r.hand.order} 与日志名次 {logged} 对不上")
                break
        else:
            ours = rules.points(r.hand.ranks())
            up = (g.settle.get("UpgradeInfo") or {}).get("Upgrade")
            pairs[(ours, up)] += 1

    dist = "、".join(f"我们{p}/日志{u}：{n}局" for (p, u), n in sorted(pairs.items()))
    same = sum(n for (p, u), n in pairs.items() if p == u)
    nl = chr(10)          # 用 chr(10)：这个文件里的 "\n" 被 shell 的 heredoc 吃过一次
    res.note = (f"       升级点核对（**只报告不判失败**，spec §13.4）：{same}/{res.total} 局一致"
                + nl + f"       {dist}"
                + nl + "       手牌记账逐手拿服务器的 LeftCardLen 校过：全对"
                       "（这一路不经过 meld.py，是独立校验）"
                + nl + f"       推断出的「过」共 {passes_all} 次"
                       "（日志里不记「过」，只能推断 —— 是**推断**不是观测）"
                + nl + "       能力边界：判「能不能出」的 rank 两侧同源，"
                       "所以**看得见轮转/接风/终局/记账的错，看不见「rank 读弱了」**")
    return res


def main(argv=None) -> int:
    utf8_stdout()          # 不调这个，GBK 控制台下「全绿 ✓」那行会抛 UnicodeEncodeError
    games = load_corpus()
    print(f"载入对局 {len(games)} 局，其中有结算的 "
          f"{sum(1 for g in games if g.settle)} 局\n")
    from tools import game_log
    print(f"语料来源：{game_log.LAST_SOURCE}\n")
    res = check_replay(games)
    print(res.report())
    ok = res.ok
    print("\n验收" + ("全绿 ✓" if ok else "**未通过** ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
