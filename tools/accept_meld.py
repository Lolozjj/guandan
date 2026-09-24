"""第一层验收（spec §6）：拿真实对局验证牌型引擎。

跑法：
    .venv/Scripts/python.exe -m tools.accept_meld

四项：
  ① 真实着法必须能枚举出来        ② 谁压谁逐条对齐
  ③ 不变量（不需要真值）          ④ 逢人配专项

失败会打印「哪一局、第几手、真实出的是什么」，便于定位。

**这一层只能证伪、不能证明**：55 局真实记录里没出现过的牌型（例如某些张数的
炸弹、某些位置的逢人配替牌）枚举错了也照样全绿。要证明得靠单测（tests/test_meld_*）。

与 plan 的偏离（Task 4~6 实装后签名漂移，逐条在此记明）：
  - `tools/decision_points.py` 的 `Snapshot` **没有 card_type 字段**（那是
    game_log 的 PlayRec 才有的）。诊断信息里的 card_type 由 `_card_type()` 从
    `g.plays[i]` 现取 —— `decision_points()` 每手一个快照、顺序与 plays 一一对应。
  - `net.cards` 显式 import，不再借道 `meld.cards` 这个间接属性。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from net import cards
from net.sim import meld
from tools.decision_points import decision_points
from tools.game_log import load_games


@dataclass
class Result:
    name: str
    total: int = 0
    bad: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bad

    def report(self, limit: int = 5) -> str:
        if self.ok:
            return f"[OK]   {self.name}  {self.total} 项全过"
        lines = [f"[FAIL] {self.name}  {len(self.bad)}/{self.total} 项不过"]
        lines += [f"       {b}" for b in self.bad[:limit]]
        if len(self.bad) > limit:
            lines.append(f"       …还有 {len(self.bad) - limit} 条")
        return "\n".join(lines)


def _stronger(a, b) -> bool:
    """同一组牌符合多个牌型时，a 是否比 b 更「强」。"""
    ca, cb = meld.bomb_class(a), meld.bomb_class(b)
    if (ca is None) != (cb is None):
        return ca is not None                 # 炸弹类优先
    if ca is not None and cb is not None and ca != cb:
        return ca > cb
    return a.kind > b.kind


def as_meld(ids, level):
    """把一组**具体**的牌判成一个 Meld；判不出返回 None。

    这里必须是精确集合 —— 参数就是那一手真实的牌，没有"代表牌"问题。

    ⚠️ **同一组牌可能符合多个牌型，必须取最强的那个：**
    5 张同花连续的牌**同时**是顺子(kind 4)与同花顺(kind 9)，
    `melds_from` 两条都产出。取第一个的话：
      - 真实打出的同花顺会被当成顺子
      - 桌面上的同花顺会被低估成顺子 -> legal_moves 会放进本该压不过的顺子
    游戏自己也把它叫同花顺（card_type 9）。
    """
    best = None
    for m in meld.melds_from(list(ids), level):
        if sorted(m.cards) != sorted(ids):
            continue
        if best is None or _stronger(m, best):
            best = m
    return best


def shape(m):
    """比较用的**形状**键：(牌型, 张数, 主点数)。**不含具体是哪几张牌。**

    为什么必须要形状键而不是精确牌组：`melds_from` 每个形状只给一个**代表**，
    而两手牌可能打出同一个形状的不同具体牌（手里同时有 ♠ 与 ♥ 两套同顶端的
    同花顺、同点数 5 张里挑哪 4 张做炸、两副牌的同名牌……）。用精确牌组比会把
    它们误报成「枚举不出」，而那正是本验收要抓的失败类的**假阳性**版本。

    炸弹归一：引擎用 `BOMB` 承载 4~10 张，而游戏协议用 card_type 8 表 4~5 张、
    10 表 6 张 —— 比较时必须先把这两种 kind 归一，否则 6 张炸会假红。
    """
    kind = meld.BOMB if m.kind in (meld.BOMB, meld.BOMB6) else m.kind
    return (kind, m.size, m.rank)


def _card_type(g, i) -> str:
    """第 i 手服务器自己标的 card_type（仅供报错信息用）。

    `Snapshot` 不带这个字段（decision_points 刻意不把「牌型」的概念引进来，
    否则与 net/sim/meld.py 循环依赖）。但报错时它有用：手牌/真实牌是真值，
    card_type 是游戏自己的判据，两边摆一起就能分辨「我们枚举漏了牌型」还是
    「这一手数据本身怪」。快照与 `g.plays` 顺序一一对应（对不上时
    `decision_points()` 会直接 raise），下标可以直取；越界返回 '?'。
    """
    return g.plays[i].card_type if i < len(g.plays) else "?"


def check_real_moves(games=None) -> Result:
    """① 每一手真实出的牌，必须在 legal_moves 里。

    真实着法一定是合法的（游戏自己认过），所以不在里面 = 我们错了。
    **局限：55 局只能证伪，不能证明**（spec §6①）。
    """
    r = Result("① 真实着法可枚举")
    games = load_games() if games is None else games
    for g in games:
        if not g.settle:
            continue
        for i, s in enumerate(decision_points(g)):
            r.total += 1
            table = None
            if s.table:
                table = as_meld(s.table, s.level)
                if table is None:
                    r.bad.append(f"{g.t0:%m-%d %H:%M} 打{s.level} 第{i}手 "
                                 f"桌面牌本身判不出牌型 {sorted(s.table)}")
                    continue
            # 两步：先看真实那一手**本身**能不能判出牌型（精确集合，能抓牌型缺口，
            # 例如「王当三带二的对子」那种）；再比**形状**是否在候选里
            # （形状比而非精确集合，因为枚举只给代表，见 shape() 的说明）。
            real = as_meld(s.actual, s.level)
            if real is None:
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 打{s.level} 第{i}手 座位{s.seat} "
                    f"真实出的 {sorted(s.actual)} 本身判不出牌型"
                    f"（card_type={_card_type(g, i)}）")
                continue
            moves = meld.legal_moves(s.hand, table=table, level=s.level)
            if not any(shape(m) == shape(real) for m in moves):
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 打{s.level} 第{i}手 座位{s.seat} "
                    f"真实出的 {sorted(s.actual)}（{shape(real)}，"
                    f"card_type={_card_type(g, i)}）不在候选里"
                    f"（手牌 {len(s.hand)} 张，"
                    f"桌面 {sorted(s.table) if s.table else '空'}，"
                    f"候选 {len(moves)} 个，形状集 {sorted({shape(m) for m in moves})}）")
    return r


def _bomb_pairs(g):
    """同一轮内「炸弹 A 之后又出了炸弹 B」的证据对。

    轮次边界判据与 tools/decision_points.py **完全一致**：
    同一座位又出牌、或队友接风（桌面主人上一手已出完）。

    ⚠️ **不要用 PlayRec.nxt 判** —— 服务器算下一手时会跳过已出完的座位，
    所以 nxt 指到队友既可能是接风、也可能只是在跳过。详见 Task 3 的原理说明。

    这条**独立于 legal_moves**：它只从出牌序列推「谁大」，所以 ① 全绿它仍可能红。

    ⚠️ 产出的是**两组牌 ID**（`(a_ids, b_ids)`），不是 PlayRec：
    plan 里写的是 `pairs.append((table, p))`，但 `table`/`p` 都是 PlayRec，
    调用方 `check_beats_from_records` 拿到的 `a_ids` 会被当牌组用，
    直接 `TypeError: 'PlayRec' object is not iterable`。
    """
    pairs = []
    table = None
    table_seat = None
    prev_left = None
    for p in g.plays:
        if table is not None:
            partner = (table_seat + 2) % 4
            # 与 tools/decision_points.py 同一判据。**不能用 nxt**：
            # 服务器算 nxt 时会跳过已出完的座位，会把它误判成接风。
            is_new_lead = (p.seat == table_seat
                           or (p.seat == partner and prev_left == 0))
            if not is_new_lead:
                pairs.append((table.cards, p.cards))
        table = p
        table_seat = p.seat
        prev_left = p.left
    return pairs


def check_beats_from_records(games=None) -> Result:
    """② 炸弹层级：真实对局里「炸弹 A 被炸弹 B 压掉」的证据必须逐条成立。

    专门验用户口述的炸弹顺序（4炸<5炸<同花顺<6炸<7炸<8炸<天王炸）。
    注意「同花顺夹在 5炸与 6炸之间」那半边**用户口述时数据没覆盖**（spec §2.1）
    —— 跑出来的条数要报出来，是 0 条就说明这段仍未验到。

    ⚠️ 桌面那一手用 `as_meld` 取**最强**解释：一手 5 张同花连续的牌在这条里
    算同花顺（炸弹），不算顺子。这与游戏自己的判据一致（card_type 9）。
    """
    r = Result("② 炸弹层级（谁压谁）")
    games = load_games() if games is None else games
    for g in games:
        if not g.settle:
            continue
        for a_ids, b_ids in _bomb_pairs(g):
            a = as_meld(a_ids, g.trump)
            b = as_meld(b_ids, g.trump)
            if a is None or b is None:
                continue
            if meld.bomb_class(a) is None and meld.bomb_class(b) is None:
                continue                      # 不是炸弹对，本检查不管
            r.total += 1
            if not meld.beats(b, a):
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 打{g.trump} {sorted(a_ids)}"
                    f"（{shape(a)}）-> {sorted(b_ids)}（{shape(b)}）"
                    f" 但 beats() 说压不过")
    return r


def check_invariants() -> Result:
    """③ 不变量：纯逻辑，不需要真值。"""
    r = Result("③ 不变量")

    r.total += 1
    if meld.legal_moves([], table=None, level=2):
        r.bad.append("空手牌不该有候选")

    r.total += 1
    hand = [16 + 5, 32 + 5]                     # 5♠ 5♥
    moves = meld.legal_moves(hand, table=None, level=9)
    if not moves:
        r.bad.append("领出时必须至少有一个候选")
    if any(m.size == 0 for m in moves):
        r.bad.append("候选里不该有「过」")

    r.total += 1
    for m in moves:
        if len(set(m.cards)) != len(m.cards):
            r.bad.append(f"候选 {m.cards} 里有重复牌")

    r.total += 1
    big = [16 + 5, 16 + 5 + 256, 32 + 5, 32 + 5 + 256]
    for m in meld.melds_from(big, level=9):      # 四张 5 的炸
        if m.kind == meld.BOMB and m.size == 4:
            if len(set(m.cards)) != 4:
                r.bad.append("四张炸里有重复牌 ID")
    return r


def check_wildcard() -> Result:
    """④ 逢人配专项。"""
    r = Result("④ 逢人配")
    level = 5
    wilds = [32 + 5, 32 + 5 + 256]               # ♥5 / ♥5(二副)
    # ⚠️ 必须用**非级牌**的点数（6，不是 5）。用 5 的话两张 ♥5 本身就是两张 5，
    # n=2 时天然就是四炸、wild_used == 0，而「天然优先」要求它必须是 0 ——
    # 断言 wild_used > 0 会与本任务的硬规格直接冲突。
    naturals = [64 + 6, 48 + 6, 16 + 6]          # ♦6 ♣6 ♠6

    for n in (0, 1, 2):
        r.total += 1
        hand = naturals + wilds[:n]
        bombs = [m for m in meld.melds_from(hand, level=level)
                 if m.kind == meld.BOMB]
        if n == 0:
            if bombs:
                r.bad.append("没有逢人配时，三张 6 不该有炸弹")
        elif not any(m.wild_used > 0 for m in bombs):
            r.bad.append(f"{n} 张逢人配时应当能补出炸弹")

    r.total += 1
    hand = [32 + 5, 15, 271, 14]                 # ♥5 + 大王 + 大王(二副) + 小王
    for m in meld.melds_from(hand, level=level):
        if m.kind == meld.BOMB and m.size == 4:
            if not all(cards.parts(c)[0] in (14, 15) for c in m.cards):
                r.bad.append("王炸里混进了逢人配")
    return r


CHECKS = [check_real_moves, check_beats_from_records, check_invariants,
          check_wildcard]

# ①② 吃语料，③④ 是纯逻辑 —— 后两个的签名里根本没有 games（见 brief 的
# Interfaces 一栏：`check_invariants() -> Result`）。
# plan 的 main() 一律写 `fn(games)`，对 ③④ 会 `TypeError: check_invariants()
# takes 0 positional arguments but 1 was given`。这里按各自签名分开调，
# 不为了让循环好看而给纯函数塞一个被忽略的参数。
_LOG_FED = (check_real_moves, check_beats_from_records)


def main() -> int:
    games = load_games()
    print(f"载入对局 {len(games)} 局，其中有结算的 "
          f"{sum(1 for g in games if g.settle)} 局\n")
    dropped = sum(g.unparsed for g in games)
    # R7：日志解析失败的行必须**可见**，不能静默 ——
    # 这个数字不为 0 就说明语料有缺失，下面四项结论都要打折看。
    print(f"解析失败的行合计 unparsed = {dropped}"
          + ("  <- 非 0！语料有缺失，结论要打折看" if dropped else ""))
    failed = 0
    for fn in CHECKS:
        res = fn(games) if fn in _LOG_FED else fn()
        print(res.report())
        failed += 0 if res.ok else 1
    print()
    if failed:
        print(f"验收不通过：{failed} 项红")
        return 1
    print("验收全绿 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
