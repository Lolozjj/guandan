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

终审修复（Task 8 之后）：
  - `as_meld` / `_stronger` 已**上收进 `guandan/sim/meld.py`**（`meld.as_meld` /
    `meld.strongest`）。理由不是「DRY 好看」：`as_meld` 是生产推理链
    （spec §3/§8.1 `guandan/advice/advise.py -> legal_moves(hand, table=…)`）**必须**用的
    原语 —— 桌上的 `Play.cards` 是一串牌 ID，喂 `beats()` 前得先判成带 rank 的
    Meld。留在离线验收目录里会让 Plan 3/4 走错方向地 import，或者再抄出第三份；
    而老 YOLO 适配层（`live/rules.py`，已删）里那份**逐字相同**的副本已经漂过一次（`9♣10♣J♣Q♣+♥2`
    应当是同花顺而不是顺子）。
  - `Result.ok` 现在要求 `total > 0`，②另记「炸弹 vs 炸弹」的条数，
    ①② 另有语料地板 —— 见各自 docstring：这三处都是「什么都没跑也算全绿」
    那个家族的防线（spec §6⑥）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from guandan.console import utf8_stdout
from guandan.sim import meld
from tools.decision_points import decision_points
from tools import game_log
from tools.game_log import load_corpus

# 语料地板：结算局少于这个数，①/② 的「全过」不足以称为结论。
# 与 tests/test_game_log.py 的 `len(settled) >= 20` 同口径 —— 那边早就定了这条线，
# 验收脚本这边却一直没设，于是 8 局没结算的语料也能打印「验收全绿 ✓」。
_MIN_SETTLED = 20


@dataclass
class Result:
    """一项检查的结果（①~④ 每项一个）。**失败要能说清是哪几条、为什么。**"""

    name: str                  # 这一项叫什么（打印用）
    total: int = 0             # 查了多少条 —— **必须 > 0**，否则算失败（见 `ok`）
    bad: list = field(default_factory=list)   # 不过的明细（空 = 全过）
    floor: str = ""            # 非空 = 语料地板没到（见 _corpus_floor）
    note: str = ""             # 附加说明行（如 ② 的条数拆分），成功失败都打

    @property
    def ok(self) -> bool:
        # total == 0 是**失败**：一项都没检查到却报「0 项全过」，就是假绿。
        return not self.bad and not self.floor and self.total > 0

    def report(self, limit: int = 5) -> str:
        if self.ok:
            lines = [f"[OK]   {self.name}  {self.total} 项全过"]
        else:
            lines = [f"[FAIL] {self.name}"]
            if self.floor:
                lines.append(f"       {self.floor}")
            if self.total == 0:
                lines.append("       **一项都没检查到**（total == 0）—— "
                             "「全过」是假绿，不许当成通过")
            if self.bad:
                lines.append(f"       {len(self.bad)}/{self.total} 项不过")
                lines += [f"       {b}" for b in self.bad[:limit]]
                if len(self.bad) > limit:
                    lines.append(f"       …还有 {len(self.bad) - limit} 条")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


def _corpus_floor(games) -> str:
    """语料地板：结算局太少就返回一句红字（空串 = 达标）。

    为什么要有：`check_real_moves` / `check_beats_from_records` 只遍历 `g.settle`
    的局。语料里结算局一少（实测本机已有 8 局没结算，日志 2 天轮转还会更糟），
    它们检查的条数就趋近 0，最后一行照样打「验收全绿 ✓」。
    """
    settled = sum(1 for g in games if g.settle)
    if settled < _MIN_SETTLED:
        return (f"结算对局只有 {settled} 局 < 地板 {_MIN_SETTLED} 局 —— "
                f"语料太少，本项的结论不成立")
    return ""


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
    否则与 guandan/sim/meld.py 循环依赖）。但报错时它有用：手牌/真实牌是真值，
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
    games = load_corpus() if games is None else games
    for g in games:
        if not g.settle:
            continue
        for i, s in enumerate(decision_points(g)):
            r.total += 1
            table = None
            if s.table:
                table = meld.as_meld(s.table, s.level)
                if table is None:
                    r.bad.append(f"{g.t0:%m-%d %H:%M} 打{s.level} 第{i}手 "
                                 f"桌面牌本身判不出牌型 {sorted(s.table)}")
                    continue
            # 两步：先看真实那一手**本身**能不能判出牌型（精确集合，能抓牌型缺口，
            # 例如「王当三带二的对子」那种）；再比**形状**是否在候选里
            # （形状比而非精确集合，因为枚举只给代表，见 shape() 的说明）。
            real = meld.as_meld(s.actual, s.level)
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

    专门验用户口述的炸弹顺序（完整阶梯：
        4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 9炸 < 10炸 < 天王炸，
    9炸/10炸 是自然延伸、只有「存在」证据，见 guandan/sim/meld.py 的
    `_BOMB_CLASS_BY_SIZE`）。注意「同花顺夹在 5炸与 6炸之间」那半边**用户口述时
    数据没覆盖**（spec §2.1）—— 跑出来的条数要报出来，是 0 条就说明这段仍未验到。

    ⚠️ **`total` 只说明「有这么多对出过炸弹」，不等于阶梯被验了这么多条。**
    只要一对里有一边是炸弹就算一条，而其中大部分是「炸弹压普通牌型」——
    那验的是「炸弹 > 普通」，跟 4炸 与 5炸 谁大毫无关系。
    所以这里**分开计数**并把两个数都打出来，另外单列出真正涉及 9炸/10炸 的条数
    （阶梯顶端最缺证据的那两格）。**不要**把 total 单独当成阶梯覆盖率读。

    ⚠️ 桌面那一手用 `meld.as_meld` 取**最强**解释：一手 5 张同花连续的牌在这条里
    算同花顺（炸弹），不算顺子。这与游戏自己的判据一致（card_type 9）。
    """
    r = Result("② 炸弹层级（谁压谁）")
    games = load_corpus() if games is None else games
    bomb_vs_bomb = 0            # 两边都是炸弹 —— 只有这些在验阶梯
    normal_vs_bomb = 0          # 一边炸弹、一边普通 —— 只验「炸弹压普通」
    with_9, with_10 = 0, 0      # 阶梯顶端（9炸 / 10炸）真正被覆盖到的条数
    for g in games:
        if not g.settle:
            continue
        for a_ids, b_ids in _bomb_pairs(g):
            a = meld.as_meld(a_ids, g.trump)
            b = meld.as_meld(b_ids, g.trump)
            if a is None or b is None:
                continue
            ca, cb = meld.bomb_class(a), meld.bomb_class(b)
            if ca is None and cb is None:
                continue                      # 不是炸弹对，本检查不管
            r.total += 1
            if ca is not None and cb is not None:
                bomb_vs_bomb += 1
            else:
                normal_vs_bomb += 1
            # 9炸/10炸 都是 BOMB 承载的「张数」；只数真正的炸弹那一侧
            for m, cls in ((a, ca), (b, cb)):
                if m.kind == meld.BOMB and cls is not None:
                    with_9 += 1 if m.size == 9 else 0
                    with_10 += 1 if m.size == 10 else 0
            if not meld.beats(b, a):
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 打{g.trump} {sorted(a_ids)}"
                    f"（{shape(a)}）-> {sorted(b_ids)}（{shape(b)}）"
                    f" 但 beats() 说压不过")
    r.note = (
        f"     其中 炸弹 vs 炸弹 {bomb_vs_bomb} 条（**只有这些在验阶梯**）；"
        f"炸弹 vs 普通 {normal_vs_bomb} 条（只验「炸弹能压普通」）\n"
        f"     阶梯顶端覆盖：涉及 9炸 {with_9} 条、10炸 {with_10} 条"
        f"（0 条 = 这两格仍未被真实证据验到）")
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

    # R21：原 brief 这里拿 [♥5, 大王, 大王(二副), 小王] 去找 **4 张炸**，可那手牌
    # 只有 **3 张王**（天王炸要 4 张），`m.size == 4` 永远不成立 —— 那句断言一次都
    # 没执行过，等于什么都没验。换成下面两条**能失败**的：
    #   1) 3 张王 + 1 张逢人配：逢人配不能当王，不该凑出天王炸
    #   2) 4 张王 + 1 张逢人配：应当**恰好**一个天王炸，且里面不能混逢人配
    # 第 2 条的 `len(jb) != 1` 是结构性判据：引擎一旦产不出天王炸、或产出多个，
    # 它立刻红，不依赖任何巧合。
    wild5 = 32 + 5                               # ♥5，与上面 wilds[0] 同一张
    three_jokers = [wild5, 14, 270, 15]          # 逢人配 + 小王×2 + 大王
    r.total += 1
    if any(meld.bomb_class(m) == meld.CLASS_JOKER_BOMB
           for m in meld.melds_from(three_jokers, level=level)):
        r.bad.append("3 张王 + 1 张逢人配 不该能出天王炸（逢人配不能当王）")

    four_jokers = [wild5, 14, 270, 15, 271]      # 逢人配 + 四张王
    r.total += 1
    jb = [m for m in meld.melds_from(four_jokers, level=level)
          if meld.bomb_class(m) == meld.CLASS_JOKER_BOMB]
    if len(jb) != 1:
        r.bad.append(f"4 张王 + 1 张逢人配 应当恰好有一个天王炸，实得 {len(jb)} 个")
    elif any(meld.is_wild(c, level) for c in jb[0].cards):
        r.bad.append("天王炸里混进了逢人配")
    return r


CHECKS = [check_real_moves, check_beats_from_records, check_invariants,
          check_wildcard]

# ①② 吃语料，③④ 是纯逻辑 —— 后两个的签名里根本没有 games（见 brief 的
# Interfaces 一栏：`check_invariants() -> Result`）。
# plan 的 main() 一律写 `fn(games)`，对 ③④ 会 `TypeError: check_invariants()
# takes 0 positional arguments but 1 was given`。这里按各自签名分开调，
# 不为了让循环好看而给纯函数塞一个被忽略的参数。
_LOG_FED = (check_real_moves, check_beats_from_records)


def main(games=None) -> int:
    """跑完四项。返回进程退出码（0 = 全绿）。

    `games` 只为测试留的口子：不传就 load_games()。终审修复的回归测试要构造
    「语料太少 / 一项都查不到」的假语料来证明这个门会红 —— 不注入的话没法构造。
    """
    utf8_stdout()
    games = load_corpus() if games is None else games
    settled = sum(1 for g in games if g.settle)
    print(f"载入对局 {len(games)} 局，其中有结算的 {settled} 局\n")
    # 换源必须可见：不能让人以为在验实时日志、其实验的是很久以前的快照。
    print(f"语料来源：{game_log.LAST_SOURCE}")
    print()
    dropped = sum(g.unparsed for g in games)
    # R7：日志解析失败的行必须**可见**，不能静默 ——
    # 这个数字不为 0 就说明语料有缺失，下面四项结论都要打折看。
    print(f"解析失败的行合计 unparsed = {dropped}"
          + ("  <- 非 0！语料有缺失，结论要打折看" if dropped else ""))
    floor = _corpus_floor(games)
    if floor:
        # 语料地板没到：明着说，别靠下面四项各自的 total 去暗示。
        print(f"[WARN] {floor}")
    failed = 0
    for fn in CHECKS:
        res = fn(games) if fn in _LOG_FED else fn()
        if fn in _LOG_FED:
            # ①② 吃语料，地板挂在它们身上：结算局太少时它们**不许**声称成功。
            res.floor = floor
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
