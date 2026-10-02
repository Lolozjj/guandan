"""把一局**自对弈**打成人类可读的战报 —— 看模型到底怎么打的。

为什么要有这个：所有判据都是数字（胜率、浪费率、动作边际），但「牌打得好不好」
最终得靠会打牌的人看几局。影子日志（`runtime/shadow.jsonl`）记的是**真实对局**里
模型的建议，这个脚本看的是**自对弈** —— 两者互补。

用法：
    .venv/Scripts/python.exe -m tools.show_game                       # 挑最新 best.pt
    .venv/Scripts/python.exe -m tools.show_game --seed 61
    .venv/Scripts/python.exe -m tools.show_game --quiet               # 只看牌不看分
    .venv/Scripts/python.exe -m tools.game_viewer --seed 61           # 图形版（可点击）
    .venv/Scripts/python.exe -m tools.show_game --opp-kind rule        # 模型 vs 规则式对手
    .venv/Scripts/python.exe -m tools.game_viewer --opp-kind rule      # 同上，图形版

⚠️ `--opp-kind`：**对手那一队不打分**。拿模型的 Q 去给规则式出的牌打分是**编数据**
（那个分数不代表规则式在想什么），比不打分有害得多 —— 所以对手帧的候选是空的，
画面/战报上写明「谁出的」。默认 None = 四家都是网络（老行为，一行不变）。

⚠️ **权重口径复用 `guandan/advice/advise.py::resolve_weights`**，不另写一份 ——
不然「面板用哪个模型」与「这个脚本看哪个模型」会漂。

⚠️ **重放逻辑只此一份**：`replay_game()` 产出 `Frame` 列表，文字版
（`render_text`）与图形版（`tools/game_viewer.py`）都吃它。两个视图各写一遍
重放，迟早会漂（本仓库为此吃过亏）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import torch

from guandan.advice import advise
from guandan.capture.cards import names_sorted
from guandan.sim import env, meld, rules
from guandan.capture.state import Play
from guandan.rl.eval import (bomb_opportunity, is_upgraded_wild, is_wasted_bomb,
                            is_wasted_wild, wild_opportunity)
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.policies import greedy_policy
from guandan.rl.rule_policy import rule_choose
from guandan.rl.selfplay import OPP_KIND_CN, OPP_KINDS   # 对手类型的唯一产地，不另写一份
from guandan.console import utf8_stdout

#: 与 `advice/reasons.py` 同口径：队友剩 ≤ 这么多张时解释为「喂队友」
FEED_LEFT = 2

#: 「擦浪费」开关 —— **看局工具与面板同一条口径**（`GUANDAN_TIDY`）。
#: 模块级读一次：走局时不再反复查环境。
_TIDY_MODE = advise.tidy_mode()
_TIDY_MARGIN = advise.tidy_margin()

TEAM_NAME = {0: "甲", 1: "乙"}


def seat_label(s: int) -> str:
    """自对弈回放用**绝对座位 + 队**（不用「我/上家」那种相对叫法）——
    复盘时要一直认得出是同一家。"""
    return f"座位{s}({TEAM_NAME[rules.TEAM[s]]})"


def policy_name(meta: dict, seat: int) -> str:
    """**这个座位是谁在打** —— 模型 / 规则式 / 贪心。

    为什么要标出来：回放里最容易搞混的就是这件事 —— 看半天以为在复盘模型，
    其实对面那一队是规则式（或者反过来）。`replay_game` 只有三种可能：
    - 没给 `opp_kind`（默认）：**四家全是模型**（自对弈）
    - 给了：`opp_team` 那一队走固定对手（`OPP_KIND_CN` 里的名字），另一队是模型

    ⚠️ **没有「随机」** —— `replay_game` 只支持 `greedy` / `rule` 两种对手
    （`OPP_KINDS` 就这两个）。想加的话得先把它接进 `--opp-kind`。
    """
    kind = (meta or {}).get("opp_kind")
    if kind and rules.TEAM[seat] == (meta or {}).get("opp_team"):
        return OPP_KIND_CN.get(kind, kind)
    return "模型"


def cards_text(ids, level) -> str:
    return " ".join(names_sorted(ids, level))


@dataclass
class Frame:
    """**一步棋的全部现场** —— 文字视图与图形视图共用它（免得两处重放漂）。"""
    step: int                       # 第几手（1 起）
    seat: int                       # 谁出手
    chosen: object                  # 选中的 Meld；None = 过
    level: int                      # 这一局的级别（打几）
    turn: object                    # 出手**之前**轮到谁（= seat；终局帧是 None）
    table: object                   # 出手**之前**台面上待压的 `Play`；None = 领出
    hands: dict                     # 出手之前四家手牌 {seat: set}
    played: dict                    # 出手之前每家已出过的 `Play` 列表
    passes: set                     # 出手之前谁已经要不起
    candidates: list = field(default_factory=list)   # [(q, meld|None)] 按 q 降序
    wasted: bool = False            # 这一手是不是「白炸」：本来能用普通牌压却出了炸
    why: str = ""                   # **为什么**（解释型建议，`advice/reasons.py`；只解释不改动作）
    over: bool = False              # 这一步之后是不是终局
    opp: object = None              # 这手是**固定对手**出的：`kind` 字符串；None = 模型自己的


def replay_game(path: str = None, seed: int = 7, level: int = None,
                top: int = 3, opp_kind: str = None, opp_team: int = 1) -> tuple:
    """打一局并记下每一步。返回 `(meta, frames)`。

    `opp_kind=None`（默认）= 自对弈：四家都是网络。给了 `"greedy"` / `"rule"` 时，
    **`opp_team` 那一队**走固定对手（与训练里 `--opp-kind` 同一套实现），
    另一队是模型 —— 这就是「看模型在像人的对手面前怎么打」。

    ⚠️ 对手那一队的帧 `candidates` 是**空的**（`opp` 字段记着是谁出的）：
    拿模型的 Q 去给对手的牌打分是编数据。

    `frames` 末尾**多一帧「终局」**（`turn=None`、没有候选），用来显示四家的
    全部出牌与最终名次。
    """
    if opp_kind is not None and opp_kind not in OPP_KINDS:
        raise ValueError(f"认不出的对手类型：{opp_kind!r}（只有 {OPP_KINDS}）")
    if opp_team not in (0, 1):
        raise ValueError(f"opp_team 只能是 0/1，给的是 {opp_team!r}")
    p = path or advise.resolve_weights()
    if not p:
        raise SystemExit("找不到权重：设 GUANDAN_WEIGHTS，或把权重放到 models/best.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = QNet()
    load_state(net, ck["net"] if isinstance(ck, dict) else ck)
    net.eval()
    lv = level if level is not None else 8
    e = env.GuandanEnv(seed=seed)
    e.reset(level=lv)

    played = {s: [] for s in rules.SEATS}
    frames = []
    obs = e.observe()
    step = bombs = chance = waste = 0
    wild_chance = wild_waste = wild_up = 0
    while not e.done:
        acts = e.legal()
        seat = e.hand.turn
        hist = env.encode_history(e.hand, seat)
        # 对手那一队走固定策略：**不做前向、不打分**（打分就是编数据）
        is_opp = opp_kind is not None and rules.TEAM[seat] == opp_team
        if is_opp:
            q = None
            i = (rule_choose(obs, acts) if opp_kind == "rule"
                 else greedy_policy(obs, acts, hist))
        else:
            q = q_values(net, obs, acts, hist)
            i = int(q.argmax())
            # 「擦浪费」开关（`GUANDAN_TIDY`）：**看局工具必须与面板同一条口径**，
            # 否则你用这个窗口看到的还是"原样乱炸"，会以为修了没效果（2026-10-02 踩到）。
            if _TIDY_MODE != "off":
                from guandan.rl.tidy import tidy_index
                i = tidy_index(q, acts, i, has_table=e.hand.table is not None,
                               bombs=_TIDY_MODE != "wilds", wilds=_TIDY_MODE != "bombs",
                               margin=_TIDY_MARGIN)
        m = acts[i]
        why = ""
        if not is_opp:
            try:
                from guandan.advice.reasons import explain
                why = explain(obs, acts, i, feed_left=FEED_LEFT)
            except Exception:            # 解释失败绝不许影响战报本身
                why = ""
        step += 1
        if m is not None and m.is_bomb:
            bombs += 1
        # 「白炸」用 guandan/rl/eval.py 的**唯一判定**，不另写一份。
        # ⚠️ 只数**模型自己**的着法 —— 对手的白炸也记进来的话，
        # 战报末尾那个数就变成两个策略混在一起，没法读
        hit_chance = (not is_opp) and bomb_opportunity(e.hand.table, acts)
        was_wasted = is_wasted_bomb(e.hand.table, acts, m)
        wild_wasted = is_wasted_wild(e.hand.table, acts, m)
        wild_upgraded = is_upgraded_wild(e.hand.table, acts, m)
        if hit_chance:
            chance += 1
            waste += int(was_wasted)
        if (not is_opp) and wild_opportunity(e.hand.table, acts):
            wild_chance += 1
            wild_waste += int(wild_wasted)
            wild_up += int(wild_upgraded)
        order = ([] if q is None
                 else sorted(range(len(acts)), key=lambda j: -float(q[j]))[:top])
        frames.append(Frame(
            step=step, seat=seat, chosen=m, level=lv, turn=seat,
            table=(Play(seat=e.hand.table_seat, cards=list(e.hand.table.cards))
                   if e.hand.table is not None else None),
            hands={s: set(e.hand.hands[s]) for s in rules.SEATS},
            played={s: list(played[s]) for s in rules.SEATS},
            passes=set(e.hand.passed),
            candidates=[(float(q[j]), acts[j]) for j in order],
            wasted=was_wasted and not is_opp, opp=(opp_kind if is_opp else None),
            why=why))
        if m is not None:
            played[seat].append(Play(seat=seat, cards=list(m.cards)))
        obs, _r, _done, _info = e.step(i)

    ranks = e.ranks                      # ⚠️ `ranks[seat] = 1..4`（按座位索引）
    winner = rules.winner_team(ranks)
    frames.append(Frame(                 # 终局帧：四家全部出牌都在，名次也定了
        step=step, seat=None, chosen=None, level=lv, turn=None, table=None,
        hands={s: set(e.hand.hands[s]) for s in rules.SEATS},
        played={s: list(played[s]) for s in rules.SEATS},
        passes=set(), over=True))
    meta = {"path": p, "games": ck.get("games") if isinstance(ck, dict) else None,
            "seed": seed, "level": lv, "first": e.hand.steps[0].seat,
            "opp_kind": opp_kind, "opp_team": opp_team,
            "ranks": ranks, "winner": winner,
            "points": rules.points(ranks), "steps": step,
            "bombs": bombs, "chance": chance, "waste": waste,
            "wild_chance": wild_chance, "wild_waste": wild_waste, "wild_up": wild_up,
            "tidy": _TIDY_MODE, "tidy_margin": _TIDY_MARGIN}
    return meta, frames


def chosen_text(f: Frame) -> str:
    """这一步出了什么（两个视图共用）。"""
    if f.over:
        return "终局"
    if f.chosen is None:
        return "过"
    return f"{cards_text(f.chosen.cards, f.level)}（{meld.describe_meld(f.chosen)}）"


def frame_hint(f: Frame) -> str:
    """底部那一行提示。**谁出的牌必须写在上面**（换源必须可见）。"""
    if f.over:
        return "本局结束 —— 点「上一步」回看"
    lead = "领出" if f.table is None else "跟牌"
    mine = f" · {OPP_KIND_CN[f.opp]}对手出牌（不打分）" if f.opp else ""
    return f"第 {f.step} 手 · {seat_label(f.seat)} {lead} {chosen_text(f)}{mine}"


def advice_of(f: Frame) -> list:
    """渲染器要的候选结构 `[{cards, kind, q}, …]`（首选在前）—— 两个视图共用。"""
    return [{"cards": [] if m is None else list(m.cards), "q": q}
            for q, m in f.candidates]


def render_text(meta: dict, frames: list, quiet: bool = False, log=print) -> None:
    """把 `frames` 打成文字（就是这个脚本原来的样子）。"""
    opp = meta.get("opp_kind")
    log(f"=== 一局：模型 vs {OPP_KIND_CN[opp]} ===" if opp else "=== 自对弈一局 ===")
    log(f"权重：{meta['path']}" + (f"（{meta['games']:,} 局）" if meta["games"] else ""))
    log(f"级别：打 {meta['level']}      先手：{seat_label(meta['first'])}      "
        f"种子：{meta['seed']}")
    log("队伍：甲队 = 座位 0、2      乙队 = 座位 1、3")
    log("谁在打：" + "    ".join(
        f"{seat_label(s)} = {policy_name(meta, s)}" for s in rules.SEATS))
    if opp:
        t = meta["opp_team"]
        seats = "、".join(str(s) for s in rules.SEATS if rules.TEAM[s] == t)
        log(f"对手：{TEAM_NAME[t]}队（座位 {seats}）= {OPP_KIND_CN[opp]}"
            f"      模型：{TEAM_NAME[1 - t]}队        "
            f"（对手的着法不打分 —— 用模型的 Q 给它打分是编数据）")
    log("")
    log("--- 发牌 ---")
    for s in rules.SEATS:
        h = sorted(frames[0].hands[s])
        log(f"  {seat_label(s)} {len(h):2d} 张：{cards_text(h, meta['level'])}")
    log("")
    log("--- 出牌 ---")
    for f in frames:
        if f.over:
            break
        lead = "领出" if f.table is None else "跟牌"
        what = chosen_text(f)
        if f.chosen is None:
            what = f"**{what}**"          # 文字版把「过」加粗便于扫（图形版不要星号）
        flag = "   ⚠️ **白炸**（有普通牌能压）" if f.wasted else ""
    if getattr(f, "why", ""):
        flag += f"   💡 {f.why}"
        who = f"   ←{OPP_KIND_CN[f.opp]}对手" if f.opp else ""
        log(f"  #{f.step:<3d} {seat_label(f.seat)} {lead}  {what}{flag}{who}")
        if not quiet:
            for r, (qv, a) in enumerate(f.candidates):
                txt = "过" if a is None else (
                    f"{cards_text(a.cards, f.level)}（{meld.describe_meld(a)}）")
                log(f"          {'★' if r == 0 else ' '} {qv:+7.3f}  {txt}")
    log("")
    log("--- 终局 ---")
    log("  名次：" + "  ".join(f"{meta['ranks'][s]} 名 {seat_label(s)}"
                              for s in rules.SEATS))
    log(f"  {TEAM_NAME[meta['winner']]}队赢 → 得 {meta['points']} 分"
        f"（双上是 3、有 3 名是 2、有 4 名是 1）")
    log(f"  共 {meta['steps']} 手，其中炸弹 {meta['bombs']} 手；"
        f"{'模型' if meta.get('opp_kind') else ''}"
        f"有「用普通牌压」的机会 {meta['chance']} 次，其中白炸 {meta['waste']} 次"
        + (f"（{meta['waste'] / meta['chance']:.0%}）" if meta["chance"] else ""))
    # 万能牌那一类（用户 2026-10-01 的第二条抱怨）+ **擦浪费开关的状态**
    # —— 开关状态必须印出来：不印的话"看了几局觉得没变"分不清是没生效还是本来如此。
    if meta.get("wild_chance"):
        log(f"  有「不用万能牌也能压」的机会 {meta['wild_chance']} 次，"
            f"其中白用万能牌 {meta['wild_waste']} 次"
            f"（{meta['wild_waste'] / meta['wild_chance']:.0%}）"
            f"，其中『用万能牌放大炸弹』{meta['wild_up']} 次")
    # **解释型建议**汇总（`advice/reasons.py`）：只解释、不改动作 ⇒ 对强度零影响
    whys = [(f.step, seat_label(f.seat), f.why) for f in frames if getattr(f, "why", "")]
    if whys:
        log(f"  💡 本次给出 {len(whys)} 条「为什么」（解释型建议，不改动作）：")
        for st, who_, w in whys[:8]:
            log(f"      #{st:<3d} {who_}：{w}")
        if len(whys) > 8:
            log(f"      …… 另有 {len(whys) - 8} 条")
    try:
        from guandan.advice.reasons import opponent_hint
        _h = opponent_hint(frames[0].obs) if getattr(frames[0], "obs", None) is not None else ""
        if _h:
            log(f"  {_h}")
    except Exception:
        pass
    if meta.get("tidy", "off") != "off":
        log(f"  **擦浪费：{meta['tidy']}**"
            + (f"（阈值 {meta['tidy_margin']:g}）" if meta.get("tidy_margin") else ""))
    else:
        log("  擦浪费：关（设 `$env:GUANDAN_TIDY=\"margin:0.25\"` 再跑一次可对比）")


def show(path: str = None, seed: int = 7, level: int = None, top: int = 3,
         quiet: bool = False, log=print, opp_kind: str = None,
         opp_team: int = 1) -> dict:
    """打一局并打印战报（`tools/game_viewer.py` 是同一个重放的图形版）。"""
    meta, frames = replay_game(path, seed=seed, level=level, top=top,
                               opp_kind=opp_kind, opp_team=opp_team)
    render_text(meta, frames, quiet=quiet, log=log)
    return meta


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--top", type=int, default=3, help="每个决策点列几个候选")
    ap.add_argument("--quiet", action="store_true", help="只看出牌，不看打分")
    ap.add_argument("--opp-kind", choices=OPP_KINDS, default=None,
                    help="对手那一队走固定对手（默认 None = 自对弈）")
    ap.add_argument("--opp-team", type=int, choices=(0, 1), default=1,
                    help="哪一队当对手（默认乙队 = 座位 1、3）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    show(a.weights, seed=a.seed, level=a.level, top=a.top, quiet=a.quiet,
         opp_kind=a.opp_kind, opp_team=a.opp_team)
    return 0


if __name__ == "__main__":
    sys.exit(main())
