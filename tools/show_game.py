"""把一局**自对弈**打成人类可读的战报 —— 看模型到底怎么打的。

为什么要有这个：所有判据都是数字（胜率、浪费率、动作边际），但「牌打得好不好」
最终得靠会打牌的人看几局。影子日志（`net/shadow.jsonl`）记的是**真实对局**里
模型的建议，这个脚本看的是**自对弈** —— 两者互补。

用法：
    # 用面板那套口径挑权重（`advise.resolve_weights`：GUANDAN_WEIGHTS 或最新的 best.pt）
    .venv/Scripts/python.exe -m tools.show_game

    # 指定权重 / 种子 / 少打一点字
    .venv/Scripts/python.exe -m tools.show_game runs/rl/20260926-1407/best.pt --seed 7
    .venv/Scripts/python.exe -m tools.show_game --quiet      # 只看出牌，不看打分

⚠️ **权重口径复用 `net/advise.py::resolve_weights`**，不另写一份 ——
不然「面板用哪个模型」与「这个脚本看哪个模型」会漂。
"""
from __future__ import annotations

import argparse
import sys

import torch

from net import advise
from net.cards import names_sorted
from net.sim import env, meld, rules
from train.eval import bomb_opportunity, is_wasted_bomb
from train.net import QNet, q_values
from tools.accept_meld import _utf8_stdout

TEAM_NAME = {0: "甲", 1: "乙"}


def _seat(s: int) -> str:
    return f"座位{s}({TEAM_NAME[rules.TEAM[s]]})"


def _cards(ids, level) -> str:
    """一手牌 -> 「大王 ♠A ♥K」这样的可读串（按掼蛋大小排）。"""
    return " ".join(names_sorted(ids, level))


def show(path: str = None, seed: int = 7, level: int = None, top: int = 3,
         quiet: bool = False, log=print) -> dict:
    """打一局并打印战报。返回 `{ranks, points, steps, bombs, w, path}`。"""
    p = path or advise.resolve_weights()
    if not p:
        raise SystemExit("找不到权重：设 GUANDAN_WEIGHTS，或先训练出一份 runs/rl/*/best.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = QNet()
    net.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)
    net.eval()
    games = ck.get("games") if isinstance(ck, dict) else None
    log(f"=== 自对弈一局 ===")
    log(f"权重：{p}" + (f"（{games:,} 局）" if games else ""))
    lv = level if level is not None else 8
    e = env.GuandanEnv(seed=seed)
    e.reset(level=lv)
    log(f"级别：打 {lv}      先手：{_seat(e.hand.turn)}      种子：{seed}")
    log(f"队伍：甲队 = 座位 0、2      乙队 = 座位 1、3")

    log("")
    log("--- 发牌 ---")
    for s in rules.SEATS:
        h = sorted(e.hand.hands[s])
        log(f"  {_seat(s)} {len(h):2d} 张：{_cards(h, lv)}")

    log("")
    log("--- 出牌 ---")
    obs = e.observe()
    step = bombs = chance = waste = 0
    while not e.done:
        acts = e.legal()
        seat = e.hand.turn
        hist = env.encode_history(e.hand, seat)
        q = q_values(net, obs, acts, hist)
        i = int(q.argmax())
        m = acts[i]
        step += 1
        if m is not None and m.is_bomb:          # ⚠️ 候选里的「过」就是 None
            bombs += 1
        lead = "领出" if e.hand.table is None else "跟牌"
        what = "**过**" if m is None else f"{_cards(m.cards, lv)}（{meld.describe_meld(m)}）"
        # 「白炸」用 train/eval.py 的**唯一判定**，不另写一份（用户最初的抱怨就是它）
        flag = ""
        if bomb_opportunity(e.hand.table, acts):
            chance += 1
            if is_wasted_bomb(e.hand.table, acts, m):
                waste += 1
                flag = "   ⚠️ **白炸**（有普通牌能压）"
        log(f"  #{step:<3d} {_seat(seat)} {lead}  {what}{flag}")
        if not quiet:
            order = sorted(range(len(acts)), key=lambda j: -float(q[j]))[:top]
            for r, j in enumerate(order):
                a = acts[j]
                txt = "过" if a is None else f"{_cards(a.cards, lv)}（{meld.describe_meld(a)}）"
                log(f"          {'★' if r == 0 else ' '} {float(q[j]):+7.3f}  {txt}")
        obs, _r, _done, _info = e.step(i)

    ranks = e.ranks          # ⚠️ `ranks[seat] = 1..4`（按座位索引，不是名次顺序表）
    log("")
    log("--- 终局 ---")
    log("  名次：" + "  ".join(f"{ranks[s]} 名 {_seat(s)}" for s in rules.SEATS))
    pts = rules.points(ranks)
    w = rules.winner_team(ranks)
    log(f"  {TEAM_NAME[w]}队赢 → 得 {pts} 分（双上是 3、有 3 名是 2、有 4 名是 1）")
    log(f"  共 {step} 手，其中炸弹 {bombs} 手；"
        f"有「用普通牌压」的机会 {chance} 次，其中白炸 {waste} 次"
        + (f"（{waste / chance:.0%}）" if chance else ""))
    return {"ranks": ranks, "points": pts, "winner": w, "steps": step,
            "bombs": bombs, "chance": chance, "waste": waste, "path": p}


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--top", type=int, default=3, help="每个决策点列几个候选")
    ap.add_argument("--quiet", action="store_true", help="只看出牌，不看打分")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    show(a.weights, seed=a.seed, level=a.level, top=a.top, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
