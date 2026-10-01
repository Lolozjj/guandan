"""结构指标（路线图 A6b）：**指出那 30% 是丢在哪一段**，而不是只报一个整局胜率。

三个读数（不训练）：

1. **名次形态**：双上率 / 1-3 / 2-4 / 末游率（两队都给）
2. **残局转化率**：所有手牌都 ≤ `--endgame-n` 的**第一个时刻**，记下「出牌权在哪一队」，
   再看终局 ⇒ `P(赢 | 进残局时有出牌权)` 的 2×2 表
3. **与规则式的着法一致率**（学生决策点上 `rule_choose` 与学生 argmax 相同的比例）

预登记（`plans/2026-09-30-beat-70-roadmap.md` §十）：这不是 A/B 判据，是**尺子** ——
「残局转化率接近 100%」⇒ 丢分在**争出牌权**（A2 的特征该压在最少手数/火力/对家剩牌上）；
「残局转化率明显低于对手」⇒ 丢分在**残局本身**（该单独练残局）。

跑法：
    .venv/Scripts/python.exe -m tools.structure_metrics
    .venv/Scripts/python.exe -m tools.structure_metrics --games 400 --seeds 1002,1003
"""
from __future__ import annotations

import argparse
import collections
import random
import statistics
import sys

from guandan import paths
from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.rule_policy import rule_choose
from guandan.sim import env, rules

DEFAULT_SEEDS = [1002, 1003, 1004, 1005, 1006, 1007]
ENDGAME_N = 8          # 「所有手牌都 ≤ N」= 进残局


def load(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n, (d.get("games") if isinstance(d, dict) else None)


def _shape(my_ranks: list) -> str:
    """名次形态 —— **六种都要列**（名次是 1..4 的排列，别漏 `1-4` 与 `2-3`）。

    ⚠️ 第一版只列了四种，把 `1-4`（**我方赢**）与 `2-3`（我方输）都塞进了「被双上」，
    于是「胜率 90.9%」与「双上 45.9%」自相矛盾 —— 是工具错，不是模型强。
    """
    return {(1, 2): "双上(赢)", (1, 3): "1-3(赢)", (1, 4): "1-4(赢)",
            (2, 3): "2-3(输)", (2, 4): "2-4(输)", (3, 4): "被双上(输)"}[
                tuple(sorted(my_ranks))]


def is_win(ranks: list, team: int) -> bool:
    """`team` 这一队赢没赢 —— **与评测器同一处口径**（`rules.winner_team`）。

    ⚠️ 别自己写 `min(我方) <= 2`：那样 `[2,4]`（我方 2、4 名 vs 对方 1、3 名）会被算成赢，
    而掼蛋是「**较好的名次更靠前**的那队赢」。我第一版就是这么错的，胜率虚高到 90.9%。
    """
    return rules.winner_team(ranks) == team


def one_seed(net, games: int, seed: int, endgame_n: int = ENDGAME_N) -> dict:
    """逐局跑（不批量）：要读中间的「残局时刻」，批量反而更绕。"""
    rng = random.Random(seed)
    shapes = collections.Counter()
    wins = 0
    agree = dec = 0
    # 残局 2×2：进残局时出牌权在我方 / 在对方 × 终局赢/输
    endgame = collections.Counter()
    for g in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        learner_team = g % 2                        # 座位对调（与评测器同口径）
        lead_holder = None                          # 进残局那一刻的出牌权归谁
        while not e.done:
            seat = e.hand.turn
            if lead_holder is None and max(len(h) for h in e.hand.hands) <= endgame_n:
                # 「出牌权」= 桌上那手是谁打的；桌上没牌就是当前该谁领出
                owner = e.hand.table_seat if e.hand.table is not None else seat
                lead_holder = owner
            obs, acts = e.observe(), e.legal()
            if rules.TEAM[seat] == learner_team:
                qs = [float(x) for x in q_values(net, obs, acts,
                                                env.encode_history(e.hand, seat))]
                i = max(range(len(qs)), key=qs.__getitem__)
                if len(acts) >= 2:
                    dec += 1
                    agree += 1 if rule_choose(obs, acts) == i else 0
            else:
                i = rule_choose(obs, acts)
            e.step(i)

        my_ranks = [e.ranks[s] for s in rules.SEATS if rules.TEAM[s] == learner_team]
        won = is_win(e.ranks, learner_team)
        wins += 1 if won else 0
        shapes[_shape(my_ranks)] += 1
        if lead_holder is not None:
            endgame[(rules.TEAM[lead_holder] == learner_team, won)] += 1
    return {"games": games, "wins": wins, "shapes": shapes,
            "endgame": endgame, "agree": agree, "dec": dec}


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="结构指标：丢分丢在哪一段")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--endgame-n", type=int, default=ENDGAME_N)
    a = ap.parse_args(argv)

    path = a.weights or str(paths.BEST)
    net, g = load(path)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"学生：{path}" + (f"（{g:,} 局）" if g else ""))
    print(f"对手：规则式；每种子 {a.games} 局（座位对调）；残局阈值 = 所有手牌 ≤ {a.endgame_n}\n")

    rows = [one_seed(net, a.games, s, a.endgame_n) for s in seeds]
    wr = [r["wins"] / r["games"] for r in rows]
    print(f"整局胜率：{statistics.mean(wr):.1%}"
          f"（逐种子 {', '.join(f'{x:.1%}' for x in wr)}）")

    tot = collections.Counter()
    for r in rows:
        tot.update(r["shapes"])
    n = sum(tot.values())
    print("\n【名次形态】（我方视角，全部种子合计；**六种都列**）")
    for k in ("双上(赢)", "1-3(赢)", "1-4(赢)", "2-3(输)", "2-4(输)", "被双上(输)"):
        print(f"  {k:10s} {tot[k]:5d}  {tot[k] / max(1, n):6.1%}")
    print("  ⚠️ 对手的双上率 = 我方的「被双上」率（互补），所以只列一方")

    eg = collections.Counter()
    for r in rows:
        eg.update(r["endgame"])
    print(f"\n【残局转化率】（进残局那一刻出牌权在哪队 -> 我方最终赢没赢）")
    for has in (True, False):
        w, l = eg[(has, True)], eg[(has, False)]
        who = "我方" if has else "对方"
        if w + l == 0:
            print(f"  出牌权在{who}：样本 0")
            continue
        print(f"  出牌权在{who}：{w:4d}/{w + l:4d} = {w / (w + l):6.1%}")

    dec = sum(r["dec"] for r in rows)
    ag = sum(r["agree"] for r in rows)
    print(f"\n【行为指纹】学生与规则式的着法一致率 {ag / max(1, dec):.1%}"
          f"（{ag}/{dec} 个决策点）")

    print("\n判读（预登记 §十）：")
    e1 = eg[(True, True)] + eg[(True, False)]
    e0 = eg[(False, True)] + eg[(False, False)]
    r1 = eg[(True, True)] / e1 if e1 else float("nan")
    r0 = eg[(False, True)] / e0 if e0 else float("nan")
    if e1 and e1 >= 30:
        if r1 >= 0.9:
            print(f"  拿到出牌权时 {r1:.0%} 能赢 ⇒ **丢分主要在「争出牌权」**（中局/配合）")
            print("  ⇒ A2 的特征该压在「最少手数 / 火力 / 对家剩牌」上")
        else:
            print(f"  拿到出牌权也只有 {r1:.0%} 赢、没有出牌权 {r0:.0%}"
                  f" ⇒ **残局转化本身也是短板** ⇒ A2 之后该单独练残局")
    else:
        print(f"  进残局的样本太少（{e1} 局有出牌权），先加 --games 再判")
    return 0


if __name__ == "__main__":
    sys.exit(main())
