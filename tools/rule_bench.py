"""**规则式的尺子** —— 强度 + 三条行为护栏，一次全量出来。

为什么要有它：台账里那张「vs 贪心 / 压队友 / 用炸率 / 白炸」的表是
**一次性脚本**量出来的（n 步自举交付台账（已归档到 `master` 分支） §7），
没留工具。调规则式时每次都要量这四样 —— 不留工具就每次重写一份，
而「副本会漂」这个仓库已经吃过一次亏了。

**口径（引用时别丢）**：

- **强度**：`match(规则式, 对手, games, seed)` —— 座位对调、同步推进、固定牌堆。
  ⚠️ 单种子 sd 约 2.1pp（400 局），所以要多给几个种子取均值。
- **压队友**：`跟牌时台面是**队友**出的`那些决策点里，我**没有过**的比例。
  分母是「队友赢着这一手、轮到我表态」，分子是「我压了」。
- **白炸 / 用炸率**：`bomb_waste` / `bomb_rate` 的老口径 —— 对**贪心**、
  60 局、`seed=3001`（与 `tools/ab_compare.py` 同一把）。

用法：

    .venv/Scripts/python.exe -m tools.rule_bench
    .venv/Scripts/python.exe -m tools.rule_bench --seeds 1002,1003,1004
    .venv/Scripts/python.exe -m tools.rule_bench --opp models/best.pt
    .venv/Scripts/python.exe -m tools.rule_bench --behavior-opp ckpt
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

import torch

from guandan.advice import advise
from guandan.sim import env, rules
from guandan.rl.eval import bomb_rate, bomb_waste, match
from guandan.rl.net import DEVICE, QNet, q_values
from guandan.rl.policies import greedy_policy
from guandan.rl.rule_policy import ally_of, rule_choose, rule_policy, table_owner
from guandan.console import utf8_stdout


def net_pol(path: str):
    """权重 -> `(obs, acts, hist) -> 下标`（走 `q_values`，与上线同一条路）。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    n.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)
    n.eval().to(DEVICE)
    games = ck.get("games") if isinstance(ck, dict) else None
    return (lambda obs, acts, hist: int(q_values(n, obs, acts, hist).argmax())), games


def behavior(opp, games: int, seed: int, rule_team: int = 0) -> dict:
    """跑 `games` 局，量**行为护栏**（压队友 / 用炸 / 白炸）。

    ⚠️ 白炸与用炸率不在这里数 —— 那两样走 `guandan/rl/eval.py` 的**唯一判定**
    （`bomb_waste` / `bomb_rate`），不另写一份。
    """
    rng = random.Random(seed)
    dec = pressed = 0
    for _ in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        while not e.done:
            acts, seat = e.legal(), e.hand.turn
            o = e.observe()
            if rules.TEAM[seat] == rule_team:
                i = rule_choose(o, acts)
                # 「队友赢着这一手」的决策点 —— 我压了没有
                if o.table and table_owner(o) == ally_of(seat):
                    dec += 1
                    pressed += int(acts[i] is not None)
            else:
                i = opp(o, acts, env.encode_history(e.hand, seat))
            e.step(i)
    return {"dec": dec, "pressed": pressed,
            "rate": pressed / dec if dec else float("nan")}


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default=None, help="对手权重（默认挑最新 best.pt）")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seeds", default="1002,1004", help="逗号分隔")
    ap.add_argument("--behavior-opp", choices=("greedy", "ckpt"), default="greedy")
    ap.add_argument("--behavior-games", type=int, default=200)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    path = a.opp or advise.newest_weights()
    opp, games = net_pol(path)
    print(f"对手：{path}" + (f"（{games:,} 局）" if games else ""))
    print(f"规则式：guandan/rl/rule_policy.py" + (f"（{games:,} 局）" if games else ""))
    print()

    r = rule_policy()
    print(f"【强度】{a.games} 局 × 各种子（座位对调、固定牌堆）")
    for name, other in (("打 1407", opp), ("打 贪心", greedy_policy)):
        wr = [match(r, other, games=a.games, seed=int(s)) for s in a.seeds.split(",")]
        tag = "  ".join(f"{x:.1%}" for x in wr)
        print(f"  {name:8s} {tag}   均值 {statistics.mean(wr):5.1%}")
    print()

    print(f"【行为护栏】{a.behavior_games} 局 seed=1002")
    b = behavior(greedy_policy if a.behavior_opp == "greedy" else opp,
                 games=a.behavior_games, seed=1002)
    print(f"  压队友   {b['pressed']}/{b['dec']} = {b['rate']:.1%}"
          f"    <- 跟牌时台面是队友出的、我还没过")
    for name, fn in (("用炸率", bomb_rate), ("炸弹浪费率", bomb_waste)):
        x, y = fn(r, games=60, seed=3001, opponent=greedy_policy)
        print(f"  {name:6s} {x:3d}/{y:4d} = {x / y:.2f}" + (" 手/局" if y else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
