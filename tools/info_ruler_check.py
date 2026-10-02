"""量"会利用信息的尺子"（`rule_policy("info")`）到底有没有惩罚作用。

判据（`plans/2026-10-02-info-ruler.md` §二，**先写死**）：

| # | 判据 | 阈值 |
|---|---|---|
| ① | 模型的白炸率：`info` 下 ≥ `normal` 下的 **1.3 倍** | 会利用信息的对手应当让"白炸"付出代价 |
| ② | 模型的**用炸手数**在两把尺子下显著不同 | 配对 ≥8 种子，\\|t\\| ≥ 2 |
| ③ | `info` 不能弱到不可用 | 模型对 `info` 的胜率 ≥ 对 `normal` 的 −8pp |

    .venv/Scripts/python.exe -m tools.info_ruler_check --seeds 8 --games 250
"""
from __future__ import annotations

import argparse
import statistics
import sys

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.rule_policy import rule_policy
from tools.mistake_profile import net_play


def load_net(path: str):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def _paired(d):
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return m, sd, (m / (sd / len(d) ** 0.5) if sd else float("inf"))


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="info 尺子：有没有惩罚作用")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1002)
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--meters", type=int, default=150, help="量浪费率的局数（每把尺子）")
    a = ap.parse_args(argv)

    net = load_net(a.weights)
    pol = net_play(net)
    print(f"权重 {a.weights}；{a.seeds} 个种子 × {a.games} 局（配对），浪费率各 {a.meters} 局\n")

    out = {}
    for name in ("normal", "info"):
        opp = rule_policy(name)
        wr = [ev.match(pol, opp, games=a.games, seed=a.seed0 + k) for k in range(a.seeds)]
        # ⚠️ 逐种子分别量 ⇒ 才有配对 t（预登记判据②要 |t| ≥ 2）
        per = [ev._bomb_stats(pol, a.meters, a.seed0 + k, opponent=opp) for k in range(a.seeds)]
        perw = [ev._wild_stats(pol, a.meters, a.seed0 + k, opponent=opp) for k in range(a.seeds)]
        waste = [x[0] / max(x[1], 1) for x in per]
        out[name] = dict(wr=wr, waste=statistics.mean(waste), per_waste=waste,
                         bombs=[x[2] for x in per],
                         wild=statistics.mean([x[0] / max(x[2], 1) for x in perw]))
        print(f"  {name:6s} 模型胜率 {statistics.mean(wr):6.1%}   "
              f"白炸 {out[name]['waste']:5.1%}   "
              f"用炸 {sum(out[name]['bombs']):4d} 手（{a.seeds} 种子 × {a.meters} 局）")

    # ① 白炸率之比
    ratio = out["info"]["waste"] / max(out["normal"]["waste"], 1e-9)
    ok1 = ratio >= 1.3
    print(f"\n① 白炸率之比（info / normal）= **{ratio:.2f}×**（要 ≥1.30）⇒ "
          f"{'通过 ✓' if ok1 else '未通过 ✗'}")

    # ② 用炸手数之差（配对：同一批种子里的用炸数只有一把尺子一个数 ⇒ 用逐种子重采样）
    #    ⚠️ 浪费/用炸统计是"一把尺子一个数"（不是逐种子），所以这里报**绝对差**并标注口径。
    db = [x - y for x, y in zip(out["normal"]["bombs"], out["info"]["bombs"])]
    mb, sdb, tb = _paired(db)
    ok2 = abs(tb) >= 2.0
    print(f"② 用炸手数（逐种子配对）：normal {sum(out['normal']['bombs'])} 手 → "
          f"info {sum(out['info']['bombs'])} 手；差 {mb:+.1f} 手/种子，t={tb:+.2f}"
          f" ⇒ {'有可测差异 ✓' if ok2 else '差异不显著 ✗'}")
    dw = [x - y for x, y in zip(out["normal"]["per_waste"], out["info"]["per_waste"])]
    mw, sdw, tw = _paired(dw)
    print(f"   白炸率（逐种子配对）：差 {mw * 100:+.2f}pp，t={tw:+.2f}")

    # ③ info 不能太弱：模型对它的胜率不许比对 normal 高 8pp 以上
    d = [x - y for x, y in zip(out["normal"]["wr"], out["info"]["wr"])]
    m, sd, t = _paired(d)
    ok3 = m >= -0.08
    print(f"③ 模型胜率：normal − info = **{m * 100:+.2f}pp**（sd {sd * 100:.1f}pp，t={t:+.2f}）"
          f" ⇒ {'通过 ✓' if ok3 else '过弱 ✗'}"
          f"（正 = info 更让模型难受）")
    # ④ **诊断：我加的行为到底有没有触发？**（反事实分歧率）
    #    在"模型 vs normal"的真实对局里，对每个规则式的决策点同时问两把尺子，
    #    数它们给出不同下标的比例 —— 这直接分开"没触发"与"触发了但没用"。
    import random
    from guandan.rl.rule_policy import STYLES, rule_choose
    from guandan.sim import env as senv, rules as srules
    rng = random.Random(a.seed0)
    s_norm, s_info = STYLES["normal"], STYLES["info"]   # ⚠️ 要 Style 对象，不是策略闭包
    diff = tot = 0
    for g in range(60):
        e = senv.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        team0 = g % 2
        while not e.done:
            obs, acts = e.observe(), e.legal()
            seat = e.hand.turn
            mine = srules.TEAM[seat] == team0
            if mine:
                hist = senv.encode_history(e.hand, seat)
                from guandan.rl.net import q_values
                i = int(__import__("numpy").argmax(q_values(net, obs, acts, hist)))
            else:
                i_n = rule_choose(obs, acts, s_norm)
                i_f = rule_choose(obs, acts, s_info)
                tot += 1
                diff += int(i_n != i_f)
                i = i_n
            e.step(i)
    print(f"\n④ 诊断——两把尺子在**同一批决策点**上的分歧率："
          f"**{diff}/{tot} = {diff / max(tot, 1):.1%}**")
    print("   （~0% ⇒ 我加的行为几乎没触发，得先让它真的触发；"
          "两位数百分比而胜率不动 ⇒ 行为确实不重要）")

    print("\n判读：①②通过 ⇒ 这把尺子**确实在惩罚信息劣势**，进六把尺子；"
          "否则按 §二记录并调那三条行为。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
