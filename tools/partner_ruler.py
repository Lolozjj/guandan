"""**陪打尺子**：我的模型 + 一个**外来队友**，对上一对强对手。

为什么需要第三把尺子（`plans/2026-10-03-human-play.md`）：
面板给的是**建议**，真实牌桌上的队友是**人** ✗ 不是另一个模型。
"两个模型配合得好"不等于"我的建议能和一个人配合好" ✓
⇒ 量"**我的那一手在陌生队友身边值多少**"，才是离产品最近的代理。

三种队友可选（按"像不像人"排）：
- `rule`：规则式（脚本，但打法像人 —— 也是**最接近人类队友**的现成东西）；
- `greedy`：贪心（弱且机械，最不像人）；
- `chk:路径.pt`：另一件检查点（"像高手但不认识我"）。

    .venv/Scripts/python.exe -m tools.partner_ruler --me models/best.pt --mate rule ^
        --opp models/best.pt --seeds 8 --games 400
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch

from guandan.console import utf8_stdout
from guandan.rl import eval as ev
from guandan.rl.net import QNet, load_state
from guandan.rl.policies import greedy_policy
from guandan.rl.rule_policy import rule_policy
from guandan.rl.teams import pair_of
from guandan.rl.tidy import tidy_net_policy
from tools.mistake_profile import net_play


def load_net(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = QNet()
    load_state(n, d["net"] if isinstance(d, dict) else d)
    n.eval()
    return n


def make_policy(spec: str):
    """`rule` / `greedy` / `chk:路径` / `tidy:margin=0.25`（默认底权重=现役）。"""
    if spec == "rule":
        return rule_policy(), "规则式"
    if spec == "greedy":
        return greedy_policy, "贪心"
    if spec.startswith("chk:"):
        p = spec.split(":", 1)[1]
        return net_play(load_net(p)), f"检查点 {p}"
    if spec.startswith("combined"):
        from guandan.rl.coop import combined_net_policy
        base = load_net("models/best.pt")
        return (combined_net_policy(base, margin=0.25, leads=True),
                f"{spec}（配合护栏 + 擦浪费，基于 models/best.pt）")
    if spec.startswith("coop"):
        from guandan.rl.coop import coop_net_policy
        # ⚠️ `partner_ruler` 没有 `--weights`（它的 `--me/--mate/--opp` 各自带路径）
        # ⇒ coop 的底权重就是现役那份（要看别的权重就写 `chk:路径` 后缀那份）
        base = load_net("models/best.pt")
        return coop_net_policy(base), f"{spec}（喂队友护栏，基于 models/best.pt）"
    if spec.startswith("tidy"):
        mg = float(spec.split("margin=")[1].split(",")[0]) if "margin=" in spec else 0.0
        return tidy_net_policy(load_net("models/best.pt"), margin=mg,
                               leads=("leads=1" in spec)), spec
    if spec.endswith(".pt"):
        return net_play(load_net(spec)), spec
    raise SystemExit(f"认不出的策略 {spec!r}（rule / greedy / chk:xx.pt / tidy:margin=0.25 / xx.pt）")


def _paired(d):
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return m, sd, (m / (sd / len(d) ** 0.5) if sd else float("inf"))


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="陪打尺子：我 + 外来队友 vs 强对手")
    ap.add_argument("--me", default="models/best.pt")
    ap.add_argument("--mate", default="rule")
    ap.add_argument("--opp", default="models/best.pt", help="对手那一队（两个座位同一策略）")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1002)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--mine-only", action="store_true",
                    help="只让**我的座位**用 --me，队友也换成 --opp（对照：队友弱不弱差多少）")
    a = ap.parse_args(argv)

    me, name_me = make_policy(a.me)
    mate, name_mate = make_policy(a.mate)
    opp, name_opp = make_policy(a.opp)
    mine_side = pair_of(me, opp if a.mine_only else mate)
    print(f"我   = {name_me}\n队友 = {name_opp if a.mine_only else name_mate}"
          f"{'（--mine-only）' if a.mine_only else ''}\n对手 = {name_opp}（两座位同策略）\n"
          f"{a.seeds} 种子 × {a.games} 局（逐局换边、逐种子）\n")

    wr = []
    for k in range(a.seeds):
        s = a.seed0 + k
        wr.append(ev.match(mine_side, opp, games=a.games, seed=s))
        print(f"  seed {s}: 我方胜率 {wr[-1]:6.1%}")
    m, sd, t = _paired([x - 0.5 for x in wr])
    print(f"\n**我方（我 + {name_opp if a.mine_only else name_mate}）的胜率 "
          f"{statistics.mean(wr):.1%}**（逐种子 sd {sd * 100:.1f}pp）"
          f"；相对 50% 的 t={t:+.2f}")
    print("判读：>50% ⇒ 我这套建议**带得动**这个队友；≈50% ⇒ 势均力敌；<50% ⇒ 带不动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
