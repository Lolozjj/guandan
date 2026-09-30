"""量多进程训练的吞吐：worker 数各跑一段，报局/秒与队列积压。

跑法：
    .venv/Scripts/python.exe -m tools.bench_train --workers 1 2 3 4 6 --seconds 45

**量的是「训练阶段」的吞吐**（`train_parallel` 返回的 `elapsed` 在收尾评测之前取），
所以短跑不会被固定开销（两次 200 局评测 + 炸弹指标）淹没。判据见
多进程自对弈设计（已归档到 `master` 分支） §5②。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from guandan.rl.selfplay import train_parallel
from tools.accept_meld import _utf8_stdout


def bench_run(workers: int, seconds: float = 45.0, out_dir: str = None,
              log=print, **kw) -> dict:
    """跑一段，返回 `{workers, games, elapsed, games_per_s, qmax}`。"""
    r = train_parallel(seconds=seconds, workers=workers,
                       out_dir=out_dir or tempfile.mkdtemp(prefix="bench-"),
                       eval_games=1, eval_every=10 ** 9,   # 不中途评测
                       log=log, **kw)
    return {"workers": workers, "games": r["games"], "elapsed": r["elapsed"],
            "games_per_s": r["games"] / max(1e-9, r["elapsed"]),
            "qmax": r.get("qmax", 0)}


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mc-mix", type=float, default=1.0,
                    help="beta：MC 与自举的混合比（1.0 = 纯 DMC）。量自举成本用")
    ap.add_argument("--n-step", type=int, default=3, help="自举往后看几步")
    ap.add_argument("--init", default=None, help="热启动权重（默认随机）")
    ap.add_argument("--algo", default="dmc", help="dmc（默认）或 pg（策略梯度）")
    ap.add_argument("--beta-ent", type=float, default=None, help="PG 的熵系数")
    ap.add_argument("--weight-sync-games", type=int, default=None,
                    help="PG 臂要调小（staleness），默认 1000")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    log = (lambda *a: None) if args.quiet else print
    rows = []
    print(f"每档跑 {args.seconds:.0f} 秒（量的是训练阶段，不含收尾评测）  "
          f"algo={args.algo} beta={args.mc_mix:g} n={args.n_step}"
          + (f"  热启动 {args.init}" if args.init else "  随机初始化"))
    print(f"{'worker':>7} {'局数':>8} {'秒':>7} {'局/秒':>8} {'队列积压峰值':>12}")
    for w in args.workers:
        extra = {"mc_mix": args.mc_mix, "n_step": args.n_step, "init": args.init,
                 "algo": args.algo}
        if args.beta_ent is not None:
            extra["beta_ent"] = args.beta_ent
        if args.weight_sync_games is not None:
            extra["weight_sync_games"] = args.weight_sync_games
        r = bench_run(w, seconds=args.seconds, log=log, **extra)
        rows.append(r)
        print(f"{r['workers']:>7} {r['games']:>8} {r['elapsed']:>7.0f} "
              f"{r['games_per_s']:>8.1f} {r['qmax']:>12}")
        sys.stdout.flush()
    if rows:
        best = max(rows, key=lambda r: r["games_per_s"])
        base = rows[0]["games_per_s"]
        print(f"\n最快：{best['workers']} 个 worker，{best['games_per_s']:.1f} 局/秒"
              f"（相对 1 个 worker 的 {base:.1f} = {best['games_per_s'] / base:.2f}×）")
        print("注：理论天花板约 23 局/秒（学习进程每局要花 27ms 重放，见设计 §2）——"
              "再加 worker 只会堆队列（看「队列积压峰值」列）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
