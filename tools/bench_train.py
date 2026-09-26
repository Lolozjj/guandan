"""量多进程训练的吞吐：worker 数各跑一段，报局/秒与队列积压。

跑法：
    .venv/Scripts/python.exe -m tools.bench_train --workers 1 2 3 4 6 --seconds 45

**量的是「训练阶段」的吞吐**（`train_parallel` 返回的 `elapsed` 在收尾评测之前取），
所以短跑不会被固定开销（两次 200 局评测 + 炸弹指标）淹没。判据见
`docs/superpowers/specs/2026-09-26-multiprocess-selfplay-design.md` §5②。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from train.selfplay import train_parallel
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
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    log = (lambda *a: None) if args.quiet else print
    rows = []
    print(f"每档跑 {args.seconds:.0f} 秒（量的是训练阶段，不含收尾评测）\n")
    print(f"{'worker':>7} {'局数':>8} {'秒':>7} {'局/秒':>8} {'队列积压峰值':>12}")
    for w in args.workers:
        r = bench_run(w, seconds=args.seconds, log=log)
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
