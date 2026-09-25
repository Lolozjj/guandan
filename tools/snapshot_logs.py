"""把当前日志语料冻结成仓库里的一份 JSON 快照。

跑法：
    .venv/Scripts/python.exe -m tools.snapshot_logs

为什么：验收脚本读游戏日志，而日志会被轮转删除。快照让 Plan 1 的回归防线
不依赖日志还活着（Plan 2/3 每次改模拟器都要跑它）。
"""
from __future__ import annotations

import sys

from tools.game_log import LOG_DIR, conserved, load_games, save_snapshot


def main() -> int:
    games = load_games()
    settled = [g for g in games if g.settle]
    bad = [g for g in settled if not conserved(g)]
    if bad:
        sys.exit(f"[FAIL] {len(bad)} 局不守恒，语料有问题，拒绝冻结快照："
                 f"{[str(g.t0) for g in bad[:3]]}")
    path = save_snapshot(games)
    import os
    print(f"已冻结 {len(games)} 局（有结算 {len(settled)} 局，守恒 "
          f"{len(settled)}/{len(settled)}）")
    print(f"  -> {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    print(f"  源: {LOG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
