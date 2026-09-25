"""自对弈吞吐实测（spec §9 风险 1）。

跑法：
    .venv/Scripts/python.exe -m tools.bench_sim            # 默认 10 秒
    .venv/Scripts/python.exe -m tools.bench_sim 60         # 跑 60 秒

**测的是「随机策略下的规则引擎 + 环境」的吞吐** —— 不含网络前向。
真实训练里每个决策点还要过一次网络，所以这里出来的数字是**上界**，
写进结论时要说明这一点，别拿它当训练速度。

输出里必须给出「一千万局要多久」—— spec §5.2 的量级是千万局，
不换算成小时的话这个数字没有决策价值。

⚠️ **测出来的数字只代表「现在这一版实现」**。`play()` 为了正确处理「同一套牌多种读法」
（见该方法的注释）会在每次出牌时再枚举一遍整手牌 —— 那是一次实打实的开销，
也是这一版吞吐的主要成本之一。要优化先从这里看。
"""
from __future__ import annotations

import random
import sys
import time

from net.sim import env


def run(seconds: float = 10.0, seed: int = 0) -> dict:
    e = env.GuandanEnv(seed=seed)
    rng = random.Random(seed)
    games = decisions = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        e.reset()
        while not e.done:
            e.step(rng.randrange(len(e.legal())))
            decisions += 1
        games += 1
    dt = time.perf_counter() - t0
    return {
        "seconds": dt,
        "games": games,
        "decisions": decisions,
        "games_per_sec": games / dt,
        "decisions_per_sec": decisions / dt,
        "decisions_per_game": decisions / max(games, 1),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    seconds = float(argv[0]) if argv else 10.0
    r = run(seconds)
    print(f"跑了 {r['seconds']:.1f} 秒：{r['games']} 局 / {r['decisions']} 个决策点")
    print(f"  局/秒        {r['games_per_sec']:,.1f}")
    print(f"  决策点/秒    {r['decisions_per_sec']:,.1f}")
    print(f"  每局决策点   {r['decisions_per_game']:.1f}")
    gps = r["games_per_sec"]
    if gps > 0:
        print(f"\n换算：一千万局（spec §5.2 的量级）"
              f"≈ {1e7 / gps / 3600:,.1f} 小时 ≈ {1e7 / gps / 86400:,.1f} 天")
    print("\n**这是上界**：不含网络前向与反向。真实训练速度一定比它低，"
          "低多少取决于 batch 与设备。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
