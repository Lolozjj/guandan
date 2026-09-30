"""把池子接进多进程：真起进程，看记录里的 `opp` 是不是跟着控制消息变。

为什么要真起进程：spawn 的 import 问题、控制队列的接线错误、以及「新成员到达前
就抽到它」这类时序问题，**只有真跑才暴露**（本项目在 Plan 3 的多进程上吃过一次）。
"""
import multiprocessing as mp

import torch

from guandan.rl import pool, worker
from guandan.rl.net import QNet


def test_worker_takes_a_new_member_from_the_control_queue():
    """喂一个成员进去，worker 下一批就用它当对手 —— 证据是记录里的 `opp`。

    这条同时证明两件事：控制消息被接住了，以及**记录里能读出用了谁**
    （PFSP 的归因就靠它）。
    """
    ctx = mp.get_context("spawn")
    send_q, ctrl_q = ctx.Queue(), ctx.Queue()
    torch.manual_seed(0)
    ctrl_q.put(("member", (7, QNet().state_dict())))
    cfg = worker.worker_cfg(seed=3, eps=0.0, opp_mix=1.0, greedy_share=0.0,
                            batch_games=2, init=None, members={},
                            pick_all="member", member_id=7)
    p = ctx.Process(target=worker.run_worker, args=(send_q, ctrl_q, cfg), daemon=True)
    p.start()
    try:
        recs = send_q.get(timeout=180)
    finally:
        ctrl_q.put(("stop", None))
        p.join(timeout=30)
    assert not p.is_alive(), "收到 stop 之后进程要退出"
    assert all(r.opp == ("member", 7) for r in recs), \
        f"记录里的对手不对：{[r.opp for r in recs]}"


def test_pool_report_warns_when_the_pool_collapses():
    """池子塌成一个成员必须**响**（spec §1.4）—— 返回值就是那个「响」，
    不是只写一行日志。"""
    from guandan.rl import selfplay
    wr = pool.WinRates()
    for _ in range(200):
        wr.record(0, True)
    lines = []
    assert selfplay.pool_report(wr, {0: 1.0}, lines.append) is True
    assert any("塌" in s for s in lines), f"日志里没提池子塌了：{lines}"


def test_pool_report_is_quiet_for_a_healthy_pool():
    from guandan.rl import selfplay
    wr = pool.WinRates()
    for i in range(3):
        for _ in range(50):
            wr.record(i, True)
    lines = []
    assert selfplay.pool_report(wr, {0: 0.34, 1: 0.33, 2: 0.33},
                                lines.append) is False
    assert lines, "健康也要打表（不然看不到池子的状态）"
