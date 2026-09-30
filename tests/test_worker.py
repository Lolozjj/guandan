"""worker 进程：取数 + 控制循环。**必须真起进程**（spawn 的 import 问题只有真跑才暴露）。"""
import multiprocessing as mp
import random

import torch

from guandan.rl import selfplay, worker
from guandan.rl.net import QNet


def _fresh_net(seed=0):
    torch.manual_seed(seed)
    return QNet().eval()


def test_worker_batch_matches_in_process_bit_for_bit():
    """同种子同权重同批大小 -> worker 产出的局与进程内 `generate_batch` **逐局相等**。

    这是多进程改动唯一的硬保证（spec 验收②）：分出去了，但打法必须一模一样。
    """
    n = 4
    # 两边必须给**同样的显式参数** —— 第一次写这条测试时我只给 worker 传了
    # opp_mix=0.5，进程内用了默认的 0.0，于是「打法不同」其实是参数不同（自己踩的）
    kw = dict(opp_mix=0.5, greedy_share=0.8)
    mine = [rec for rec, _p, _y in selfplay.generate_batch(
        _fresh_net(1), random.Random(7), eps=0.0, n_games=n, capture=False, **kw)]
    got = worker.worker_batch(_fresh_net(1), random.Random(7), eps=0.0, n_games=n, **kw)
    assert len(got) == n
    for a, b in zip(mine, got):
        assert (a.level, a.first, a.hands, a.actions) == (b.level, b.first, b.hands, b.actions)


def test_worker_runs_in_a_real_process_and_sends_records():
    """真起一个进程：发一批记录回来，然后能被干净地停掉。"""
    ctx = mp.get_context("spawn")
    send_q, ctrl_q = ctx.Queue(), ctx.Queue()
    cfg = worker.worker_cfg(seed=3, eps=1.0, opp_mix=0.0, greedy_share=0.8, batch_games=4)
    p = ctx.Process(target=worker.run_worker, args=(send_q, ctrl_q, cfg), daemon=True)
    p.start()
    try:
        recs = send_q.get(timeout=180)
        assert len(recs) == 4 and hasattr(recs[0], "actions")
    finally:
        ctrl_q.put(("stop", None))
        p.join(timeout=30)
    assert not p.is_alive(), "收到 stop 之后进程要退出"


def test_worker_uses_cpu_even_when_the_learner_uses_cuda():
    """worker 只做推理 —— 显存留给 learner（spec §4）。"""
    assert worker.worker_device() == "cpu"
