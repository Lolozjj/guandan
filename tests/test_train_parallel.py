"""学习进程的编排：收记录 -> 训练 -> 广播；worker 死了必须炸。"""
import os

import pytest

from guandan.rl import selfplay


def test_worker_death_raises_loudly(tmp_path):
    """worker 死了**必须报错**，不许静默变慢（本仓库纪律：失败必须响）。"""
    with pytest.raises(RuntimeError, match="worker"):
        selfplay.train_parallel(seconds=60, workers=1, out_dir=str(tmp_path),
                                batch_games=2, eval_games=1, log=lambda *a: None,
                                _kill_worker_after=3.0)      # 测试钩子：3 秒后杀它


def test_train_parallel_runs_and_saves(tmp_path):
    """短跑要走通：有局数、有 loss、存下 last.pt、返回结构与非并行一致。"""
    r = selfplay.train_parallel(seconds=25, workers=2, out_dir=str(tmp_path),
                                batch_games=2, eval_games=1, eval_every=10_000,
                                log=lambda *a: None)
    assert r["games"] > 0 and r["curve"] == []
    assert os.path.exists(os.path.join(str(tmp_path), "last.pt"))
    assert set(r) >= {"games", "curve", "best_greedy", "wr_random", "wr_greedy", "out_dir"}
