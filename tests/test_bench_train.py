"""bench 工具：能报出局/秒（薄测试 —— 真正的曲线在交付台账里）。"""
import os


def test_bench_reports_games_per_second(tmp_path):
    from tools.bench_train import bench_run
    r = bench_run(workers=1, seconds=8, out_dir=str(tmp_path), log=lambda *a: None)
    assert r["games"] > 0 and r["games_per_s"] > 0
    assert os.path.exists(str(tmp_path))
