"""快照必须**嵌套 + 不叫 best.pt** —— 否则面板会误加载（静默换源）。

`net/advise.py::newest_weights()` 按修改时间挑 `runs/rl/*/best.pt`。
快照要是长得像 `best.pt`、或者落在那一层，面板会**在你眼皮底下**换一个模型给建议，
而日志上一概看不出来 —— 这是本项目「换源必须可见」纪律的反面教材。

为什么要定期快照：现在只在「刷新最好」时存 `best.pt`，
1407 那 144 万局只落了几个点 —— 池子原料不够（spec §3.1）。
"""
import glob
import os

from train import pool


def test_snapshot_path_is_nested_and_not_named_best(tmp_path):
    p = pool.snapshot_path(str(tmp_path), 20000)
    assert os.path.basename(p).startswith("snap_")
    assert os.path.dirname(p).endswith("pool"), "快照要嵌在 pool/ 下，别平铺在 run 目录里"


def test_snapshots_are_invisible_to_newest_weights(tmp_path):
    """`newest_weights()` 必须**看不见**快照 —— 看见了就是静默换源。"""
    from net.advise import newest_weights
    p = pool.snapshot_path(str(tmp_path), 20000)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()
    assert newest_weights(str(tmp_path)) is None


def test_prune_keeps_the_newest_and_reports_what_it_dropped(tmp_path):
    for g in range(10, 110, 10):                 # 10000 .. 100000，共 10 个
        p = pool.snapshot_path(str(tmp_path), g * 1000)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()
        os.utime(p, (g, g))                      # 让修改时间有先后（取「最新」靠它）
    dropped = pool.prune_snapshots(str(tmp_path), keep=3)
    left = {os.path.basename(p)
            for p in glob.glob(os.path.join(str(tmp_path), "pool", "snap_*.pt"))}
    # ⚠️ 别用 sorted(...)[-1] 判「留了最新的」—— basename 是**字典序**，
    # `snap_100000.pt` 排在 `snap_80000.pt` 前面。用集合比。
    assert left == {"snap_80000.pt", "snap_90000.pt", "snap_100000.pt"}
    assert len(dropped) == 7


def test_prune_is_a_noop_below_the_limit(tmp_path):
    p = pool.snapshot_path(str(tmp_path), 20000)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()
    assert pool.prune_snapshots(str(tmp_path), keep=20) == []
