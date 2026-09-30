"""训练快照必须**嵌套 + 不叫 best.pt** —— 面板不许看见它们。

面板只认 `models/best.pt` 一个路径（`guandan/advice/advise.py::resolve_weights`），
所以「快照被误加载」在结构上已经不可能发生。这条测试钉两件事：
① 快照落在 `<run>/pool/` 下、名字是 `snap_*.pt`；
② 权重解析只有三档**写死的**路径，没有「扫目录挑最新」那套魔法。

为什么要定期快照：老版本只在「刷新最好」时存 `best.pt`，
一次 144 万局的训练只落了几个点 —— 池子原料不够。
"""
import glob
import os

from guandan.rl import pool


def test_snapshot_path_is_nested_and_not_named_best(tmp_path):
    p = pool.snapshot_path(str(tmp_path), 20000)
    assert os.path.basename(p).startswith("snap_")
    assert os.path.dirname(p).endswith("pool"), "快照要嵌在 pool/ 下，别平铺在 run 目录里"


def test_weights_resolve_to_exactly_one_path(monkeypatch, tmp_path):
    """权重解析 = 传参 → `GUANDAN_WEIGHTS` → `models/best.pt`，三档都是写死的路径。

    ⚠️ 这里**不断言「扫目录扫不到快照」** —— 现在压根没有扫目录这件事，
    断言它等于给一个不存在的机制写测试。
    """
    from guandan import paths
    from guandan.advice import advise

    got = advise.resolve_weights()
    assert got is None or got == str(paths.BEST), "默认只认 models/best.pt"
    monkeypatch.setenv("GUANDAN_WEIGHTS", str(tmp_path / "x.pt"))
    assert advise.resolve_weights() == str(tmp_path / "x.pt")
    assert advise.resolve_weights("yy.pt") == "yy.pt"


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
