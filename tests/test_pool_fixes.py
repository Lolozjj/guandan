"""修复轮：评审出的 Critical + Important（外加一条被实验证据升格的 Minor）。

每一条都是**先有会失败的测试**。四条各自的来龙去脉：

- **C1** 池子的贪心份额用的还是「老二分」那个 0.8 —— 池子在混合局里只占 20%，
  全局只有 10% 的局打池成员（设计是 40%）。**整轮 A/B 只给了 1/4 剂量**，
  结论因此无从谈起。
- **I1** 种子池为空时，`--pfsp` 会**静默**退化成一个纯贪心臂（连 20% 随机也没了），
  而唯一的那句话把「空」误报成「塌陷」。
- **I2** 池子塌陷只写进 `r["pool_collapsed"]`，没人看 —— 退出码照样是 0、
  照样印「判据过了」。规矩是失败必须响。
- **I3** `--pool-size` 只约束运行时新增的成员，种子那 5 个不受它管 ——
  想按文档调小内存的人调不动。
- **M3**（被实验证据升格）预热成员拿的是 PFSP 的**最大**项（p=0.5 → 0.25），
  实测在真跑里一个 0 局的新成员吃掉了 **38.4%** 的采样权重。
  文档写的是「按均匀」，实现不是 —— 两套规则并存正是本仓库最忌讳的。
"""
import multiprocessing as mp
import os

import pytest
import torch

from guandan.rl import pool, selfplay, worker
from guandan.rl.net import QNet


# ---------------------------------------------------------------- C1：剂量

def test_worker_cfg_defaults_the_pool_share_to_the_pool_constant():
    """池子的贪心份额必须来自 `pool.GREEDY_SHARE`，不是老二分那个 0.8。"""
    cfg = worker.worker_cfg(seed=0, eps=0.0, opp_mix=1.0, greedy_share=0.8,
                            batch_games=2)
    assert cfg["pool_greedy_share"] == pool.GREEDY_SHARE
    assert cfg["pool_greedy_share"] != cfg["greedy_share"], \
        "两个份额必须是**两个**值 —— 混用就是那个 1/4 剂量的 bug"


def test_the_pool_share_is_honoured_end_to_end():
    """把池子的贪心份额设成 0，每个混合局的对手都该是池成员。

    （0.8 / 0.2 的比例要真跑才看得出来，所以这里取端点：0 就是「一个贪心都不许有」。）
    """
    ctx = mp.get_context("spawn")
    send_q, ctrl_q = ctx.Queue(), ctx.Queue()
    torch.manual_seed(0)
    cfg = worker.worker_cfg(seed=3, eps=0.0, opp_mix=1.0, greedy_share=0.8,
                            batch_games=2, members={9: QNet().state_dict()},
                            use_pool=True, pool_greedy_share=0.0)
    p = ctx.Process(target=worker.run_worker, args=(send_q, ctrl_q, cfg), daemon=True)
    p.start()
    try:
        recs = send_q.get(timeout=180)
    finally:
        ctrl_q.put(("stop", None))
        p.join(timeout=30)
    assert recs and all(r.opp and r.opp[0] == "member" for r in recs), \
        f"池份额设成 0 却还有非池对手：{[r.opp for r in recs]}"


# ---------------------------------------------------------------- A/M3：预热量纲

def test_warmup_members_get_exactly_the_uniform_share():
    """预热成员拿的就是**均匀那一份**（1/K），不是 PFSP 的最大项。

    原实现给它 p=0.5 —— 那是 `p(1−p)` 的**最大**值，等于让没测过的成员
    比健康成员重 4 倍。实测（`runs/B_pool.log`）：一个 0 局的新成员
    吃掉了 **38.4%** 的采样权重。

    ⚠️ 健康成员的 p **不能取 0.5** —— 那样「最大项」与「均匀」在数值上撞在一起，
    这个测试就测不出任何东西（第一版就是这么假绿的）。
    """
    w = pool.pfsp_weights({0: 0.9, 1: 0.0}, games={0: 999, 1: 3}, min_games=20)
    assert w[1] == pytest.approx(0.5), "预热成员该拿 1/K"
    assert w[0] == pytest.approx(0.5)


def test_warmup_share_does_not_depend_on_how_many_are_warm():
    """3 个成员、2 个在预热 -> 各拿 1/3，剩下的那个也拿 1/3。"""
    w = pool.pfsp_weights({0: 0.9, 1: 0.5, 2: 0.5},
                          games={0: 999, 1: 1, 2: 1}, min_games=20)
    assert w[1] == pytest.approx(1 / 3)
    assert w[2] == pytest.approx(1 / 3)
    assert w[0] == pytest.approx(1 / 3)
    assert sum(w.values()) == pytest.approx(1.0)


def test_all_warmup_is_plain_uniform():
    w = pool.pfsp_weights({0: 0.9, 1: 0.1}, games={0: 0, 1: 0}, min_games=20)
    assert w[0] == pytest.approx(0.5) and w[1] == pytest.approx(0.5)


# ---------------------------------------------------------------- I1/I3：种子池

def _fake_ckpt(path):
    torch.manual_seed(0)
    torch.save({"net": QNet().state_dict()}, path)
    return path


def test_seed_pool_respects_pool_size(tmp_path):
    """`--pool-size` 必须也管住**种子** —— 否则调小内存的旋钮是假的。

    原实现只在「新增成员的 FIFO」里剪枝，种子的 5 个一路全留；
    而文档（现用方案 §五）明写「池子留最近 `--pool-size` 个」。
    """
    paths = [_fake_ckpt(str(tmp_path / f"b{i}.pt")) for i in range(5)]
    sds, order = selfplay.load_seed_pool(paths, pool_size=2)
    assert len(sds) == 2 and len(order) == 2


def test_seed_pool_size_zero_yields_nothing(tmp_path):
    paths = [_fake_ckpt(str(tmp_path / "b0.pt"))]
    sds, order = selfplay.load_seed_pool(paths, pool_size=0)
    assert sds == {} and order == []


def test_pfsp_with_an_empty_seed_pool_raises_loudly(tmp_path, monkeypatch):
    """`--pfsp` 但一个种子都没有时**必须炸**。

    原来会静默退化成一个纯贪心臂（`pick_opponent` 对空池返回 `("greedy",)`，
    连老二分里那 20% 随机都没了），而唯一那句话把「空」误报成「塌陷」——
    一整夜的 `--pfsp` 跑出来其实是个标错的对照臂。
    """
    monkeypatch.setattr(selfplay, "RUNS_DIR", str(tmp_path / "空的"))
    os.makedirs(str(tmp_path / "空的"), exist_ok=True)
    with pytest.raises(RuntimeError, match="种子池是空的"):
        selfplay.train_parallel(seconds=1, workers=1, pfsp=True,
                                out_dir=str(tmp_path / "o"), buffer_games=10,
                                batch_games=2, eval_games=1, eval_every=10 ** 9,
                                log=lambda *a: None)


# ---------------------------------------------------------------- I2：塌陷要响

def test_a_collapsed_pool_fails_the_run(tmp_path, monkeypatch, capsys):
    """池子塌了就不能印「判据过了」并返回 0 —— 那是失败，不是一行日志。"""
    fake = {"games": 1, "curve": [], "best_score": 0.9, "wr_random": 0.9,
            "wr_greedy": 0.99, "out_dir": str(tmp_path), "elapsed": 1.0,
            "qmax": 0, "pool_collapsed": True}
    monkeypatch.setattr(selfplay, "train_parallel", lambda **kw: fake)
    rc = selfplay.main(["1", "--workers", "2"])
    out = capsys.readouterr().out
    assert rc == 1, "池子塌了却返回 0"
    assert "池子塌了" in out, f"没有报出来：{out}"
