"""`--mc-mix` 等开关必须真的透传到训练函数（写错了在日志里看不出来）。"""
import inspect

import guandan.rl.selfplay as sp


def test_cli_forwards_the_bootstrap_knobs(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--mc-mix", "0.5", "--n-step", "2", "--tgt-sync", "500"])
    assert (seen["mc_mix"], seen["n_step"], seen["tgt_sync"]) == (0.5, 2, 500)


def test_defaults_are_the_old_behaviour():
    """默认必须是**老的 DMC**：β=1（自举关，n 根不起作用）、每 1000 局同步。

    ⚠️ `n_step` 的默认值**从 3 改成了 2**（2026-09-28 评审的视角修正）——
    它是偶数：`V(s_{t+n})` 是「那一刻出手的人」那一队的分，而出手顺序是 0→3→2→1，
    奇数 n 的自举源落在对家（实测命中率 16% vs 89%）。**这条与「默认行为不变」不冲突**：
    β=1 时 n 完全不参与运算。见 `replay._boot_source_ok` 与 spec §3.1 的修正块。
    """
    sig = inspect.signature(sp.train)
    assert sig.parameters["mc_mix"].default == 1.0
    assert sig.parameters["n_step"].default == 2
    assert sig.parameters["tgt_sync"].default == 1000
