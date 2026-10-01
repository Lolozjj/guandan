"""B4：评测口径的两个开关必须真的透传。

`best.pt` 就是按**训练内评测**挑的，所以这两个数直接影响"挑出来的最好那份有多可信"：
默认 200 局的 sd ≈ 2pp，与一代的进步（~3pp）同量级 —— 等于在抽签。
"""
import inspect

import guandan.rl.selfplay as sp


def _fake_train(seen):
    def f(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}
    return f


def test_cli_forwards_eval_knobs(monkeypatch):
    seen = {}
    monkeypatch.setattr(sp, "train", _fake_train(seen))
    sp.main(["1", "--eval-games", "600", "--eval-every", "2000",
             "--eval-rule-games", "400"])
    assert (seen["eval_games"], seen["eval_every"], seen["eval_rule_games"]) == (600, 2000, 400)


def test_defaults_are_unchanged(monkeypatch):
    """没传的时候参数**不进 kw**（走签名默认值）—— 默认口径不许悄悄变。"""
    assert inspect.signature(sp.train).parameters["eval_games"].default == 200
    assert inspect.signature(sp.train).parameters["eval_every"].default == 10_000
    assert inspect.signature(sp.train_parallel).parameters["eval_games"].default == 200
    seen = {}
    monkeypatch.setattr(sp, "train", _fake_train(seen))
    sp.main(["1"])
    assert "eval_games" not in seen and "eval_every" not in seen
