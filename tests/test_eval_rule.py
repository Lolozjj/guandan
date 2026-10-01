"""训练内评测要把 **vs 规则式** 也报出来 —— 它才是现在的主尺子。

为什么这件事值得一条测试：

- `vs 贪心` 早在 90%+ 饱和（量不出代差），而规则式「像人」（不压队友、留炸、算剩牌），
  还留着几十个百分点余量 —— `tools/ruler.py` 就是拿它做多臂配对的主尺子。
- 但规则式是**纯 Python 启发式**，比贪心慢一个量级 ⇒ 它必须**可关**
  （`rule_games=0`），否则量吞吐的短跑（`tools/bench_train.py`）会把慢尺子算进开销。

所以这里钉两件事：**默认要报**、**要能关**。
"""
import inspect

import pytest

import guandan.rl.selfplay as sp
from guandan.rl.net import QNet


def _sentinel(*_a):
    """冒充规则式策略 —— 靠它认「这一把是不是拿规则式当的对手」。"""
    return 0


@pytest.fixture
def wired(monkeypatch):
    """把 `match` 与 `rule_policy` 换成探针，返回调用记录 `[(对手, 局数, seed), …]`。"""
    seen = []

    def fake_match(_pol, opp, games, seed):
        seen.append((opp, games, seed))
        return 0.5

    monkeypatch.setattr(sp, "match", fake_match)
    monkeypatch.setattr(sp, "rule_policy", lambda: _sentinel)
    return seen


def test_periodic_eval_reports_all_three_rulers(tmp_path, wired):
    """每 1 万局那行要同时有三个数；曲线的第 4 列也要带上规则式。"""
    curve, lines = [], []
    sp._maybe_eval(QNet().eval(), 10_000, curve, -1.0, 4, 10_000, 32, str(tmp_path),
                   lines.append, snap_every=0, rule_games=4)
    assert any(opp is _sentinel for opp, _g, _s in wired), "没有拿规则式当对手评过"
    assert "vs 规则式" in lines[0], lines[0]
    assert len(curve[0]) == 4 and curve[0][3] == 0.5


def test_rule_ruler_can_be_switched_off(tmp_path, wired):
    """`rule_games=0` ⇒ 不跑规则式；那一列写 `None`，不凭空造个数。"""
    curve, lines = [], []
    sp._maybe_eval(QNet().eval(), 10_000, curve, -1.0, 4, 10_000, 32, str(tmp_path),
                   lines.append, snap_every=0, rule_games=0)
    assert not any(opp is _sentinel for opp, _g, _s in wired)
    assert "vs 规则式" not in lines[0], lines[0]
    assert curve[0][3] is None


def test_rule_ruler_uses_the_same_boards_as_vs_greedy(tmp_path, wired):
    """收尾那两次用**同样的 seed**（= 同一副牌）⇒ 两组数可以直接对着看。"""
    lines = []
    r = sp._finish(QNet().eval(), games=10, steps=1, t0=0.0, curve=[],
                   best_greedy=-1.0, eval_games=4, out_dir=str(tmp_path),
                   log=lines.append, bomb_games=0, rule_games=4)
    seeds = [s for _opp, _g, s in wired]
    assert 2001 in seeds and 2002 in seeds
    assert r["wr_rule"] == 0.5
    assert any("vs 规则式" in line for line in lines)


def test_bench_does_not_measure_the_slow_ruler():
    """`tools/bench_train.py` 量的是吞吐 —— 它必须把规则式那把尺子关掉。"""
    import tools.bench_train as bt
    assert "eval_rule_games=0" in inspect.getsource(bt.bench_run)


def test_cli_forwards_eval_rule_games(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0, "wr_rule": 0.7,
                "games": 0, "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--eval-rule-games", "0"])
    assert seen["eval_rule_games"] == 0


def test_defaults_are_none_so_old_calls_keep_working():
    """两条训练路线的默认值都是 `None`（= 跟 `eval_games` 一样）—— 老调用方不受影响。"""
    for fn in (sp.train, sp.train_parallel):
        assert inspect.signature(fn).parameters["eval_rule_games"].default is None
