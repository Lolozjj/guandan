"""信用分配 spec 的设计 1：**势函数 shaping**。

理论上最重要的两条：① 默认 0.0 逐位不变；② 望远镜化 ⇒
**加进去的这一项对"同一局面里哪个动作更好"没有影响**（推理的 argmax 口径不变），
它改的只是回归目标。所以这里既钉默认值，也钉那条"不改排序"的性质。
"""
import inspect

import pytest

from guandan.rl import replay
from guandan.sim import meld, rules

A = meld.cid_from_name


def _seq(moves):
    """`moves = [(座位, [牌名...] or None), ...]` -> `[(座位, Meld or None), ...]`。"""
    return [(s, None if cards is None else meld.as_meld([A(c) for c in cards], 8))
            for s, cards in moves]


def test_default_is_off_and_matches_the_plain_formula():
    """`shaping=0.0`（含完全不给）必须与老公式**逐点相等**。"""
    seq = _seq([(0, ["S3"]), (1, ["S4"]), (2, None), (3, ["S5", "H5"])])
    ranks = [1, 3, 2, 4]
    hands0 = [(5, 5, 5, 5)] * 4
    base = replay.mc_targets(seq, ranks, bomb_cost=0.0)
    assert replay.mc_targets(seq, ranks, bomb_cost=0.0, shaping=0.0) == base
    assert replay.mc_targets(seq, ranks, bomb_cost=0.0, hands0=hands0,
                             shaping=0.0) == base
    assert base == [rules.reward(ranks, s) for s, _m in seq]


def test_telescoping_matches_the_increment_form():
    """`β(Φ_T − Φ_t)` 必须等于「逐步势差之和」—— 这就是势函数那种"安全"形式。"""
    seq = _seq([(0, ["S3"]), (1, ["S4"]), (0, ["S5", "H5"]), (2, ["S6"])])
    ranks = [1, 2, 3, 4]
    hands0 = [(2, 2, 2, 2)] * 4
    beta = 0.5
    got = replay.mc_targets(seq, ranks, hands0=hands0, shaping=beta)
    plain = replay.mc_targets(seq, ranks, hands0=hands0, shaping=0.0)

    played = {s: 0 for s in rules.SEATS}
    base = {s: len(hands0[s]) for s in rules.SEATS}
    phis = []
    for s, m in seq:
        phis.append(replay._phi(base, played, s))
        played[s] += 0 if m is None else len(m.cards)
    phi_end = [replay._phi(base, played, s) for s, _m in seq]
    for i in range(len(seq)):
        assert got[i] == pytest.approx(plain[i] + beta * (phi_end[i] - phis[i]))


def test_shaping_cannot_change_the_ranking_within_a_state():
    """⚠️ 设计上最关键的一条：同一局面里两个**同张数、都非炸**的动作，
    shaping 给它们加的是**同一个常数** ⇒ 相对次序不变（推理口径不变）。

    这也是为什么它的判据第一位是「边际」而不是胜率 —— 它不改排序，只改拟合难度。
    """
    seq_a = _seq([(0, ["S3"]), (1, ["S4"])])
    seq_b = _seq([(0, ["S3"]), (1, ["H4"])])          # 只换了一张同花色的牌，张数一样
    ranks = [1, 2, 3, 4]
    hands0 = [(2, 2, 2, 2)] * 4
    p_a = replay.mc_targets(seq_a, ranks, hands0=hands0, shaping=0.0)
    p_b = replay.mc_targets(seq_b, ranks, hands0=hands0, shaping=0.0)
    s_a = replay.mc_targets(seq_a, ranks, hands0=hands0, shaping=0.7)
    s_b = replay.mc_targets(seq_b, ranks, hands0=hands0, shaping=0.7)
    assert [x - y for x, y in zip(s_a, s_b)] == pytest.approx(
        [x - y for x, y in zip(p_a, p_b)])


def test_shaping_without_hands0_must_raise():
    """给了 β 却没给 `hands0` ⇒ 势函数算不出来，**必须响**（不许静默变成没开）。"""
    seq = _seq([(0, ["S3"])])
    with pytest.raises(ValueError, match="hands0"):
        replay.mc_targets(seq, [1, 2, 3, 4], shaping=0.3)


def test_cli_and_defaults():
    assert inspect.signature(replay.mc_targets).parameters["shaping"].default == 0.0
    import guandan.rl.selfplay as sp
    assert inspect.signature(sp.train).parameters["shaping"].default == 0.0
    assert inspect.signature(sp.train_parallel).parameters["shaping"].default == 0.0
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}

    orig = sp.train
    sp.train = fake_train
    try:
        sp.main(["1"])
        assert "shaping" not in seen
        sp.main(["1", "--shaping", "0.3"])
        assert seen["shaping"] == 0.3
    finally:
        sp.train = orig


def test_training_smoke_with_shaping(tmp_path):
    """开 β 之后训练循环能跑、标签不炸（守门会查 |标签|）。"""
    import guandan.rl.selfplay as sp
    lines = []
    r = sp.train(seconds=4, log=lines.append, eval_games=2, eval_every=10 ** 9,
                 opp_mix=0.0, eval_rule_games=0, shaping=0.3, out_dir=str(tmp_path))
    assert r["games"] > 0
    assert any("势函数 shaping" in l for l in lines), "日志必须如实写换了什么"
