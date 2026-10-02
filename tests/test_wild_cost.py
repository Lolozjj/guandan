"""**万能牌代价**（`wild_cost`）：标签的唯一实现，与 `bomb_cost` 同构。

它要修的是用户 2026-10-01 实机反馈的第 2 条：
「炸个 8888 就行了，却用万能牌凑成五个 8」——
终局回报**看不见**这种浪费（脚本对手也不惩罚它），只能把代价显式挂在动作上。

三件事必须钉住：
1. `wild_cost=0` 时**逐点等于**老标签（默认行为不变）；
2. 代价按**后缀**累计：用了 k 张万能牌，从这一步起的标签少 `λ·k`；
3. **只算自己**用掉的（别人用万能牌不该改我的标签）。
"""
import pytest

from guandan.sim import meld, rules
from guandan.rl import replay

#: 打 2 时 ♥2 是逢人配（万能牌）。
#:
#: ⚠️ **口径坑（实测踩到）**：牌型点数**等于级牌**时，♥级牌不算"替身"
#: （`_wild_used_same_rank`：`idx == level` 直接返回 0）⇒ 用「♥2 + ♠2 凑一对 2」
#: 造不出 `wild_used=1`。要造它得用**非级牌**的牌型：一对外 5 里补一张 ♥2。
LV = 2
WILD = 34              # ♥2（打 2 时的逢人配）—— 实测 id，不是猜的
PAIR_NAT = [21, 53]    # ♠5 + ♣5：天然一对 5
PAIR_WILD = [21, 34]   # ♠5 + ♥2：同一对 5，但**用掉一张万能牌**（wind_used=1）
RANKS = [1, 2, 3, 4]


def _m(ids, level=LV):
    m = meld.as_meld(list(ids), level)
    assert m is not None, ids
    return m


def test_wild_card_is_recognized():
    assert meld.is_wild(WILD, LV), "前提错了：♥2 打 2 时应当是万能牌"
    assert _m(PAIR_WILD).wild_used == 1 and _m(PAIR_NAT).wild_used == 0
    assert not meld.is_wild(WILD, 3)


def test_lambda_zero_reproduces_the_old_label():
    seq = [(0, _m(PAIR_WILD)), (1, _m(PAIR_NAT)), (0, None)]
    assert replay.mc_targets(seq, RANKS, wild_cost=0.0) == \
        [rules.reward(RANKS, s) for s, _m_ in seq]


def test_cost_is_charged_from_that_step_on():
    """用了一张万能牌 ⇒ 从这一步起的标签都少 λ。"""
    wild = _m(PAIR_WILD)
    assert wild.wild_used == 1, f"前提错了：wild_used={wild.wild_used}"
    plain = _m(PAIR_NAT)
    seq = [(0, wild), (0, plain)]
    y = replay.mc_targets(seq, RANKS, wild_cost=0.5)
    base = rules.reward(RANKS, 0)
    assert y[0] == pytest.approx(base - 0.5)
    assert y[1] == pytest.approx(base)          # 后面那手没用万能牌


def test_it_counts_only_my_own_wildcards():
    """别人用万能牌**不该**改我的标签（与 bomb_cost 的口径一致）。"""
    other = [(1, _m(PAIR_WILD)), (0, _m(PAIR_NAT))]
    y = replay.mc_targets(other, RANKS, wild_cost=0.5)
    assert y[1] == pytest.approx(rules.reward(RANKS, 0))
    mine = [(0, _m(PAIR_WILD)), (1, _m(PAIR_NAT))]
    y2 = replay.mc_targets(mine, RANKS, wild_cost=0.5)
    assert y2[0] == pytest.approx(rules.reward(RANKS, 0) - 0.5)


def test_negative_is_rejected():
    with pytest.raises(ValueError, match="wild_cost"):
        replay.mc_targets([(0, None)], RANKS, wild_cost=-0.1)


def test_cli_forwards_wild_cost():
    import guandan.rl.selfplay as sp
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0, "games": 0,
                "curve": [], "best_score": 1.0, "elapsed": 0.0}

    orig = sp.train
    sp.train = fake_train
    try:
        sp.main(["1", "--wild-cost", "0.3"])
        assert seen["wild_cost"] == 0.3
    finally:
        sp.train = orig


def test_worker_cfg_carries_wild_cost():
    from guandan.rl import worker
    cfg = worker.worker_cfg(1, 0.1, 0.5, 0.8, 4, wild_cost=0.4)
    assert cfg["wild_cost"] == 0.4
    assert worker.worker_cfg(1, 0.1, 0.5, 0.8, 4)["wild_cost"] == 0.0
