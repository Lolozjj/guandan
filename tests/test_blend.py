"""`blend` —— MC 标签与自举值的混合（spec §3.1）。"""
import pytest

from train import replay


def test_beta_one_is_exactly_the_old_labels():
    """β=1 = 现在的 DMC（默认）。**逐点相等**，不是「约等于」。"""
    y = [-3.0, 1.0, 2.0]
    assert replay.blend(y, [9.9, 9.9, 9.9], 1.0) == y


def test_beta_one_ignores_a_poisoned_bootstrap_value():
    """β=1 时**连看**都不看自举值 —— 结构上相等，不是浮点凑出来的。

    把自举值塞成 inf：若实现写成 (1-β)·v + β·y，会算出 0.0*inf = nan。"""
    assert replay.blend([1.0], [float("inf")], 1.0) == [1.0]


def test_beta_zero_is_pure_bootstrap():
    assert replay.blend([1.0, -3.0], [2.0, 4.0], 0.0) == [2.0, 4.0]


def test_half_mixes_halfway():
    assert replay.blend([0.0], [4.0], 0.5) == [2.0]


def test_none_falls_back_to_the_mc_label():
    """越过终局、或本轮不算自举的点，**整项退回 MC**（不许编一个 0）。"""
    assert replay.blend([1.0, -3.0], [None, 5.0], 0.0) == [1.0, 5.0]


def test_all_none_is_the_mc_labels():
    assert replay.blend([1.0, -3.0], [None, None], 0.5) == [1.0, -3.0]


def test_no_boot_at_all_is_the_mc_labels():
    """`boot=None` 的整条路（`n=0`）也要能用。"""
    assert replay.blend([1.0, -3.0], None, 0.5) == [1.0, -3.0]


def test_length_mismatch_raises():
    """错开一格就是**拿别人的未来当自己的标签** —— 必须炸，不许 zip 截断。"""
    with pytest.raises(ValueError):
        replay.blend([1.0, 2.0], [1.0], 0.5)


@pytest.mark.parametrize("beta", [-0.1, 1.5])
def test_beta_out_of_range_raises(beta):
    with pytest.raises(ValueError):
        replay.blend([1.0], [1.0], beta)
