"""探针 `tools/opp_headroom.py` 里**会给出反向结论**的那一步：分位。

`rank_pct` 算反了，判读会整个倒过来 —— 「两者本就很像」和「学生强烈不同意」
是同一组数字的两种读法。而它**跑得通、不报错**，所以必须有牙。
"""
import pytest

from tools.opp_headroom import rank_pct


def test_best_and_worst_are_the_ends():
    qs = [3.0, 2.0, 1.0]
    assert rank_pct(qs, 0) == 1.0          # 学生自己的首选
    assert rank_pct(qs, 2) == 0.0          # 最差那个
    assert rank_pct(qs, 1) == pytest.approx(0.5)


def test_ties_are_not_called_worse():
    """三个并列 —— **没有谁比谁严格更好**，所以都是 1.0。

    按名次比例算的实现会在这里给出 1.0 / 0.5 / 0.0，凭空造出「分歧」。
    """
    assert rank_pct([1.0, 1.0, 1.0], 2) == 1.0


def test_single_candidate_is_not_a_choice():
    """只有一个候选时**谈不上分歧** —— 返回 1.0，调用方靠 `len>=2` 那道门排除它。"""
    assert rank_pct([7.0], 0) == 1.0
