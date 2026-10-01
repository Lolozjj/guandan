"""结构指标（A6b）里**两个我自己写错过**的地方：胜负判据、名次形态。

两个错都会让报告**看起来更好**（或更整齐），而且跑得通、不报错 —— 所以必须有牙。
"""
import pytest

from tools import structure_metrics as sm


def test_win_criterion_matches_the_ruler():
    """⚠️ `min(我方) <= 2` 是错的：掼蛋是「**较好的名次更靠前**的那队赢」。

    我第一版这么写，`[2,4]`（我方 2、4 名 vs 对方 1、3 名）被算成赢 ⇒ 胜率虚高到 90.9%
    （尺子是 71.2%）。
    """
    assert sm.is_win([1, 3, 2, 4], 0)          # 我方（座位 0/2）= 1、2 名
    assert not sm.is_win([2, 1, 3, 4], 0)      # 我方 = 2、3 名；对方 = 1、4 名
    assert sm.is_win([1, 2, 4, 3], 0)          # 我方 = 1、4 名 —— 较好的名次是 1
    assert not sm.is_win([2, 4, 1, 3], 1)      # team1 = 3、4 名


def test_shape_lists_all_six_patterns():
    """名次是 1..4 的排列 ⇒ **六种都要能标**。第一版只列四种，把 `1-4`（我方赢）
    和 `2-3`（我方输）都塞进「被双上」，于是自相矛盾（胜率 90.9% 而双上只有 45.9%）。"""
    assert sm._shape([1, 2]) == "双上(赢)"
    assert sm._shape([2, 1]) == "双上(赢)"        # 与座位顺序无关
    assert sm._shape([1, 3]) == "1-3(赢)"
    assert sm._shape([1, 4]) == "1-4(赢)"
    assert sm._shape([2, 3]) == "2-3(输)"
    assert sm._shape([2, 4]) == "2-4(输)"
    assert sm._shape([3, 4]) == "被双上(输)"
    with pytest.raises(KeyError):
        sm._shape([1, 1])                         # 不可能的名次：宁可炸


def test_every_pattern_is_covered_and_consistent_with_is_win():
    """六种形态的胜负标注必须与 `is_win` 一致（免得再出现"标注说赢、判据说输"）。"""
    cases = {(1, 2): 0, (1, 3): 0, (1, 4): 0, (2, 3): 1, (2, 4): 1, (3, 4): 1}
    for pattern, winner in cases.items():
        ranks = [0, 0, 0, 0]
        ranks[0], ranks[2] = pattern[0], pattern[1]
        left = [r for r in (1, 2, 3, 4) if r not in pattern]
        ranks[1], ranks[3] = left
        assert sm.is_win(ranks, winner)
        assert not sm.is_win(ranks, 1 - winner)
        assert sm._shape([ranks[0], ranks[2]]) in ("双上(赢)", "1-3(赢)", "1-4(赢)",
                                                  "2-3(输)", "2-4(输)", "被双上(输)")
