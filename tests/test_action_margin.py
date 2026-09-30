"""动作边际：随机初始化的网络应当是 0（p90 也接近 0），训练过的应当明显大于 0。"""
import torch

from tools.action_margin import margins_of, measure
from guandan.rl.net import QNet


def test_untrained_net_has_no_margin():
    """随机初始化的同结构网络 —— 记录在案的数字是 0.000（p90 0.001）。"""
    torch.manual_seed(0)
    _n, st = measure(QNet().eval(), games=2, seed=0)
    assert st["median_margin"] < 0.01


def test_measure_reports_the_fields_the_criterion_needs():
    torch.manual_seed(0)
    n, st = measure(QNet().eval(), games=2, seed=0)
    assert n > 0 and st["n_margins"] > 0
    for k in ("median_margin", "p90_margin", "tie_share", "median_spread"):
        assert k in st


def test_margins_are_never_negative():
    """`top1 >= top2` 是排序的性质，负的边际只可能是取错了下标。"""
    torch.manual_seed(0)
    _n, st = measure(QNet().eval(), games=2, seed=0)
    assert st["median_margin"] >= 0.0


def test_margin_definition_is_top1_minus_top2():
    """钉住口径：边际 = 排序后**前两名之差**（不是「首选与均值之差」）。"""
    import tools.action_margin as am

    def fake_q(_net, _obs, acts, _hist):
        assert len(acts) == 4, "这一条只对 4 个候选有意义"
        return torch.tensor([0.0, 1.0, 2.0, 10.0])

    orig, am.q_values = am.q_values, fake_q
    try:
        # 4 个候选、Q 是 0/1/2/10：前两名之差 8，而「首选减均值」是 6.75 ——
        # 用 3 个候选的话这两个口径会撞成同一个数，测不出区别
        got = margins_of(object(), [("o", ["a", "b", "c", "d"], "h")])
    finally:
        am.q_values = orig
    assert got == [8.0]
