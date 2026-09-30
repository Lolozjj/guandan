"""配对测量工具 `tools/ruler.py` 里**会误导结论的那部分**。

工具本身只是把 `match` 串起来跑；真正能悄悄给出错结论的是统计那两下：
`paired` 算错 ⇒ 「涨了 4pp」和「分辨不出」会互换，而**跑得通、不报错**。
所以这里只钉纯函数，不钉跑局（跑局那层 `match` 有自己的测试）。
"""
import math

import pytest

from tools.ruler import label_of, paired


def test_paired_reports_mean_sd_and_t():
    """手算：差 [2,4,6,8]pp，均值 5pp、sd 2.58pp、t=3.87。"""
    m, sd, t = paired([0.02, 0.04, 0.06, 0.08])
    assert m == pytest.approx(0.05)
    assert sd == pytest.approx(0.02582, abs=1e-5)
    assert t == pytest.approx(3.873, abs=1e-3)


def test_paired_does_not_make_a_claim_from_one_seed():
    """**一个种子算不出离散度** —— 这里必须返回 nan，不许硬报一个 t。

    这是这个工具最容易骗人的地方：单种子 t 会是 ±inf 或某个假的大数，
    看上去像「铁证」。
    """
    m, sd, t = paired([0.04])
    assert m == pytest.approx(0.04)
    assert math.isnan(sd) and math.isnan(t)


def test_paired_survives_zero_variance():
    """四个种子给出**一模一样**的差（散度 0）—— 不能 ZeroDivisionError。"""
    m, sd, t = paired([0.03] * 4)
    assert sd == 0.0
    assert t == math.inf                      # 方向对：一致为正就是铁证


def test_paired_calls_a_consistent_loss_a_loss():
    """散度 0 且一致为负 —— t 必须是负无穷，**不是**正无穷（符号别丢）。"""
    _, _, t = paired([-0.03] * 4)
    assert t == -math.inf


def test_label_of_keeps_the_arm_and_the_file():
    """`runs/R8_rule/best.pt` -> `R8_rule/best.pt`（报告里两臂要能分开）。"""
    assert label_of("runs/R8_rule/best.pt") == "R8_rule/best.pt"
    assert label_of(r"runs\ab\R10_rule\pool\snap_300000.pt") == "R10_rule/snap_300000.pt"
