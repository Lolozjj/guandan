"""信用分配 spec 的设计 2：**按胶着度加权**。

两条最要紧的保证：① 默认 1.0 必须**逐位不变**（走原来那一行 `F.mse_loss`）；
② 权重必须**归一化**（否则改权重会顺手改掉有效学习率 —— 两个变量一起动就说不清）。
"""
import inspect

import pytest
import torch

import guandan.rl.selfplay as sp


def test_weights_only_hit_blowouts():
    """`|标签| = 3` ⟺ 双上 / 被双上（一边倒）；`|标签| < 3` 才是胶着局。"""
    t = [3.0, -3.0, 1.0, -2.0, 2.0, -1.0]
    w = sp._close_weights(t, 2.0, torch.device("cpu"))
    assert list(w) == [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert list(sp._close_weights(t, 1.0, torch.device("cpu"))) == [1.0] * 6


def test_all_ones_weights_reproduce_plain_mse():
    """全 1 权重 == `F.mse_loss`（等式，不是近似）—— 归一化那一项不能歪。"""
    torch.manual_seed(0)
    y_hat, y = torch.randn(7), torch.randn(7)
    w = torch.ones(7)
    weighted = (w * (y_hat - y) ** 2).sum() / w.sum()
    assert torch.allclose(weighted, torch.nn.functional.mse_loss(y_hat, y))


def test_reweighting_actually_shifts_the_gradient():
    """加权要真的把重点挪到胶着样本上：胶着样本误差大时，加权损失必须更高。"""
    y = torch.tensor([3.0, 1.0])            # 一边倒 / 胶着
    y_hat = torch.tensor([3.0, 0.0])        # 只在**胶着**那条上有误差
    plain = torch.nn.functional.mse_loss(y_hat, y)
    w = sp._close_weights([3.0, 1.0], 4.0, torch.device("cpu"))
    weighted = (w * (y_hat - y) ** 2).sum() / w.sum()
    assert weighted > plain, "加权没有把重点挪到胶着样本上"


def test_default_is_one_and_goes_through_the_old_code_path():
    """默认 1.0：签名默认值 + 没传开关就不进 `kw`（默认口径不许悄悄变）。"""
    assert inspect.signature(sp.train).parameters["close_weight"].default == 1.0
    assert inspect.signature(sp.train_parallel).parameters["close_weight"].default == 1.0
    assert inspect.signature(sp._learn_step).parameters["close_weight"].default == 1.0
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}

    orig = sp.train
    sp.train = fake_train
    try:
        sp.main(["1"])
        assert "close_weight" not in seen
        sp.main(["1", "--close-weight", "2"])
        assert seen["close_weight"] == 2.0
    finally:
        sp.train = orig


def test_pg_refuses_the_weight_instead_of_silently_ignoring_it():
    """PG 的损失是策略梯度，没有样本权重 ⇒ 必须**响**，不能静默失效。"""
    with pytest.raises(ValueError, match="close-weight"):
        sp.train(seconds=0, algo="pg", close_weight=2.0)
