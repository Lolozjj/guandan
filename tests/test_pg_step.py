"""PG 训练步：`R` 的滑动均值基线 + 熵/logits 两处守门 + **一步真的动参数**。"""
import random

import pytest
import torch

from tests.test_pg_logprob import _samples
from guandan.rl.net import DEVICE, QNet
from guandan.rl.selfplay import BETA_ENT, _pg_step, _RunningMean


def test_running_mean_moves_toward_the_data():
    b = _RunningMean(window=10)
    for _ in range(10):
        b.update(1.0)
    assert abs(b.value - 1.0) < 1e-9
    b.update(-1.0)
    assert b.value < 1.0
    assert abs(b.value - 0.8) < 1e-9          # 10 个 1.0 加 1 个 -1.0


def test_running_mean_starts_at_zero_and_windows():
    """空的时候基线是 0（= 不做基线），窗口满了就丢掉最老的。"""
    b = _RunningMean(window=2)
    assert b.value == 0.0
    b.update(3.0)
    b.update(3.0)
    assert b.value == 3.0
    b.update(-3.0)
    assert b.value == 0.0                    # (3 −3)/2


def test_pg_step_changes_the_parameters_and_reports():
    """⚠️ 与 Task 1 那条同一个命门：**一步必须真的动参数**。"""
    net = QNet().to(DEVICE)      # ⚠️ 跟 train() 一样搬上去
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    samples = _samples(3)
    out = _pg_step(net, opt, samples, [2.0] * len(samples), _RunningMean(), games=32)
    assert set(out) == {"loss", "entropy", "adv"}
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters())), \
        "训一步参数没变 —— 梯度断了"


def test_pg_step_reports_the_advantage_after_the_baseline():
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.0)      # 不更新，只看报告
    samples = _samples(2)
    base = _RunningMean()
    base.update(1.0)
    out = _pg_step(net, opt, samples, [3.0] * len(samples), base, games=0)
    assert out["adv"] == pytest.approx(2.0)


def test_pg_step_raises_when_the_policy_collapses(monkeypatch):
    """把输出层权重放大 1e4 倍 ⇒ logits 随候选剧变 ⇒ 策略退化成 one-hot
    ⇒ **守门必须响亮地炸**（熵塌 或 `|logits|` 超限，两个 RuntimeError 都算数）。

    ⚠️ 我第一版把最后一层**清零** ⇒ 输出恒等 ⇒ softmax 均匀 ⇒ 熵**最大**，
    既不塌也不炸 —— 名字写着"会炸"、断言却是"不炸"。自己写岔了，改掉。
    """
    # ⚠️ 第一版是 `net.mlp[-1].weight.mul_(1e4)` —— **依赖初始化**：主干输出若恰好
    # 接近 0，logits 就摊平了、根本不会塌 ⇒ 单跑绿、全量红（随机种子不同）。
    # 改成 monkeypatch 直接造极端 logits ⇒ 完全确定，与初始化无关。
    import guandan.rl.net as net_mod

    def _extreme(_net, pending, grad=False):
        """每个局面都把**最后一个候选**抬到 200 ⇒ one-hot ⇒ 熵≈0。

        ⚠️ 量级要**低于** `LOGIT_ABS_MAX`（1e4），否则响的是 logits 守门、
        测不到熵那一条（评审 I3）；而且必须 `requires_grad=True`，
        否则把两个守门都删掉之后 `backward()` 会报另一种 RuntimeError，
        测试照样"通过" —— 那它就没有牙了。
        """
        rows, counts = [], []
        for _o, a, _h in pending:
            counts.append(len(a))
            rows += [0.0] * (len(a) - 1) + [200.0]
        return torch.tensor(rows, requires_grad=True), counts

    monkeypatch.setattr(net_mod, "_flat_scores", _extreme)
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.0)
    samples = _samples(2)
    with pytest.raises(RuntimeError, match="熵"):
        _pg_step(net, opt, samples, [1.0] * len(samples), _RunningMean(), games=0)


def test_beta_ent_default_is_small_and_positive():
    assert 0.0 < BETA_ENT <= 0.05


def test_pg_step_feeds_the_baseline():
    """⚠️ **评审抓到的 Critical**：`base.update` 原来一次都没被调用过 ⇒ `adv ≡ R`
    （而 ~95% 的局都赢 ⇒ R 几乎恒正 ⇒ 更新退化成"无条件抬升采样到的那一手"）。

    这条钉住「一步之后基线必须落在这一批 R 上」。
    """
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.0)
    samples = _samples(2)
    base = _RunningMean()
    assert base.value == 0.0
    _pg_step(net, opt, samples, [5.0] * len(samples), base, games=0)
    assert base.value == pytest.approx(5.0)


def test_adv_is_centered_once_the_baseline_is_warm():
    """基线喂热之后，同一批 R 的 `adv` 必须**接近于 0** —— 这才叫"中心化"。"""
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.0)
    samples = _samples(2)
    base = _RunningMean()
    for _ in range(3):
        base.update(5.0)
    out = _pg_step(net, opt, samples, [5.0] * len(samples), base, games=0)
    assert abs(out["adv"]) < 1e-6
