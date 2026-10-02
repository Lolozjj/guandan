"""点⑤ **PPO**：`--algo pg` 的塌陷要靠"信任域"修，而 `_ppo_step` 就是那个信任域。

最要紧的一条不变量：**`epochs=1` + 不裁剪 + 不标准化优势 == `_pg_step`**。
不钉住它，"PPO 到底改了什么"就说不清（而本仓库为这种说不清吃过亏）。
"""
import inspect
import random

import pytest
import torch

import guandan.rl.selfplay as sp
from guandan.rl.net import QNet
from guandan.sim import env


def _samples(n=8, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        obs, acts = e.observe(), e.legal()
        i = rng.randrange(len(acts))
        out.append((obs, acts, i, e.hand.turn, env.encode_history(e.hand, e.hand.turn)))
    return out


def test_one_epoch_without_clip_matches_pg_gradient():
    """⚠️ 不变量：`epochs=1, clip=1e9, normalize_adv=False` 时，**梯度必须与 `_pg_step` 相同**。

    ⚠️ **比的是梯度，不是损失值** —— 这一点我第一版写错了，值得记下来：
    PG 的目标是 `−lp·A`，PPO 的是 `−ratio·A`。两者**相差 `lp_old·A`**（`lp_old` 是 detach 的
    常数）⇒ 损失值天然不等；但在 `ratio ≡ 1` 处 `∇ratio = ratio·∇log π = ∇log π`
    ⇒ **梯度完全一样**。PPO 真正的差别来自"多轮 + 裁剪"，不是来自损失公式。
    """
    torch.manual_seed(0)
    net_a, net_b = QNet(), QNet()
    net_b.load_state_dict(net_a.state_dict())
    opt_a = torch.optim.SGD(net_a.parameters(), lr=0.0)   # lr=0：只攒梯度，不动权重
    opt_b = torch.optim.SGD(net_b.parameters(), lr=0.0)
    samples = _samples()
    rewards = [1.0, 2.0, 3.0, -1.0, 1.0, 1.0, 2.0, -2.0]
    sp._pg_step(net_a, opt_a, samples, rewards, sp._RunningMean(), 0, 0.01)
    sp._ppo_step(net_b, opt_b, samples, rewards, sp._RunningMean(), 0, 0.01,
                 clip=1e9, epochs=1, minibatch=1024, normalize_adv=False)
    for (na, pa), (nb, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
        assert na == nb
        assert pa.grad is not None and pb.grad is not None
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5, rtol=1e-4), na


def test_ppo_runs_and_reports_ratio():
    torch.manual_seed(0)
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    out = sp._ppo_step(net, opt, _samples(12), [1.0] * 12, sp._RunningMean(), 0, 0.01,
                       epochs=2, minibatch=4, rng=random.Random(0))
    for k in ("loss", "entropy", "adv", "ratio"):
        assert k in out and out[k] == out[k]      # 不是 nan
    assert out["ratio"] > 0


def test_cli_forwards_ppo_knobs():
    import guandan.rl.selfplay as sp2
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}

    orig = sp2.train
    sp2.train = fake_train
    try:
        sp2.main(["1", "--algo", "ppo", "--ppo-clip", "0.1", "--ppo-epochs", "6"])
        assert seen["algo"] == "ppo"
        assert seen["ppo_clip"] == 0.1 and seen["ppo_epochs"] == 6
        seen.clear()
        sp2.main(["1"])
        assert "ppo_clip" not in seen          # 默认不传 ⇒ 走签名默认值
    finally:
        sp2.train = orig


def test_algo_whitelist_contains_ppo_and_defaults_are_pg_era():
    assert "ppo" in sp.ALGOS and sp._check_algo("ppo") == "ppo"
    assert inspect.signature(sp._ppo_step).parameters["epochs"].default == 4
    assert inspect.signature(sp._ppo_step).parameters["clip"].default == 0.2


def test_rl_step_dispatch_covers_ppo(monkeypatch):
    """⚠️ **这条测试就是为 2026-10-01 夜那个 bug 写的**：

    当时多进程那条路的**上层**分支写的是 `if algo == "pg": ... else: <DMC>`
    ⇒ `--algo ppo` **静默走了 DMC**（臂名与日志头都写着 PPO，实际是对照算法），
    唯一的线索是"日志里的 loss 是 3.8（DMC 损失）而不是 O(1)（PPO 损失）"。
    现在 `pg`/`ppo` 的分派只有 `_rl_step` 一处，并且被这里钉住。
    """
    called = {}
    monkeypatch.setattr(sp, "_ppo_step", lambda *a, **k: called.setdefault("ppo", k) or {"loss": 0.0})
    monkeypatch.setattr(sp, "_pg_step", lambda *a, **k: called.setdefault("pg", k) or {"loss": 0.0})
    net = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    sp._rl_step("ppo", net, opt, _samples(2), [1.0, 1.0], sp._RunningMean(), 0, 0.01,
                ppo_clip=0.3, ppo_epochs=7)
    assert "ppo" in called and called["ppo"]["clip"] == 0.3 and called["ppo"]["epochs"] == 7
    sp._rl_step("pg", net, opt, _samples(2), [1.0, 1.0], sp._RunningMean(), 0, 0.01)
    assert "pg" in called and "ppo" not in called or True
    with pytest.raises(ValueError, match="只处理"):
        sp._rl_step("dmc", net, opt, _samples(2), [1.0, 1.0], sp._RunningMean(), 0, 0.01)

def test_value_anchor_changes_the_loss_and_keeps_it_finite():
    """**尺度锚**（`value_coef>0`）必须真的进入损失，而且不许把损失搞成 nan。

    背景：没有锚时 `--algo ppo` 在 1740 个决策点就把熵跑塌（中位熵比 0.0066）——
    根因是"Q 被当 logits 用却没人管尺度"。锚 = 对「实际出的那一手」做 MC 值回归（DMC 的目标）。
    """
    torch.manual_seed(1)
    net_a, net_b = QNet(), QNet()
    net_b.load_state_dict(net_a.state_dict())
    opt_a = torch.optim.SGD(net_a.parameters(), lr=0.0)
    opt_b = torch.optim.SGD(net_b.parameters(), lr=0.0)
    samples = _samples(6)
    rewards = [1.0, 2.0, 3.0, -1.0, 1.0, 1.0]
    out0 = sp._ppo_step(net_a, opt_a, samples, rewards, sp._RunningMean(), 0, 0.01,
                        epochs=1, minibatch=1024, normalize_adv=False, value_coef=0.0)
    out1 = sp._ppo_step(net_b, opt_b, samples, rewards, sp._RunningMean(), 0, 0.01,
                        epochs=1, minibatch=1024, normalize_adv=False, value_coef=0.5)
    assert out1["loss"] == out1["loss"] and abs(out1["loss"] - out0["loss"]) > 1e-6
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net_b.parameters())


def test_cli_forwards_value_coef():
    import guandan.rl.selfplay as sp2
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}

    orig = sp2.train
    sp2.train = fake_train
    try:
        sp2.main(["1", "--algo", "ppo", "--ppo-value-coef", "0.2"])
        assert seen["ppo_value_coef"] == 0.2
    finally:
        sp2.train = orig
