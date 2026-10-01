"""集成（`rl/ensemble.py`）：**两个不变量**先钉住 —— 它们错了整条实验就是假的。

1. 两个**完全相同**的网络做输出集成，必须与单个逐位一致（平均不改变排序）。
2. 权重平均（soup）在成员完全相同时必须还原成那份权重、输出逐位一致。

另外钉住"成员形状不同也能装"（老件 (700,143) 与现役 (727,146) 混着用是常态）。
"""
import numpy as np
import pytest
import torch

from guandan.rl.ensemble import ensemble_policy, load_net, soup
from guandan.rl.net import QNet, q_values
from guandan.sim import env


def _obs_acts(seed=3):
    e = env.GuandanEnv(seed=seed)
    e.reset(level=8)
    return e, e.observe(), e.legal(), env.encode_history(e.hand, e.hand.turn)


def test_ensemble_of_two_identical_nets_equals_one():
    net = QNet().eval()
    e, obs, acts, hist = _obs_acts()
    single = int(np.argmax(q_values(net, obs, acts, hist)))
    both = ensemble_policy([net, net])(obs, acts, hist)
    assert both == single


def test_soup_of_identical_weights_is_that_net():
    net = QNet().eval()
    out = soup([net, net])
    for k, v in net.state_dict().items():
        assert torch.allclose(out.state_dict()[k], v, atol=1e-6), k
    e, obs, acts, hist = _obs_acts()
    assert np.allclose(q_values(out, obs, acts, hist), q_values(net, obs, acts, hist), atol=1e-5)


def test_members_with_different_layouts_can_be_loaded(tmp_path):
    """老件 (700,143) 与现役 (727,146) 混用是常态 —— 装载必须透明（零填充）。"""
    from guandan.rl.net import load_state
    r11 = QNet()
    sd = r11.state_dict()
    old = {k: v.clone() for k, v in sd.items()}
    # 造一个"老形状"：砍掉状态块多出来的 27 列与动作块多出来的 3 列
    w = old["mlp.0.weight"]
    s_old = env.STATE_DIM - 27
    a_old = env.ACTION_BASE_DIM
    old["mlp.0.weight"] = torch.cat(
        [w[:, :s_old], w[:, env.STATE_DIM:env.STATE_DIM + a_old],
         w[:, env.STATE_DIM + env.ACTION_DIM:]], dim=1)
    p = tmp_path / "old.pt"
    torch.save({"net": old}, p)
    n = load_net(str(p))                      # 不该抛
    e, obs, acts, hist = _obs_acts()
    assert len(q_values(n, obs, acts, hist)) == len(acts)


def test_weights_length_must_match():
    net = QNet().eval()
    with pytest.raises(ValueError, match="个数"):
        ensemble_policy([net], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="正"):
        ensemble_policy([net], weights=[0.0])
