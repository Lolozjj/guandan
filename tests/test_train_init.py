"""`--init` 必须**真的**把权重装进去。

否则「热启动」是假的：两臂都从随机初始化开始，A/B 量的是别的东西。
而这件事**在日志上看不出来** —— 所以必须用测试钉住，不能只靠调用一次 `load_state_dict`。

为什么池子实验离不开它：池子只有在模型足够强时才有意义（对手打不打得动你）。
没有热启动，验证池子得先练 9 小时；而且 A/B 两臂必须从**同一个起点**出发才可比。
"""
import os

import pytest

import torch

from train import selfplay
from train.net import QNet


def test_init_warm_starts_from_the_checkpoint(tmp_path):
    """端到端：1 秒训练后，权重离 checkpoint **近**、离随机初始化**远**。"""
    torch.manual_seed(0)
    src = QNet()
    with torch.no_grad():                      # 造一份「好认」的权重
        for p in src.parameters():
            p.mul_(3.0)
    ck = os.path.join(str(tmp_path), "init.pt")
    torch.save({"net": src.state_dict()}, ck)

    r = selfplay.train(seconds=1, init=ck, out_dir=str(tmp_path),
                       log=lambda *a: None, eval_games=1, eval_every=10 ** 9,
                       batch_games=2, opp_mix=0.0)
    got = torch.load(os.path.join(r["out_dir"], "last.pt"), map_location="cpu")["net"]
    fresh = QNet().state_dict()
    near = max((got[k] - v).abs().max().item() for k, v in src.state_dict().items())
    far = max((got[k] - v).abs().max().item() for k, v in fresh.items())
    assert near < far * 0.5, f"热启动没生效（near={near:.3f} far={far:.3f}）"


def test_load_init_is_a_noop_without_a_path():
    """不传 `--init` 时不许改变任何东西（默认行为不变）。"""
    torch.manual_seed(0)
    net = QNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    selfplay.load_init(net, None)
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} 被改了"


# ---------------------------------------------------------------- ε 起点（2026-09-27）

def test_resolve_eps_start_lowers_it_for_a_warm_start():
    """热启动的 ε 起点必须低。

    这是三臂 A/B/C **每一臂都比起点差**（95.2% → 90~94.5%）的直接原因：
    1407 那版 95.2% 的模型，热启动之后 ε 仍从 1.0 起 —— 头一万多局近乎随机，
    把热启动冲掉了。**不修它，任何「从 1407 出发」的实验都测不出别的东西。**
    """
    from train.selfplay import EPS_END, EPS_START, EPS_START_WARM, resolve_eps_start
    assert resolve_eps_start(init=None) == EPS_START
    assert resolve_eps_start(init="runs/rl/x/best.pt") == EPS_START_WARM
    assert EPS_END < EPS_START_WARM < EPS_START, "要落在「退到底」与「全新开跑」之间"
    assert resolve_eps_start(init="x.pt", explicit=0.5) == 0.5, "显式给了就用显式的"
    assert resolve_eps_start(init=None, explicit=0.5) == 0.5


def test_eps_for_honours_an_explicit_start():
    from train.selfplay import EPS_END, eps_for
    assert eps_for(0, 1000, start=0.3) == pytest.approx(0.3)
    assert eps_for(1000, 1000, start=0.3) == pytest.approx(EPS_END)
    assert eps_for(500, 1000, start=0.3) < 0.3, "中途要真的在退"
    assert eps_for(0, 1000) == pytest.approx(1.0), "默认起点不许变（老行为）"


def test_a_warm_start_logs_where_eps_starts(tmp_path):
    """端到端：日志头必须说清这一轮 ε 从哪起。

    否则「为什么热启动之后反而变差了」在日志上**查不出来** —— 这正是上一轮
    三臂实验花了 3 小时才看出来的那件事。
    """
    torch.manual_seed(0)
    ck = os.path.join(str(tmp_path), "init.pt")
    torch.save({"net": QNet().state_dict(), "games": 1234}, ck)
    lines = []
    selfplay.train(seconds=0.0, init=ck, out_dir=os.path.join(str(tmp_path), "o"),
                   log=lines.append, eval_games=1, eval_every=10 ** 9,
                   batch_games=2, opp_mix=0.0)
    head = "\n".join(lines[:3])
    assert "ε 起点" in head, f"日志头没说 ε 起点：{head}"
    assert "0.30" in head, f"热启动的 ε 起点不是 0.30：{head}"


def test_a_cold_start_still_starts_eps_at_one(tmp_path):
    """不热启动时 ε 起点仍是 1.0（默认行为不变）。"""
    lines = []
    selfplay.train(seconds=0.0, out_dir=os.path.join(str(tmp_path), "o"),
                   log=lines.append, eval_games=1, eval_every=10 ** 9,
                   batch_games=2, opp_mix=0.0)
    head = "\n".join(lines[:3])
    assert "1.00" in head, f"冷启动的 ε 起点被改了：{head}"
