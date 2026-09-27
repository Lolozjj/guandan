"""`--init` 必须**真的**把权重装进去。

否则「热启动」是假的：两臂都从随机初始化开始，A/B 量的是别的东西。
而这件事**在日志上看不出来** —— 所以必须用测试钉住，不能只靠调用一次 `load_state_dict`。

为什么池子实验离不开它：池子只有在模型足够强时才有意义（对手打不打得动你）。
没有热启动，验证池子得先练 9 小时；而且 A/B 两臂必须从**同一个起点**出发才可比。
"""
import os

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
