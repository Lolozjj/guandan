"""出手分组：谁出手 → 用哪份权重。**纯函数单测 + 真跑一次**。

为什么这是本改动唯一的架构级改动（spec §3.3）：worker 现在把每个决策点**二分**
（学习队问网络 / 固定对手走便宜的 Python）。池子要求它按**谁在出手**分组，
每组各做一次批量前向 ——「当前权重」只是众多组里的一组。

⚠️ 分组错了会**静默用错权重**（池子里全变成一个模型），日志上完全看不出来。
所以 `plan_step` 抽成纯函数单独钉，而不是埋在循环里。
"""
import random

import pytest
import torch

from train import selfplay
from train.net import QNet


def test_plan_step_dispatches_by_who_acts():
    """分组是纯函数，先把这张表钉死。"""
    learn = (0, 2)
    assert selfplay.plan_step(learn, 0, None) == ("learner", None)
    assert selfplay.plan_step(learn, 2, ("greedy", 1)) == ("learner", None)
    assert selfplay.plan_step(learn, 1, ("greedy", 1)) == ("fixed", "greedy")
    assert selfplay.plan_step(learn, 3, ("random", 1)) == ("fixed", "random")
    assert selfplay.plan_step(learn, 1, ("member", 7)) == ("member", 7)
    # 纯自对弈（四家都学）：谁都走 learner
    assert selfplay.plan_step((0, 1, 2, 3), 1, None) == ("learner", None)


def test_plan_step_refuses_an_impossible_state():
    """「不属学习队、又没有固定对手」是不该出现的 —— 必须炸，不许猜。

    猜的后果是**静默用错策略**：它会退回贪心，而日志上只看到「池子里的对手
    好像有点弱」。本仓库纪律：失败必须响。
    """
    with pytest.raises(ValueError):
        selfplay.plan_step((0, 2), 1, None)


def test_a_missing_member_raises_loudly():
    """抽到不在 `members` 里的成员**必须炸**。

    静默退回贪心会让池子悄悄少一个成员，而 A/B 的结果就没法解释。
    """
    torch.manual_seed(0)
    with pytest.raises(KeyError):
        selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=2,
                                capture=False, opp_mix=1.0, members={},
                                pick_fixed=lambda r: ("member", 7))


def test_members_actually_play_with_their_own_weights():
    """不同成员的出手必须**真的不同** —— 都一样说明权重没用对。

    用桩网络：`q_argmax_batch` 只要求 `net(state, action, hist)` 返回每行一个标量，
    并且**能从 `net.parameters()` 取到设备**（所以桩必须有一个哑参数，否则
    `next(net.parameters())` 抛 StopIteration）。
    """
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))   # 只为 next(parameters())

    class First(_Stub):
        def forward(self, state, action, hist):
            return torch.zeros(state.shape[0])                # argmax 恒取第 0 个候选

    class Last(_Stub):
        def forward(self, state, action, hist):
            return torch.arange(state.shape[0], dtype=torch.float32)   # 恒取最后一个

    def run(member):
        torch.manual_seed(0)
        out = selfplay.generate_batch(QNet().eval(), random.Random(3), eps=0.0,
                                      n_games=4, capture=False, opp_mix=1.0,
                                      greedy_share=0.0, members={9: member},
                                      pick_fixed=lambda r: ("member", 9))
        return [rec.actions for rec in (o[0] for o in out)]

    assert run(First()) != run(Last()), "两个极端成员打出了一模一样的东西"
