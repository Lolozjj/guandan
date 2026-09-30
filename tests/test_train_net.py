"""批量打分必须与「逐个打分」**完全一致** —— 不然就是给网络喂了错的局面。"""
import random

import torch

from guandan.sim import env
from guandan.rl.net import QNet, q_argmax_batch, q_values
from guandan.rl.policies import greedy_policy
from guandan.rl import replay


def test_batched_argmax_equals_one_by_one():
    """32 局各自推进若干步，攒一批决策点：批量 argmax 必须与逐个 `q_values` 一致。

    这条不一致的话，自对弈会按错的分数选牌 —— 而且**不报错**。
    """
    torch.manual_seed(0)
    net = QNet()                       # 留在 CPU 上（设备从 net 上取，见 q_values 的注释）
    rng = random.Random(0)

    pending = []
    for _ in range(8):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset(level=rng.randint(1, 13))
        obs = e.observe()
        for _step in range(3):
            if e.done:
                break
            acts = e.legal()
            pending.append((obs, acts, env.encode_history(e.hand, e.hand.turn)))
            obs, _r, _d, _i = e.step(greedy_policy(obs, acts))

    batched = q_argmax_batch(net, pending)
    one_by_one = [int(q_values(net, o, a, h).argmax()) for o, a, h in pending]
    assert batched == one_by_one, "批量打分与逐个打分的 argmax 不一致"
    assert len(batched) == len(pending)
