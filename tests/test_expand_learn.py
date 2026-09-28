"""`expand` 必须只产出**学习那一队**的决策点。

钉的是一个真 bug（2026-09-27 实测）：`generate_batch` 的现场抓取（`caps`）确实只记
学习那一队，但**单进程那条路几乎不用现场抓取** —— `buf.sample` 从 5 万局里有放回抽
32 局，撞上刚打那 32 局的期望是 **0.02 局/步**，所以实际训练数据 ~100% 来自 `expand`。
而 `GameRecord` 没有「哪一队是学习的」这个信息，`expand` 只能把四家全产出成训练点。

实测污染率：`opp_mix=0.5` 时 6008 个重放训练点里 **1040 个（17.3%）是固定对手的着法**，
标签却是那一局的胜负 —— 等于拿对手当老师。池子会放大它（对手会变成强模型，
从「拿弱基线当老师」变成「蒸馏强策略」），所以它是池子的前置。
"""
import random

import torch

from train import replay, selfplay
from train.net import QNet


def _fresh_net(seed=0):
    torch.manual_seed(seed)
    return QNet().eval()


def _play(n_games=8, **kw):
    return selfplay.generate_batch(_fresh_net(1), random.Random(0), eps=0.0,
                                   n_games=n_games, capture=True, **kw)


def test_expand_only_yields_the_learner_team():
    """`opp_mix=1.0`：重放产出的座位必须**只有学习那一队**。"""
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0):
        learner = {s for (_o, _a, _i, s, _h) in caps}
        pts, y, _b = replay.expand(rec)
        seats = {s for (_o, _a, _i, s, _h) in pts}
        assert seats <= learner, \
            f"重放产出了非学习队的座位：{sorted(seats - learner)}"
        assert len(pts) == len(caps)
        assert y == _y


def test_expand_reproduces_the_learner_points_bit_for_bit():
    """过滤之后，剩下的每一点仍要与现场抓到的**逐点相同**（顺序、座位、下标）。

    过滤要是把局面推进也一起跳过了，这里立刻对不上 ——
    `expand` 的 `e.step(i)` 必须**每一步都走**，过滤只能发生在 `points.append` 那一行。
    """
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0):
        pts, _y2, _b = replay.expand(rec)
        assert [(s, i) for (_o, _a, i, s, _h) in pts] == \
               [(s, i) for (_o, _a, i, s, _h) in caps]


def test_expand_is_unchanged_for_pure_selfplay():
    """纯自对弈（`opp_mix=0`）时四家都学 —— 老行为不许变。"""
    for rec, _caps, _y in _play(opp_mix=0.0):
        pts, _y2, _b = replay.expand(rec)
        assert {s for (_o, _a, _i, s, _h) in pts} == {0, 1, 2, 3}


def test_learn_all_seats_restores_the_old_behaviour():
    """`learn_all_seats=True` = A/B 的**对照臂**：对手照旧换，但四家照旧都学。"""
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0, learn_all_seats=True):
        pts, _y2, _b = replay.expand(rec)
        assert {s for (_o, _a, _i, s, _h) in pts} == {0, 1, 2, 3}
        assert {s for (_o, _a, _i, s, _h) in caps} == {0, 1, 2, 3}, \
            "对照臂连现场抓取也必须是四家 —— 否则两臂差的就不止一处"
