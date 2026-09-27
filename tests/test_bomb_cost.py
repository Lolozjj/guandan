"""动作计价（炸弹代价）—— 标签的唯一实现。

规格：`docs/superpowers/specs/2026-09-27-bomb-cost-shaping-design.md`。

**为什么这条路**：一局 ~132 个决策点共享同一个终局标签 ⇒ 动作那一维只剩残差
（实测首选次选之差中位 0.123，标签尺度 ±3）。把代价**挂在动作上**才能造出
「炸 vs 不炸」的对比 —— 挂在状态上的（势函数）在 MC 里会塌成状态基线。
"""
import pytest

from net.sim import meld, rules
from train import replay

#: 真炸弹：5♠5♥5♣5♦（同副四张同点），`is_bomb` 为 True。
#: ⚠️ **同花顺也算 `is_bomb=True`**（掼蛋里它本来就是炸），代价会一并计上它。
BOMB = [21, 37, 53, 69]
#: 5♠5♥，一对，`is_bomb` 为 False。
PLAIN = [21, 37]
RANKS = [1, 2, 3, 4]        # 座位 0（队 0）拿第 1 名 -> 队 0 赢，最差名次 3 -> points=2


def _m(card_ids, level=2):
    m = meld.as_meld(list(card_ids), level)
    assert m is not None, card_ids
    return m


def test_lambda_zero_reproduces_the_old_label():
    """λ=0 必须**逐点等于**老标签（`rules.reward`）—— 默认行为不变。"""
    seq = [(0, _m(BOMB)), (1, _m(PLAIN)), (0, None), (2, _m(BOMB))]
    assert replay.mc_targets(seq, RANKS, bomb_cost=0.0) == \
           [rules.reward(RANKS, s) for s, _mm in seq]


def test_a_bomb_lowers_the_label_by_exactly_lambda():
    """出炸那一步，比同座位不出炸低正好 λ。**这是整条路线的全部意义。**"""
    with_bomb = replay.mc_targets([(0, _m(BOMB))], RANKS, bomb_cost=0.2)
    without = replay.mc_targets([(0, _m(PLAIN))], RANKS, bomb_cost=0.2)
    assert without[0] - with_bomb[0] == pytest.approx(0.2)


def test_the_suffix_counts_bombs_that_come_later():
    """`B_t` 是**从第 t 步往后**的计数：后面那手炸也要算进前面决策点的标签。"""
    both = replay.mc_targets([(0, _m(PLAIN)), (0, _m(BOMB))], RANKS, bomb_cost=0.2)
    alone = replay.mc_targets([(0, _m(PLAIN))], RANKS, bomb_cost=0.2)
    assert alone[0] - both[0] == pytest.approx(0.2), "第 1 步没把后面那手炸算进去"


def test_the_cost_only_counts_the_acting_seats_own_bombs():
    """`B_t` 只数**该座位自己**的 —— 别的座位出炸不该动它的标签。"""
    mine = replay.mc_targets([(0, _m(PLAIN)), (1, _m(BOMB))], RANKS, bomb_cost=0.2)
    alone = replay.mc_targets([(0, _m(PLAIN))], RANKS, bomb_cost=0.2)
    assert mine[0] == pytest.approx(alone[0]), "把别人座位的炸算到自己头上了"


def test_learn_filters_inside_the_function():
    """过滤在函数里面做（别让调用方各写一份 —— 那就是「副本会漂」的入口）。"""
    seq = [(0, _m(PLAIN)), (1, _m(PLAIN)), (2, _m(PLAIN)), (3, _m(PLAIN))]
    assert len(replay.mc_targets(seq, RANKS, learn=(0, 2))) == 2
    assert len(replay.mc_targets(seq, RANKS)) == 4


def test_a_negative_cost_is_rejected():
    with pytest.raises(ValueError):
        replay.mc_targets([(0, _m(PLAIN))], RANKS, bomb_cost=-0.1)


def test_the_two_label_producers_agree_bit_for_bit():
    """现场抓取与 buffer 重放**必须逐点给出同一个标签**（λ=0 与 λ>0 都要）。

    两条路各写一份标签算法就是「副本会漂」—— 漂了的后果是「刚打的那批」与
    「从 buffer 里取的」学到两个不同的目标，且**静默**（这个仓库为此反复吃过亏）。
    """
    import random

    import torch

    from train import selfplay
    from train.net import QNet

    torch.manual_seed(0)
    for lam in (0.0, 0.2):
        out = selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0,
                                      n_games=8, capture=True, opp_mix=1.0,
                                      greedy_share=1.0, bomb_cost=lam)
        for rec, caps, y_live in out:
            pts, y_replay = replay.expand(rec, bomb_cost=lam)
            assert y_live == y_replay, f"λ={lam}：现场与重放的标签不一致"
            assert len(pts) == len(caps) == len(y_live), "条数对不上"


def test_play_capturing_also_uses_the_single_label_source():
    """**第三个产地**（`play_capturing`）也要收进 `mc_targets`。

    它今天只有测试在调，但形状和另外两条路一模一样 —— 留着就是「副本会漂」的地基：
    改了一处忘了另一处，而漂了是**静默**的。
    """
    import random

    from train.policies import greedy_policy

    rec, _pts, y = replay.play_capturing(greedy_policy, random.Random(0), level=5,
                                         capture=True, bomb_cost=0.2)
    _pts2, y_expand = replay.expand(rec, bomb_cost=0.2)
    assert y == y_expand, "第三个产地与重放的标签不一致"
    assert all(isinstance(v, float) for v in y)
