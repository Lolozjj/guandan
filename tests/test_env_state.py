import numpy as np
import pytest

from guandan.capture import cards
from guandan.sim import env, meld, rules

A = meld.cid_from_name


def _obs(seat=0, hand=(A("S3"),), played=None, left=None, table=(),
         table_kind=0, table_rank=-1, passed=None, turn=0, level=2):
    n = 4
    return env.Observation(
        seat=seat, hand=frozenset(hand),
        played=tuple(frozenset(x) for x in (played or [()] * n)),
        left=tuple(left or [1] * n), table=tuple(table),
        table_kind=table_kind, table_rank=table_rank,
        passed=tuple(passed or [False] * n), turn=turn, level=level)


def test_state_dim_is_what_the_spec_says():
    assert env.STATE_DIM == 700          # spec §4.1：108+4*108+4+108+10+15+4+4+15
    assert env.encode_state(_obs()).shape == (700,)
    assert env.encode_state(_obs()).dtype == np.float32


def test_hand_and_table_land_in_the_right_slots():
    v = env.encode_state(_obs(hand=[A("S3"), A("DK", deck=2)],
                              table=[A("H5")]))
    assert v[cards.slot(A("S3"))] == 1 and v[cards.slot(A("DK", deck=2))] == 1
    assert v.sum() >= 3
    t0 = 108 + 4 * 108 + 4                     # 桌面牌段的起点
    assert v[t0 + cards.slot(A("H5"))] == 1


def test_seats_are_relativised_so_one_policy_can_play_all_four():
    """**座位必须相对化**（spec §3）：索引 0 = 自己、1 = 下家、2 = 对家、3 = 上家。

    同一个「我的牌 + 我的下家出过什么」的局面，无论坐在哪个绝对座位上，
    编码必须**一模一样**，否则四个座位就得有四套权重。"""
    played_abs = [(), (), (), ()]
    played_abs[2] = (A("S7"),)                 # 绝对座位 2 出过 7♠
    # a：我坐 0、轮到我、座位 2（我的**对家**）出过 7♠
    a = env.encode_state(_obs(seat=0, turn=0, played=played_abs))
    # c：我坐 1、轮到我、座位 3（此时也是我的**对家**）出过 7♠
    played_c = [(), (), (), (A("S7"),)]
    c = env.encode_state(_obs(seat=1, turn=1, played=played_c))
    assert np.array_equal(a, c)                # 可观测状态相同 -> 编码必须一模一样
    # b：我坐 1，出过 7♠ 的座位 2 是**下家**（不是对家）-> 与 a 不同
    b = env.encode_state(_obs(seat=1, turn=1, played=[(), (), (A("S7"),), ()]))
    assert not np.array_equal(a, b)


def test_level_card_uses_point_value_not_the_raw_index():
    """**打 2 的时候 2 是级牌、比 A 大。** 桌面主点数的编码必须走 `meld.point_value`，
    否则「桌面是一对 2」与「桌面是一对 3」在打 2 时会被编成相邻的两格，
    而它们实际差着整整一个层级。"""
    kind, size = meld.PAIR, 2
    v2 = env.encode_state(_obs(table=(A("S2"), A("H2")), table_kind=kind,
                               table_rank=meld.point_value(2, 2), level=2))
    v3 = env.encode_state(_obs(table=(A("S3"), A("H3")), table_kind=kind,
                               table_rank=meld.point_value(3, 2), level=2))
    seg = slice(108 + 4 * 108 + 4 + 108 + 10, 108 + 4 * 108 + 4 + 108 + 10 + 15)
    assert v2[seg].argmax() != v3[seg].argmax()
    assert v2[seg].argmax() == 13              # POINT_LEVEL -> 第 13 格
    assert v3[seg].argmax() == 1               # 点数 3 -> _POINT[3]=2 -> 槽位 1


def test_encode_state_rejects_an_out_of_range_level():
    """级别必须是 1..13。**14 与 None 都不许静默写进预留槽位。**

    14 会落到第 14 格（那一格是预留的），None 会直接 TypeError ——
    两者都是「一声不响地编出错的向量」，而级别错了**逢人配就全错**。
    """
    for bad in (14, None, 0, 99):
        with pytest.raises(ValueError):
            env.encode_state(_obs(level=bad))
