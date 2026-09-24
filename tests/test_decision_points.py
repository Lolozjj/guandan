import os
import pytest
from tools.game_log import LOG_DIR, load_games
from tools.decision_points import decision_points

# 本文件 5 个测试全部要读真实日志：它们都在 _settled()（= load_games()）上跑，
# 没有一个是纯计算。所以模块级 skipif 在这里是成立的，不存在「被顺带 skip 掉」的测试。
pytestmark = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                reason="本机没有游戏日志")


def _settled():
    return [g for g in load_games() if g.settle]


def test_last_snapshot_hand_matches_settlement():
    """每个座位最后一次出牌后剩下的牌，必须等于结算里的 LeftCards。

    这是重建正确性最直接的证据 —— 手牌少算或多算都会在这里露出来。
    """
    for g in _settled()[:20]:
        snaps = decision_points(g)
        assert snaps, f"{g.t0} 一个决策点都没有"
        left = {i: set(e.get("Cards") or [])
                for i, e in enumerate(g.settle["LeftCards"])}
        for seat in range(4):
            seat_snaps = [s for s in snaps if s.seat == seat]
            if not seat_snaps:
                continue
            remain = set(seat_snaps[-1].hand) - set(seat_snaps[-1].actual)
            assert remain == left[seat], (
                f"{g.t0} 座位{seat} 重建剩 {sorted(remain)} "
                f"!= 结算 {sorted(left[seat])}")


def test_hand_never_grows():
    """一局之内手牌只减不增。"""
    for g in _settled()[:20]:
        prev = {}
        for s in decision_points(g):
            if s.seat in prev:
                assert len(s.hand) <= prev[s.seat], \
                    f"{g.t0} 座位{s.seat} 手牌变多了"
            prev[s.seat] = len(s.hand)


def test_same_seat_playing_again_is_a_new_lead():
    """同一座位连续出两手 = 其余三家都要不起 = 他重新领出，桌面必须清空。

    漏了这条，会把「他重新领出」误当成「他压自己」，验收①就会报假红。
    """
    for g in _settled()[:20]:
        snaps = decision_points(g)
        prev = None
        for s in snaps:
            if prev is not None and s.seat == prev.seat:
                assert s.table is None, \
                    f"{g.t0} 座位{s.seat} 连续出牌，桌面却没清空"
            prev = s


def test_first_snapshot_is_a_lead():
    """本局第一手的桌面必须是空的。"""
    for g in _settled()[:10]:
        assert decision_points(g)[0].table is None


def test_actual_cards_are_in_hand():
    """真实出的牌必须在他当时的手里 —— 否则重建错了。"""
    for g in _settled()[:20]:
        for s in decision_points(g):
            missing = set(s.actual) - set(s.hand)
            assert not missing, \
                f"{g.t0} 座位{s.seat} 出了手上没有的牌 {sorted(missing)}"
