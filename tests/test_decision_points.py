import os
import pytest
from tools.game_log import LOG_DIR, load_games
from tools.decision_points import decision_points

# 本文件 7 个测试全部要读真实日志：它们都在 _settled()（= load_games()）上跑，
# 没有一个是纯计算。所以模块级 skipif 在这里是成立的，不存在「被顺带 skip 掉」的测试。
pytestmark = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                reason="本机没有游戏日志")

# 实测：8=炸弹(4~5 张)、9=同花顺、10=大炸弹(6~8 张)、11=天王炸。
# 这四类是炸弹 —— 炸弹可以压任何牌型，所以「与桌面不同型」在它们身上是合法的。
_BOMB_TYPES = {8, 9, 10, 11}


def _settled():
    """全部有结算的局（实测 55 局）。

    ⚠️ **不要加 `[:20]` 之类的切片**：报告一直说的是「55 局 x 4 座位」，
    而前 4 条测试只跑前 20 局 —— 剩下 35 局的重建错了也照样绿，
    「跑了」与「报的」不是一回事（终审修复 R）。整份文件全量跑实测只要几秒
    （重建是 O(手数)，1706 手），省这点时间换不来任何东西。
    """
    return [g for g in load_games() if g.settle]


def test_last_snapshot_hand_matches_settlement():
    """每个座位最后一次出牌后剩下的牌，必须等于结算里的 LeftCards。

    这是重建正确性最直接的证据 —— 手牌少算或多算都会在这里露出来。
    """
    for g in _settled():
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
    for g in _settled():
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
    for g in _settled():
        snaps = decision_points(g)
        prev = None
        for s in snaps:
            if prev is not None and s.seat == prev.seat:
                assert s.table is None, \
                    f"{g.t0} 座位{s.seat} 连续出牌，桌面却没清空"
            prev = s


def test_first_snapshot_is_a_lead():
    """本局第一手的桌面必须是空的。"""
    for g in _settled():
        assert decision_points(g)[0].table is None


def test_actual_cards_are_in_hand():
    """真实出的牌必须在他当时的手里 —— 否则重建错了。"""
    for g in _settled():
        for s in decision_points(g):
            missing = set(s.actual) - set(s.hand)
            assert not missing, \
                f"{g.t0} 座位{s.seat} 出了手上没有的牌 {sorted(missing)}"


# ---------------------------------------------------------------------------
# 桌面（领出/压牌）的回归防线。
#
# 上面 5 个测试只校验**手牌**，桌面的错误一个都露不出来 —— 上面 5 个全绿的同时，
# 领出判定曾经真错过（拿服务器 NextTurnSeatID 判「队友接风」，误清 34 手、
# 漏清 27 手）。下面两条用**独立于实现的口径**把这两类都钉住。

def test_no_illegal_response_within_a_trick():
    """轮内不许出现「既不同型、又不是炸弹」的一手 —— 掼蛋里那不是合法响应。

    口径独立于实现：只看 Snapshot 记下的桌面牌**内容**，再去 plays 里反查它的
    牌型（每张牌本局只会出现一次，所以牌面 == 唯一一手），不复述实现怎么判领出。
    一旦某手被误判成压牌（该清的桌没清），它就会以非法响应的形式露出来。

    实测：旧判据 27 处违反，现判据 0 处。
    """
    for g in _settled():
        type_of = {tuple(p.cards): p.card_type for p in g.plays}
        for i, (s, p) in enumerate(zip(decision_points(g), g.plays)):
            if s.table is None or p.card_type in _BOMB_TYPES:
                continue                     # 领出；或炸弹（什么牌型都能压）
            table_type = type_of.get(tuple(s.table))
            assert p.card_type == table_type, (
                f"{g.t0} 第{i}手 座位{s.seat} 用牌型 {p.card_type} 去压桌面牌型 "
                f"{table_type}（桌面 {sorted(s.table)}）—— 既不同型又不是炸弹，"
                f"不可能是响应；说明这一手被误判成压牌了")


def test_no_false_clear_while_owner_still_has_cards():
    """上一手的主人没出完、这一手又是别人出的 —— 那不可能是新领出。

    口径独立于实现：这一轮要结束，只能等轮次转回主人手上（他手上还有牌，
    就一定会接回领出）。所以「主人没出完 + 换人出牌」时桌面**必须**非空。
    同座位连出、以及主人已出完后的队友接风，才是仅有的两种清桌理由。

    实测：旧判据 34 处违反，现判据 0 处。
    """
    for g in _settled():
        snaps, plays = decision_points(g), g.plays
        for i in range(1, len(snaps)):
            prev, s = plays[i - 1], snaps[i]
            if prev.left != 0 and s.seat != prev.seat:
                assert s.table is not None, (
                    f"{g.t0} 第{i}手 座位{s.seat}：上一手座位{prev.seat} 还剩 "
                    f"{prev.left} 张，这一手却把桌面清空了 —— 主人没出完、又不是"
                    f"他本人接回，这轮不可能结束")

