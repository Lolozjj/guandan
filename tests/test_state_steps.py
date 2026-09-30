"""动作流水（`GameState.steps`）—— 影子模式的历史就建在它上面。

⚠️ 这一份流水**不是**「把事件抄一遍」：我自己「要不起」时服务器**不发事件**
（实测，见计划文档「本次实测到的既有事实」第 2 条），只能从轮转推出来。
"""
from guandan.sim import meld
from guandan.capture.state import GameState

A = meld.cid_from_name


def _play(st, seat, nxt, played, rest, mine=False):
    """一个出牌事件。`rest` = 这手之后该家还剩的牌（服务器报的 `LeftCardLen` = len(rest)）。
    `mine=True` 时同时带上 `LeftCardList`（只有我自己的出牌有）。
    """
    st.on_play(seat, list(played), 0, nxt, len(rest), sorted(rest) if mine else None)


def test_plays_and_notified_passes_are_recorded_in_order():
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4"), A("S5")], mine=True)   # 我出 S3 -> 轮到 0
    assert st.me == 1 and st.me_confirmed
    _play(st, 0, 3, [A("S9")], [A("SK")])                        # 0 出牌 -> 轮到 3
    st.on_pass(3, 2)                                             # 3 要不起（我会收到）
    assert [s for s, _c in st.steps] == [1, 0, 3]
    assert st.steps == [(1, [A("S3")]), (0, [A("S9")]), (3, None)]


def test_my_silent_pass_is_inferred_from_the_rotation():
    """我自己要不起时没有事件 —— 必须在下一个事件到来时从轮转补上，且顺序在它前面。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)             # 我出牌 -> 轮到 0
    _play(st, 0, 3, [A("S9")], [A("SK")])                        # 0 出牌 -> 轮到 3
    _play(st, 3, 2, [A("SJ")], [A("SQ")])                        # 3 出牌 -> 轮到 2
    st.on_pass(2, 1)                                             # 2 要不起 -> 轮到我
    st.on_pass(0, 3)                                             # 0 要不起（我会收到）
    assert [s for s, _c in st.steps] == [1, 0, 3, 2, 1, 0]
    assert st.steps[4] == (1, None), "我自己那一步是推断出来的，位置必须在 0 那一步之前"


def test_a_finished_seat_is_not_recorded_as_a_passer():
    """已出完的座位没有「过」这个动作（`rules.Hand._advance` 会跳过他不记步）。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)
    _play(st, 0, 3, [A("S9")], [A("SK")])
    _play(st, 3, 2, [A("SK")], [])                               # 座位 3 出完（left=0）
    assert st.finish_order == [3]
    st.turn = 1                                                  # 手工摆一个中间态：从 1 走到 2 会经过 0 和 3
    st.on_pass(2, 3)
    assert (3, None) not in st.steps, "出完的人不该被记成「过」"
    assert [s for s, _c in st.steps][-3:] == [1, 0, 2], \
        "1 是我（推断出来的过）、0 是真的过了、2 是自己要不起：中间的 3 被跳过"


def test_deal_change_resets_the_stream_and_reidentifies_my_seat():
    """换局：新手牌不是上一局的子集 -> 清场。**清完必须把座位也认下来** ——
    座位号每局都会变（实测 11:11 那局我是 seat1、11:15 那局我是 seat2），
    漏认的话整局方位都会反（用户报过两次）。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)             # 上一局我是 seat1
    assert st.me == 1 and st.deal_seq == 0
    _play(st, 2, 1, [A("H5")], [A("H6")], mine=True)             # 这一局我是 seat2
    assert st.deal_seq == 1
    assert st.me == 2 and st.me_confirmed, "换局后座位必须重认"
    assert st.steps == [(2, [A("H5")])], "新一局的流水要从零开始"
    assert st.finish_order == []
