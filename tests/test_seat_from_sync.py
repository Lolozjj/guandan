"""座位从「手牌/状态同步」（msgid 3019）里认 —— 用户报的那个 bug。

用户原话：「好像只有在我出了一手之后，你那个座位才会是正确的，不然有时候会是错误的」

**根因**：状态机只靠出牌报文里的 `LeftCardList` 认座位 —— 而它**只有自己出牌时才有**。
在那之前用的是写死的 `ME_SEAT = 1` 占位。实测 8 段抓包里 **6 段的真实座位不是 1**
（3/0/2/3/2）⇒ 所以「对的那两次纯属碰巧」。

**座位号一直在 3019 的 `3.26.2` 里，开局就到，只是没人读它。**
验证：8 段会话 / **16 局**，与 `LeftCardList` 给出的真值**全部一致**（含中途接进来的局面）。
protobuf **省略 0** ⇒ 字段缺席即座位 0（段2 第一局实测如此）。

⚠️ 定位它**不能用 `decode_hand`** —— 那个有「手牌 18~27 张」的窗口，
而抓包经常从中局开始（实测段4 手牌只剩 9 张）。见下面第 3 条测试用的 fixture。
"""
import io
import os

import net.protocol as protocol
import net.state as state_mod
from net.state import GameState
from net.panel import apply_event

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _parse(name):
    body = bytes.fromhex(
        io.open(os.path.join(FIX, name), encoding="utf-8").read().strip())
    return protocol.parse(body)


# ---------------------------------------------------------------- 解码那一层

def test_seat_comes_from_the_sync_frame():
    m = _parse("msg3019_seat3_27cards.hex")
    assert protocol.decode_seat_sync(m["fields"]) == 3


def test_an_absent_field_means_seat_zero():
    """protobuf 省略 0 —— 字段真的不在，而 0 是正确结论（段2 第一局实测座位 0）。"""
    m = _parse("msg3019_seat0_absent.hex")
    assert not [p for k, p, _ in m["fields"] if k == "int" and tuple(p) == (3, 26, 2)]
    assert protocol.decode_seat_sync(m["fields"]) == 0


def test_the_hand_size_window_does_not_matter():
    """⚠️ 这条 fixture 是**中局**接进来的（手牌只剩 9 张）：
    `decode_hand` 因为 18~27 那个窗口读不出来，但座位照样得认出来。"""
    m = _parse("msg3019_seat3_9cards.hex")
    assert protocol.decode_hand(m["fields"]) is None, "前提：这个窗口确实读不出来"
    assert protocol.decode_seat_sync(m["fields"]) == 3


def test_it_returns_none_for_other_messages():
    assert protocol.decode_seat_sync([]) is None


# ---------------------------------------------------------------- 状态机那一层

def test_the_sync_sets_my_seat_without_any_play():
    """**用户报的那条**：一次牌都还没出，座位就该是对的。"""
    st = GameState()
    assert st.me == state_mod.ME_SEAT and not st.me_confirmed   # 修之前的状态
    st.on_seat(3)
    assert st.me == 3 and st.me_confirmed
    assert st.seat_label(3) == "我"
    assert st.seat_label(0) == "左对手"          # 出牌顺序 0→3→2→1，(0-3)%4 = 1


def test_the_hand_event_can_carry_the_seat_too():
    st = GameState()
    st.on_hand([1, 2, 3], seat=2)
    assert st.me == 2 and st.me_confirmed


def test_a_new_deal_resets_it_and_the_next_sync_re_establishes_it():
    """座位号**每局都会变**（实测段6：第一局 2、第二三局 0）—— 换局必须重认。"""
    st = GameState()
    st.on_seat(2)
    st.on_deal()
    assert not st.me_confirmed                  # 换局清零
    st.on_seat(0)
    assert st.me == 0 and st.me_confirmed


def test_the_panel_applies_it_before_any_play():
    """端到端（面板走的那条路 `apply_event`）：**不经过任何出牌**，座位已经对。"""
    st = GameState()
    m = _parse("msg3019_seat3_27cards.hex")
    apply_event(st, {"type": "seat", "seat": 3})
    hd = protocol.decode_hand(m["fields"])
    apply_event(st, {"type": "hand", "cards": hd["cards"],
                     "seat": protocol.decode_seat_sync(m["fields"])})
    assert st.me == 3 and st.me_confirmed
