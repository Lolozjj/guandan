"""每局开头要认得出**谁领出** —— 否则我领出那一局，模型不给建议。

用户 2026-09-29 报的：「位置不会乱了，但是我的第一首出牌，模型不会有建议」。

**根因**：`on_deal` 把 `turn` 清成 `None`（新一局谁先出由服务器重分配，
上一局的轮次是陈旧数据），而**在第一手出牌之前没有任何事件能告诉状态机轮到谁**。
影子模式的门是 `if st.turn != st.me: return` —— 我领出时 `turn` 是 `None`，
于是那一手**不算决策点、不算建议**。
实测 16 局里 **4 局**是这种（都是 `steps=0`，即我领出）；别人先出的 12 局正常，
所以现象是「有时候」。

**信号**：msgid **3004** 在**每局开桌前**来一条，`3.5.1` = 本局领出者的座位
（protobuf 省略 0 ⇒ 字段缺席即座位 0）。
验证：16 条 3004 / 16 局，其中 11 条带字段、5 条不带 —— **「不带」正好对应领出者=0**；
能与「该局第一手出牌者」对上的 12 局**全部一致**（另外 4 局是抓包窗口截断、根本没收到）。
"""
import io
import os

from net import protocol
from net.panel import apply_event
from net.state import GameState

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _parse(name):
    body = bytes.fromhex(
        io.open(os.path.join(FIX, name), encoding="utf-8").read().strip())
    return protocol.parse(body)


# ---------------------------------------------------------------- 解码那一层

def test_the_leader_comes_from_the_table_open_frame():
    assert protocol.decode_leader(_parse("msg3004_leader1.hex")["fields"]) == 1


def test_an_absent_field_means_seat_zero():
    m = _parse("msg3004_leader0_absent.hex")
    assert not [p for k, p, _ in m["fields"] if k == "int" and tuple(p) == (3, 5, 1)]
    assert protocol.decode_leader(m["fields"]) == 0


def test_it_does_not_claim_a_leader_for_other_messages():
    """认不出就返回 `None` —— 不许在别的消息上瞎认一个座位出来。"""
    assert protocol.decode_leader(_parse("msg3019_seat3_27cards.hex")["fields"]) is None
    assert protocol.decode_leader([]) is None


# ---------------------------------------------------------------- 状态机那一层

def test_the_open_frame_sets_the_turn():
    st = GameState()
    st.on_deal()                       # 换局：turn 被清成 None（这是有意的）
    assert st.turn is None
    st.on_leader(3)
    assert st.turn == 3


def test_it_does_not_override_an_established_turn():
    """⚠️ 中局来一条陈旧的 3004 不许把**已经建立**的轮次改掉 ——
    改了会让 `_advance_to` 凭空补出几个不存在的「要不起」。"""
    st = GameState()
    st.turn = 2
    st.on_leader(0)
    assert st.turn == 2


# ---------------------------------------------------------------- 用户报的那条

def test_the_advice_gate_opens_for_my_lead():
    """**用户报的那条**：我领出时，影子模式那道门（`st.turn == st.me`）必须是开的。

    门的原文在 `ShadowLog.after_event`：`if st.turn != st.me: return`。
    """
    st = GameState()
    apply_event(st, {"type": "seat", "seat": 3})          # 3019 认座位（今天刚修的）
    apply_event(st, {"type": "hand", "cards": list(range(1, 28)), "seat": 3})
    st.on_deal()                                          # 换局 -> turn 清成 None
    assert st.turn != st.me, "前提：不修的话这时候门是关的"
    apply_event(st, {"type": "leader", "seat": 3})        # 开桌那条 3004
    assert st.turn == st.me, "门必须开着，否则我领出那一手没有建议"
