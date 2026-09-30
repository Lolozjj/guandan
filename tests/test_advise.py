"""推理链：`GameState -> Observation` 必须与模拟器那条路**编出同一个向量**。

白名单式的字段比对会漏（漏一个字段只是少一维，看不出来），只有
「700 维逐位相等」才能保证喂给网络的东西和训练时**完全一样**。
"""
import numpy as np

from guandan.advice import advise
from guandan.sim import env, meld, rules
from guandan.capture.state import GameState

A = meld.cid_from_name
LEVEL = 9
MY_SEAT = 1

#: 测试用的一手牌。**选材有两条硬约束**（都踩过）：
#:   1. **每家都要出过至少一手** —— 线上 `left` 对没出过的座位是「27 - 出过张数」的兜底，
#:      小手牌会撞上这个兜底，于是「两条路同向量」会被测试素材本身弄红；
#:   2. 打的牌一律**避开级牌点数与它的红桃** —— 一个是 9（打 9 时 9 比 J 还大，
#:      素材会莫名其妙「压不过桌面」），一个是 ♥9（逢人配，同一组牌在
#:      `melds_from(整手牌)` 与 `as_meld(这一手)` 下可能有多种读法）。后者是引擎的已知偏差，
#:      不该混进「适配层对不对」这条测试里。
_HANDS = [{A("S7"), A("C4")}, {A("S3"), A("H5"), A("D6")},
          {A("S10"), A("C8")}, {A("SJ"), A("D8")}]

#: 八步走完：四家都出过一手、我自己有一次**要不起**（服务器不发事件，靠推断补）。
#:   1 我领出 ♠3 -> 0 用 ♠7 压 -> 3 用 ♠J 压 -> 2 要不起 -> 我要不起（静默）
#:   -> 0 要不起 -> 3 领出 ♦8 -> 2 用 ♠10 压 -> 轮到我
_ACTS = [(1, [A("S3")]), (0, [A("S7")]), (3, [A("SJ")]), (2, None),
         (1, None), (0, None), (3, [A("D8")]), (2, [A("S10")])]


def _drive(actions, hands=None):
    """同一局，两条路各走一遍 —— 一边是模拟器（明牌 `Hand`），一边是网络状态机（事件流）。

    `actions = [(座位, 牌列表 or None)]`（座位是绝对座位号，`None` = 要不起）。

    三条对齐纪律（不对齐就会拿两条不同局面的向量去比，红得莫名其妙）：
      - **网络侧的「下一手」取自真值**（服务器报的就是这个值），不能填常数；
      - 网络侧**先走真值再发事件**，因为事件里要用到走完之后的轮次；
      - 我自己要不起时**不发事件**（服务器本来就不发，实测），让推断那条路也被压进测试。
    返回 `(hand, st, played)`：走到同一位置的两边状态，以及各家出过的牌。
    """
    hands = hands or _HANDS
    hand = rules.Hand(hands=[set(h) for h in hands], level=LEVEL, turn=actions[0][0])
    st = GameState()
    st.level = LEVEL
    played = {s: set() for s in rules.SEATS}
    for seat, cs in actions:
        cs = list(cs) if cs else None
        m = None if cs is None else meld.as_meld(cs, LEVEL)
        assert cs is None or m is not None, f"测试素材本身有问题：{cs}"
        # 「这一手之后还剩几张」要在动手**之前**算，否则多减了一手
        rest = sorted(hand.hands[seat] - set(cs or ()))
        # ---- 真值侧先走 ----
        if m is None:
            hand.pass_turn(seat)
        else:
            hand.play(seat, m)
            played[seat] |= set(cs)
        nxt = -1 if hand.over else hand.turn
        # ---- 网络侧 ----
        if cs is None:
            if seat != MY_SEAT:
                st.on_pass(seat, nxt if 0 <= nxt <= 3 else None)   # 别人的要不起会通知我
        else:
            st.on_play(seat, cs, 0, nxt, len(rest),
                       rest if seat == MY_SEAT else None)
    return hand, st, played


def test_network_path_and_simulator_path_encode_the_same_vector():
    """`encode_state` 内部按出牌人相对化，所以两套座位编号只要差一个旋转就等价 ——
    两边的编号都遵守 `0→3→2→1` 的出牌顺序，因此可以直接逐位比。
    """
    hand, st, played = _drive(_ACTS)
    assert st.turn == MY_SEAT, "这一局的收尾必须正好轮到我（不然比的是不同局面）"
    assert st.steps[4] == (MY_SEAT, None), "我自己那一步是推断出来的，位置在 0 那一步之前"
    b = advise.build(st)
    assert not b.reason, f"不该跳过：{b.reason}"
    truth = env.observe(hand, MY_SEAT, played, hand.table)
    assert np.array_equal(env.encode_state(b.obs), env.encode_state(truth))
    assert np.array_equal(b.hist, env.encode_history(hand, MY_SEAT))


def test_pass_is_a_candidate_only_when_there_is_a_table():
    _hand, st, _p = _drive(_ACTS)                      # 桌上有一张 ♠10，轮到我（跟牌）
    b = advise.build(st)
    assert None in advise.candidates(b), "跟牌时「能压也可以过」，None 永远在候选里"
    # 领出：桌上没牌时候选里**不许**有「过」
    st2 = GameState()
    st2.level = LEVEL
    st2.on_play(MY_SEAT, [A("S3")], 0, 0, 2, [A("S4"), A("S5")])  # 认座位
    st2.table = None
    st2.turn = MY_SEAT
    assert None not in advise.candidates(advise.build(st2))


def test_every_uncomputable_case_returns_a_reason_and_never_raises():
    st = GameState()
    st.level = LEVEL
    assert advise.build(st).reason == advise.SKIP_SEAT, "座位没认出来之前不许算"
    st.on_play(MY_SEAT, [A("S3")], 0, 0, 2, [A("S4"), A("S5")])
    assert st.me_confirmed
    st.turn = 0
    assert advise.build(st).reason == advise.SKIP_NOTURN
    st.turn = MY_SEAT
    st.table = None
    st.hand = []
    assert advise.build(st).reason == advise.SKIP_HAND
    st.hand = [A("S4")]
    st.level = None
    assert advise.build(st).reason == advise.SKIP_LEVEL, "级别未知必须跳过，不许猜"
    st.level = 14
    assert advise.build(st).reason == advise.SKIP_LEVEL, "14 是日志里 A 的另一种写法，越界"
    st.level = LEVEL
    # 桌面这两张牌组不成任何合法牌型（不是对子、不是顺子）—— 不许当成「桌上无牌」
    from guandan.capture.state import Play
    st.table = Play(seat=0, cards=[A("S3"), A("S5")], card_type=0, next_seat=0, left=26)
    assert advise.build(st).reason == advise.SKIP_TABLE
