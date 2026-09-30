import numpy as np

from guandan.capture import cards
from guandan.sim import env, meld, rules

A = meld.cid_from_name


def test_history_shape_and_zero_padding_at_the_front():
    # 别写 `[{A("S3")}] * 4` —— 那是**同一个 set 对象**出现四次，改一家会改四家
    h = rules.Hand(hands=[{A("S3")} for _ in range(4)], level=2, turn=0)
    v = env.encode_history(h, 0)
    assert v.shape == (env.HISTORY_LEN, env.HISTORY_DIM)
    assert v.sum() == 0, "还没出过牌，历史必须全 0"


def test_recent_steps_are_right_aligned_and_carry_the_relative_seat():
    """最近一手必须在**最后一行**（右对齐），且带出牌人的**相对座位**。

    历史里不带「谁出的」会丢掉一半信息 —— 同样一张 9♠ 是下家出的还是对家出的，
    对判断局面完全不同。所以每行是 `encode_action(m) ⊕ 相对座位 one-hot(4)`。
    """
    h = rules.Hand(hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))
    v = env.encode_history(h, 1)                      # 从座位 1 的视角看
    assert v[:env.HISTORY_LEN - 1].sum() == 0         # 只有一步，前面全是 0
    row = v[-1]
    assert row[:env.ACTION_DIM].sum() > 0             # 有动作
    assert row[env.ACTION_DIM:].argmax() == 3         # 座位 0 对座位 1 来说是**上家**（相对 3）


def test_history_is_truncated_to_the_last_15_steps():
    h = rules.Hand(hands=[{A("S3"), A("S4"), A("S5")},
                          {A("S6"), A("S7")}, {A("S8")}, {A("S9"), A("ST")}],
                   level=2, turn=0)
    for _ in range(3):
        if h.over:
            break
        s = h.turn
        acts = [a for a in h.actions(s) if a is not None]
        # 走不动时就过 —— 轮到座位 2（只有一张 8♠）面对 9♠ 时，候选只有「过」
        h.play(s, acts[0] if acts else None)
    v = env.encode_history(h, 0)
    assert v.shape == (env.HISTORY_LEN, env.HISTORY_DIM)


def test_history_does_not_leak_opponent_hands():
    """`encode_history` 收的是**明牌的 `Hand`** —— 所以必须钉住它只读公开的 `steps`。

    两个只差「对手手上还有什么」的局面，历史必须一模一样。
    """
    def build(opp):
        h = rules.Hand(hands=[{A("S3")}, set(opp), {A("S6")}, {A("S7")}],
                       level=2, turn=0)
        h.play(0, meld.as_meld([A("S3")], 2))
        return env.encode_history(h, 0)

    a = build({A("S9"), A("ST")})
    b = build({A("HJ"), A("HQ")})
    assert np.array_equal(a, b), "历史里混进了对手手牌 —— 明牌泄漏"
