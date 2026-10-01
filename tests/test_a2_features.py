"""A2 的显式特征：**不许泄漏、维度对、老权重零填充后等价**。

三条都是"错了不会报错、只会静默变差"的类型，所以必须有牙。
"""
import numpy as np
import torch

from guandan.capture import cards
from guandan.rl.net import LSTM_HIDDEN, QNet, load_state
from guandan.sim import env, features, meld

A = meld.cid_from_name


def _obs(**kw):
    n = 4
    base = dict(seat=0, hand=(A("S3"),), played=None, left=None, table=(),
                table_kind=0, table_rank=-1, passed=None, turn=0, level=2)
    base.update(kw)
    return env.Observation(
        seat=base["seat"], hand=frozenset(base["hand"]),
        played=tuple(frozenset(base["played"] or ()) for _ in range(n)),
        left=tuple(base["left"] or [27] * n),
        table=tuple(base["table"]), table_kind=base["table_kind"],
        table_rank=base["table_rank"],
        passed=tuple(base["passed"] or [False] * n), turn=base["turn"],
        level=base["level"])


def test_dim_and_range():
    obs = _obs()
    f = features.extra_features(obs)
    assert len(f) == features.EXTRA_DIM
    assert env.STATE_DIM == 700 + features.EXTRA_DIM
    v = env.encode_state(obs)
    assert v.shape == (env.STATE_DIM,)
    # 剩牌对比那一块可以是负的；其余都在 [0,1]，整体不许跑出 [-1,1]
    assert all(-1.0 <= x <= 1.0 for x in f), f


def test_default_is_off_and_costs_nothing():
    """⚠️ **默认关**（2026-09-30 的 A2 中期判读之后）：那一块恒为 0，且前 700 维不受影响。

    这条钉的是两件事：① 关着时**零开销**（不调 `extra_features`）；
    ② 架构不变 —— 状态还是 727 维，于是 A2 与老权重都装得上。
    """
    assert features.ENABLED is False
    obs = _obs(hand=(A("S5"), A("S6"), A("S7"), A("S8"), A("S9")))
    v = env.encode_state(obs)
    assert v.shape == (727,)
    assert not v[700:].any(), "关掉时那一块必须是 0（不然就是白付了开销）"
    # 前 700 维仍是老编码：手牌 one-hot 在正确的位置
    assert v[cards.slot(A("S5"))] == 1.0


def test_switch_turns_the_block_back_on(monkeypatch):
    """开关一开就完全恢复 A2（单变量、随时可复验）。"""
    obs = _obs(hand=(A("S3"), A("H5")), left=[4, 20, 20, 20])
    assert not env.encode_state(obs)[700:].any()
    monkeypatch.setattr(features, "ENABLED", True)
    v = env.encode_state(obs)
    assert v[700:].any(), "开了开关那一块必须有内容"
    assert list(v[700:]) == list(features.extra_features(obs))


def test_features_react_to_the_public_state_they_should():
    """手数、未见牌、残局信号要对**公开信息 + 我的手牌**有反应。"""
    made = _obs(hand=(A("S5"), A("S6"), A("S7"), A("S8"), A("S9")), level=2)
    scat = _obs(hand=(A("S3"), A("H5"), A("C7"), A("D9"), A("SJ")), level=2)
    assert features.extra_features(made)[0] < features.extra_features(scat)[0]
    # 我剩 4 张 -> 残局信号那三个都是 1（<=11/<=8/<=5）
    late = _obs(hand=(A("S3"), A("H5")), left=[4, 20, 20, 20])
    assert features.extra_features(late)[-3:] == (1.0, 1.0, 1.0)
    early = _obs(hand=(A("S3"), A("H5")), left=[27, 27, 27, 27])
    assert features.extra_features(early)[-3:] == (0.0, 0.0, 0.0)


def test_no_leak_from_opponents_hands():
    """⚠️ 改**别人手里具体是哪些牌**（张数不变）不许影响编码 —— 那是明牌。

    这是"不泄漏"最直接的行为测试：`Observation` 只带张数，`unseen_pool` 只减
    我的手牌与已出过的牌，两处都不碰对手手牌的内容。
    """
    e = env.GuandanEnv(seed=3)
    e.reset(level=5, first=0)
    before = env.encode_state(e.observe())
    me = e.hand.turn
    opp = next(s for s in (0, 1, 2, 3) if s != me)
    other = sorted(c for c in range(1, 300) if cards.is_card(c))
    keep = len(e.hand.hands[opp])
    new_hand = set(other[:keep])                      # 换一批牌，**张数不变**
    assert new_hand != e.hand.hands[opp]
    e.hand.hands[opp] = new_hand
    after = env.encode_state(e.observe())
    assert np.array_equal(before, after), "对手手牌的内容影响了编码 —— 泄漏了"


def test_old_weights_load_by_zero_padding_and_stay_equivalent():
    """⚠️ A2 可比的前提：**老权重（少 27 维）装上之后，行为必须一字不差。**

    构造方式（第一版我写错了，记下来）：老 checkpoint 的逻辑含义是
    「**新特征那几列根本不存在**」⇒ 等价于那些列是 0。
    所以先**把新特征列清零**，再切掉它们模拟老 ckpt —— 装回来之后输出必须与清零后完全相同。

    （反面教训：直接切掉**随机非零**的那几列，函数当然会变 —— 那是测试构造错，不是实现错。）
    """
    net = QNet().eval()
    s_new = env.STATE_DIM
    s_old = s_new - features.EXTRA_DIM
    with torch.no_grad():
        net.mlp[0].weight[:, s_old:s_new] = 0.0          # 老网络没有这些特征 ⇒ 列是 0
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    old_w = torch.cat([sd["mlp.0.weight"][:, :s_old],
                       sd["mlp.0.weight"][:, s_new:]], dim=1)   # 模拟老 ckpt（971 列）
    assert old_w.shape[1] == s_old + env.ACTION_DIM + LSTM_HIDDEN

    st = torch.randn(2, s_new)
    ac = torch.randn(2, env.ACTION_DIM)
    hi = torch.randn(2, 15, env.HISTORY_DIM)
    with torch.no_grad():
        want = net(st, ac, hi)

    grown = QNet().eval()
    load_state(grown, {**sd, "mlp.0.weight": old_w})
    # 新特征那几列必须是 0（贡献为 0）
    assert torch.count_nonzero(grown.mlp[0].weight[:, s_old:s_new]) == 0
    with torch.no_grad():
        got = grown(st, ac, hi)
    assert torch.allclose(want, got, atol=1e-6), "零填充之后输出变了 —— A2 不可比"


def test_wider_checkpoints_are_rejected_loudly():
    """只允许「长大」：比当前网络**更宽**的 checkpoint 一律抛，不静默缩维凑合。"""
    net = QNet().eval()
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    w = sd["mlp.0.weight"]
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 1)], dim=1)   # 多一列
    try:
        load_state(net, sd)
    except ValueError:
        return
    raise AssertionError("比当前网络更宽时应当抛 ValueError")
