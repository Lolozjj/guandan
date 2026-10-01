"""动作侧特征（2026-09-30）：语义 + **训练/推理编码一致性**。

为什么要这一份：这是全项目唯一"同一组权重、两条路各编一遍"的地方（训练侧 `_tensors`、
推理侧 `q_values`）。两边编得不一样**不会报错**，只会静默把权重用歪 —— 所以必须有牙。
"""
import numpy as np
import torch

from guandan.rl.net import QNet, load_state, q_values
from guandan.rl.selfplay import _tensors
from guandan.sim import env, features, meld

A = meld.cid_from_name


def _obs_level():
    e = env.GuandanEnv(seed=11)
    e.reset(level=8)
    return e


def test_consequence_semantics():
    """三档语义：用掉一整手计划 / 拆了一手计划 / 拆自己的炸 / 过。"""
    lv = 2
    # ① 三个 3 是计划里的一手（三张）⇒ uses=1
    hand = [A("S3"), A("H3"), A("D3"), A("S5"), A("S6"), A("S7"), A("S8"), A("S9")]
    whole = meld.as_meld([A("S3"), A("H3"), A("D3")], lv)
    assert features.consequence_features(hand, whole, lv) == (1.0, 0.0, 0.0)
    # ② 计划是**三连对**（3-3 4-4 5-5）时只出其中一对 ⇒ 拆了它（也没拆炸）
    lian = [A("S3"), A("H3"), A("S4"), A("H4"), A("S5"), A("H5")]
    pair3 = meld.as_meld([A("S3"), A("H3")], lv)
    assert pair3 is not None, "这对 3 必须是合法牌型（测试自身的前提）"
    assert features.consequence_features(lian, pair3, lv) == (0.0, 1.0, 0.0)
    # ③ 手里 4 张 3 时出三张 3 ⇒ 拆炸
    four = [A("S3"), A("H3"), A("D3"), A("C3"), A("S5")]
    assert features.consequence_features(four, whole, lv)[2] == 1.0
    # ④ 过：全 0
    assert features.consequence_features(hand, None, lv) == (0.0, 0.0, 0.0)


def test_training_and_inference_encode_actions_identically():
    """⚠️ **两条路必须编出同一个向量**：训练侧 `_tensors` vs 推理侧 `q_values`。

    做法：用**生产函数** `_tensors` 编训练侧那一行，和 `q_values` 的同一个候选比 Q。
    两边若给动作侧特征喂了不同的输入（比如一边忘了传 hand），这里就会炸。
    """
    net = QNet().eval()
    e = _obs_level()
    obs, acts = e.observe(), e.legal()
    hist = env.encode_history(e.hand, e.hand.turn)
    i = min(3, len(acts) - 1)

    st, ac, hi = _tensors([(obs, acts, i, e.hand.turn, hist)])
    assert ac.shape[1] == env.ACTION_DIM
    with torch.no_grad():
        q_train = float(net(st, ac, hi)[0])
    q_infer = float(q_values(net, obs, acts, hist)[i])
    assert abs(q_train - q_infer) < 1e-4, (q_train, q_infer)


def test_unknown_layout_is_rejected_loudly():
    """布局反推不出来就必须抛 —— 猜错就是静默把权重接在错的输入上。"""
    net = QNet().eval()
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    sd["mlp.0.weight"] = torch.cat(
        [sd["mlp.0.weight"], torch.zeros(sd["mlp.0.weight"].shape[0], 7)], dim=1)
    try:
        load_state(net, sd)
    except ValueError as exc:
        assert "布局" in str(exc)
        return
    raise AssertionError("反推不出的布局应当抛 ValueError")


def test_history_rows_keep_the_base_width():
    """历史行**不能**带动作侧特征（那里拿不到当时的手牌）—— 宽度是基础 143+4。"""
    e = _obs_level()
    h = env.encode_history(e.hand, e.hand.turn)
    assert h.shape == (env.HISTORY_LEN, env.HISTORY_DIM)
    assert env.HISTORY_DIM == env.ACTION_BASE_DIM + 4
    assert env.HISTORY_DIM != env.ACTION_DIM + 4, "两个宽度混了"
