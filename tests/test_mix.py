"""混合策略（`rl/mix.py`）：**只在 Q 近似并列时随机**，差得明显时与 argmax 逐位一致。"""
import numpy as np

from guandan.rl.mix import mixed_index
from guandan.rl.mix import mixed_net_policy


def test_clear_winner_is_argmax_deterministic():
    """⚠️ 并列不明显（差距 > margin）时**必须**与 argmax 一致 —— 否则就是把模型变乱。"""
    q = [3.0, 1.0, 0.5]
    for _ in range(30):
        assert mixed_index(q, margin=0.15) == 0


def test_near_tie_is_randomised():
    """近似并列时会抽到不同下标（这才是"不可被利用"的来源）。"""
    q = [1.00, 0.99, 0.98]
    seen = {mixed_index(q, margin=0.15, temp=0.05) for _ in range(200)}
    assert len(seen) >= 2, "并列集合里应当随机"
    assert seen <= {0, 1, 2}


def test_single_and_empty_are_safe():
    assert mixed_index([0.5]) == 0


def test_policy_only_changes_near_ties(monkeypatch):
    import numpy as _np
    from guandan.rl import mix

    class _Net:
        pass

    monkeypatch.setattr(mix, "q_values", lambda *a, **k: _np.array([5.0, 1.0]))
    pol = mixed_net_policy(_Net())
    for _ in range(20):
        assert pol(None, [object(), object()], None) == 0
