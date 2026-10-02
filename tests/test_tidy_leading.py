"""⚠️ **回归测试**：「领出」（桌面空）时"擦浪费"**不许介入**。

`Observation.table` 在领出时是**空元组**（不是 None）—— 2026-10-01 我在 `tidy_index` 里
修过一次同类错，2026-10-03 又在**调用方**（`tidy_net_policy` / `advise`）写成了
`obs.table is not None` ⇒ 恒为真 ⇒ 连领出都被当成"有牌要压" ⇒ 主动出炸/用万能牌被降级。
这条测试把"领出 = 不介入"钉死。
"""
import random

from guandan.rl import tidy
from guandan.sim import env


def test_leading_means_no_intervention():
    e = env.GuandanEnv(seed=5)
    obs = e.reset()
    assert obs.table == (), "前提：领出时 table 是空元组"
    acts = e.legal()
    q = [0.1] * len(acts)
    # 随便挑一个"最贵的"候选当网络首选（炸弹优先），若错误地认为"有牌要压"，就会被降级
    bomb_i = next((i for i, m in enumerate(acts) if m is not None and m.is_bomb), None)
    if bomb_i is None:
        return                      # 这一手没有炸，测不了（换一局由上层保证覆盖）
    got = tidy.tidy_index(q, acts, bomb_i, has_table=bool(obs.table))
    assert got == bomb_i, "领出时不许把主动出炸降级"


def test_policy_wrapper_uses_bool_table(monkeypatch):
    """包装器必须把 `bool(obs.table)`（而不是 `is not None`）传下去。"""
    from guandan.rl import tidy as T

    seen = {}

    class _Net:
        pass

    class _Obs:
        table = ()                  # 领出

    def fake_q(*a, **k):
        import numpy as np
        return np.array([5.0, 1.0])

    monkeypatch.setattr(T, "q_values", fake_q)
    orig = T.tidy_index

    def spy(q, acts, i, **kw):
        seen.update(kw)
        return orig(q, acts, i, **kw)

    monkeypatch.setattr(T, "tidy_index", spy)

    class _M:
        def __init__(self, bomb):
            self.is_bomb = bomb
            self.wild_used = 0
            self.kind = 1
            self.size = 1
            self.rank = 5
            self.cards = (1,)

    T.tidy_net_policy(_Net())(_Obs(), [_M(True), _M(False)], None)
    assert seen.get("has_table") is False, f"领出必须 has_table=False，实得 {seen.get('has_table')}"
