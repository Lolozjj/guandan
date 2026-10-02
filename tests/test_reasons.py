"""解释型建议：**只解释、不改动作**（零强度影响）。"""
from guandan.advice.reasons import explain
from guandan.sim import env, meld, rules


def _obs_with(seed=7):
    e = env.GuandanEnv(seed=seed)
    o = e.reset()
    return e, o


def test_no_reason_is_empty_string():
    """没有可说的地方必须返回空串（不许硬凑理由）。"""
    e, o = _obs_with()
    acts = e.legal()
    r = explain(o, acts, 0)
    assert isinstance(r, str)


def test_feed_reason_when_mate_has_two():
    """队友剩 2 张 + 我领出 + 我出对子 ⇒ 必须给出「喂队友」的理由。"""
    e, o = _obs_with()
    acts = e.legal()
    # 把队友手数改成 2、桌面清空，然后挑一个对子当"已选定"
    o2 = type(o)(**{**o.__dict__, "left": tuple(2 if i == rules.PARTNER[o.seat] else v
                                               for i, v in enumerate(o.left)), "table": ()})
    pair_i = next((i for i, m in enumerate(acts) if m is not None and m.kind == meld.PAIR), None)
    if pair_i is None:
        return                       # 这手没有对子 ⇒ 换局由上层覆盖
    assert "喂队友" in explain(o2, acts, pair_i)


def test_explain_never_changes_action():
    """这个模块**只读**：不该碰 acts（原地对比）。"""
    e, o = _obs_with()
    acts = e.legal()
    before = [None if m is None else m.cards for m in acts]
    explain(o, acts, 0)
    assert [None if m is None else m.cards for m in acts] == before
