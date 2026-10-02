"""对手情报提示：**只用公开信息、零动作影响**。"""
from guandan.advice.reasons import opponent_hint
from guandan.sim import env, rules


def test_hint_is_public_info_only_and_starts_with_prefix():
    e = env.GuandanEnv(seed=11)
    o = e.reset()
    h = opponent_hint(o)
    assert h.startswith("对手："), h
    assert "剩" in h


def test_hint_flags_short_opponent():
    """有人 ≤3 张时必须点出来（这是人要数的东西）。"""
    e = env.GuandanEnv(seed=11)
    o = e.reset()
    opp = next(s for s in rules.SEATS if rules.TEAM[s] != rules.TEAM[o.seat])
    o2 = type(o)(**{**o.__dict__,
                    "left": tuple(2 if i == opp else v for i, v in enumerate(o.left))})
    h = opponent_hint(o2)
    assert "只剩 2 张" in h


def test_hint_never_touches_actions():
    e = env.GuandanEnv(seed=11)
    o = e.reset()
    acts = e.legal()
    before = [None if m is None else m.cards for m in acts]
    opponent_hint(o)
    assert [None if m is None else m.cards for m in acts] == before
