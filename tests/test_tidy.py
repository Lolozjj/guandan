"""`rl/tidy.py`（擦浪费）与 `advice.advise.tidy_mode`（开关）—— 用户实机抱怨的那两件事。

用户 2026-10-01 原话：①「有牌可以压制的情况下会使用炸弹进行压制」；
②「实际炸个 8888 就行了，但他会用万能牌构造成五个 8 去炸」。

两条最容易写错的地方：
- **不能降低"压得过"的能力**：候选已按 `beats` 过滤，任何候选都压得过；擦的只是"用更贵的资源"。
- **不浪费时不许动**：擦浪费只在判定成立时介入 ⇒ 不成立时必须与 `argmax` **逐位一致**。
"""
import os

import pytest

from guandan.rl import tidy


def _meld(kind, size, rank, cards=(1,), wild=0, bomb=False):
    """造一个够用的假 Meld（测试只关心这几个字段）。"""
    from guandan.sim.meld import Meld
    m = Meld(kind, size, rank, tuple(cards), wild_used=wild)
    return m


def _acts():
    """[普通对子, 炸弹(天然), 炸弹(用万能牌放大)] —— 桌面有牌要压的场景。"""
    import guandan.sim.meld as M
    plain = _meld(M.PAIR, 2, 5, (1, 2))
    bomb4 = _meld(M.BOMB, 4, 8, (10, 11, 12, 13), bomb=True)
    bomb5w = _meld(M.BOMB, 5, 8, (10, 11, 12, 13, 99), wild=1, bomb=True)
    return [plain, bomb4, bomb5w]


def test_no_waste_means_no_intervention():
    """⚠️ **不浪费时必须逐位不动**（否则"擦浪费"会把所有决策都搅一遍）。"""
    acts = _acts()
    q = [3.0, 1.0, 0.5]
    assert tidy.tidy_index(q, acts, has_table=True, chosen_i=0) == 0


def test_tidy_removes_wasted_bomb():
    """① 白炸：有普通牌能压却选了炸弹 ⇒ 换成 Q 最高的**非炸弹**。"""
    acts = _acts()
    q = [0.5, 3.0, 2.0]                      # argmax 是天然炸弹（浪费）
    assert tidy.tidy_index(q, acts, has_table=True, chosen_i=1) == 0


def test_tidy_prefers_natural_bomb_over_wild_enlarged_bomb():
    """② 用户那一手：天然 8888 就在候选里，却用万能牌凑成 5 张炸 ⇒ **换回天然炸弹**。

    ⚠️ 这条以前被我写成"两条 if 顺序判断"，结果第一条（白炸）先把局面降级成普通牌，
    第二条**永远轮不到**。改成资源字典序之后才成立 —— 这个 bug 是测试逼出来的。
    """
    import guandan.sim.meld as M
    bomb4 = _meld(M.BOMB, 4, 8, (10, 11, 12, 13), bomb=True)
    bomb5w = _meld(M.BOMB, 5, 8, (10, 11, 12, 13, 99), wild=1, bomb=True)
    q = [1.0, 3.0]                           # 网络更偏爱"万能牌放大"的那一手
    assert tidy.tidy_index(q, [bomb4, bomb5w], has_table=True, chosen_i=1) == 0


def test_tidy_keeps_the_bomb_when_it_is_the_cheapest_option():
    """只有炸弹可压（没有普通牌）时**仍然要炸** —— 擦浪费不是"从此不炸"。

    但同是炸弹时，**天然炸弹优先于万能牌放大的炸弹**（这就是②那条）。
    """
    import guandan.sim.meld as M
    bomb4 = _meld(M.BOMB, 4, 8, (10, 11, 12, 13), bomb=True)
    bomb8 = _meld(M.BOMB, 8, 15, (30, 31, 32, 33, 34, 35, 36, 37), bomb=True)
    q = [1.0, 3.0]
    # 两个都是天然炸弹、都不用万能牌 ⇒ **一动不动**（没有更省的可选）
    assert tidy.tidy_index(q, [bomb4, bomb8], has_table=True, chosen_i=1) == 1
    # 若"最省"是普通牌（普通对子），才会被降级 —— 这里没有普通牌，所以不降级
    pair = _meld(M.PAIR, 2, 5, (1, 2))
    assert tidy.tidy_index(q, [pair, bomb8], has_table=True, chosen_i=1) == 0


def test_tidy_net_policy_only_changes_wasteful_decisions(monkeypatch):
    """包一层之后：**不浪费的局面**必须与裸网络给出同一下标。"""
    import numpy as np
    from guandan.rl.tidy import tidy_net_policy

    class _Net:      # 只占位，q_values 被替换
        pass

    acts = _acts()
    monkeypatch.setattr("guandan.rl.tidy.q_values", lambda *a, **k: np.array([3.0, 1.0, 0.5]))
    class _Obs:
        table = None                       # 领出（没牌要压）⇒ 擦浪费不介入

    pol = tidy_net_policy(_Net())
    assert pol(_Obs(), acts, None) == 0    # 不浪费 ⇒ 与裸网络一致


# ---------------------------------------------------------------- 开关

def test_tidy_mode_parsing_and_loud_rejection():
    from guandan.advice.advise import tidy_mode
    old = os.environ.get("GUANDAN_TIDY")
    try:
        for raw, want in (("", "off"), ("0", "off"), ("off", "off"), ("1", "all"),
                          ("all", "all"), ("bombs", "bombs"), ("wilds", "wilds")):
            os.environ["GUANDAN_TIDY"] = raw
            assert tidy_mode() == want, raw
        os.environ["GUANDAN_TIDY"] = "maybe"
        with pytest.raises(ValueError, match="GUANDAN_TIDY"):
            tidy_mode()
    finally:
        if old is None:
            os.environ.pop("GUANDAN_TIDY", None)
        else:
            os.environ["GUANDAN_TIDY"] = old

def test_pass_is_never_a_tidy_alternative():
    """⚠️ **回归测试**：「过」不能被当成"更省资源"的选择。

    2026-10-01 的真 bug：`None` 既不是炸弹也不用万能牌 ⇒ 在字典序里被当成最省 ⇒
    该炸的局面被判成浪费、改成**过牌** ⇒ 实测 300 局**一次炸都不出**。
    """
    import guandan.sim.meld as M
    bomb = _meld(M.BOMB, 4, 8, (10, 11, 12, 13), bomb=True)
    acts = [None, bomb]                        # 候选里有"过"（跟牌时才有）
    q = [5.0, 1.0]                             # 网络其实想炸（q[1] 是它选的）
    got = tidy.tidy_index(q, acts, has_table=True, chosen_i=1)
    assert acts[got] is not None, "不许把出炸改成过牌"
    assert got == 1, "没有更省的实体牌 ⇒ 一动不动"
