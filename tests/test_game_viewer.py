"""自对弈回放器：翻页逻辑 + 「喂给渲染器的形状」是否齐全。

**不开窗口**（本仓库规矩：面板的显示逻辑要能不打开窗口就测）。
真建窗口那一步由 `python -m tools.game_viewer --selftest` 手动跑。
"""
import ast
import inspect
import textwrap

import pytest

from net import table
from net.sim import rules
from net.state import GameState
from tools.game_viewer import Cursor, _View
from tools.show_game import advice_of, frame_hint, replay_game

_has_weights = bool(__import__("glob").glob("runs/rl/*/best.pt"))


def test_cursor_clamps_at_both_ends():
    c = Cursor(3)
    assert (c.i, c.at_start, c.at_end) == (0, True, False)
    c.prev()                                   # 开头再往前 -> 不动（不许越到帧外面）
    assert (c.i, c.at_start) == (0, True)
    c.next(); c.next()
    assert (c.i, c.at_end) == (2, True)
    c.next()
    assert c.i == 2
    c.goto(99)
    assert c.i == 2
    c.goto(-5)
    assert c.i == 0


def test_cursor_rejects_an_empty_reel():
    with pytest.raises(ValueError):
        Cursor(0)


def test_view_provides_everything_the_renderer_actually_touches():
    """**从渲染器自己身上把字段清单抽出来**，而不是手抄一份。

    `TableWindow.draw` / `_seat_area` 里所有 `st.X` 就是契约。将来渲染器多要一个
    字段，这条**自动**变红 —— 不用等人去界面上看出「半个牌桌画不出来」。
    """
    need = set()
    for fn in (table.TableWindow.draw, table.TableWindow._seat_area):
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "st"):
                need.add(node.attr)
    assert need, "没抽出任何字段？渲染器的形参名变了？"

    class _F:                                  # 只求形状，不求内容
        step, seat, level, turn = 1, 0, 8, 0
        chosen = None
        table = None                           # 类命名空间里的，不遮外面的模块
        hands = {s: set() for s in range(4)}
        played = {s: [] for s in range(4)}
        passes = set()
        candidates, over = [], False

    v = _View(_F())
    missing = sorted(n for n in need if not hasattr(v, n))
    assert not missing, f"`_View` 少了渲染器要的字段：{missing}"


@pytest.mark.skipif(not _has_weights, reason="还没训练出 runs/rl/*/best.pt")
def test_replay_gives_one_frame_per_play_plus_a_final_one():
    """**真打一局**（约 5 秒）：帧数 = 手数 + 1，最后一帧是终局。"""
    meta, frames = replay_game(seed=61, top=3)
    assert len(frames) == meta["steps"] + 1
    assert frames[-1].over and frames[-1].turn is None
    assert not any(f.over for f in frames[:-1])
    # 名次是「按座位索引」的 1..4
    assert sorted(meta["ranks"]) == [1, 2, 3, 4]
    # 每一帧的四家手牌加起来 + 已经出掉的 = 108 张
    f = frames[-1]
    assert sum(len(v) for v in f.hands.values()) + \
        sum(len(p.cards) for ps in f.played.values() for p in ps) == 108


@pytest.mark.skipif(not _has_weights, reason="还没训练出 runs/rl/*/best.pt")
def test_advice_is_the_shape_the_panel_wants():
    """右侧栏吃 `[{cards, q}, …]`，且**首选在前**（面板按 i==0 画大图）。"""
    _meta, frames = replay_game(seed=61, top=3)
    f = frames[0]
    adv = advice_of(f)
    assert adv and all("cards" in a and "q" in a for a in adv)
    assert [a["q"] for a in adv] == sorted((a["q"] for a in adv), reverse=True)
    assert f.wasted in (True, False) and frame_hint(f)


# ------------------------------------------------ 四家手牌（2026-09-29 用户要的）

def test_the_viewer_exposes_all_four_hands():
    """用户报的：「想直接看到每个人打过的牌，和当前手牌，而不是**到谁了才能看到谁的手牌**」。

    ⚠️ 这是**回放器独有**的能力：真实对局里拿不到别人的手牌，所以实机面板
    （`GameState`）没有 `hand_of` —— 见下面那条测试。
    """
    _, frames = replay_game(seed=7, level=8)
    f = next(x for x in frames if not x.over)
    v = _View(f)
    for s in rules.SEATS:
        assert v.hand_of(s) == sorted(f.hands[s]), f"座位{s} 的手牌不对"


def test_the_live_panel_state_has_no_hand_of():
    """**隔离**：`GameState` 没有 `hand_of` ⇒ `_seat_area` 走原来那条分支
    （出过的牌铺满整个框、不画手牌）⇒ **实机面板的样子一字不变**。"""
    assert not hasattr(GameState(), "hand_of")


def test_hand_of_returns_a_copy_not_the_live_set():
    """返回 `sorted(...)` 而不是原集合 —— 渲染器不该能改到帧里的数据。"""
    _, frames = replay_game(seed=7, level=8)
    f = next(x for x in frames if not x.over)
    v = _View(f)
    got = v.hand_of(3)
    got.append(999999)
    assert v.hand_of(3) == sorted(f.hands[3]), "改了返回值却影响到了帧"
