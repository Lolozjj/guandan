"""牌桌面板的**右侧建议栏**：用牌面图片显示模型建议，不写数字。

用户 2026-09-26 的要求（原来是底部一行文字，他说「不够明显」）：
右侧单独一栏、首选大图 + 两个备选小图、**不显示 Q 值/名次这些数字**。

显示逻辑里能脱离窗口的那部分拆成纯函数（`table._wrap`）离线测 ——
项目规矩：面板的显示逻辑要能不打开窗口就测（HANDOFF 第八节第 6 条）。
"""
from net import shadow
from net.sim.meld import cid_from_name as A
from net.state import GameState
from net import table
from train.net import QNet


def test_wrap_chunks_long_melds():
    """一手最多画 5 张一行，多的换行（8 张的炸也得放得下）。"""
    assert table._wrap([1, 2, 3, 4, 5, 6, 7, 8], 5) == [[1, 2, 3, 4, 5], [6, 7, 8]]
    assert table._wrap([1, 2, 3], 5) == [[1, 2, 3]]
    assert table._wrap([], 5) == []


def _play(st, seat, nxt, played, rest, mine=False):
    st.on_play(seat, list(played), 0, nxt, len(rest), sorted(rest) if mine else None)


def test_the_recorder_hands_the_panel_structured_advice(tmp_path):
    """面板要的是**牌 ID 列表**（它自己画牌面），不是拼好的文字。

    生命周期与 `last_advice` 一致：算出建议时有值，我一出手就清空。
    """
    log = shadow.ShadowLog(net=QNet().eval(), out_path=str(tmp_path / "s.jsonl"),
                           weights="", show_advice=True)
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4"), A("H5")], mine=True)
    _play(st, 0, 1, [A("S4")], [A("S9")])
    log.after_event(st, {"type": "play"})
    assert log.advice_top, "面板该拿到结构化的建议"
    assert all("cards" in it and "q" in it for it in log.advice_top)
    _play(st, 1, 0, [A("H5")], [A("S4")], mine=True)
    log.after_event(st, {"type": "play"})
    assert log.advice_top == [], "出手之后要清空"


def test_drawing_the_advice_panel_does_not_crash(tmp_path):
    """真开一个窗口，把三种情况都画一遍：有建议 / 过 / 没建议。

    视觉只能靠离线冒烟看（`python -m net.table --replay`），这里守的是「别崩」。
    """
    import tkinter as tk
    try:
        root = tk.Tk()
    except Exception:                      # noqa: BLE001 - 没有可用显示时跳过
        import pytest
        pytest.skip("没有可用的 Tk 显示")
    try:
        win = table.TableWindow(root)
        st = GameState()
        st.level = 9
        st.hand = [A("S3"), A("H5")]
        # ① 有建议：首选一把 8 张炸（要换行）+ 两个备选
        win.draw(st, "提示", 3, "影子：测试",
                 advice=[{"cards": [A("S3")] * 1 + [A("H5")] * 1, "kind": 8, "q": 0.1},
                         {"cards": [], "kind": 0, "q": 0.05},          # 「过」
                         {"cards": [A("S3")], "kind": 1, "q": 0.01}])
        # ② 没建议（还没轮到我）
        win.draw(st, "提示", 3, "影子：测试", advice=None)
        tk.Label(root).destroy()            # 只是让 root 有个子控件，避免空窗口的怪行为
    finally:
        root.destroy()
