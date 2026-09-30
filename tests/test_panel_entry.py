"""面板**入口**那条路（实机体上会走的就是它）：签名、接线、别崩。

⚠️ 独立评审抓出来的：Task 4 只改了函数体、没改签名（一次失败的补丁脚本把前面
三次成功替换一起丢掉了），于是 `python -m guandan.ui.panel --replay` 一启动就
`TypeError`；而**实机入口 `guandan.launcher` 起面板时压根没传记录器** ——
用户按交付台账打几十局会得到空文件。原来的接线测试只测
`apply_event + after_event`，所以 324 个测试全绿也照样漏。
"""
import json
import os

from guandan import paths
from guandan.ui import panel

from guandan.advice import shadow
from guandan.sim.meld import cid_from_name as A
from guandan.capture.state import GameState
from guandan.rl.net import QNet

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = str(paths.RAW)


def _recorder(tmp_path, **kw):
    return shadow.ShadowLog(net=QNet().eval(), out_path=str(tmp_path / "s.jsonl"),
                            weights="", **kw)


def _reader(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def test_open_shadow_reports_the_weights_it_actually_loads(tmp_path, monkeypatch):
    """`open_shadow` 是**唯一**一处建记录器的地方（launcher 与两个面板都走它）。

    它记进 session 行的 `weights` 必须是**真正要加载的那一份** ——
    原来面板传的是 `newest_weights()`，而 `load_net` 用的是
    `GUANDAN_WEIGHTS or newest_weights()`：设了环境变量之后，溯源字段说的是别人。
    """
    monkeypatch.setenv("GUANDAN_WEIGHTS", str(tmp_path / "nope.pt"))
    log = shadow.open_shadow(out_path=str(tmp_path / "s.jsonl"))
    assert not log.enabled
    assert log.weights == str(tmp_path / "nope.pt"), "溯源字段要说真正加载的那份"
    assert "不存在" in log.last_line


def test_text_panel_live_takes_shadow_log_and_stops_after_seconds(tmp_path,
                                                                 monkeypatch):
    """文本面板的实时入口要能吃 level / shadow_log / seconds（seconds 是给测试用的）。

    事件**要在它跑起来之后再写**：`Tailer` 是从文件末尾开始跟读的（生产上就该这样，
    不然会把上一局的旧事件重放一遍）。所以这里开一个线程，过一会儿再追加。
    """
    import threading
    import time as _time

    ev_path = tmp_path / "events.jsonl"
    ev_path.write_text("", encoding="utf-8")          # 先有个空文件，尾部 = 0
    events = (
        {"type": "play", "seat": 1, "cards": [A("S3")], "card_type": 0,
         "next": 0, "left": 1, "left_cards": [A("S4")]},
        {"type": "play", "seat": 0, "cards": [A("S2")], "card_type": 0,
         "next": 1, "left": 26},
        {"type": "play", "seat": 1, "cards": [A("S4")], "card_type": 0,
         "next": 0, "left": 0, "left_cards": []},
    )

    def writer():
        _time.sleep(0.2)
        with open(ev_path, "a", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                fh.flush()

    # 实时那条路会去读**真实**的游戏日志拿级别 —— 测试里钉成「读不到」，
    # 否则断言会取决于用户昨天打的是几（打 A 时 level=14 会被判越界而跳过）
    monkeypatch.setattr(panel.LevelWatcher, "poll", lambda self: None)
    threading.Thread(target=writer, daemon=True).start()
    log = _recorder(tmp_path)
    panel.run_live(GameState(), str(ev_path), level=9, shadow_log=log, seconds=1.0)
    dec = [r for r in _reader(log.out_path) if r["type"] == "decision"]
    assert len(dec) == 1 and dec[0]["actual"] == [A("S4")]


def test_text_panel_replay_runs_with_level_and_shadow(tmp_path):
    """`python -m guandan.ui.panel --replay` 这条命令要能跑通并真的落盘。

    评审实测它的失败长这样：`TypeError: run_replay() got an unexpected keyword
    argument 'level'`（签名没改）。
    """
    if not os.path.exists(RAW):
        import pytest
        pytest.skip("没有全量抓包（runtime/raw.jsonl）")
    cap = tmp_path / "cap.jsonl"
    n = 0
    with open(RAW, encoding="utf-8") as src, open(cap, "w", encoding="utf-8") as dst:
        for line in src:
            r = json.loads(line)
            if r.get("k") == "frame" and r.get("dir") == "S→C":
                dst.write(json.dumps({"k": "ws_msg", "host": "hlxyxws",
                                      "dir": "S→C", "hex": r["hex"],
                                      "t": r["t"]},
                                     ensure_ascii=False) + "\n")
                n += 1
            if n >= 40:
                break
    log = _recorder(tmp_path)
    panel.run_replay(GameState(), str(cap), 0, level=9, shadow_log=log)
    assert [r for r in _reader(log.out_path) if r["type"] == "decision"], \
        "前 40 帧里至少该有一个决策点"
