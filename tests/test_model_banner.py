"""启动时**控制台要写清「加载的是哪一份权重」**（2026-09-29 用户提的）。

为什么需要：面板挑权重有两级 —— `GUANDAN_WEIGHTS` 环境变量 → `runs/rl/*/best.pt` 里
最新的一份（按修改时间）。用户设了环境变量去试一份**中途快照**之后，
从控制台**看不出到底加载了哪一份**，也就无法确认「试完有没有回到 1407」。

本仓库的纪律就是**换源必须可见**（这句在 HANDOFF 和 spec 里都写过）。

⚠️ 这一行会同时出现在三个入口：`net.launcher` 与两个面板的 `main()` ——
它们打的是同一个 `ShadowLog.last_line`，所以**只改一处**，不会漂。
"""
from net import advise
from net import shadow


def test_the_banner_names_the_file_and_the_training_amount(monkeypatch):
    monkeypatch.setattr(advise, "weights_info", lambda p: {"games": 160000})
    line = shadow.model_banner("runs/ab/R7_rule/pool/snap_160000.pt")
    assert "snap_160000.pt" in line, "得写清是哪一份"
    assert "160,000" in line, "得写清训练到多少局（同目录里 best 和 snap 是两个东西）"
    assert "R7_rule" in line, "目录也要有 —— 光看文件名分不清是哪次训练"


def test_it_still_names_the_file_when_the_metadata_is_unreadable(monkeypatch):
    """拿不到局数（旧格式 / 读不了）也要报出文件名，不许整个吞掉。"""
    monkeypatch.setattr(advise, "weights_info", lambda p: {})
    line = shadow.model_banner("runs/rl/20260926-1407/best.pt")
    assert "best.pt" in line and "20260926-1407" in line


def test_no_model_is_stated_explicitly():
    line = shadow.model_banner("")
    assert "没有模型" in line


def test_the_logger_uses_the_banner_as_its_startup_line(monkeypatch):
    """接进 `ShadowLog.last_line` —— 三个入口打的都是它。"""
    monkeypatch.setattr(advise, "weights_info", lambda p: {"games": 1})
    sh = shadow.ShadowLog(weights="runs/ab/R7_rule/pool/snap_160000.pt")
    assert "snap_160000.pt" in sh.last_line


def test_a_load_failure_keeps_saying_why(monkeypatch):
    """加载失败时**不许**被这句顶掉 —— 那时候用户更需要知道原因。"""
    sh = shadow.ShadowLog(weights="x/y.pt", weights_note="权重加载失败：坏文件")
    assert sh.last_line == "权重加载失败：坏文件"
