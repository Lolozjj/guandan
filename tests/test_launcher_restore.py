"""还原路径：**演示结束关面板时不许卡、不许弹 traceback、不许报假成功**。

2026-09-29 用 `net.launcher --selftest 30` 实机自检抓到：

    subprocess.TimeoutExpired: Command '['certutil', '-user', '-delstore', 'Root',
    '20B1DD17...']' timed out after 60 seconds

实测**证书其实已经删掉了**（复核过：库里没有了、代理还原成了 127.0.0.1:7897、
mitmdump 已退出）—— 是 `certutil` 删完**自己卡住不退**（删不存在的目标只要 0.29 秒，
删真实存在的会卡 >60 秒，像是弹了个确认框）。
后果不是「没还原」，是「**关面板时卡一分钟 + 弹红色报错**」，
而且 `remove_ca` 后面那两行「证书已卸干净 / 已还原代理」**全被跳过**。

⇒ 定下的两条纪律：**不等 certutil 退出**（轮询证书库，数到 0 就收工并 kill 掉它）、
**查不到就不许说成功**。
"""
import subprocess
import time

import net.launcher as launch


class _Stuck:
    """一个**永远不退出**的假 certutil（模拟那个确认框）。"""

    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def test_a_stuck_certutil_neither_blocks_nor_raises(monkeypatch):
    stuck = _Stuck()
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: stuck)
    monkeypatch.setattr(launch, "ca_installed_count", lambda: 0)
    t = time.monotonic()
    assert launch.remove_ca("ABC") is True          # 复核干净了 -> 就是干净了
    assert time.monotonic() - t < 2.0, "它还在等 certutil 退出（演示时会卡住）"
    assert stuck.killed, "卡住的进程必须收掉，不许留着"


def test_an_unknown_count_is_not_reported_as_success(monkeypatch):
    """查不到（`None`）**不等于**删干净了 —— 原来 `int(out) if out.isdigit() else 0`
    在 PowerShell 起不来时静默返回 0，也就是在还原路径上报**假成功**。"""
    monkeypatch.setattr(launch, "DELSTORE_WAIT_S", 0.3)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: _Stuck())
    monkeypatch.setattr(launch, "ca_installed_count", lambda: None)
    assert launch.remove_ca("ABC") is False


def test_tries_by_name_when_the_fingerprint_left_something(monkeypatch):
    """按指纹没删干净 -> 按名字再试（原行为保留），返回 False。"""
    calls = []
    monkeypatch.setattr(launch, "DELSTORE_WAIT_S", 0.3)

    def popen(cmd, **kw):
        calls.append(cmd)
        return _Stuck()

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "ca_installed_count", lambda: 1)
    assert launch.remove_ca("ABC") is False
    assert len(calls) == 2
    assert any(launch.CA_CN in c for c in calls[-1])


def test_a_popen_failure_is_not_success(monkeypatch):
    def boom(*a, **k):
        raise OSError("certutil 不见了")
    monkeypatch.setattr(launch, "DELSTORE_WAIT_S", 0.3)
    monkeypatch.setattr(launch.subprocess, "Popen", boom)
    monkeypatch.setattr(launch, "ca_installed_count", lambda: 1)
    assert launch.remove_ca("ABC") is False


# ---------------------------------------------------------------- 复核本身

def test_the_count_never_lies_when_certutil_fails(monkeypatch):
    """`certutil` 起不来（或返回非 0）-> `None`，**不是 0**。"""
    def boom(*a, **k):
        raise OSError("certutil 不见了")
    monkeypatch.setattr(launch.subprocess, "run", boom)
    assert launch.ca_installed_count() is None
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", ""))
    assert launch.ca_installed_count() is None


def test_the_count_reads_subject_lines(monkeypatch):
    """只数 `Subject:` 行 —— 同一张证 `Issuer:`/`Subject:` 各一行，数错会翻倍。"""
    out = ("================ Certificate 0 ================\n"
           "Serial Number: 00\n"
           "Issuer: CN=mitmproxy\n"
           " Subject: CN=mitmproxy\n"
           "Subject: CN=别的\n")
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, out, ""))
    assert launch.ca_installed_count() == 1
