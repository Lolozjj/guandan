"""第一层验收脚本的自动化门（Task 7 / spec §6）。

跑法：
    .venv/Scripts/python.exe -m pytest tests/test_accept.py -q
    .venv/Scripts/python.exe -m tools.accept_meld

⚠️ skipif **逐条挂**，不用模块级 `pytestmark`：
`check_invariants()` 与 `check_wildcard()` 是纯逻辑、一次日志都不读，目录没了
它们照样该跑。模块级 marker 会把这两条一起静默 skip 掉 —— 本项目
tests/test_decision_points.py 顶上已经把标准写清楚了：模块级 skip 只在
「全部测试都要读真实日志」时才成立，这里不成立。
（这条是相对 brief 的有意偏离，见 task-7-report.md。）
"""
import os
import subprocess
import sys

import pytest

from tools.accept_meld import (check_beats_from_records, check_invariants,
                               check_real_moves, check_wildcard)
from tools.game_log import LOG_DIR

# 只有下面三条要真日志：check_real_moves / check_beats_from_records 内部会
# load_games()，main() 更是上来就 load_games()。目录没了它们就是 skip
# （真阻塞，不是通过）。
_needs_logs = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                 reason="本机没有游戏日志")


def test_invariants_pass():
    r = check_invariants()
    assert r.ok, r.report()


def test_wildcard_checks_pass():
    r = check_wildcard()
    assert r.ok, r.report()


@_needs_logs
def test_real_moves_are_all_legal():
    """55 局真实记录里，每一手真实出的牌都必须在合法着法集合里。"""
    r = check_real_moves()
    assert r.ok, r.report()


@_needs_logs
def test_bombs_beat_what_records_say():
    """真实对局里「炸弹 A 之后又出了炸弹 B」的证据必须逐条成立。"""
    r = check_beats_from_records()
    assert r.ok, r.report()


@_needs_logs
def test_main_exits_zero_through_a_pipe():
    """`python -m tools.accept_meld` 全绿返回 0 —— **包括输出被接管道时**。

    这条守的是一个真踩过的坑：main() 末行 `print("验收全绿 ✓")` 的 U+2713 在
    GBK 里没有，stdout 一旦被重定向/接管道（CI、`> out.txt`）就
    UnicodeEncodeError -> 四项全绿却 exit 1。控制台直连时不会犯，所以肉眼
    跑一遍看不出来，必须有这条。子进程的 stdout 就是管道。
    """
    p = subprocess.run([sys.executable, "-m", "tools.accept_meld"],
                       capture_output=True, timeout=300)
    out = p.stdout.decode("utf-8", errors="replace")
    assert p.returncode == 0, out + p.stderr.decode("utf-8", errors="replace")
    assert out.count("[OK]") == 4, out
