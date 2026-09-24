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
from datetime import datetime

import pytest

import tools.accept_meld as accept
from tools.accept_meld import (check_beats_from_records, check_invariants,
                               check_real_moves, check_wildcard)
from tools.game_log import GameLog, LOG_DIR, PlayRec

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


# ---------------------------------------------------------------------------
# 终审修复：「验收全绿」必须建立在**真的检查过东西**上。
#
# `Result.ok` 原来只是 `not bad`，于是「一项都没检查到」会打成
# `[OK] … 0 项全过`、main() 记成功、exit 0。构造非常简单：语料里所有局都没结算
# （`g.settle is None` —— 本机实测就有 8 局如此，日志 2 天轮转只会更糟），
# ①② 一条都查不到。本项目已经被「绿因为什么都没跑」咬过三次（spec §6⑥）。
#
# 下面两条都用**注入语料**跑完整的 main()（`games=` 参数就是为这个加的），
# 断言的是进程退出码与输出文字 —— 不是内部函数，是那个门本身。

_LEFTCARDS = [{"Cards": [16 + 5]} for _ in range(4)]      # 每座位剩一张 ♠5


def _game(plays=(), settle=None, trump=9):
    return GameLog(t0=datetime(2026, 9, 24, 17, 0, 0), trump=trump,
                   my_cards=[], plays=list(plays), settle=settle)


def _bomb_game():
    """一局「4 张 5 的炸被 4 张 6 的炸压掉」——①② 各至少有一条可查。"""
    p1 = PlayRec(seat=0, cards=[16 + 5, 16 + 5 + 256, 32 + 5, 32 + 5 + 256],
                 card_type=8, left=26, nxt=1)
    p2 = PlayRec(seat=1, cards=[48 + 6, 48 + 6 + 256, 64 + 6, 64 + 6 + 256],
                 card_type=8, left=26, nxt=-1)
    settle = {"LeftCards": [{"Cards": list(p1.cards)}, {"Cards": list(p2.cards)},
                            {"Cards": []}, {"Cards": []}]}
    return _game(plays=[p1, p2], settle=settle)


def _run_main(monkeypatch, games) -> int:
    # main() 上来就把 stdout 切到 UTF-8；pytest 的捕获流不一定支持 reconfigure。
    # 这里测的是「门会不会红」，不是编码，所以把它换掉。
    monkeypatch.setattr(accept, "_utf8_stdout", lambda: None)
    return accept.main(games=games)


def test_zero_items_examined_is_red(monkeypatch, capsys):
    """语料够多、却一手都没得查时，①② 必须红 —— 不许报「0 项全过」。

    语料是 `_MIN_SETTLED` 局「已结算但没有一手出牌」的局：结算局数达标（地板
    不响），但 `decision_points()` 给出 0 个快照、`_bomb_pairs()` 给出 0 对。
    这正是 reviewer 用「所有局都没结算」构造出来的那个假绿。
    """
    games = [_game(settle={"LeftCards": _LEFTCARDS})
             for _ in range(accept._MIN_SETTLED)]
    rc = _run_main(monkeypatch, games)
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "验收全绿" not in out
    assert "一项都没检查到" in out


def test_too_little_corpus_is_red(monkeypatch, capsys):
    """结算局低于语料地板时，①② 不许声称「全过」（哪怕一条错都没查出来）。

    这里 ①② 各自的 total 都 > 0 且 bad 为空 —— 唯一让它变红的就是地板。
    """
    rc = _run_main(monkeypatch, [_bomb_game()])           # 只有 1 局结算
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "验收全绿" not in out
    assert "地板" in out and "结算对局只有 1 局" in out


def test_floor_is_not_a_blind_red(monkeypatch, capsys):
    """地板达标 + 一条错都没有 -> 必须**绿**（别把修复做成「永远红」）。"""
    rc = _run_main(monkeypatch, [_bomb_game() for _ in range(accept._MIN_SETTLED)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "验收全绿" in out
    # ② 的条数拆分必须打出来：「530 项全过」不能再被读成阶梯覆盖率
    assert "炸弹 vs 炸弹" in out
