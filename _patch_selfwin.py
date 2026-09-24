"""修 capture.py：不要把面板自己的窗口当成游戏。

用户关了游戏之后，面板的窗口标题「掼蛋助手」也含「掼蛋」这个关键字，
于是 find_window 找到了自己 —— 面板开始抓自己的画面（448x927），
显示「0 张手牌」。必须把「本进程自己的窗口」排除掉。
"""
from pathlib import Path

p = Path('live/capture.py')
s = p.read_text(encoding='utf-8')

old = """import argparse
import ctypes
import sys
import time
from pathlib import Path"""
new = """import argparse
import ctypes
import os
import sys
import time
from pathlib import Path"""
assert old in s
s = s.replace(old, new)

old = """import win32con     # noqa: E402
import win32gui     # noqa: E402
import win32ui      # noqa: E402"""
new = """import win32api     # noqa: E402
import win32con     # noqa: E402
import win32gui     # noqa: E402
import win32process  # noqa: E402
import win32ui      # noqa: E402"""
assert old in s
s = s.replace(old, new)

old = """def find_window(title_substr: str = DEFAULT_TITLE) -> int | None:"""
new = """def _is_own_window(hwnd: int) -> bool:
    \"\"\"是不是本进程自己的窗口。

    必须排除：面板窗口的标题「掼蛋助手」也含「掼蛋」，游戏关掉之后
    find_window 会找到面板自己，于是面板开始抓自己的画面。
    \"\"\"
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid == os.getpid()
    except Exception:
        return False


def find_window(title_substr: str = DEFAULT_TITLE) -> int | None:"""
assert old in s
s = s.replace(old, new)

# 两处枚举回调都加上排除
old = """    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        if w > 300 and h > 300:
            found.append((w * h, hwnd, title, (l, t, r, b)))"""
new = """    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or _is_own_window(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        if w > 300 and h > 300:
            found.append((w * h, hwnd, title, (l, t, r, b)))"""
assert old in s
s = s.replace(old, new)

old = """    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if r - l > 300 and b - t > 300:
            out.append((hwnd, title, (l, t, r, b), r - l, b - t))"""
new = """    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or _is_own_window(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if r - l > 300 and b - t > 300:
            out.append((hwnd, title, (l, t, r, b), r - l, b - t))"""
assert old in s
s = s.replace(old, new)

# grab 里再加一道保险：抓到的是自己就返回 None
old = """        img = None
        if self.offscreen:"""
new = """        if _is_own_window(self.hwnd):
            # 找不到游戏、只找到面板自己的时候别自己拍自己
            self.hwnd = None
            return None

        img = None
        if self.offscreen:"""
assert old in s
s = s.replace(old, new)

p.write_text(s, encoding='utf-8')
print('capture.py 已排除自身窗口')
