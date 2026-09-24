"""列出当前所有可见窗口，找出掼蛋游戏窗口。

用法: python live/find_window.py
"""
import sys

import win32gui


def main():
    hits = []
    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        cls = win32gui.GetClassName(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if w < 200 or h < 200:
            return
        hits.append((hwnd, cls, title, rect, w, h))

    win32gui.EnumWindows(cb, None)

    print(f"可见窗口共 {len(hits)} 个，其中尺寸 >200x200 的：\n")
    for hwnd, cls, title, rect, w, h in sorted(hits, key=lambda t: -t[4] * t[5]):
        mark = " <== 可能是游戏" if ("掼蛋" in title or "guandan" in title.lower()) else ""
        print(f"  hwnd={hwnd:<10} {w:>5}x{h:<5} pos=({rect[0]},{rect[1]})"
              f"  class={cls:<24} title={title!r}{mark}")


if __name__ == "__main__":
    main()
