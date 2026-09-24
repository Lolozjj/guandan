"""抓取掼蛋游戏窗口的画面。

设计要点：

1. **进程声明 DPI 感知**。Windows 有显示缩放（125%/150%）时，未声明感知的进程
   拿到的窗口坐标是「逻辑像素」，比截图里的物理像素小一圈，按那个区域抓会抓偏。
   声明之后 GetWindowRect 返回的就是物理像素，和截图对得上。

2. **只抓游戏窗口，不抓整个屏幕**。否则旁边那个信息面板也会被拍进去，
   模型看到一堆文字框会乱认。

3. **抓整个窗口（含标题栏），不是客户区**。用户给的截图里是带 "大掼蛋（腾讯）"
   那条标题栏的，模型见过的就是那个版式。

用法:
    python live/capture.py                 # 找窗口并保存一张快照
    python live/capture.py --title 掼蛋     # 指定窗口标题关键字
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --- 必须在使用任何窗口 API 之前声明 DPI 感知 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import mss          # noqa: E402
import win32api     # noqa: E402
import win32con     # noqa: E402
import win32gui     # noqa: E402
import win32process  # noqa: E402
import win32ui      # noqa: E402

PW_RENDERFULLCONTENT = 2      # Win 8.1+ 才有，能抓到硬件加速窗口的内容

DEFAULT_TITLE = "掼蛋"


def _is_own_window(hwnd: int) -> bool:
    """是不是本进程自己的窗口。

    必须排除：面板窗口的标题「掼蛋助手」也含「掼蛋」，游戏关掉之后
    find_window 会找到面板自己，于是面板开始抓自己的画面。
    """
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid == os.getpid()
    except Exception:
        return False


def find_window(title_substr: str = DEFAULT_TITLE) -> int | None:
    """按标题关键字找可见窗口，返回面积最大的那个的 hwnd。"""
    found = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or _is_own_window(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        if w > 300 and h > 300:
            found.append((w * h, hwnd, title, (l, t, r, b)))

    win32gui.EnumWindows(cb, None)
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1]


def list_candidates(title_substr: str = DEFAULT_TITLE) -> list[tuple]:
    out = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or _is_own_window(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title_substr and title_substr not in title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if r - l > 300 and b - t > 300:
            out.append((hwnd, title, (l, t, r, b), r - l, b - t))

    win32gui.EnumWindows(cb, None)
    return out


def grab_window_printwindow(hwnd: int) -> np.ndarray | None:
    """用 PrintWindow 抓窗口内容（BGR）。被别的窗口盖住也能抓到。

    抓不到（部分老程序对 WM_PRINT 支持不好）时返回 None，调用方回退到抓屏幕。
    """
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        return None
    hwnd_dc = None
    save_dc = None
    bmp = None
    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        # 返回 0 一般表示这个窗口不支持 WM_PRINT，交给调用方回退
        if not ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(),
                                                PW_RENDERFULLCONTENT):
            return None
        info = bmp.GetInfo()
        buf = bmp.GetBitmapBits(True)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(info["bmHeight"],
                                                         info["bmWidth"], 4)
        out = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # 全黑说明没画上东西（有些窗口会「成功」返回但内容为空）
        if out.size == 0 or float(out.mean()) < 1.0:
            return None
        return out.copy()
    except Exception:
        return None
    finally:
        try:
            if bmp is not None:
                win32gui.DeleteObject(bmp.GetHandle())
            if save_dc is not None:
                save_dc.DeleteDC()
            if hwnd_dc is not None:
                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


class GameCapture:
    """持续抓取游戏窗口画面。"""

    def __init__(self, title_substr: str = DEFAULT_TITLE,
                 canonical_width: int | None = None,
                 offscreen: bool = True):
        self.title_substr = title_substr
        self.canonical_width = canonical_width
        # offscreen=True: 用 PrintWindow 抓，被别的窗口盖住也认；
        #                 抓不到时自动回退到抓屏幕。
        self.offscreen = offscreen
        self.last_mode = ""          # 上一次实际用的是哪种方式，便于排查
        self.hwnd: int | None = None
        self._sct = mss.mss()
        self._last_rect = None

    def ensure_window(self) -> bool:
        """窗口失焦/被关掉后重新找。返回是否可用。"""
        if self.hwnd is not None and win32gui.IsWindow(self.hwnd):
            return True
        self.hwnd = find_window(self.title_substr)
        return self.hwnd is not None

    def rect(self) -> tuple[int, int, int, int]:
        l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        return l, t, r, b

    def grab(self) -> np.ndarray | None:
        """抓一帧（BGR）。窗口不可用时返回 None。"""
        if not self.ensure_window():
            return None
        l, t, r, b = self.rect()
        if r - l < 300 or b - t < 300:
            return None

        if _is_own_window(self.hwnd):
            # 找不到游戏、只找到面板自己的时候别自己拍自己
            self.hwnd = None
            return None

        img = None
        if self.offscreen:
            img = grab_window_printwindow(self.hwnd)
            self.last_mode = "printwindow" if img is not None else "screen(mss)"
        if img is None:
            raw = np.asarray(self._sct.grab({"left": l, "top": t,
                                             "width": r - l, "height": b - t}))
            img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            self.last_mode = self.last_mode or "screen(mss)"
        if img is None:
            return None
        if self.canonical_width and img.shape[1] != self.canonical_width:
            s = self.canonical_width / img.shape[1]
            img = cv2.resize(img, (self.canonical_width, int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        self._last_rect = (l, t, r, b)
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--save", default="live/snapshot.png")
    ap.add_argument("--n", type=int, default=1, help="连拍几张看稳定性")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    cands = list_candidates(args.title)
    if not cands:
        print(f"没找到标题含 {args.title!r} 的窗口。当前尺寸>300x300 的可见窗口：")
        for hwnd, title, rect, w, h in list_candidates(""):
            print(f"  {w:>5}x{h:<5} hwnd={hwnd:<10} {title!r}")
        raise SystemExit("请把游戏窗口打开后再试，或用 --title 指定别的关键字")

    print("候选窗口：")
    for hwnd, title, rect, w, h in cands:
        print(f"  {w}x{h} hwnd={hwnd} {title!r}")

    cap = GameCapture(args.title)
    if not cap.ensure_window():
        raise SystemExit("[FAIL] 拿不到窗口句柄")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img = cap.grab()
        if img is None:
            raise SystemExit("[FAIL] 抓屏失败")
        # 帧间差异，判断画面是否稳定（动画中不该识别）
        if i and img.shape == prev.shape:
            d = cv2.absdiff(img, prev).mean()
            print(f"  第{i}帧  {img.shape[1]}x{img.shape[0]}  与上一帧差异 {d:.2f}")
        else:
            print(f"  第{i}帧  {img.shape[1]}x{img.shape[0]}")
        prev = img
        if args.n > 1:
            time.sleep(args.interval)

    cv2.imwrite(args.save, img)
    print(f"\n快照已存: {args.save}")
    print("请打开这张图和 shots/ 下的截图对比 —— 版式应该一致（含标题栏、手牌在下方）。")


if __name__ == "__main__":
    main()
