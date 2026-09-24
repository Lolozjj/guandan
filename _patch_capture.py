"""给 capture.py 加「被挡住也能抓」的 PrintWindow 模式。

原来的实现用 mss 抓屏幕区域 —— 游戏窗口一旦被别的窗口盖住，抓到的就是
盖住它的那个窗口（实测：PyCharm 盖住游戏时，抓到的整张图全是编辑器画面，
面板显示「0 张手牌」）。用户不会一直把游戏放最前面，所以这个是必须解决的。

PrintWindow(hwnd, dc, PW_RENDERFULLCONTENT=2) 让窗口自己把内容画到 DC 上，
不经过屏幕，被盖住/移出屏幕都不影响。默认用它，失败再回退到抓屏幕。
"""
from pathlib import Path

p = Path('live/capture.py')
s = p.read_text(encoding='utf-8')

# 1) 导入
old = """import mss          # noqa: E402
import win32gui     # noqa: E402"""
new = """import mss          # noqa: E402
import win32con     # noqa: E402
import win32gui     # noqa: E402
import win32ui      # noqa: E402

PW_RENDERFULLCONTENT = 2      # Win 8.1+ 才有，能抓到硬件加速窗口的内容"""
assert old in s
s = s.replace(old, new)

# 2) 新增抓窗口内容的函数
anchor = """class GameCapture:"""
func = '''def grab_window_printwindow(hwnd: int) -> np.ndarray | None:
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


class GameCapture:'''
assert anchor in s
s = s.replace(anchor, func, 1)

# 3) 构造函数加开关
old = """    def __init__(self, title_substr: str = DEFAULT_TITLE,
                 canonical_width: int | None = None):
        self.title_substr = title_substr
        self.canonical_width = canonical_width"""
new = """    def __init__(self, title_substr: str = DEFAULT_TITLE,
                 canonical_width: int | None = None,
                 offscreen: bool = True):
        self.title_substr = title_substr
        self.canonical_width = canonical_width
        # offscreen=True: 用 PrintWindow 抓，被别的窗口盖住也认；
        #                 抓不到时自动回退到抓屏幕。
        self.offscreen = offscreen
        self.last_mode = ""          # 上一次实际用的是哪种方式，便于排查"""
assert old in s
s = s.replace(old, new)

# 4) grab() 里优先用 PrintWindow
old = """        l, t, r, b = self.rect()
        if r - l < 300 or b - t < 300:
            return None
        raw = np.asarray(self._sct.grab({"left": l, "top": t,
                                         "width": r - l, "height": b - t}))
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)"""
new = """        l, t, r, b = self.rect()
        if r - l < 300 or b - t < 300:
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
            return None"""
assert old in s
s = s.replace(old, new)

p.write_text(s, encoding='utf-8')
print('capture.py 已加 PrintWindow 模式')
