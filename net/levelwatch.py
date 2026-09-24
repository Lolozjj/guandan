"""从游戏日志里读级别（打几）。

为什么走日志而不是网络：级别在网络报文里找不到（3006/3009/3004/3019 都翻过了），
但**级别一局之内根本不变**，而日志的延迟只有约 20 秒 —— 对这个字段毫无影响。
这样就不用截图、不用视觉识别，面板保持「纯数据」。
"""

import glob
import os
import re
import time

LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

DEAL_RE = re.compile(r"SendCardsService set roundID:([\d,]+) k : "
                     r"\{.*?\"Trump\":(\d+)")
SETTLE_RE = re.compile(r"结算协议 = \{")


class LevelWatcher:
    """跟读日志最新那几个文件，取出最近一次的级别。

    只看每个文件的尾部：日志动辄几 MB，全扫一遍没必要。
    """

    TAIL = 300_000          # 每次读文件最后这么多字节

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        self.level = None
        self.last_seen = 0.0
        self._pos = {}

    def poll(self):
        """返回级别（没变就返回 None）。"""
        files = sorted(glob.glob(os.path.join(self.log_dir, "*.log")))[-2:]
        found = None
        for f in files:
            try:
                size = os.path.getsize(f)
            except OSError:
                continue
            start = self._pos.get(f)
            if start is None or size < start:
                start = max(0, size - self.TAIL)     # 首次只看尾部
            if size <= start:
                continue
            with open(f, "rb") as fh:
                fh.seek(start)
                raw = fh.read()
            cut = raw.rfind(b"\n")
            if cut == -1:
                continue
            self._pos[f] = start + cut + 1
            text = raw[:cut + 1].decode("utf-8", "replace")
            for line in text.splitlines():
                m = DEAL_RE.search(line)
                if m:
                    found = int(m.group(2))
        if found is not None and found != self.level:
            self.level = found
            self.last_seen = time.time()
            return found
        return None


if __name__ == "__main__":
    w = LevelWatcher()
    print("盯着日志里的级别，20 秒内看到就打印……")
    t0 = time.time()
    while time.time() - t0 < 20:
        lv = w.poll()
        if lv is not None:
            print(f"  级别 = 打{lv}")
        time.sleep(1)
    print(f"结束，最后已知级别 = {w.level}")
