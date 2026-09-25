"""从游戏日志里读级别（打几）。

⚠️ **游戏 3.2.2（2026-09-26）把发牌行删了** —— 老格式那条
`SendCardsService set roundID:… k : {"Trump":N,…}` 现在一条都没有
（实测命中 0），于是「级别」这个字段一度全空，面板一个决策点都算不出来。
现在的主来源是**结算行**（还在）里的 `UpgradeInfo`，语义实测如下
（拿 34 局老格式样本对齐，那时本局 Trump 是确定的）：

    UpgradeInfo.TrumpValue = **赢家升级后、下一局要打的级别**   ← 34/34 对上
    UpgradeInfo.trump[]    = 升级后每座位的级别（两队各一个）

⚠️ **它不是本局级别**（34 局里 0 局相等）。踩过这个坑：拿 TrumpValue 当本局级别，
算出来的建议连牌型都判不出来（那一局打 9，我按 11 算，`9♥ AAA` 这种逢人配牌全解不出）。

发牌行那条路**保留**（游戏回退时还得能用），两条都在时以**后出现的那条**为准。
"""

import glob
import json
import os
import re
import time

LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

DEAL_RE = re.compile(r"SendCardsService set roundID:([\d,]+) k : "
                     r"\{.*?\"Trump\":(\d+)")
#: 结算行（3.2.2 之后级别的主要来源）。只关心 UpgradeInfo。
SETTLE_RE = re.compile(r"EVA1B001结算协议 = (\{.*)")


class LevelWatcher:
    """跟读日志最新那几个文件，取出最近一次的级别。

    只看每个文件的尾部：日志动辄几 MB，全扫一遍没必要。
    """

    #: 稳态跟读时每次回看这么多字节（够接上上一次的位置）
    TAIL = 300_000
    #: **首次**读往前看这么多字节。别只按 TAIL 看尾部 —— 实测踩过：
    #: 今天那局日志 7 分钟长了 2.4 MB，06:51 的结算行（级别的唯一来源）
    #: 根本不在最后 300 KB 里，级别读成 None，于是「一局 20 个决策点全跳过」。
    FIRST_TAIL = 8_000_000

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        self.level = None
        #: 这个级别是从哪读来的（"发牌行（本局）" / "结算行（下一局）"）——
        #: 影子日志会记下它，离线上分得清「这条记录的级别可不可信」。
        self.level_src = ""
        #: 最近一次结算行里每座位的级别（两队各一个）
        self.last_levels = None
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
                # 首次往前多看一截（见 FIRST_TAIL 的注释）
                start = max(0, size - self.FIRST_TAIL)
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
            # 按行序处理：两条都在时**后出现的那条**算 —— 结算行总是在发牌行之后
            for line in text.splitlines():
                m = DEAL_RE.search(line)
                if m:
                    found, src = int(m.group(2)), "发牌行（本局）"
                    continue
                j = SETTLE_RE.search(line)
                if j:
                    got = self._from_settle(j.group(1))
                    if got is not None:
                        found, src = got, "结算行（下一局）"
        if found is not None and found != self.level:
            self.level, self.level_src = found, src
            self.last_seen = time.time()
            return found
        return None

    def _from_settle(self, payload: str):
        """从结算 JSON 里取「下一局要打的级别」；取不到返回 None。

        语义见模块 docstring —— `TrumpValue` 是**下一局**的级别，不是本局的。
        """
        try:
            info = json.loads(payload).get("UpgradeInfo") or {}
        except ValueError:
            return None
        v = info.get("TrumpValue")
        lv = info.get("trump")
        if isinstance(lv, list) and len(lv) == 4:
            self.last_levels = [int(x) for x in lv]
        return int(v) if isinstance(v, int) and 1 <= v <= 13 else None


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
