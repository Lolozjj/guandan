"""判断当前画面处于哪个阶段 —— 用来屏蔽换局/结算时的误读。

**为什么必须做**：换局和结算画面上也有牌形的东西，但它们不是「出牌」。
实测不做屏蔽时面板会：

    结算/升段画面   凭空报出「我: ♣5」
    进贡画面        把那张贡牌当成「机器人1 出了 ♠8」

三种非正常阶段，各有一个实测出来的判据（阈值都在 337 帧真实画面上验过）：

    横幅（等待还贡 / 玩家X接风）  画面中部出现深蓝色长条，占比 > 5%
    结算 / 升段 / 胜利            桌面区整体变亮，亮度中位数 > 200（正常约 145）
    进贡 / 还贡                   出现红色「贡」小标记，模板匹配 NCC > 0.85

三者的分离度（不做的对照见上面注释）：
    横幅      换局帧 0.17~0.18   vs 正常帧 0.000~0.0003
    结算      换局帧 238~239     vs 正常帧 145~148
    进贡      换局帧 0.935~1.000  vs 其余帧 <= 0.695
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BADGE_PATH = ROOT / "assets" / "gong_badge.png"
REVEAL_PATH = ROOT / "assets" / "reveal_text.png"

# 横幅：画面中部那条深蓝长条所在的搜索范围
BANNER_ROI = (300, 200, 1400, 460)      # x1, y1, x2, y2
BANNER_RATIO = 0.05                     # 实测换局 0.17+ / 正常 0.0003-

# 结算：桌面区整体变亮
RESULT_ROI = (200, 200, 1500, 600)
RESULT_MEDIAN = 200                     # 实测结算 238 / 正常 145

# 进贡：「贡」小标记
BADGE_THR = 0.85                        # 实测 0.935~1.0 / 其余 <=0.695

# 亮手牌：「队友手牌 / 对手手牌」的标题
REVEAL_BAND = (760, 900)        # y 范围（画面宽 1698 时）
REVEAL_THR = 0.75               # 实测正例 0.885~1.0 / 正常帧 0.315~0.351

_badge: np.ndarray | None = None
_reveal: np.ndarray | None = None


def _get_badge() -> np.ndarray | None:
    global _badge
    if _badge is None:
        img = cv2.imread(str(BADGE_PATH))
        _badge = img if img is not None else np.zeros((1, 1, 3), np.uint8)
    return _badge if _badge.size > 3 else None


def _get_reveal() -> np.ndarray | None:
    """亮手牌的标题模板 —— 只取右半边的「手牌」两个字，兼容队友/对手。"""
    global _reveal
    if _reveal is None:
        img = cv2.imread(str(REVEAL_PATH))
        _reveal = img[:, img.shape[1] // 2:] if img is not None \
            else np.zeros((1, 1, 3), np.uint8)
    return _reveal if _reveal.size > 3 else None


def detect_phase(img: np.ndarray) -> tuple[str, str]:
    """返回 (阶段, 说明)。阶段是 normal / banner / result / tribute 之一。

    normal 之外的都不该拿模型结果去显示 —— 那时候的牌不是「出牌」。
    """
    if img is None or img.size == 0:
        return "normal", ""

    # 1) 结算 / 升段：整片变亮
    x1, y1, x2, y2 = RESULT_ROI
    roi = img[y1:y2, x1:x2]
    if roi.size:
        med = float(np.median(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]))
        if med > RESULT_MEDIAN:
            return "result", f"结算/升段（亮度 {med:.0f}）"

    # 2) 系统横幅：画面中部的深蓝色长条
    x1, y1, x2, y2 = BANNER_ROI
    roi = img[y1:y2, x1:x2]
    if roi.size:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
        bar = ((h >= 95) & (h <= 135) & (s > 60) & (v > 40) & (v < 120))
        if float(bar.mean()) > BANNER_RATIO:
            return "banner", "换局横幅（还贡/接风）"

    # 3) 亮手牌：队友/对手剩下的牌被亮出来（一局结束后）
    tpl = _get_reveal()
    if tpl is not None:
        sc = img.shape[1] / 1698.0
        y1, y2 = int(REVEAL_BAND[0] * sc), int(REVEAL_BAND[1] * sc)
        band = img[y1:y2, :]
        if band.shape[0] >= tpl.shape[0] and band.shape[1] >= tpl.shape[1]:
            r = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
            if float(r.max()) > REVEAL_THR:
                return "reveal", "亮手牌（队友/对手剩下的牌）"

    # 4) 进贡：红色「贡」小标记
    tpl = _get_badge()
    if tpl is not None and img.shape[0] > tpl.shape[0] and img.shape[1] > tpl.shape[1]:
        r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        if float(r.max()) > BADGE_THR:
            return "tribute", "进贡/还贡"

    return "normal", ""
