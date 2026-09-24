"""判断现在轮到谁出牌。

「轮到我」的判据：画面中部出现橙色「出牌」按钮。
实测 181 帧里只有 14 帧出现（橙色占比 0.36~0.39，其余接近 0）—— 非常干净。
游戏只在轮到你的时候才显示 不出/提示/出牌 这三个按钮。

用法:
    python live/turn.py live/frames
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

# 「出牌」按钮所在的区域（相对画面尺寸的百分比）
BTN_ROI = (0.60, 0.80, 0.36, 0.48)     # x1, x2, y1, y2
ORANGE_RATIO = 0.20                    # 橙色占比超过这个值算"按钮在"

# 倒计时图标可能出现的四个区域（谁出牌就出现在谁那边）
COUNTDOWN_ZONES = {
    "mine":   (0.34, 0.66, 0.34, 0.54),
    "left":   (0.08, 0.28, 0.28, 0.58),
    "top":    (0.34, 0.66, 0.14, 0.40),
    "right":  (0.72, 0.92, 0.28, 0.58),
}


def _orange_ratio(img: np.ndarray, roi: tuple) -> float:
    x1, x2, y1, y2 = roi
    h, w = img.shape[:2]
    sub = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    if sub.size == 0:
        return 0.0
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (hh >= 10) & (hh <= 25) & (ss > 140) & (vv > 190)
    return float(m.mean())


def is_my_turn(img: np.ndarray) -> tuple[bool, float]:
    """轮到我 -> (True, 橙色占比)。"""
    r = _orange_ratio(img, BTN_ROI)
    return r >= ORANGE_RATIO, r


def countdown_side(img: np.ndarray) -> tuple[str | None, float]:
    """靠倒计时的位置判断轮谁。返回 (位置, 得分)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 倒计时是个金黄色圆形图标
    gold = ((hh >= 15) & (hh <= 32) & (ss > 120) & (vv > 180))
    h, w = img.shape[:2]
    best, bs = None, 0.0
    for side, (x1, x2, y1, y2) in COUNTDOWN_ZONES.items():
        sub = gold[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        if sub.size == 0:
            continue
        s = float(sub.mean())
        if s > bs:
            bs, best = s, side
    return best, bs


def main() -> None:
    files = []
    for a in sys.argv[1:] or ["live/frames"]:
        p = Path(a)
        files += sorted(p.glob("*.png")) if p.is_dir() else [p]
    hit = 0
    for f in files[:20]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        mine, r = is_my_turn(img)
        side, s = countdown_side(img)
        hit += mine
        print(f"  {f.name:<22} 轮到我={mine}  (橙色 {r:.2f})   倒计时位置={side} ({s:.3f})")
    print(f"\n前 {min(20, len(files))} 帧里 {hit} 帧轮到我")


if __name__ == "__main__":
    main()
