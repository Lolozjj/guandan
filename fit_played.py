"""从真实画面量『出牌区牌面』的排版：卡框 + 各字形的相对位置和缩放。

出牌区的牌是白底圆角卡，比手牌小、而且重度重叠。这里找**最右那张完全可见的卡**，
量出它的卡框，再在卡内做连通域，得到点数 / 小花色 / 大花色的精确摆位。
"""
from __future__ import annotations
import sys, cv2, numpy as np

def card_bbox(img, x1, y1, x2, y2, min_w=70, min_h=110):
    """在窗口内找最右那张亮色圆角卡的外接框。"""
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    mask = ((s < 60) & (v > 150)).astype(np.uint8)     # 近白 = 牌面
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w < min_w or h < min_h:
            continue
        if best is None or (x + w) > (best[0] + best[2]):   # 最右
            best = (x, y, w, h, a)
    if best is None:
        return None
    x, y, w, h, _ = best
    return (x1 + x, y1 + y, w, h)

def glyphs(img, bx, by, w, h):
    """卡内非白像素连通域（相对卡左上）。"""
    c = img[by:by + h, bx:bx + w]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((s > 60) | (v < 130)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, a = stats[i]
        if a < 60 or cw < 8 or ch < 8:
            continue
        if cw > w * 0.9 and ch < 12:      # 卡的上/下边缘残留
            continue
        out.append((int(x), int(y), int(cw), int(ch), int(a)))
    out.sort(key=lambda t: (t[1], t[0]))
    return out

CASES = [
    ("live/frames/f00028_0021.png", (600, 300, 1050, 520), "f00028 右侧 8♠"),
    ("live/frames/f00047_0031.png", (600, 170, 1100, 400), "f00047 3♠"),
    ("live/frames/f00069_0041.png", (1100, 330, 1560, 560), "f00069 右侧 2♣"),
    ("live/frames/f00087_0051.png", (150, 250, 620, 480), "f00087 左侧 K♣"),
]
for path, win, name in CASES:
    img = cv2.imread(path)
    if img is None:
        print(name, "读不到"); continue
    bb = card_bbox(img, *win)
    if bb is None:
        print(f"\n=== {name}: 没找到卡框"); continue
    bx, by, w, h = bb
    print(f"\n=== {name}   卡框 x={bx} y={by}  {w}x{h}   (手牌 132x174, 比例 {w/132:.3f})")
    for g in glyphs(img, bx, by, w, h):
        print(f"     字形 相对卡左上 ({g[0]:>3},{g[1]:>3})  {g[2]:>3}x{g[3]:<3} 面积 {g[4]}")
