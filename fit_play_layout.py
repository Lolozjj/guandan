"""拟合『出牌区』牌面排版（字形掩码 IoU 版）。

出牌卡 = 手牌 132x174 缩 0.803 -> 106x140，间距 50。
靶子取 3 帧里最右那张完全可见的牌，只用「字形掩码」比对，避免被卡底色差异干扰。
"""
from __future__ import annotations
import sys, cv2, numpy as np
sys.path.insert(0, 'synth')
from compose import sprite, card_background, _paste

CW, CH = 106, 140
REF = 132.0
CASES = [   # (帧, 卡左上x, 卡左上y, 类别)
    ("live/frames/f00028_0021.png", 893, 355, "S8"),
    ("live/frames/f00047_0031.png", 843, 199, "S3"),
    ("live/frames/f00087_0051.png", 404, 312, "C13"),
]
HUASE = {"S": "huase_1", "H": "huase_2", "C": "huase_3", "D": "huase_4"}
COLOR = {"S": "hei", "H": "hong", "C": "hei", "D": "hong"}

def target_mask(path, x, y):
    c = cv2.imread(path)[y:y + CH, x:x + CW]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((s > 70) | (v < 140)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m[:4, :] = 0; m[-4:, :] = 0; m[:, :4] = 0; m[:, -4:] = 0   # 去掉卡边缘
    return m

def synth_mask(cls, params):
    (rx, ry, rs), (sx, sy, ss), (bx, by, bs) = params
    suit, rank = cls[0], cls[1:]
    n = {"A": 1, "T": 10, "J": 11, "Q": 12, "K": 13}.get(rank, None)
    n = int(rank) if n is None else n
    m = np.zeros((CH, CW), np.uint8)
    for name, (x, y), sc in [
            (f"LargeCard_commom_shuzi_{COLOR[suit]}_{n}", (rx, ry), rs),
            (f"LargeCard_{HUASE[suit]}", (sx, sy), ss),
            (f"LargeCard_{HUASE[suit]}", (bx, by), bs)]:
        sp = sprite(name)
        sp = cv2.resize(sp, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(x)), int(round(y))
        h, w = sp.shape[:2]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(CW, x0 + w), min(CH, y0 + h)
        if xb <= xa or yb <= ya:
            continue
        a = sp[ya - y0:yb - y0, xa - x0:xb - x0, 3]
        m[ya:yb, xa:xb] = np.maximum(m[ya:yb, xa:xb], (a > 100).astype(np.uint8) * 255)
    m[:4, :] = 0; m[-4:, :] = 0; m[:, :4] = 0; m[:, -4:] = 0
    return m

TARGETS = [(cls, target_mask(p, x, y)) for p, x, y, cls in CASES]

def loss(params):
    tot = 0.0
    for cls, t in TARGETS:
        m = synth_mask(cls, params)
        inter = int(np.logical_and(t > 0, m > 0).sum())
        union = int(np.logical_or(t > 0, m > 0).sum())
        tot += 1.0 - inter / max(union, 1)
    return tot / len(TARGETS)

params = [(9.0, 6.0, 0.80), (6.0, 52.0, 0.95), (50.0, 78.0, 1.70)]
steps = [4.0, 4.0, 0.10]
best = loss(params)
print(f"初值 IoU-loss = {best:.4f}")
for it in range(200):
    improved = False
    for pi in range(3):
        for k in range(3):
            for sgn in (+1, -1):
                t = [list(p) for p in params]
                t[pi][k] += sgn * steps[k]
                t = [tuple(v) for v in t]
                l = loss(t)
                if l < best - 1e-5:
                    best, params, improved = l, t, True
    if not improved:
        steps = [s / 2 for s in steps]
        if max(steps) < 0.02:
            break
print(f"拟合后 IoU-loss = {best:.4f}   (IoU = {1-best:.4f})")
(rx, ry, rs), (sx, sy, ss), (bx, by, bs) = params
k = REF / CW
print(f"\n出牌卡坐标 (106x140):")
print(f"  PLAY_RANK       ({rx:.1f}, {ry:.1f})   x{rs:.3f}")
print(f"  PLAY_SUIT_SMALL ({sx:.1f}, {sy:.1f})   x{ss:.3f}")
print(f"  PLAY_SUIT_BIG   ({bx:.1f}, {by:.1f})   x{bs:.3f}")
print(f"\n归一化到 132x174 (x{k:.4f}):")
print(f"  PLAY_RANK       ({rx*k:.1f}, {ry*k:.1f})   x{rs*k:.3f}")
print(f"  PLAY_SUIT_SMALL ({sx*k:.1f}, {sy*k:.1f})   x{ss*k:.3f}")
print(f"  PLAY_SUIT_BIG   ({bx*k:.1f}, {by*k:.1f})   x{bs*k:.3f}")

# 逐帧看 IoU，并出对比图
tiles = []
for cls, t in TARGETS:
    m = synth_mask(cls, params)
    iou = np.logical_and(t > 0, m > 0).sum() / max(np.logical_or(t > 0, m > 0).sum(), 1)
    print(f"  {cls}: IoU {iou:.3f}")
    tiles.append(np.hstack([255 - t, np.full((CH, 8), 128, np.uint8), 255 - m]))
sep = np.full((10, tiles[0].shape[1], 1), 128, np.uint8)
cv2.imwrite('audit/playfit_mask.png', cv2.resize(np.vstack(
    [np.vstack([t, sep]) for t in tiles]), None, fx=2.0, fy=2.0,
    interpolation=cv2.INTER_NEAREST))
print("掩码对比 audit/playfit_mask.png")
