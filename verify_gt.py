"""独立核验真值。

手牌的左边缘精确落在 x0 + 95k 上，卡片左边缘在图像里是一条强竖直边缘
（实测：有牌的层梯度 190~300，没牌的层 0~55，阈值取 120 分得很干净）。

这个方法不依赖任何列检测逻辑，所以能独立验证真值有没有漏牌。

用法: python verify_gt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))

import cv2
import numpy as np

from eval_real import GROUND_TRUTH

# 手牌层级：最底行顶边 714，每层往上 62px。
# 最多 8 层 —— 两副牌同一点数有 8 张（8 炸），级联就要摞 8 层。
# 之前硬编码成 7 层漏掉了最上面那层，所以这里直接算出来，避免再漏。
STEP_Y, Y_BOTTOM, MAX_LEVELS = 62, 714, 8
LEVELS = [Y_BOTTOM - STEP_Y * i for i in range(MAX_LEVELS - 1, -1, -1)]
PITCH, EDGE_TH = 95, 120


def detect(img, x0, ncols):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    found: set = set()
    for k in range(ncols):
        x = x0 + PITCH * k
        if x + 132 > img.shape[1] - 20:
            break
        edges = set()
        for L in LEVELS:
            if L + 50 > img.shape[0]:
                continue
            # 真值里的 x0 来自「白像素左边界」，和真实卡边可能差几个像素，
            # 所以在 x 附近 ±6px 搜索竖直边缘
            best = max(gx[L + 10:L + 45, x + dx - 3:x + dx + 4].mean()
                       for dx in range(-6, 7))
            if best > EDGE_TH:
                edges.add(L)
        # 手牌分组是从最底层(713)连续往上摞的，中断处以上不是手牌
        # （否则按钮、出牌区这些也会被算进来）
        for L in reversed(LEVELS):
            if L not in edges:
                break
            found.add((k, L))
    return found


def unmatched(a, b):
    return sorted(x for x in a
                  if not any(x[0] == y[0] and abs(x[1] - y[1]) <= 3 for y in b))


bad = 0
for shot, truth in GROUND_TRUTH.items():
    img = cv2.imread(f"shots/{shot}.png")
    x0 = min(cx for _c, cx, _y in truth)
    ncols = max(c for _c, c, _y in truth) + 1
    det = detect(img, x0, ncols)
    gt = {((cx - x0) // PITCH, y) for _cls, cx, y in truth}

    only_det = unmatched(det, gt)
    only_gt = unmatched(gt, det)
    ok = not only_det and not only_gt
    bad += not ok
    print(f"{shot:<7} 真值 {len(gt):>2}  边缘检测 {len(det):>2}   "
          f"{'一致' if ok else '** 不一致 **'}")
    if only_det:
        print(f"    边缘检测到但真值没有: {only_det}")
    if only_gt:
        print(f"    真值有但边缘检测不到: {only_gt}")

print()
print("结论:", "5 张截图的真值全部通过独立核验" if bad == 0
      else f"{bad} 张不一致，需复查")
