"""用独立方法核对模型在截图上的检出：边缘检测枚举卡片位置，与模型检出对照。

能发现两类问题：模型漏检 / 模型重复检出。

用法: python check_shots.py img_6 img_7
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))

import cv2
import numpy as np

from layout import CLASSES
from predict_cards import dedup, split_hand_table

# 手牌层级：最底行顶边 714，每层往上 62px。
# 最多 8 层 —— 两副牌同一点数有 8 张（8 炸），级联就要摞 8 层。
# 之前硬编码成 7 层漏掉了最上面那层，所以这里直接算出来，避免再漏。
STEP_Y, Y_BOTTOM, MAX_LEVELS = 62, 714, 8
LEVELS = [Y_BOTTOM - STEP_Y * i for i in range(MAX_LEVELS - 1, -1, -1)]
PITCH, EDGE_TH = 95, 120


def main(shots):
    from ultralytics import YOLO

    m = YOLO("runs/detect/guandan4/weights/best.pt")
    m.model.names = {i: c for i, c in enumerate(CLASSES)}
    names = m.names

    for shot in shots:
        img = cv2.imread(f"shots/{shot}.png")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))

        res = m.predict(f"shots/{shot}.png", imgsz=960, conf=0.25, verbose=False)[0]
        dets = dedup([(names[int(b.cls)], float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                       float((b.xyxy[0][1] + b.xyxy[0][3]) / 2), float(b.conf))
                      for b in res.boxes])
        hand, table = split_hand_table(dets)

        # 用模型检出反推精确的栅格起点：找一个 x0 让所有卡左边缘都尽量接近 95 的倍数
        base = [d[1] - 54 for d in hand]
        b0 = min(base)
        x0 = min(((sum(abs((b - c) / 95 - round((b - c) / 95)) for b in base), c)
                  for c in np.arange(b0 - 6, b0 + 6, 0.5)))[1]

        edge_set = set()
        for k in range(14):
            x = int(round(x0)) + PITCH * k
            if x + 132 > img.shape[1] - 20:
                break
            edges = set()
            for L in LEVELS:
                if L + 60 > img.shape[0]:
                    continue
                # 卡片的左边缘是贯穿整条可见带（62px）的**长**竖直边；
                # 花色图案的边也能落在栅格线附近，但只有几十像素高。
                # 所以除了强度，还要看这条边在竖直方向覆盖了多少行。
                hit = 0
                total = 0
                for dy in range(4, 58):
                    y = L + dy
                    cols = gx[y, x - 3:x + 4]
                    total += 1
                    if cols.max() > EDGE_TH:
                        hit += 1
                if total and hit / total > 0.80:
                    edges.add(L)
            for L in reversed(LEVELS):       # 从底部连续往上
                if L not in edges:
                    break
                edge_set.add((k, L))

        det_set = set()
        for d in hand:
            k = int(round((d[1] - 54 - x0) / PITCH))
            L = min(LEVELS, key=lambda v: abs(v - (d[2] - 34)))
            det_set.add((k, L))

        print(f"{shot}:  模型 {len(hand)} 张   边缘检测 {len(edge_set)} 张   "
              f"桌面 {len(table)} 张")
        od, oe = sorted(det_set - edge_set), sorted(edge_set - det_set)
        if od:
            print(f"   模型有、边缘没有（疑似重复检出）: {od}")
        if oe:
            print(f"   边缘有、模型没有（疑似漏检）    : {oe}")
        if not od and not oe:
            print("   位置完全一致")
        low = [(d[0], round(d[3], 2)) for d in hand if d[3] < 0.7]
        if low:
            print(f"   低置信度: {low}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["img_6", "img_7"])
