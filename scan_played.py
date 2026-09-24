"""扫描全部真实帧：出牌区的客观统计（只看四个方位窗口内）。"""
from __future__ import annotations
import glob, json
import cv2, numpy as np

ZONES = {  # 名称: (中心x, 中心y, 半宽, 半高)
    "队友":   (843, 295, 230, 105),
    "机器人1": (280, 400, 200, 105),
    "机器人3": (1420, 400, 200, 105),
    "我":     (845, 480, 230, 105),
}

def card_blobs(img, cx, cy, hw, hh):
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((s < 55) & (v > 145)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (85 <= h <= 155 and 25 <= w <= 700):
            continue
        if a < 0.45 * w * h:
            continue
        out.append((int(x1 + x), int(y1 + y), int(w), int(h)))
    return out

rows = []
for f in sorted(glob.glob('live/frames/*.png')):
    img = cv2.imread(f)
    if img is None:
        continue
    z = {k: card_blobs(img, *v) for k, v in ZONES.items()}
    rows.append({"frame": f, "zones": {k: v for k, v in z.items()},
                 "n": sum(len(v) for v in z.values())})

have = [r for r in rows if r["n"]]
print(f"总帧 {len(rows)}   至少一个方位有出牌的帧 {len(have)}  ({len(have)/len(rows):.1%})")
print(f"出牌组总数 {sum(r['n'] for r in rows)}")

hs = np.array([g[3] for r in rows for v in r["zones"].values() for g in v])
ws = np.array([g[2] for r in rows for v in r["zones"].values() for g in v])
print(f"\n卡高 h: 中位 {np.median(hs):.0f}  5% {np.percentile(hs,5):.0f}  95% {np.percentile(hs,95):.0f}  范围 {hs.min()}~{hs.max()}")
print(f"缩放 h/174: 中位 {np.median(hs)/174:.3f}  范围 {hs.min()/174:.3f}~{hs.max()/174:.3f}")
hist, edges = np.histogram(hs, bins=np.arange(85, 160, 10))
for c, e in zip(hist, edges):
    print(f"  h {e:>4.0f}-{e+10:<4.0f} {'#'*int(c/6):<44} {c}")

print("\n每个方位出现在多少帧里:")
for k in ZONES:
    nn = sum(1 for r in rows if r["zones"][k])
    print(f"  {k:<7} {nn:>3} 帧 ({nn/len(rows):.1%})")

# 每帧的出牌组数分布
from collections import Counter
cnt = Counter(r["n"] for r in rows)
print("\n每帧出牌组数分布:", dict(sorted(cnt.items())))
json.dump(rows, open('played_scan.json', 'w'))
