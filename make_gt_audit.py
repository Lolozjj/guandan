"""手工核对出牌真值：把模板匹配的结果画在真实画面上，供逐帧核对。

模板匹配在真实画面上 NCC 0.95~0.99（已抽样验过），拿它当「建议」比从零读图快得多；
核对者只需要判断「框对不对、有没有漏、有没有多」。
注意：模板匹配认不出王（王牌面还原度差），所以王必须人工补。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'synth'))
import layout as L                                       # noqa: E402
from match_played import build_templates, detect, MARK_W, MARK_H  # noqa: E402

BAND = (60, 150, 1630, 640)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=17)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.88)
    ap.add_argument("--out", default="audit")
    args = ap.parse_args()

    tpl = build_templates()
    files = sorted((ROOT / 'live' / 'frames').glob('*.png'))[::args.every]
    x1, y1, x2, y2 = BAND
    tiles = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        hits = detect(img, tpl, thr=args.thr)
        c = img[y1:y2, x1:x2].copy()
        # 画四个区的位置参考线
        for k, z in L.PLAYED_ZONES.items():
            cv2.line(c, (0, z['y'] - y1), (c.shape[1], z['y'] - y1),
                     (255, 255, 255), 1)
            cv2.putText(c, k, (4, z['y'] - y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        for cls, lv, sc, x, y in hits:
            col = (0, 0, 255)
            cv2.rectangle(c, (x - x1, y - y1), (x - x1 + MARK_W, y - y1 + MARK_H), col, 2)
            cv2.putText(c, cls if lv is None else f"{cls}", (x - x1, y - y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
        bar = np.full((24, c.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{f.name}   模板建议 {len(hits)} 张",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        tiles.append(np.vstack([bar, c]))

    for i in range(0, len(tiles), args.per):
        page = tiles[i:i + args.per]
        cv2.imwrite(str(ROOT / args.out / f"gt_{i // args.per:02d}.png"),
                    np.vstack(page))
    print(f"共 {len(tiles)} 帧，{len(range(0, len(tiles), args.per))} 张核对图")


if __name__ == "__main__":
    main()
