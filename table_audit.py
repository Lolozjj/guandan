"""给出牌区检出率定基准：把桌面区域裁出来拼成图，人工核对。

问题：『能检出桌面牌的帧占比』这个指标的分母里混着『本来就没出牌』的帧
（刚开局、别家思考中）。要判断模型好不好，必须先知道每帧实际有几张出牌。

用法:
    python table_audit.py --every 10 --out audit/
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
from layout import CLASSES  # noqa: E402
from predict_cards import dedup, split_hand_table  # noqa: E402

# 桌面出牌区（四个方位的中心实测值）张开后的外接框
BAND = (60, 195, 1630, 580)   # x1, y1, x2, y2
SCALE = 0.62


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/guandan6/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--per", type=int, default=8, help="每张拼图放几帧")
    ap.add_argument("--out", default="audit")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.model.names = {i: c for i, c in enumerate(CLASSES)}

    files = sorted(glob.glob("live/frames/*.png"))[::args.every]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = BAND
    tiles = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2), float(b.conf))
                for b in r.boxes]
        # 原始框（用于画图）
        raw = [(model.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf))
               for b in r.boxes]
        hand, table = split_hand_table(dedup(dets))
        crop = img[y1:y2, x1:x2].copy()
        # 画所有框：桌面=红，手牌=蓝灰
        th = {(round(d[1]), round(d[2])) for d in table}
        for cls, box, conf in raw:
            bx1, by1, bx2, by2 = box
            is_table = any(abs((bx1 + bx2) / 2 - t[0]) < 6 and
                           abs((by1 + by2) / 2 - t[1]) < 6 for t in th)
            if is_table:
                cv2.rectangle(crop, (int(bx1 - x1), int(by1 - y1)),
                              (int(bx2 - x1), int(by2 - y1)), (0, 0, 255), 2)
                cv2.putText(crop, cls, (int(bx1 - x1), int(by1 - y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        small = cv2.resize(crop, None, fx=SCALE, fy=SCALE,
                           interpolation=cv2.INTER_AREA)
        bar = np.full((26, small.shape[1], 3), 255, np.uint8)
        txt = f"{Path(f).name}   模型判为桌面 {len(table)} 张"
        cv2.putText(bar, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1)
        tiles.append(np.vstack([bar, small]))

    for i in range(0, len(tiles), args.per):
        page = tiles[i:i + args.per]
        h = max(t.shape[0] for t in page)
        page = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 200, np.uint8)])
                for t in page] if any(t.shape[0] != h for t in page) else page
        sheet = np.vstack(page)
        p = out / f"audit_{i // args.per:02d}.png"
        cv2.imwrite(str(p), sheet)
        print(f"{p}   {sheet.shape[1]}x{sheet.shape[0]}  ({len(page)} 帧)")


if __name__ == "__main__":
    main()
