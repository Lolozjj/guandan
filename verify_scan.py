"""交叉验证：blob 扫描说有出牌、但模型报 0 张的帧，裁出来看。"""
import json, sys, cv2, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'synth')
from layout import CLASSES
from ultralytics import YOLO
from predict_cards import dedup, split_hand_table

rows = json.load(open('played_scan.json'))
m = YOLO('runs/detect/guandan6/weights/best.pt')
m.model.names = {i: c for i, c in enumerate(CLASSES)}

ZONES = {"队友": (843, 295), "机器人1": (280, 400),
         "机器人3": (1420, 400), "我": (845, 480)}

cands = []
for r in rows:
    real = {k: [g for g in v if g[3] >= 125] for k, v in r["zones"].items()}
    if sum(len(v) for v in real.values()) == 0:
        continue
    img = cv2.imread(r["frame"])
    if img is None:
        continue
    res = m.predict(img, imgsz=960, conf=0.30, verbose=False)[0]
    d = dedup([(m.names[int(b.cls)], float((b.xyxy[0][0]+b.xyxy[0][2])/2),
                float((b.xyxy[0][1]+b.xyxy[0][3])/2), float(b.conf)) for b in res.boxes])
    _, table = split_hand_table(d)
    if len(table) == 0:
        cands.append((r["frame"], real))

print(f"『扫描有牌 且 模型报 0』的帧: {len(cands)}")
tiles = []
for f, real in cands[::max(1, len(cands)//6)][:6]:
    img = cv2.imread(f)
    crop = img[195:580, 60:1630].copy()
    for k, v in real.items():
        cx, cy = ZONES[k]
        for (x, y, w, h) in v:
            cv2.rectangle(crop, (x-60, y-195), (x+w-60, y+h-195), (0, 0, 255), 2)
        cv2.putText(crop, k, (cx-60-40, cy-195-110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    s = cv2.resize(crop, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
    bar = np.full((24, s.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{f.split(chr(92))[-1]}  实际有出牌(红框)  模型报 0 张",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    tiles.append(np.vstack([bar, s]))
if tiles:
    cv2.imwrite('audit/miss_check.png', np.vstack(tiles))
    print("saved audit/miss_check.png")
