"""模型报的『桌面牌』里，有多少真的落在真实出牌组的框里？"""
import json, sys, cv2, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'synth')
from layout import CLASSES
from ultralytics import YOLO
from predict_cards import dedup, split_hand_table

rows = {r["frame"]: r for r in json.load(open('played_scan.json'))}
m = YOLO('runs/detect/guandan6/weights/best.pt')
m.model.names = {i: c for i, c in enumerate(CLASSES)}

TP = FP = 0
fn_boxes = fn_tot = 0
frames_with_det = 0
for f, r in sorted(rows.items()):
    img = cv2.imread(f)
    if img is None:
        continue
    boxes = [g[:4] for v in r["zones"].values() for g in v if g[3] >= 125]
    res = m.predict(img, imgsz=960, conf=0.30, verbose=False)[0]
    d = dedup([(m.names[int(b.cls)], float((b.xyxy[0][0]+b.xyxy[0][2])/2),
                float((b.xyxy[0][1]+b.xyxy[0][3])/2), float(b.conf)) for b in res.boxes])
    _, table = split_hand_table(d)
    if table:
        frames_with_det += 1
    hits = set()
    for t in table:
        x, y = t[1], t[2]
        ok = False
        for i, (bx, by, bw, bh) in enumerate(boxes):
            if bx - 6 <= x <= bx + bw + 6 and by - 6 <= y <= by + bh + 6:
                ok = True; hits.add(i)
        TP += ok; FP += (not ok)
    # 有真牌但一张没被覆盖的组 = 完全漏掉的组
    fn_boxes += len([i for i in range(len(boxes)) if i not in hits])
print(f"模型报出的桌面牌: 命中真实出牌区 {TP}   不在任何真实出牌区(误报) {FP}")
if TP + FP:
    print(f"  -> 精确率 {TP/(TP+FP):.1%}")
print(f"真实出牌组 {sum(len([g for g in v if g[3]>=125]) for r in rows.values() for v in r['zones'].values())} 个, "
      f"其中一张都没被模型碰到的 {fn_boxes} 个")
print(f"模型有桌面检出的帧 {frames_with_det}/{len(rows)} = {frames_with_det/len(rows):.1%}")
