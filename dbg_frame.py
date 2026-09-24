import sys
sys.path.insert(0,'.'); sys.path.insert(0,'synth')
from layout import CLASSES
from ultralytics import YOLO
from predict_cards import dedup, split_hand_table, display
m = YOLO('runs/detect/guandan6/weights/best.pt'); m.model.names={i:c for i,c in enumerate(CLASSES)}
import cv2
for f in sys.argv[1:]:
    img = cv2.imread(f)
    r = m.predict(img, imgsz=960, conf=0.30, verbose=False)[0]
    dets = [(m.names[int(b.cls)], float((b.xyxy[0][0]+b.xyxy[0][2])/2), float((b.xyxy[0][1]+b.xyxy[0][3])/2), float(b.conf)) for b in r.boxes]
    d = dedup(dets)
    hand, table = split_hand_table(d)
    print(f"\n=== {f}   原始框 {len(dets)}  去重后 {len(d)}  手牌 {len(hand)}  桌面 {len(table)}")
    print("  -- 判为桌面 --")
    for x in sorted(table, key=lambda t:-t[3]):
        print(f"     {display(x[0]):<5} {x[3]:.2f}  @({x[1]:.0f},{x[2]:.0f})")
    print("  -- 判为手牌 --")
    for x in sorted(hand, key=lambda t:-t[3]):
        print(f"     {display(x[0]):<5} {x[3]:.2f}  @({x[1]:.0f},{x[2]:.0f})")
