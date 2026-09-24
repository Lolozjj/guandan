"""旋转鲁棒性测试：验证模型对旋转后的牌面是否还能识别。

背景: shrimantasatpati/yolov11_playing_cards_detection 的训练配置里 degrees=0.0，
      即完全没做旋转增强。而掼蛋四家手牌朝向不同（对手的牌转 90/180 度）。
      这个脚本把同一张图转 0/90/180/270 度分别推理，看精度是否崩掉。

用法:
    python test_rotation.py <权重.pt> <图片>
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2


def run(model, names, img, imgsz, conf):
    res = model.predict(source=img, imgsz=imgsz, conf=conf, verbose=False)[0]
    cls = res.boxes.cls.tolist()
    confs = res.boxes.conf.tolist()
    labels = [names[int(c)] for c in cls]
    return Counter(labels), sum(confs) / len(confs) if confs else 0.0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="runs/rotation_test")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    names = model.names
    src = cv2.imread(args.image)
    if src is None:
        raise SystemExit(f"[FAIL] 读不到图片: {args.image}")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"模型: {args.model}")
    print(f"图片: {args.image}  {src.shape[1]}x{src.shape[0]}")
    print(f"imgsz={args.imgsz}  conf={args.conf}\n")

    results = {}
    for deg in (0, 90, 180, 270):
        if deg == 0:
            img = src.copy()
        elif deg == 90:
            img = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        elif deg == 180:
            img = cv2.rotate(src, cv2.ROTATE_180)
        else:
            img = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

        counts, mean_conf, res = run(model, names, img, args.imgsz, args.conf)
        results[deg] = (counts, mean_conf)
        res.save(filename=Path(args.out) / f"rot{deg:03d}.jpg")

        total = sum(counts.values())
        print(f"旋转 {deg:>3}°: {total:>3} 个框, 平均置信度 {mean_conf:.3f}")
        if counts:
            print(f"          {dict(counts)}")

    # --- 结论 ---
    base_counts, base_conf = results[0]
    base_total = sum(base_counts.values())
    print(f"\n{'=' * 62}")
    if base_total == 0:
        print("基准（0°）就检不出东西，无法比较。换张更清晰的图。")
        return

    for deg in (90, 180, 270):
        counts, conf = results[deg]
        total = sum(counts.values())
        ratio = total / base_total if base_total else 0
        print(f"{deg:>3}°: 检出 {total}/{base_total} 个 ({ratio:.0%}), "
              f"置信度 {conf:.3f} vs 基准 {base_conf:.3f} ({conf / base_conf:.0%})")

    # 判断损坏程度
    worst = min(sum(results[d][0].values()) / base_total for d in (90, 180, 270))
    print()
    if worst > 0.85:
        print("[OK] 旋转基本不影响 -> 模型自带了旋转不变性，微调时 degrees 可以不开")
    elif worst > 0.4:
        print("[WARN] 旋转后明显掉点 -> 微调时务必开 degrees=90（YOLO 会取 90/180/270 三个角度）")
    else:
        print("[FAIL] 旋转后几乎失效 -> 证实 degrees=0.0 的缺陷，微调时 degrees 是必需项，不是可选项")
    print(f"\n结果图: {args.out}/rot{{000,090,180,270}}.jpg")


if __name__ == "__main__":
    main()
