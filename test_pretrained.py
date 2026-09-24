"""用现成的扑克牌检测权重，在掼蛋截图上做零样本评估。

用法:
    python test_pretrained.py <权重.pt> <图片目录或单张图片> [选项]

例:
    python test_pretrained.py yolov8n-playing-cards.pt shots/ --imgsz 960

它会做三件事:
    1. 打印模型自带的类别表，并和本项目的 54 类做对比（列出缺失/多余）
    2. 逐张图推理，打印检出的类别和数量
    3. 把画了框的结果图存到 runs/pretrained_test/

注意: 现成权重几乎都是「单副牌 52 类」体系，和我们「双副牌 54 类」不一致。
      这个脚本只做零样本摸底，不修正类别差异——本来就是要看它错得多离谱。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 本项目的 54 类，与 dataset/classes.txt 一致
OUR_CLASSES = (
    [f"S{r}" for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")]
    + [f"H{r}" for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")]
    + [f"D{r}" for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")]
    + [f"C{r}" for r in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")]
    + ["JOKER_S", "JOKER_B"]
)


def collect_images(src: Path) -> list[Path]:
    """src 可以是单张图，也可以是目录（不递归）。"""
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    raise SystemExit(f"[FAIL] 路径不存在: {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="预训练权重路径 (.pt)")
    ap.add_argument("source", help="图片目录或单张图片")
    ap.add_argument("--imgsz", type=int, default=960, help="推理分辨率，默认 960")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值，默认 0.5")
    ap.add_argument("--out", default="runs/pretrained_test", help="结果图输出目录")
    ap.add_argument("--max", type=int, default=0, help="最多处理几张图，0 = 全部")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  (CUDA 可用: {torch.cuda.is_available()})")

    # --- 载入模型，先看它的类别表 ---
    model = YOLO(args.model)
    model_names = model.names  # {idx: name}
    print(f"\n模型类别数: {len(model_names)}")
    print(f"模型类别  : {sorted(model_names.values())}")

    model_upper = {n.upper() for n in model_names.values()}
    ours_upper = {n.upper() for n in OUR_CLASSES}
    missing = sorted(ours_upper - model_upper)   # 我们需要但模型没有
    extra = sorted(model_upper - ours_upper)     # 模型有但我们不需要
    print(f"\n本项目有、模型没有 ({len(missing)}): {missing}")
    print(f"模型有、本项目不要 ({len(extra)}): {extra}")
    if missing:
        print("  ^ 这些类别现成权重认不出来，属于预期内，不用管")

    # --- 逐张推理 ---
    images = collect_images(Path(args.source))
    if args.max:
        images = images[: args.max]
    if not images:
        raise SystemExit(f"[FAIL] 在 {args.source} 里没找到图片")
    print(f"\n待测图片: {len(images)} 张")

    total = Counter()
    empty = 0

    for i, img in enumerate(images, 1):
        res = model.predict(
            source=str(img), imgsz=args.imgsz, conf=args.conf,
            iou=args.iou, device=device, verbose=False,
        )[0]
        names = [model_names[int(c)] for c in res.boxes.cls.tolist()]
        counts = Counter(names)
        total.update(counts)
        if not counts:
            empty += 1

        tag = " ".join(f"{k}x{v}" for k, v in sorted(counts.items())) or "(未检出)"
        print(f"  [{i:>3}/{len(images)}] {img.name}: {len(names)} 个框 | {tag}")

        res.save(filename=Path(args.out) / img.name)

    # --- 汇总 ---
    print(f"\n{'=' * 60}")
    print(f"总检出: {sum(total.values())} 个框 / {len(images)} 张图")
    print(f"平均每张: {sum(total.values()) / len(images):.1f} 个框")
    print(f"完全没检出的图: {empty} / {len(images)}")

    if total:
        print(f"\n最常检出的类别 (top 15):")
        for name, cnt in total.most_common(15):
            print(f"  {name:<10} {cnt}")
        unrecognized = [n for n in total if n.upper() not in ours_upper]
        if unrecognized:
            print(f"\n模型检出但我们没有的类别: {unrecognized}")

    print(f"\n结果图已存到: {args.out}/")


if __name__ == "__main__":
    main()
