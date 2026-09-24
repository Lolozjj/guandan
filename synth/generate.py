"""生成掼蛋牌面识别的合成训练数据集（含 YOLO 自动标注）。

原理：牌面用游戏原生素材拼装、布局按实测参数生成、标注由坐标算出 —— 全程无需人工。

用法:
    python synth/generate.py --n 2000                    # 生成 2000 张到 dataset/
    python synth/generate.py --n 12 --preview            # 只看效果，不写数据集
    python synth/generate.py --n 2000 --no-played        # 只生成手牌，不要出牌区
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout as L                      # noqa: E402
from compose import _paste, render_card  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BG_DIR = ROOT / "assets" / "table_bg"

# 掼蛋两副牌：54 类 × 2
FULL_DECK = L.CLASSES * 2
LEVELS_ALL = list("23456789TJQKA")


def load_backgrounds() -> list[np.ndarray]:
    bgs = []
    for p in sorted(BG_DIR.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            bgs.append(img)
    if not bgs:
        raise SystemExit(f"[FAIL] {BG_DIR} 下没有背景图，先跑 synth/make_bg.py")
    return bgs


def sample_hand(rng: random.Random, size: int) -> list[str]:
    """从两副牌里抽 size 张（不放回）。"""
    return rng.sample(FULL_DECK, size)


def sample_hand_size(rng: random.Random) -> int:
    """手牌张数分布：偏向大牌手，但小牌手也要有。"""
    r = rng.random()
    if r < 0.55:
        return rng.randint(13, 27)     # 常见
    if r < 0.85:
        return rng.randint(5, 12)      # 残局
    return rng.randint(1, 4)           # 极限残局


def place_played(rng, no_played):
    """随机布置出牌区，返回 [(区名, [(类别, x, y, scale), ...]), ...]。

    按「一组」返回而不是平铺 —— 因为动画/庆祝效果是**整组一起**变化的
    （变暗、染色），逐张处理的话后画的那张会把前一张的效果盖掉。

    要点（都是实测出来的）：
      - 牌 106x140（= 手牌 0.804 倍），缩放基本恒定
      - 牌之间**重度重叠**，步长 50，每张只露左边 50px
      - 四个区各有固定的锚点和排列方向（左家左对齐 / 右家右对齐 / 上下居中）
      - 位置加抖动，别让模型把某个固定坐标当成信号
    """
    if no_played:
        return []
    groups = []
    # 每区独立有牌的概率 0.25：实测真实画面是「每帧平均 0.81 个牌组」
    # （0 组 37.4% / 1 组 45.7% / 2 组 15.7% / 3 组 1.2%），对应 p≈0.20~0.27。
    # 之前写的是 0.35，配上「4 个区」相当于每帧 1.4 组，牌出得太密，
    # 会教模型在空桌面上瞎认。
    for zname, zone in L.PLAYED_ZONES.items():
        if rng.random() < 0.25:        # 大多数时候该区是空的
            continue
        # 真实牌组张数分布（实测 275 组）：1 张 49、2 张 34、3 张 27、
        # 4 张 120（炸弹多）、5 张 39、6~8 张各几个
        n = rng.choices([1, 2, 3, 4, 5, 6, 7, 8],
                        weights=[50, 34, 27, 120, 39, 3, 1, 2])[0]
        cards = rng.sample(FULL_DECK, n)
        sc = rng.uniform(*L.PLAYED_SCALE_RANGE)
        total_w = L.PLAYED_PITCH * (n - 1) + L.CARD_W * sc
        anchor = zone["anchor"]
        if anchor == "center":
            x0 = zone["x"] - total_w / 2
        elif anchor == "right":
            x0 = zone["x"] - total_w
        else:
            x0 = zone["x"]
        x0 += rng.randint(-14, 14)
        y0 = float(zone["y"]) + rng.randint(-8, 8)
        groups.append((zname, [(cls, x0 + i * L.PLAYED_PITCH, y0, sc)
                               for i, cls in enumerate(cards)]))
    return groups


def render_sample(bgs, card_cache, scaled_cache, rng, args):
    """渲染一张合成图，返回 (图像, 标注框列表)。

    标注框是 (类别, x1, y1, x2, y2) 的绝对坐标，最后统一在扰动之后换算成 YOLO 行 —— 
    因为后面会对整图做亚像素平移，标注必须跟着一起动。
    """
    bg = bgs[rng.randrange(len(bgs))]
    img = bg.copy()
    h, w = img.shape[:2]
    boxes: list[tuple[str, float, float, float, float]] = []

    hand = sample_hand(rng, sample_hand_size(rng))
    level = rng.choice(LEVELS_ALL)               # 随机级牌：影响排序，也决定哪张牌用金色
    placed = L.layout(hand, w, level=level)

    # 出牌区先画。整组画完再统一做动画/庆祝效果 —— 逐张做的话后画的牌
    # 会把前一张的效果盖掉。
    card_h, card_w = scaled_cache[(L.CLASSES[0], level)].shape[:2]
    for _zname, cards in place_played(rng, args.no_played):
        before = img.copy()
        gx1 = gy1 = 10 ** 9
        gx2 = gy2 = -1
        for cls, x, y, sc in cards:
            xi, yi = int(round(x)), int(round(y))
            _paste(img, scaled_cache[(cls, level)], xi, yi, 1.0)
            gx1, gy1 = min(gx1, xi), min(gy1, yi)
            gx2 = max(gx2, xi + card_w)
            gy2 = max(gy2, yi + card_h)
            boxes.append((cls, *L.played_marker_box(x, y, sc)))

        x1, y1 = max(0, gx1), max(0, gy1)
        x2, y2 = min(img.shape[1], gx2), min(img.shape[0], gy2)
        if x2 <= x1 or y2 <= y1:
            continue

        # (1) 出牌动画中间态：整组被半透明层压住、发灰。
        #     实测约 22% 的帧是这个状态，不让模型见过的话真实画面里一律认不出。
        if rng.random() < 0.22:
            bg = before[y1:y2, x1:x2].astype(np.float32)
            fg = img[y1:y2, x1:x2].astype(np.float32)
            a = rng.uniform(0.45, 0.8)
            img[y1:y2, x1:x2] = np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)
        # (2) 打出炸弹/顺子时，整组牌会被庆祝光效**染上一层暖色**。
        #     实测 f00015 那组：牌面 BGR 从 (220,220,220) 变成 (223,227,248)，
        #     也就是红通道 +28、蓝绿基本不动 —— 大约是在牌面上叠 11% 的红色。
        #     这种帧只持续 1s 左右，实时面板靠 --stable-frames 会跳过，
        #     但万一正好在稳定期被读到，模型得有抵抗力。
        elif rng.random() < 0.12:
            tint = np.array([rng.uniform(20, 90), rng.uniform(30, 110), 255.0])  # BGR
            a = rng.uniform(0.06, 0.20)
            sub = img[y1:y2, x1:x2].astype(np.float32)
            img[y1:y2, x1:x2] = np.clip(sub * (1 - a) + tint * a, 0, 255).astype(np.uint8)

    # 手牌后画。水平/垂直各加一点整数抖动：真实截图里手牌的居中位置
    # 有 ±2.5px 的浮动（游戏用的是带内边距的容器），靠抖动覆盖掉。
    jx = rng.randint(-3, 4)
    jy = rng.randint(-2, 3)
    for pc in L.draw_order(placed):
        _paste(img, card_cache[(pc.cls, level)], pc.x + jx, pc.y + jy, 1.0)
        boxes.append((pc.cls, *L.marker_box(pc.x + jx, pc.y + jy)))

    img, _shift = augment(img, rng, args)
    return img, boxes


def augment(img, rng, args):
    """模拟真实截图的成像差异。

    实测：位置偏 1px 就让角标区差异从 4.66 涨到 9.77/255，
    所以亚像素平移是关键的一项 —— 它让模型不再对像素级对齐敏感。
    """
    h, w = img.shape[:2]
    nprng = np.random.default_rng(rng.getrandbits(32))

    # 亚像素平移
    dx, dy = rng.uniform(-3.5, 3.5), rng.uniform(-2.5, 2.5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # 轻微模糊（真实渲染有抗锯齿/缩放）
    if rng.random() < 0.45:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))

    # 亮度/对比度微扰
    if rng.random() < 0.5:
        alpha = 1.0 + rng.uniform(-0.06, 0.06)
        beta = rng.uniform(-6, 6)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 极轻的噪声
    if rng.random() < 0.4:
        noise = nprng.normal(0, rng.uniform(0.8, 2.2), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, (dx, dy)


def _paste_return(img, sprite, x, y, scale):
    _paste(img, sprite, x, y, scale)
    return img


def boxes_to_lines(boxes, w, h):
    """绝对坐标框 -> YOLO 标注行。"""
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        line = L.yolo_line(cls, x1, y1, x2, y2, w, h)
        if line:
            out.append(line)
    return out


def build_cache(classes):
    """预渲染全部牌面（手牌原尺寸 + 出牌区版）。

    每个类别都要按 13 种级牌各渲染一份 —— 级牌红桃的小花色符号是金色，
    是同一类别内部的合法外观差异，必须让模型都见过。

    **出牌区必须单独渲染**（style="played"）：它的小花色符号在点数正下方，
    直接把手牌那张缩一下是画不出来的。之前就是这里漏了，导致模型
    在真实出牌上置信度只有 0.05、77% 的出牌组认不出来。
    """
    native, scaled = {}, {}
    sw, sh = L.PLAYED_CARD_W, L.PLAYED_CARD_H
    for cls in classes:
        for lv in LEVELS_ALL:
            native[(cls, lv)] = render_card(cls, level=lv)
            played = render_card(cls, level=lv, style="played")
            scaled[(cls, lv)] = cv2.resize(played, (sw, sh),
                                           interpolation=cv2.INTER_AREA)
    return native, scaled


def write_data_yaml(out_root: Path) -> None:
    """从 CLASSES 生成 dataset/data.yaml。

    类别顺序必须和 CLASSES 完全一致 —— 错一位的话模型学到的映射是对的、
    但读出来会全体错位（坑 #3 实测过一次：26 张牌全部 rank+1）。
    所以这个文件由脚本生成，不要手写。
    """
    lines = [
        "# 掼蛋牌面识别 —— YOLO 数据集配置（由 synth/generate.py 自动生成，不要手改）",
        "# 类别顺序 = synth/layout.py 的 CLASSES：先花色 S H D C，花色内 A,2,...,K",
        "",
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(L.CLASSES)]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"data.yaml 已生成: {out_root / 'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--no-played", action="store_true", help="不生成出牌区的牌")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="只生成 --n 张预览图")
    ap.add_argument("--jpg-quality", type=int, default=92)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bgs = load_backgrounds()
    print(f"背景 {len(bgs)} 张，尺寸 {bgs[0].shape[1]}x{bgs[0].shape[0]}")
    native, scaled = build_cache(L.CLASSES)
    print(f"已预渲染 {len(native)} 种牌面\n")

    if args.preview:
        for i in range(args.n):
            img, boxes = render_sample(bgs, native, scaled, rng, args)
            lines = boxes_to_lines(boxes, img.shape[1], img.shape[0])
            out = ROOT / "assets" / "preview"
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"prev_{i:03d}.jpg"), img[:, :, :3],
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            print(f"  预览 {i}: {img.shape[1]}x{img.shape[0]}  {len(lines)} 个框")
        print(f"\n预览图: {ROOT / 'assets' / 'preview'}")
        return

    out_root = Path(args.out)
    n_val = int(args.n * args.val_ratio)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in L.CLASSES}
    total_boxes = 0
    for i in range(args.n):
        img, boxes = render_sample(bgs, native, scaled, rng, args)
        lines = boxes_to_lines(boxes, img.shape[1], img.shape[0])
        split = "val" if i < n_val else "train"
        stem = f"g{i:06d}"
        q = rng.randint(args.jpg_quality - 15, args.jpg_quality)
        cv2.imwrite(str(out_root / "images" / split / f"{stem}.jpg"), img[:, :, :3],
                    [cv2.IMWRITE_JPEG_QUALITY, q])
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_boxes += len(lines)
        for ln in lines:
            counts[L.CLASSES[int(ln.split()[0])]] += 1
        if (i + 1) % 200 == 0:
            print(f"  已生成 {i + 1}/{args.n}")

    write_data_yaml(out_root)

    missing = [c for c, n in counts.items() if n == 0]
    print(f"\n完成: {args.n} 张图, {total_boxes} 个标注框, "
          f"平均每张 {total_boxes / max(1, args.n):.1f} 个")
    print(f"train/val 划分: {args.n - n_val} / {n_val}")
    if missing:
        print(f"[WARN] 未出现的类别 ({len(missing)}): {missing} —— 请加大 --n")
    lo = min(counts.values())
    hi = max(counts.values())
    print(f"类别样本数: 最少 {lo}, 最多 {hi}")
    print(f"输出: {out_root.resolve()}")


if __name__ == "__main__":
    main()
