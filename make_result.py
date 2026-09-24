"""生成一眼能看懂的效果图：截图 + 检测框（按花色配色）+ 识别出的手牌列表。

框上不打标签 —— 角标框挨得很近，标签会糊成一片。手牌单独列在下方面板里，
花色用红黑区分，方便和原图对照。

用法:
    python make_result.py <权重.pt> <图片> [--imgsz 960] [--level 2] [--out results/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
from layout import CLASSES  # noqa: E402
from predict_cards import card_value, dedup, display, split_hand_table  # noqa: E402

# 两个字体各有所长，缺一个都会渲染成方框：
#   msyh.ttc  —— 有中文，但没有 ♠♥♦♣
#   arial.ttf —— 有 ♠♥♦♣ 和 ASCII，但没有中文
# 所以按字符选字体。
FONT_CJK = "C:/Windows/Fonts/msyh.ttc"
FONT_SYM = "C:/Windows/Fonts/arial.ttf"
SUIT_CHARS = "♠♥♦♣"
_font_cache: dict = {}


def _font(size: int, ch: str) -> ImageFont.FreeTypeFont:
    key = (size, ch in SUIT_CHARS or ch.isascii())
    if key not in _font_cache:
        path = FONT_SYM if key[1] else FONT_CJK
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def draw_text(dr: ImageDraw.ImageDraw, x: int, y: int, text: str,
              size: int, color: tuple) -> int:
    """逐字符选字体绘制，返回结尾 x。"""
    for ch in text:
        f = _font(size, ch)
        dr.text((x, y), ch, font=f, fill=color)
        x += int(f.getlength(ch))
    return x
SUIT_COLOR = {"S": (40, 40, 40), "C": (40, 40, 40),
              "H": (30, 30, 200), "D": (30, 30, 200)}
JOKER_COLOR = (0, 130, 200)
RED, BLACK = (30, 30, 200), (40, 40, 40)


def draw_boxes(img: np.ndarray, hand: list[tuple],
               table: list[tuple]) -> np.ndarray:
    """手牌框画粗、桌面出牌区画细，一眼能区分。"""
    out = img.copy()
    for dets, thick in ((table, 1), (hand, 2)):
        for cls, cx, cy, conf in dets:
            col = JOKER_COLOR if cls.startswith("JOKER") else SUIT_COLOR[cls[0]]
            w, h = 93, 66
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            cv2.rectangle(out, (x1, y1), (x1 + w, y1 + h), col, thick)
    return out


def build_panel(pairs: list[tuple[str, float]], width: int, level: str) -> Image.Image:
    """下方信息面板。pairs 是 (类别, 置信度)。"""
    S_BIG, S_MID, S_SML = 30, 28, 24
    line_h, pad = 42, 24

    groups: dict[str, list[tuple[str, float]]] = {}
    for cls, c in pairs:
        key = cls if cls.startswith("JOKER") else cls[1:]
        groups.setdefault(key, []).append((cls, c))

    order = sorted(groups, key=lambda k: card_value(
        groups[k][0][0], level))
    height = pad * 2 + 46 + line_h * len(order)

    img = Image.new("RGB", (width, height), (250, 250, 248))
    dr = ImageDraw.Draw(img)

    n = len(pairs)
    avg = sum(c for _, c in pairs) / n if n else 0
    draw_text(dr, pad, pad - 4,
              f"识别到 {n} 张手牌   平均置信度 {avg:.2f}", S_BIG, (20, 20, 20))

    y = pad + 50
    for k in order:
        cards = sorted(groups[k], key=lambda t: 0 if t[0].startswith("JOKER")
                       else "SCDH".index(t[0][0]))
        label = "王" if k.startswith("JOKER") else ("10" if k == "T" else k)
        draw_text(dr, pad, y + 2, f"x{len(cards)}", S_SML, (130, 130, 130))
        x = pad + 62
        for cls, c in cards:
            col = JOKER_COLOR if cls.startswith("JOKER") else SUIT_COLOR[cls[0]]
            if cls.startswith("JOKER"):
                x = draw_text(dr, x, y, "大王" if cls == "JOKER_B" else "小王",
                              S_MID, col) + 14
            else:
                rank = cls[1:]
                x = draw_text(dr, x, y, SUIT_CHARS["SHDC".index(cls[0])], S_MID, col)
                x = draw_text(dr, x, y, "10" if rank == "T" else rank,
                              S_MID, BLACK) + 14
        y += line_h
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("image")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--level", default="2")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.model.names = {i: c for i, c in enumerate(CLASSES)}
    names = model.names

    res = model.predict(source=args.image, imgsz=args.imgsz, conf=args.conf,
                        verbose=False)[0]
    dets = dedup([(names[int(b.cls)], (b.xyxy[0][0] + b.xyxy[0][2]) / 2,
                   (b.xyxy[0][1] + b.xyxy[0][3]) / 2, float(b.conf))
                  for b in res.boxes])

    src = cv2.imread(args.image)
    hand_dets, table_dets = split_hand_table(dets)
    marked = draw_boxes(src, hand_dets, table_dets)
    hand = sorted([(d[0], d[3]) for d in hand_dets],
                  key=lambda t: card_value(t[0], args.level))

    top = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    panel = build_panel(hand, top.width, args.level)

    out = Image.new("RGB", (top.width, top.height + panel.height), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(panel, (0, top.height))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) / f"{Path(args.image).stem}_result.png"
    out.save(dest)
    print(f"{args.image}: {len(hand)} 张手牌 -> {dest}")


if __name__ == "__main__":
    main()
