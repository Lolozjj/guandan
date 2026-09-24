"""出牌区验收对比图：真实画面 + 「每家出了什么牌」的真值/检出对照。

用户要的是「每家出过什么牌」，所以图里除了画框，更重要的是把读出来的牌
逐家列出来跟真值对照 —— 一眼就能看出认没认出来、认对了没有。

用法:
    python make_table_fig.py <权重.pt> [--out audit/table_result.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'synth'))
import layout as L                                            # noqa: E402
from layout import CLASSES                                    # noqa: E402
from predict_cards import (dedup, split_hand_table,           # noqa: E402
                           table_by_player, display)
from table_gt import GROUND_TRUTH                             # noqa: E402

# msyh（微软雅黑）**没有** ♠♥♦♣ 这几个字形，直接画会变成方框；
# seguisym 有花色但没有中文。所以按字符切换字体混排。
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONT_SYM = 'C:/Windows/Fonts/seguisym.ttf'
SYMBOLS = set("♠♥♦♣")


def draw_mixed(d, x, y, text, font, font_sym, fill):
    """中文用 font、花色符号用 font_sym，逐段画。"""
    i = 0
    while i < len(text):
        sym = text[i] in SYMBOLS
        j = i
        while j < len(text) and (text[j] in SYMBOLS) == sym:
            j += 1
        seg = text[i:j]
        f = font_sym if sym else font
        d.text((x, y), seg, font=f, fill=fill)
        x += d.textlength(seg, font=f)
        i = j
    return x
ZONE_ORDER = ("left", "right", "top", "bottom")
ZONE_CN = {"left": "机器人1", "right": "机器人3", "top": "队友", "bottom": "我"}
ZONE_XY = {"left": (204, 312), "right": (1485, 312),
           "top": (846, 199), "bottom": (846, 355)}
BAND = (60, 150, 1630, 660)
SCALE = 0.52


def cards_text(cls_list):
    if not cls_list:
        return "—"
    return " ".join(display(c) for c in cls_list)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--out", default="audit/table_result.png")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.model.names = {i: c for i, c in enumerate(CLASSES)}

    x1, y1, x2, y2 = BAND
    panels = []
    for name, gt in GROUND_TRUTH.items():
        img = cv2.imread(str(ROOT / 'live' / 'frames' / name))
        if img is None:
            continue
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = dedup([(model.names[int(b.cls)],
                       float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                       float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                       float(b.conf)) for b in res.boxes])
        _, table = split_hand_table(dets)
        by_player = table_by_player(table)

        c = img[y1:y2, x1:x2].copy()
        # 真值画绿框
        for z, cls_list in gt.items():
            zx, zy = ZONE_XY[z]
            n = len(cls_list)
            tw = int(L.PLAYED_PITCH * (n - 1) + L.PLAYED_CARD_W)
            gx = zx - tw if z == "right" else (int(zx - tw / 2) if n and z in ("top", "bottom") else zx)
            for i in range(n):
                bx = gx + L.PLAYED_PITCH * i
                cv2.rectangle(c, (bx - x1, zy - y1),
                              (bx - x1 + L.PLAYED_CARD_W, zy - y1 + L.PLAYED_CARD_H),
                              (0, 220, 0), 3)
        # 模型检出画红框
        for who, v in by_player.items():
            for d in v:
                cx, cy = int(d[1] - x1), int(d[2] - y1)
                cv2.rectangle(c, (cx - 16, cy - 34), (cx + 16, cy + 34), (0, 0, 255), 2)
        c = cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

        got = {z: sorted(by_player[ZONE_CN[z]], key=lambda t: t[1]) for z in ZONE_ORDER}
        want = {z: gt.get(z, []) for z in ZONE_ORDER}
        # 只统计「真值或检出非空」的区；两边都空的区不算数（否则空区会白送正确）
        def zone_ok(z):
            if not want[z] and not got[z]:
                return None
            return sorted(x[0] for x in got[z]) == sorted(want[z])

        n_ok = sum(1 for z in ZONE_ORDER if zone_ok(z) is True)
        n_z = sum(1 for z in ZONE_ORDER if zone_ok(z) is not None)
        panels.append((name, c, want, got, n_ok, n_z))

    PW, PH = 620, 300            # 文字面板尺寸
    IMG_H = panels[0][1].shape[0]
    TILE_H = IMG_H + PH + 40
    f_sym = ImageFont.truetype(FONT_SYM, 19)
    f_t = ImageFont.truetype(FONT, 21)
    f_b = ImageFont.truetype(FONT, 19)
    f_h = ImageFont.truetype(FONT, 17)

    tiles = []
    for name, c, want, got, n_ok, n_z in panels:
        tile = Image.new('RGB', (c.shape[1], TILE_H), (252, 252, 252))
        tile.paste(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), (0, 0))
        d = ImageDraw.Draw(tile)
        y = IMG_H + 8
        d.text((6, y), f"{name}    整组全对 {n_ok}/{n_z}", font=f_t, fill=(20, 20, 20))
        y += 30
        for z in ZONE_ORDER:
            w = cards_text(want[z])
            g = cards_text([x[0] for x in got[z]])
            ok = sorted(x[0] for x in got[z]) == sorted(want[z])
            d.text((6, y), f"{ZONE_CN[z]:<5}", font=f_h, fill=(90, 90, 90))
            d.text((72, y), "真值", font=f_h, fill=(0, 130, 0))
            draw_mixed(d, 126, y, w, f_b, f_sym, (0, 110, 0))
            y += 24
            d.text((72, y), "检出", font=f_h, fill=(190, 0, 0))
            draw_mixed(d, 126, y, g, f_b, f_sym, (190, 0, 0))
            if zone_ok(z) is not None:
                # 用中文「对/错」—— msyh 没有 ✓✗ 的字形，会画成方框
                d.text((PW - 34, y), "对" if zone_ok(z) else "错", font=f_b,
                       fill=(0, 150, 0) if zone_ok(z) else (200, 0, 0))
            y += 30
        tiles.append(tile)

    W = sum(t.width for t in tiles[:args.per]) + 12 * (args.per - 1)
    rows = []
    for i in range(0, len(tiles), args.per):
        page = tiles[i:i + args.per]
        row = Image.new('RGB', (W, TILE_H), (255, 255, 255))
        x = 0
        for t in page:
            row.paste(t, (x, 0))
            x += t.width + 12
        rows.append(row)
    H = sum(r.height for r in rows) + 12 * len(rows) + 60
    out = Image.new('RGB', (max(r.width for r in rows), H), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.text((10, 12), "出牌区验收：绿框 = 人工真值，红框 = 模型检出",
           font=ImageFont.truetype(FONT, 30), fill=(20, 20, 20))
    y = 52
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 12
    out.save(ROOT / args.out)
    print(f"已保存 {ROOT / args.out}  ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
