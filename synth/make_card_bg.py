"""从真实截图里抠出一张「干净牌面」（含 alpha 通道），作为合成的底图。

    真实牌边界：截图 img.png 的 x=1290~1421, y=715~888  ->  132 x 174
    圆角半径约 8 px
    图形元素在该坐标系下的位置（由模板匹配 + 坐标下降拟合得到）：
        点数     (9, 5)   38x58
        小花色   (58, 17) 36x36
        大花色   (58, 94) 72x72

流程：裁出真实牌 -> 把图形元素区域 inpaint 掉 -> 生成圆角矩形 alpha -> 存成 BGRA PNG。

用法:
    python synth/make_card_bg.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHOT = ROOT / "shots" / "img.png"
OUT = ROOT / "assets" / "card_bg.png"

X, Y, W, H = 1290, 715, 132, 174
RADIUS = 8

# 图形元素（相对卡片左上角），用于挖掉后 inpaint
GLYPHS = [(9, 5, 38, 58), (58, 17, 36, 36), (58, 94, 72, 72)]
PAD = 4


def rounded_alpha(w: int, h: int, r: int) -> np.ndarray:
    """圆角矩形 alpha（0~255）。"""
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(a, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(a, (cx, cy), r, 255, -1)
    # 边缘羽化 1px，避免贴图时出现硬锯齿
    return cv2.GaussianBlur(a, (3, 3), 0.6)


def main() -> None:
    shot = cv2.imread(str(SHOT))
    if shot is None:
        raise SystemExit(f"[FAIL] 读不到 {SHOT}")

    card = shot[Y:Y + H, X:X + W].copy()
    print(f"裁出真实牌: {card.shape[1]}x{card.shape[0]}  (x{X}, y{Y})")

    # 左边缘可能有相邻牌残留，用第 4 列外推抹掉
    card[:, 0:3] = card[:, 3:4]

    mask = np.zeros((H, W), np.uint8)
    for gx, gy, gw, gh in GLYPHS:
        mask[max(0, gy - PAD):min(H, gy + gh + PAD),
             max(0, gx - PAD):min(W, gx + gw + PAD)] = 255
    print(f"待 inpaint 面积占比: {mask.mean() / 255:.1%}")

    rgb = cv2.inpaint(card, mask, 5, cv2.INPAINT_TELEA)
    rgb[:, 0:3] = rgb[:, 3:4]

    alpha = rounded_alpha(W, H, RADIUS)
    bgra = np.dstack([rgb, alpha])
    cv2.imwrite(str(OUT), bgra)
    print(f"已写出 {OUT}  (BGRA, 圆角 r={RADIUS})")

    # 角落检查：alpha 是 0 的地方不该留下绿底
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    for cy, cx in corners:
        print(f"  角({cx},{cy}): BGR={rgb[cy, cx].tolist()} alpha={int(alpha[cy, cx])}")


if __name__ == "__main__":
    main()
