"""拟合王牌面的精灵摆位（坐标下降，靶子=真实王）。

靶子：img_3 的大王、img_5 的小王/大王 —— 都是最底层、完全可见。
旧参数是手调的，角标区残差 48/255（普通牌只有 3）。

掩码用的是「和牌底色的差」而不是色饱和度 —— 小王的小丑是**银灰色**的，
用饱和度阈值会整个漏掉（第一版就是这么错的）。
"""
from __future__ import annotations
import sys
import cv2
import numpy as np

sys.path.insert(0, 'synth')
from compose import sprite, card_background, _paste, CARD_W, CARD_H

VISIBLE_W = 95          # 右手边那张牌压在上面，每张王只有左边 95px 是自己的
BG = card_background().astype(np.int16)[:, :, :3]
BGTOL = 26              # 与牌底色的差超过这个算「有内容」

TARGETS = [
    ("shots/img_3.png", 542, 714, "JOKER_B"),
    ("shots/img_5.png", 541, 713, "JOKER_S"),
    ("shots/img_5.png", 446, 713, "JOKER_B"),
]
SPRITE = {"JOKER_B": ("LargeCard_king_huase_15", "LargeCard_king_15"),
          "JOKER_S": ("LargeCard_king_huase_14", "LargeCard_king_14")}


def _mask_from_img(img):
    d = np.abs(img.astype(np.int16) - BG).max(axis=2)
    m = (d > BGTOL).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m[:3, :] = 0; m[-3:, :] = 0; m[:, :3] = 0
    m[:, VISIBLE_W:] = 0
    return m


def target_mask(path, x, y):
    return _mask_from_img(cv2.imread(path)[y:y + CARD_H, x:x + CARD_W])


def render(cls, p):
    jx, jy, js, tx, ty, ts = p
    canvas = card_background()
    jester, text = SPRITE[cls]
    _paste(canvas, sprite(text), tx, ty, ts)
    _paste(canvas, sprite(jester), jx, jy, js)
    out = np.full((CARD_H, CARD_W, 3), 255, np.uint8)
    _paste(out, canvas, 0, 0, 1.0)
    return out


def synth_mask(cls, p):
    return _mask_from_img(render(cls, p))


T = [(cls, target_mask(x_path, x, y)) for x_path, x, y, cls in TARGETS]


def loss(p):
    tot = 0.0
    for cls, t in T:
        m = synth_mask(cls, p)
        inter = int(np.logical_and(t > 0, m > 0).sum())
        union = int(np.logical_or(t > 0, m > 0).sum())
        tot += 1.0 - inter / max(union, 1)
    return tot / len(T)


def main():
    p = [36.0, 35.0, 0.85, 10.0, -2.0, 0.95]
    steps = [5.0, 5.0, 0.10, 5.0, 5.0, 0.10]
    best = loss(p)
    print(f"旧参数 IoU-loss = {best:.4f}  (IoU {1-best:.3f})")
    for _ in range(500):
        improved = False
        for i in range(6):
            for sgn in (+1, -1):
                t = list(p); t[i] += sgn * steps[i]
                l = loss(t)
                if l < best - 1e-5:
                    best, p, improved = l, t, True
        if not improved:
            steps = [s / 2 for s in steps]
            if max(steps) < 0.02:
                break
    print(f"拟合后 IoU-loss = {best:.4f}  (IoU {1-best:.3f})")
    jx, jy, js, tx, ty, ts = p
    print(f"\nJOKER_JESTER_XY = ({jx:.1f}, {jy:.1f})   JOKER_JESTER_SCALE = {js:.3f}")
    print(f"JOKER_TEXT_XY   = ({tx:.1f}, {ty:.1f})   JOKER_TEXT_SCALE   = {ts:.3f}")
    for cls, t in T:
        m = synth_mask(cls, p)
        iou = np.logical_and(t > 0, m > 0).sum() / max(1, np.logical_or(t > 0, m > 0).sum())
        print(f"   {cls}: IoU {iou:.3f}")

    # 对比图
    tiles = []
    for (path, x, y, cls), (_c, t) in zip(TARGETS, T):
        real = cv2.imread(path)[y:y + CARD_H, x:x + CARD_W].copy()
        syn = render(cls, p)
        m = synth_mask(cls, p)
        vis = np.full((CARD_H, CARD_W, 3), 255, np.uint8)
        vis[t > 0] = (90, 90, 90)
        vis[(m > 0) & (t == 0)] = (0, 0, 255)
        vis[(t > 0) & (m == 0)] = (0, 190, 0)
        real[:, VISIBLE_W:] = (real[:, VISIBLE_W:] * 0.4 + 200 * 0.6).astype(np.uint8)
        gap = np.full((CARD_H, 8, 3), 255, np.uint8)
        tiles += [real, gap, syn, gap, vis, np.full((CARD_H, 16, 3), 255, np.uint8)]
    cv2.imwrite('audit/joker_fit.png', cv2.resize(np.hstack(tiles), None, fx=2.0,
                                                  fy=2.0, interpolation=cv2.INTER_NEAREST))
    print("\n对比图 audit/joker_fit.png")


if __name__ == "__main__":
    main()
