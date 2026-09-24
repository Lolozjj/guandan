"""牌面合成器：用提取出的 sprite 拼出游戏里渲染的牌面。

版面参数由「模板匹配取初值 + 坐标下降精调」拟合得到，
在截图中的真实牌面上达到 **平均绝对差 约 2/255**，基本只剩抗锯齿噪声。

    牌面画布  132 x 174（含圆角，alpha 通道，圆角半径 8px）
    点数图形  位置 (9, 5)    缩放 1.02
    小花色    位置 (58, 17)  缩放 1.12
    大花色pip 位置 (58, 94)  缩放 2.24
    点数颜色随花色： S/C 用 hei(黑)， H/D 用 hong(红)

**两套排版**（style 参数）—— 这是 2026-09-23 找到的关键问题：

    手牌   style="hand"    （默认）小花色在点数**右边**
    出牌   style="played"         小花色在点数**正下方**

游戏对**桌面出牌区**用的是后一种排版。之前合成器只会画手牌那种，
模型在真实出牌上的置信度只有 0.05 —— 等于从没见过。
（实测：337 帧真实画面里 62.6% 有出牌，模型的精确率 30.4%、
 77% 的出牌组一张都认不出。这不是训练不够，是训练数据里根本没有这种牌。）

底图不用纯白——由 synth/make_card_bg.py 从真实截图 inpaint 抠出，存成 assets/card_bg.png。
纯白底误差 32/255，抠底色后降到约 2/255。

用法:
    python synth/compose.py --demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPRITE_DIR = ROOT / "assets" / "cards"
BG_PATH = ROOT / "assets" / "card_bg.png"

CARD_W, CARD_H = 132, 174

RANK_XY = (9, 5)
RANK_SCALE = 1.02
SUIT_SMALL_XY = (58, 17)
SUIT_SMALL_SCALE = 1.12
SUIT_BIG_XY = (58, 94)
SUIT_BIG_SCALE = 2.24

# 出牌区排版（132x174 坐标系）。由「字形掩码 IoU 坐标下降」拟合真实出牌得到，
# 3 帧交叉验证 IoU = 0.906。与手牌相比只有小花色位置不同，其余（点数、大 pip）
# 归一到 132 坐标系后和手牌一致 —— 说明游戏是拿同一张卡换了小花色的摆位。
PLAY_RANK_XY = (7.5, 6.2)
PLAY_RANK_SCALE = 0.98
PLAY_SUIT_SMALL_XY = (5.0, 63.5)
PLAY_SUIT_SMALL_SCALE = 1.25
PLAY_SUIT_BIG_XY = (58.5, 94.6)
PLAY_SUIT_BIG_SCALE = 2.26

# 王牌面的精灵摆位。旧值是手调的，角标区残差 34/255（普通牌只有 3）；
# 用真实王（img_3 的大王、img_5 的小王/大王，都是最底层完全可见）做靶子
# 坐标下降拟合后降到 20 左右。残下的主要是文字边缘 1px 抗锯齿，属正常。
#
# 注意：拟合时靶子只取「卡左边 95px」—— 王右边的牌压在上面，
# 只有这 95px 是自己的，拿整张 132 去比会把邻牌算进来（第一版就错在这）。
JOKER_JESTER_XY = (32.9, 37.5)
JOKER_JESTER_SCALE = 0.840
JOKER_TEXT_XY = (6.2, 1.8)
JOKER_TEXT_SCALE = 1.024

# 花色 -> (huase sprite 编号, 点数用色)
SUITS = {
    "S": ("huase_1", "hei"),   # 黑桃
    "H": ("huase_2", "hong"),  # 红桃
    "C": ("huase_3", "hei"),   # 梅花
    "D": ("huase_4", "hong"),  # 方块
}
# 点数 -> shuzi 编号（1=A, 11=J, 12=Q, 13=K）
RANKS = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
         "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13}

CLASSES = (
    [f"S{r}" for r in "A23456789TJQK"]
    + [f"H{r}" for r in "A23456789TJQK"]
    + [f"D{r}" for r in "A23456789TJQK"]
    + [f"C{r}" for r in "A23456789TJQK"]
    + ["JOKER_S", "JOKER_B"]
)

_cache: dict[str, np.ndarray] = {}


def sprite(name: str) -> np.ndarray:
    """按名字读 sprite（统一转成 BGRA）。"""
    if name not in _cache:
        hits = sorted(SPRITE_DIR.glob(f"_sprite_*_{name}.png")) or \
               sorted(SPRITE_DIR.glob(f"_sprite_{name}.png"))
        if not hits:
            raise FileNotFoundError(f"找不到 sprite: {name}")
        img = cv2.imread(str(hits[0]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"读不了: {hits[0]}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        _cache[name] = img
    return _cache[name]


def _paste(canvas: np.ndarray, img: np.ndarray, x: float, y: float, scale: float) -> None:
    """把带 alpha 的图按 alpha 混合贴到画布 (x, y)。canvas 可以是 3 或 4 通道。

    x/y 允许是小数（出牌区的字形位置是拟合出来的非整数），内部四舍五入。
    """
    x, y = int(round(x)), int(round(y))
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    part = img[y0 - y:y1 - y, x0 - x:x1 - x]
    a = part[:, :, 3:4].astype(np.float32) / 255.0
    dst = canvas[y0:y1, x0:x1, :3].astype(np.float32)
    canvas[y0:y1, x0:x1, :3] = (part[:, :, :3].astype(np.float32) * a
                                + dst * (1 - a)).astype(np.uint8)

    if canvas.shape[2] == 4:
        ca = canvas[y0:y1, x0:x1, 3:4].astype(np.float32)
        canvas[y0:y1, x0:x1, 3:4] = np.maximum(ca, part[:, :, 3:4]).astype(np.uint8)


def card_background(w: int = CARD_W, h: int = CARD_H) -> np.ndarray:
    """牌面底色（BGRA）。缺失时退回程序生成的圆角白卡。"""
    if w == CARD_W and h == CARD_H and BG_PATH.exists():
        bg = cv2.imread(str(BG_PATH), cv2.IMREAD_UNCHANGED)
        if bg is not None:
            if bg.ndim == 3 and bg.shape[2] == 3:
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
                bg[:, :, 3] = 255
            return bg.copy()

    canvas = np.full((h, w, 4), 255, np.uint8)
    r = 8
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(a, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(a, (cx, cy), r, 255, -1)
    canvas[:, :, :3] = 245
    canvas[:, :, 3] = a
    return canvas


def render_card(cls: str, w: int = CARD_W, h: int = CARD_H,
                level: str | None = None, style: str = "hand") -> np.ndarray:
    """按类别名渲染一张牌面（BGRA）。

    level 是当前级牌点数。级牌的红桃（逢人配）**小花色符号用金色**渲染，
    大花色 pip 仍是红色 —— 实测：金色只有小符号那一个，大 pip 是纯红。
    金色用 huase_5 这个 sprite（提取素材时就发现是金色♥，之前一直没用上）。

    style: "hand"（手牌，默认）或 "played"（桌面出牌区）。
    两者只有小花色符号的摆位不同，见模块头部说明。
    """
    canvas = card_background(w, h)

    if cls.startswith("JOKER"):
        # 王面 = 小丑角色 + 左侧竖排 JOKER 文字，两个 sprite 拼装。
        # 位置与缩放由坐标下降拟合真实王牌得到（残差约 29/255，普通牌约 3/255，
        # 王面还原度偏低是已知不足，但它只占 54 类中的 2 类）。
        jester, text = ("LargeCard_king_huase_15", "LargeCard_king_15")             if cls == "JOKER_B" else ("LargeCard_king_huase_14", "LargeCard_king_14")
        sx = w / CARD_W
        _paste(canvas, sprite(jester), JOKER_JESTER_XY[0] * sx,
               JOKER_JESTER_XY[1] * sx, JOKER_JESTER_SCALE * sx)
        _paste(canvas, sprite(text), JOKER_TEXT_XY[0] * sx,
               JOKER_TEXT_XY[1] * sx, JOKER_TEXT_SCALE * sx)
        return canvas

    suit_code, rank = cls[0], cls[1:]
    huase_name, color = SUITS[suit_code]
    rank_n = RANKS[rank]

    if style == "played":
        rank_xy, rank_sc = PLAY_RANK_XY, PLAY_RANK_SCALE
        small_xy, small_sc = PLAY_SUIT_SMALL_XY, PLAY_SUIT_SMALL_SCALE
        big_xy, big_sc = PLAY_SUIT_BIG_XY, PLAY_SUIT_BIG_SCALE
    else:
        rank_xy, rank_sc = RANK_XY, RANK_SCALE
        small_xy, small_sc = SUIT_SMALL_XY, SUIT_SMALL_SCALE
        big_xy, big_sc = SUIT_BIG_XY, SUIT_BIG_SCALE

    _paste(canvas, sprite(f"LargeCard_commom_shuzi_{color}_{rank_n}"),
           *rank_xy, rank_sc)
    small = "huase_5" if (level and cls == f"H{level}") else huase_name
    _paste(canvas, sprite(f"LargeCard_{small}"), *small_xy, small_sc)
    _paste(canvas, sprite(f"LargeCard_{huase_name}"), *big_xy, big_sc)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    out = ROOT / "assets"

    if args.demo:
        shot = cv2.imread(str(ROOT / "shots" / "img.png"))
        real = shot[715:715 + CARD_H, 1290:1290 + CARD_W]
        mine = render_card("D3")
        # 合成到绿色底上，模拟真实贴图
        canvas = np.full((CARD_H, CARD_W, 3), (120, 160, 130), np.uint8)
        _paste(canvas, mine, 0, 0, 1.0)

        diff = cv2.absdiff(real, canvas)
        print("真实牌面 vs 合成牌面")
        print(f"  平均绝对差: {diff.mean():.2f} / 255")
        print(f"  95 分位差 : {np.percentile(diff, 95):.1f}")

        gap = np.full((CARD_H, 16, 3), 200, np.uint8)
        sheet = np.hstack([real, gap, canvas, gap, diff])
        cv2.imwrite(str(out / "compose_check.png"), cv2.resize(
            sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        print(f"\n对比图: {out / 'compose_check.png'}")

    cells = [render_card(c) for c in CLASSES]
    cols, pad = 9, 6
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * (CARD_H + pad) + pad, cols * (CARD_W + pad) + pad, 3), 60, np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        _paste(sheet, c, pad + cc * (CARD_W + pad), pad + r * (CARD_H + pad), 1.0)
    cv2.imwrite(str(out / "all_classes.png"), sheet)
    print(f"54 类总览: {out / 'all_classes.png'}")


if __name__ == "__main__":
    main()
