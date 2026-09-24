"""从游戏画面里读出「打几」（当前级牌）。

原理：游戏自带 UI 数字字体（从资源包 artcontent_font_assets_bmp 里提取的
A0123456789JQK 图集），和面板上显示的完全一致，所以直接做模板匹配，
不需要 OCR、也不会认错字形。

定位方式：左上角有两个面板「我方 / 对方」，各自下面可能有一个白色三角 ▲。
▲ 标在哪个面板下，那个面板里的数字就是当前级牌。

实测字形编号（在 blue 图集里的连通域序号）：
    0:A  1:5  2:7  3:Q  4:6  5:3  6:K  7:0
    8:2  9:4  10:8  11:9  12:J  13:1
（没有单独的「10」字形，打10 时是「1」+「0」两个字形并排。）

用法:
    python live/level.py live/frames/f00100_0060.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "debug" / "font"

# 图集里连通域序号 -> 牌面字符
GLYPH_ORDER = {0: "A", 1: "5", 2: "7", 3: "Q", 4: "6", 5: "3", 6: "K",
               7: "0", 8: "2", 9: "4", 10: "8", 11: "9", 12: "J", 13: "1"}

# 级别面板区域（相对于 1690x1000 的画面）
PANEL_Y = (55, 205)
PANEL_X = (60, 345)
# 两个面板里数字所在的区域（相对 roi）。不硬编码数字位置，而是在这个范围里
# 自动找深色块 —— 硬编码过一次位置，结果「对方」的取框整个错过了数字。
# 两个面板的横向范围（相对 roi）。里面**自动找出所有字形再按 x 排序拼接** ——
# 级别是 10 的时候面板上是「1」和「0」两个字，固定宽度裁框只装得下一个。
# 往里缩一圈是为了避开面板边框/阴影形成的大连通块。
DIGIT_BOXES = {"ours": (56, 112), "theirs": (136, 192)}
DIGIT_ROI_Y = (54, 96)
INSET = 3
TRI_ROI_Y = (95, 130)   # ▲ 三角的纵向范围（相对 roi）

_templates: dict[str, list[np.ndarray]] = {}
DEBUG_DIR: Path | None = None


def _load_templates() -> dict[str, list[np.ndarray]]:
    """从两个图集切出字形，按字符归类。"""
    if _templates:
        return _templates
    for tag in ("blue", "grey"):
        p = FONT_DIR / f"A0123456789JQK{'_blue' if tag == 'blue' else ''}-export.png"
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        m = (img[:, :, 3] > 40).astype(np.uint8)
        # 图集是透明背景的。cv2.imread 读到透明区是黑色 (0,0,0)，直接二值化会把
        # 整块都当成前景、包围盒撑满 -> 模板变成一坨（真发生过）。
        # 所以先按 alpha 把字形合到白底上，模拟它在面板上的样子。
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        flat = (img[:, :, :3].astype(np.float32) * alpha
                + 255.0 * (1 - alpha)).astype(np.uint8)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
        boxes = [(x, y, w, h) for i, (x, y, w, h, a) in enumerate(stats)
                 if i > 0 and a > 300]
        boxes.sort(key=lambda b: (b[1] // 60, b[0]))
        for i, (x, y, w, h) in enumerate(boxes):
            ch = GLYPH_ORDER.get(i)
            if ch is None:
                continue
            # 归一化成 32x40 的二值图，尺寸和颜色都不影响匹配
            g = cv2.cvtColor(flat[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (32, 40), interpolation=cv2.INTER_AREA)
            _templates.setdefault(ch, []).append(_binarize(g))
    return _templates


def _binarize(g: np.ndarray) -> np.ndarray:
    """数字是深色的粗体字 -> 二值化并对齐到统一尺寸。"""
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    # 用 Otsu 自动定阈值：数字是深色粗体、背景是浅色面板，两者对比很强
    _t, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = b.astype(np.uint8)
    ys, xs = np.where(b)
    if len(ys) == 0:
        return np.zeros((40, 32), np.float32)
    crop = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    s = min(32 / max(cw, 1), 40 / max(ch, 1))
    crop = cv2.resize(crop.astype(np.float32), (max(1, int(cw * s)), max(1, int(ch * s))),
                      interpolation=cv2.INTER_AREA)
    out = np.zeros((40, 32), np.float32)
    h, w = crop.shape
    y0, x0 = (40 - h) // 2, (32 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop
    return out


def _normalize(chars: str) -> str:
    """把拼接出来的字形串规范成级别字符。"""
    if chars == "10":
        return "T"          # 打 10
    if chars == "1":
        return "T"          # 只切到「1」也算 10（另一半可能被裁掉）
    return chars


def _match(cell: np.ndarray) -> tuple[str, float]:
    tpl = _load_templates()
    q = _binarize(cell)
    best, bs = None, -1.0
    for ch, lst in tpl.items():
        for t in lst:
            # 归一化相关，对亮度/颜色不敏感
            a = q - q.mean()
            b = t - t.mean()
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-6:
                continue
            s = float((a * b).sum() / denom)
            if s > bs:
                bs, best = s, ch
    return best, bs


def _find_triangle(panel_strip: np.ndarray) -> int | None:
    """在面板下方找白色三角，返回它的 x 中心（相对坐标）。没找到返回 None。"""
    s = panel_strip
    if s.size == 0:
        return None
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 2] > 210) & (hsv[:, :, 1] < 45)).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 80 or w < 12 or h < 5:
            continue
        # 三角标记：宽高比约 2:1（实测 22x11）。
        # 方向不固定 —— 实测是 ▲（尖端朝上、底边在下），所以不能只认 ▼。
        if 1.2 < w / max(h, 1) < 3.5:
            cands.append((a, int(cent[i][0]), int(cent[i][1])))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def read_level(img: np.ndarray) -> tuple[str | None, float, dict]:
    """读当前级牌。返回 (级别字符, 匹配置信度, 调试信息)。"""
    x1, x2 = PANEL_X
    y1, y2 = PANEL_Y
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0, {}
    if DEBUG_DIR:
        cv2.imwrite(str(DEBUG_DIR / "roi.png"),
                    cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))

    tx = _find_triangle(roi[TRI_ROI_Y[0]:TRI_ROI_Y[1]])

    info = {"roi": (x1, y1, x2, y2), "tri_x": tx}
    if tx is None:
        # 没找到三角：退回「取两个面板里较大的数字」，并标记低置信
        info["fallback"] = True
    else:
        info["panel"] = "ours" if tx < roi.shape[1] // 2 else "theirs"

    cells = []
    for side, (cx1, cx2) in DIGIT_BOXES.items():
        region = roi[DIGIT_ROI_Y[0]:DIGIT_ROI_Y[1], cx1 + INSET:cx2 - INSET]
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _t, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8), 8)
        rh, rw = region.shape[:2]
        glyphs = []
        for i in range(1, n):
            gx, gy, gw, gh, ga = stats[i]
            # 数字块：够大、但不横跨整区（边框会被排除）
            if ga < 80 or gh < 0.4 * rh or gw > 0.6 * rw:
                continue
            glyphs.append((gx, gy, gw, gh))
        if not glyphs:
            continue
        glyphs.sort()
        parts = []
        for gx, gy, gw, gh in glyphs:
            pad = 2
            cell = gray[max(0, gy - pad):gy + gh + pad, max(0, gx - pad):gx + gw + pad]
            if cell.size:
                parts.append(_match(cell))
        if parts:
            cells.append((side, parts))

    # cells[0] 是「我方」面板，cells[1] 是「对方」面板
    # 多个字形按顺序拼起来（「1」+「0」= 10）
    scored = []
    for side, parts in cells:
        chars = "".join(ch for ch, _sc in parts)
        conf = min(sc for _ch, sc in parts)
        scored.append((side, (_normalize(chars), conf)))
    info["candidates"] = [(side, ch, round(sc, 3)) for side, (ch, sc) in scored]
    if not scored:
        return None, 0.0, info

    # ▲ 标在哪个面板下，那个面板的数字就是当前级牌
    if tx is not None:
        centers = {k: (v[0] + v[1]) / 2 for k, v in DIGIT_BOXES.items()}
        want = min(centers, key=lambda k: abs(centers[k] - tx))
        info["panel"] = want
        for side, (ch, sc) in scored:
            if side == want:
                return ch, sc, info
    # 没定位到 ▲ 时不猜。
    # 一开始退回「置信度最高的那个面板」，结果是错的：对方那个数字的匹配置信度
    # 常常比我方高，于是把对方的级别当成了当前级别。级别在一局里是恒定的，
    # 猜错的代价远大于"这次读不出来" —— 交给调用方沿用上一次的可信读数。
    info["panel"] = "unknown"
    return None, max((sc for _s, (_c, sc) in scored), default=0.0), info


def main() -> None:
    global DEBUG_DIR
    DEBUG_DIR = Path("assets/debug/leveldbg")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for path in sys.argv[1:] or ["live/frames"]:
        p = Path(path)
        files = sorted(p.glob("*.png")) if p.is_dir() else [p]
        for f in files[:8]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            ch, sc, info = read_level(img)
            print(f"  {f.name:<22} 级牌={ch}  置信 {sc:.3f}  {info.get('panel','-')}  "
                  f"{info.get('candidates')}")


if __name__ == "__main__":
    main()
