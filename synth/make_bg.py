"""从真实截图抠出干净桌面背景（抹掉手牌区域）。

算法：
    逐行处理。对每一行，取横跨手牌区的所有像素，剔除「白色」（那是牌），
    只保留「绿色」像素当作桌面（牌的红色/黑色笔画饱和度或色相都不符合），
    再对它们做线性拟合 —— 线性拟合对文字水印、渐变、少量残留都很稳健。
    得到逐行的桌面色后，再沿 y 做一次中值滤波，最后把这行填满。

    上方区域（玩家头像、按钮等 UI）原样保留 —— 合成图要保留真实 UI，
    否则模型会学到「没有 UI 的图」这个不存在的分布。

用法:
    python synth/make_bg.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "shots"
OUT = ROOT / "assets" / "table_bg"

HAND_X1, HAND_X2 = 190, 1470      # 需要覆盖所有可能牌位的横向范围
HAND_Y1, HAND_Y2 = 465, 896       # 纵向范围（上界避开按钮下沿）
ROW_MEDIAN_K = 3                 # 沿 y 的中值滤波核


def row_table_color(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐行估算桌面底色。

    桌面在水平方向有轻微暗角（边缘偏暗），所以不能只按行取一个常数。
    对每一行，用该行「非牌」的像素做一次 color ~ x 的线性拟合：
    牌盖住中间时，剩下的边缘像素正好能把斜率估出来。

    返回 (a, b)，填充值为 a + b * x。
    """
    x1 = max(0, HAND_X1)
    x2 = min(img.shape[1], HAND_X2)
    band = img[HAND_Y1:HAND_Y2, x1:x2]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_table = (hh >= 35) & (hh <= 110) & (ss >= 55) & (vv >= 60) & (vv <= 218)
    xs = np.arange(x2 - x1, dtype=np.float32)
    rows = band.shape[0]

    a = np.zeros((rows, 3), np.float32)
    b = np.zeros((rows, 3), np.float32)
    valid = np.zeros(rows, bool)

    for i in range(rows):
        m = is_table[i]                      # 只认绿色像素为桌面（牌的笔画是红/黑，不能算）
        if m.sum() < 40 or (xs[m].max() - xs[m].min()) < 200:
            continue                         # 该行几乎被牌占满，估不出斜率

        # 稳健两点定线：把该行绿色像素按 x 分成左右两半，各取中位数。
        # 牌盖住中间时，剩下的正是两侧边缘像素 —— 斜率就是这么估出来的。
        # 用中位数而不是最小二乘，水印文字之类的离群点带不偏结果。
        mx = xs[m]
        mid = np.median(mx)
        lm, rm = mx <= mid, mx > mid
        if lm.sum() < 15 or rm.sum() < 15:
            continue
        for c in range(3):
            yl = float(np.median(band[i][m, c][lm]))
            yr = float(np.median(band[i][m, c][rm]))
            xl, xr = float(mx[lm].mean()), float(mx[rm].mean())
            slope = (yr - yl) / max(1e-3, xr - xl)
            b[i, c] = slope
            a[i, c] = yl - slope * xl
        valid[i] = True

    if not valid.any():
        raise RuntimeError("没有任何一行能估出桌面底色，截图可能全是牌")

    # 无效行用最近的有效行补上，再沿 y 做中值滤波抹掉文字/UI 造成的跳变
    idx = np.where(valid)[0]
    for i in range(rows):
        if not valid[i]:
            j = idx[np.argmin(np.abs(idx - i))]
            a[i], b[i] = a[j], b[j]

    k = ROW_MEDIAN_K
    pad = np.pad(np.hstack([a, b]), ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.zeros_like(pad[:rows])
    for c in range(6):
        for i in range(rows):
            sm[i, c] = np.median(pad[i:i + k, c])
    return sm[:, :3], sm[:, 3:]


def find_clean_patch(img: np.ndarray, size: int = 192, step: int = 32) -> tuple[int, int]:
    """在桌面上找一块最干净的方形区域（全是桌面色、没有 UI 和牌）。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (hh >= 35) & (hh <= 110) & (ss >= 55) & (vv >= 60) & (vv <= 218)

    best, best_score = (0, 0), -1
    # 只在桌面上半部找，避开手牌区
    for y in range(170, min(470, h - size), step):
        for x in range(20, w - size, step):
            g = green[y:y + size, x:x + size]
            if g.mean() < 0.93:          # 必须几乎全是桌面
                continue
            v = img[y:y + size, x:x + size].astype(np.float32).std()
            score = -v                      # 越平滑越好（避开水印文字）
            if score > best_score:
                best_score, best = score, (x, y)
    if best_score < 0:
        return (0, 0)
    return best


def table_grain(img: np.ndarray, patch: tuple[int, int], size: int, sigma: float = 7.0):
    """取干净桌面块的高频纹理（原图 - 低通），用作填充区的质感。"""
    x, y = patch
    if x == 0 and y == 0:
        return None
    box = img[y:y + size, x:x + size].astype(np.float32)
    low = cv2.GaussianBlur(box, (0, 0), sigma)
    return box - low


def tile_grain(grain: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """把纹理块平铺到 h x w，随机偏移 + 随机翻转，避免出现明显重复图案。"""
    gh, gw = grain.shape[:2]
    ox, oy = rng.integers(0, gw), rng.integers(0, gh)
    fy = rng.random() < 0.5
    fx = rng.random() < 0.5
    big = np.tile(grain, (h // gh + 3, w // gw + 3, 1))
    big = big[oy:oy + h, ox:ox + w]
    if fy:
        big = big[::-1]
    if fx:
        big = big[:, ::-1]
    return big


def clean(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if img.shape[1] < HAND_X2 or img.shape[0] < HAND_Y2:
        raise RuntimeError(f"截图为 {img.shape[1]}x{img.shape[0]}，小于手牌区范围")

    a, b = row_table_color(img)
    x1, x2 = HAND_X1, HAND_X2
    y1, y2 = HAND_Y1, HAND_Y2

    xs = np.arange(x2 - x1, dtype=np.float32)[None, :, None]
    fill = a[:, None, :] + b[:, None, :] * xs
    noise = rng.normal(0, 1.0, fill.shape).astype(np.float32)
    filled = fill + noise

    # 叠上从真实桌面取的高频纹理，否则填充区会和周围桌面在质感上明显不同
    patch = find_clean_patch(img)
    if patch != (0, 0):
        grain = table_grain(img, patch, 192)
        if grain is not None:
            filled = filled + tile_grain(grain, y2 - y1, x2 - x1, rng)
            print(f"    纹理源: x={patch[0]} y={patch[1]}")

    filled = np.clip(filled, 0, 255).astype(np.float32)

    # 边缘羽化：填充区四周 24px 内与原图渐变过渡，消除矩形硬边。
    # 边界那一圈在原图里基本是桌面（牌只占中间），所以过渡是干净的。
    F = 24
    wgt = np.ones((y2 - y1, x2 - x1), np.float32)
    ramp = np.linspace(0.0, 1.0, F, dtype=np.float32)
    wgt[:F, :] *= ramp[:, None]
    wgt[:, :F] *= ramp[None, :]
    wgt[:, -F:] *= ramp[::-1][None, :]
    wgt[-F:, :] *= ramp[::-1][:, None]
    w3 = wgt[:, :, None]

    orig = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(filled * w3 + orig * (1 - w3), 0, 255).astype(np.uint8)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shots = sorted(p for p in SHOTS.glob("*.png") if not p.name.startswith("_"))
    if not shots:
        raise SystemExit(f"[FAIL] {SHOTS} 下没有截图")

    for p in shots:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  跳过（读不了）: {p.name}")
            continue
        bg = clean(img, rng)
        cv2.imwrite(str(OUT / p.name), bg)
        print(f"  {p.name}  {img.shape[1]}x{img.shape[0]}")

    print(f"\n背景已存到: {OUT}  (共 {len(list(OUT.glob('*.png')))} 张)")


if __name__ == "__main__":
    main()
