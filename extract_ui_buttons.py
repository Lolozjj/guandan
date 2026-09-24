"""从真实截图里抠出「不出 / 提示 / 出牌」三个按钮，存成带 alpha 的 PNG。

为什么要抠真实的：出牌区牌面的排版是「点数在上、花色在下」，轮到我出牌时
这三个按钮横在屏幕中间，会把左右两家伸进来的牌的花色符号盖掉 ——
模型看到「一张没有花色的 6」就不敢认了。实测第 5 张 6♠ 置信度只有 0.108
（同组其他 4 张是 0.74~0.81），被阈值过滤掉，于是 5 张 6 只报出 4 张。

要把这种情况加进合成数据，就得有真实的按钮图。

alpha 的取法：按钮是蓝/橙/白/黄，桌面是绿色（hue 85~95），
所以「在按钮框里、且不是桌面绿」的就是按钮本体（含它那圈光晕）。
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = Path('C:/Users/17837/AppData/Local/Temp/pasted-image-2.png')
OUT = ROOT / 'assets' / 'ui_buttons'

# 从截图里量出来的按钮外接框 (x1, y1, x2, y2)，留了点余量把光晕包进去
BUTTONS = {
    'buchu': (388, 367, 620, 436),      # 不出
    'tishi': (813, 367, 1054, 436),     # 提示
    'chupai': (1082, 367, 1300, 436),   # 出牌
    'countdown': (648, 316, 810, 470),  # 倒计时的小鸡（也会盖住牌）
}


def extract(img: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    table_green = (h >= 78) & (h <= 100) & (s > 60) & (v > 90)
    alpha = np.where(table_green, 0, 255).astype(np.uint8)
    # 补掉内部的小空洞（文字边缘、按钮上的高光）
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 只保留最大的那块连通域（把旁边误收进来的小碎片去掉）
    n, lab, st, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n > 1:
        keep = 1 + int(np.argmax(st[1:, 4]))
        alpha = np.where(lab == keep, 255, 0).astype(np.uint8)
    out = np.dstack([crop, alpha])
    return out


def main() -> None:
    img = cv2.imread(str(SRC))
    if img is None:
        raise SystemExit('读不到源截图: %s' % SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, box in BUTTONS.items():
        rgba = extract(img, box)
        p = OUT / ('%s.png' % name)
        cv2.imwrite(str(p), rgba)
        cover = float((rgba[:, :, 3] > 0).mean())
        print('%-10s %s  尺寸 %dx%d  不透明占比 %.2f'
              % (name, p.name, rgba.shape[1], rgba.shape[0], cover))
    print('输出目录:', OUT)


if __name__ == '__main__':
    main()
