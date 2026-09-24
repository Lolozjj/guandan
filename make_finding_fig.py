"""生成结论图：出牌区识别失败的根因（真实 vs 旧合成 vs 新合成）。"""
import sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, 'synth')
from compose import render_card, _paste, CARD_W, CARD_H

FONT = 'C:/Windows/Fonts/msyh.ttc'
def f(sz): return ImageFont.truetype(FONT, sz)

CW, CH = 106, 140
def card(cls, style):
    c = np.full((CARD_H, CARD_W, 3), (120, 160, 130), np.uint8)
    _paste(c, render_card(cls, level='2', style=style), 0, 0, 1.0)
    return cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA)

img = cv2.imread('live/frames/f00028_0021.png')
real = img[355:355 + CH, 893:893 + CW]          # 完全可见的那张 8♠
old = card('S8', 'hand')                         # 旧：手牌排版
new = card('S8', 'played')                       # 新：出牌排版

S = 3
tiles = [(real, "真实画面里的出牌", (40, 40, 40)),
         (old,  "旧合成器（点数/花色并排）", (150, 30, 30)),
         (new,  "新合成器（花色在点数下方）", (20, 110, 40))]
pad, top, lab = 26, 96, 54
W = pad + len(tiles) * (CW * S + pad)
H = top + CH * S + pad + 70
canvas = Image.new('RGB', (W, H), (250, 250, 250))
d = ImageDraw.Draw(canvas)
d.text((pad, 22), "出牌区认不出来的根因：合成数据里画的牌面和游戏里长得不一样",
       font=f(30), fill=(20, 20, 20))
for i, (t, label, col) in enumerate(tiles):
    x = pad + i * (CW * S + pad)
    big = cv2.resize(t, (CW * S, CH * S), interpolation=cv2.INTER_NEAREST)
    canvas.paste(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), (x, top))
    d.rectangle([x, top, x + CW * S, top + CH * S], outline=col, width=4)
    d.text((x, top + CH * S + 10), label, font=f(24), fill=col)
canvas.save('audit/结论图.png')
print('saved audit/结论图.png', canvas.size)
