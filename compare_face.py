"""真实牌面 vs 合成牌面：手牌、出牌两套渲染并排看。"""
import sys, cv2, numpy as np
sys.path.insert(0, 'synth')
from compose import render_card, _paste, CARD_W, CARD_H

def synth(cls):
    c = np.full((CARD_H, CARD_W, 3), (120, 160, 130), np.uint8)
    _paste(c, render_card(cls), 0, 0, 1.0)
    return c

def label(txt, w):
    bar = np.full((30, w, 3), 255, np.uint8)
    cv2.putText(bar, txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return bar

img = cv2.imread('live/frames/f00028_0021.png')
# 真实出牌（f00028 第5张 8♠，完全可见）：左 889 顶 354 宽 112 高 148
real_play = cv2.resize(img[354:354+148, 889:889+112], (CARD_W, CARD_H),
                       interpolation=cv2.INTER_CUBIC)
# 真实手牌（f00028 最底层 ♥Q @(785,751)，卡 132x174，中心对齐）
real_hand = img[751-87:751+87, 785-66:785+66].copy()

panels = [("真实·出牌区 8S", real_play),
          ("真实·手牌 HQ", real_hand),
          ("合成 HQ", synth("HQ")),
          ("合成 S8", synth("S8"))]
gap = np.full((CARD_H + 30, 16, 3), 255, np.uint8)
row = []
for i, (t, p) in enumerate(panels):
    if i: row.append(gap)
    row.append(np.vstack([label(t, CARD_W), p]))
sheet = np.hstack(row)
cv2.imwrite('audit/face_cmp4.png',
            cv2.resize(sheet, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST))
print('saved audit/face_cmp4.png', sheet.shape)
