"""量牌面排版：把非白色像素做连通域，看点数/花色相对卡片的偏移。"""
import sys, cv2, numpy as np

def face_layout(img, x1, y1, x2, y2, name):
    c = img[y1:y2, x1:x2]
    h, w = c.shape[:2]
    # 非白 = 不是接近白/灰的像素（牌底是白/浅灰渐变）
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(int); val = hsv[:, :, 2].astype(int)
    mask = ((sat > 60) | (val < 120)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    comps = [(stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4])
             for i in range(1, n) if stats[i, 4] >= 40]
    comps.sort(key=lambda s: (s[1], s[0]))
    print(f"\n=== {name}  卡片 {w}x{h}  (裁剪自 {x1},{y1})")
    for x, y, cw, ch, area in comps[:8]:
        print(f"   相对卡左上 ({x:>3},{y:>3})  尺寸 {cw:>3}x{ch:<3} 面积 {area}")

img = cv2.imread('live/frames/f00028_0021.png')
# 出牌：8♣ 是第 3 张，orig pitch 49，起点 x≈693
face_layout(img, 693 + 98, 354, 693 + 98 + 112, 354 + 148, "出牌 8♣ (第3张, 整张可见)")
face_layout(img, 693, 354, 693 + 112, 354 + 148, "出牌 8♥ (第1张)")
# 手牌：♣Q @(784,564)，卡 132x174，只能看见上面 62px
face_layout(img, 784 - 66, 564 - 87, 784 + 66, 564 - 87 + 70, "手牌 ♣Q (只露顶部 70px)")
# 手牌最下面那张 Q♥ @(785,751)，可见多一点
face_layout(img, 785 - 66, 751 - 87, 785 + 66, 751 + 60, "手牌 ♥Q (最底, 露 147px)")
