"""用「检出框的形状」加强手牌/出牌的划分 —— 这次是真·根本性的判据。

用户报：出了 5♥6♠7♣8♠9♦ 的顺子，只认出 1 张（♥5）。

实测发现：**模型 5 张全检出了**（♥5 0.77、其余 0.38~0.48），
但其中 4 张被 split_hand_table 判成了手牌。原因是它们的位置正好撞上手牌栅格：
- 实时窗口 1698x1009 比训练素材高 9px，手牌层级整体下移，
  最上面那层的标注框中心落在 y≈385，而出牌的「我」区中心在 y≈391
  —— 差 6px，落进了 0.12 的容差里，于是「在层级上」这一条成立了。
- x 方向 pitch 50 的出牌用 pitch 95 的手牌栅格去套，也会撞上某些列。

**真正可靠的判据是检出框的形状**，因为手牌和出牌的标注框本来就是两种形状：

    手牌标注框  93 x 66   宽高比 1.45（横的）
    出牌标注框  40 x 86   宽高比 0.47（竖的）

实测这一帧：出牌那 5 张框是 32x73 / 36x54 / 32x53 / 33x51 / 30x50
（宽高比 0.44~0.66），手牌框是 93x64 / 91x63（1.43~1.45）—— 中间空得很开。

所以给 dets 元组加上 (w, h)，split_hand_table 里加一条「手牌的框必须是横的」。
元组长度兼容：只有 4 个元素的（老调用方）就跳过这条判据。
"""
from pathlib import Path

# --- predict_cards.py ---
p = Path('predict_cards.py')
s = p.read_text(encoding='utf-8')

old = """    cols_with_bottom = {col_of(d) for d in dets if abs(d[2] - BOTTOM_CY) < 20}

    hand, table = [], []
    for d in dets:
        if col_of(d) in cols_with_bottom and on_level(d):
            hand.append(d)
        else:
            table.append(d)
    return hand, table"""
new = """    cols_with_bottom = {col_of(d) for d in dets if abs(d[2] - BOTTOM_CY) < 20}

    hand, table = [], []
    for d in dets:
        if col_of(d) in cols_with_bottom and on_level(d) and looks_like_hand_box(d):
            hand.append(d)
        else:
            table.append(d)
    return hand, table"""
assert old in s
s = s.replace(old, new)

old = """ON_LEVEL_TOL = 0.12     # 检出中心离手牌层级的相对容差"""
new = """ON_LEVEL_TOL = 0.12     # 检出中心离手牌层级的相对容差

# 手牌和出牌的标注框形状完全不同，这是比位置可靠得多的判据：
#     手牌  93 x 66   宽高比 1.41（横的，因为手牌只露出最上面一条）
#     出牌  40 x 86   宽高比 0.47（竖的，因为出牌露出的是左边一条）
# 实测真实帧上：出牌框宽高比 0.44~0.66，手牌框 1.43~1.45，中间空得很开。
HAND_BOX_ASPECT = 1.0
# 元组里带框尺寸时才算形状；老调用方给 4 元组就跳过这条
_W_IDX, _H_IDX = 4, 5


def looks_like_hand_box(d: tuple) -> bool:
    \"\"\"这个检出框像不像手牌（横的）。没带框尺寸就返回 True（不拦）。\"\"\"
    if len(d) <= _H_IDX or d[_H_IDX] <= 0:
        return True
    return d[_W_IDX] / d[_H_IDX] >= HAND_BOX_ASPECT"""
assert old in s
s = s.replace(old, new)

# predict_cards.main 里造 dets 时带上框尺寸
old = """    dets = [(names[int(b.cls)],
             float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
             float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
             float(b.conf))
            for b in res.boxes]"""
new = """    dets = [(names[int(b.cls)],
             float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
             float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
             float(b.conf),
             float(b.xyxy[0][2] - b.xyxy[0][0]),
             float(b.xyxy[0][3] - b.xyxy[0][1]))
            for b in res.boxes]"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('predict_cards.py 已加框形状判据')

# --- live/main.py ---
p = Path('live/main.py')
s = p.read_text(encoding='utf-8')
old = """        return [(names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf)) for b in res.boxes]"""
new = """        return [(names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf),
                 float(b.xyxy[0][2] - b.xyxy[0][0]),
                 float(b.xyxy[0][3] - b.xyxy[0][1])) for b in res.boxes]"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('live/main.py 已带上框尺寸')

# --- eval_table.py ---
p = Path('eval_table.py')
s = p.read_text(encoding='utf-8')
old = """        return [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf)) for b in res.boxes]"""
new = """        return [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf),
                 float(b.xyxy[0][2] - b.xyxy[0][0]),
                 float(b.xyxy[0][3] - b.xyxy[0][1])) for b in res.boxes]"""
assert old in s
s = s.replace(old, new)
# 模板匹配那条路径也补上等价尺寸（模板就是标注框，尺寸已知）
old = """        def f(img):
            return [(cls, x + PLAY_MARK_CX, y + PLAY_MARK_CY, score)
                    for cls, _lv, score, x, y in detect(img, tpl, thr=args.conf)]"""
new = """        def f(img):
            # 模板 = 出牌标注框，尺寸是已知的常量
            return [(cls, x + PLAY_MARK_CX, y + PLAY_MARK_CY, score,
                     PLAY_MARK_CX * 2, PLAY_MARK_CY * 2)
                    for cls, _lv, score, x, y in detect(img, tpl, thr=args.conf)]"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('eval_table.py 已带上框尺寸')
