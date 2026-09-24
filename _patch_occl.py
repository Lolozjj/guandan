"""把「按钮遮挡容忍」推广成「通用遮挡容忍」。

原来只判「落在 UI 按钮矩形内」。但实测还有第二种遮挡：
自己的手牌如果有高列（比如 J×6 顶到 y=404），会盖住自己刚出的牌的花色符号 ——
牌面排版是「点数在上、花色在下」，花色正好在下半部分，最容易被压住。
用户报的「打了 4 个 5 只认出 2 个」就是这个。

改成：把「按钮矩形」和「每一张检出的手牌的外接框」都当成遮挡物，
任何标注框和遮挡物相交的出牌，置信度门槛都从常态降到 0.10。
"""
from pathlib import Path

p = Path('predict_cards.py')
s = p.read_text(encoding='utf-8')

old = '''def under_ui_button(cx: float, cy: float, img_w: int) -> bool:
    """检出框中心是否落在 UI 按钮上。坐标按画面宽度等比换算。"""
    sc = img_w / UI_BASE_W
    if not (UI_BAND_Y_1698[0] * sc <= cy <= UI_BAND_Y_1698[1] * sc):
        return False
    return any(x1 * sc <= cx <= x2 * sc for x1, x2 in UI_BUTTON_BOXES_1698)


def tolerate_ui_occlusion(dets: list[tuple], img_w: int, mine: bool,
                          conf: float) -> list[tuple]:
    """按钮盖住的检出放低门槛，其余保持原门槛。

    mine=False（没轮到我）时按钮不存在，什么都不放宽。
    调用方要用 `min(conf, CONF_UNDER_UI)` 去跑模型，否则低分框根本出不来。
    """
    if not mine:
        return [d for d in dets if d[3] >= conf]
    out = []
    for d in dets:
        if d[3] >= conf:
            out.append(d)
        elif d[3] >= CONF_UNDER_UI and under_ui_button(d[1], d[2], img_w):
            out.append(d)
    return out'''

new = '''# 手牌检出的标注框相对卡左上角是 (8,1)~(101,67)，中心 (54.5,34)；
# 但整张牌是 132x174，遮挡要用整张牌的范围算，不能用标注框。
_HAND_BOX = (-54.5, -34.0, 77.5, 140.0)


def markup_box(cx: float, cy: float) -> tuple[float, float, float, float]:
    """一张出牌的标注框（含一点余量），用来判它有没有被盖住。"""
    return (cx - 19, cy - 37, cx + 19, cy + 37)


def _overlaps(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def occluders(img_w: int, mine: bool, hand_dets: list[tuple]) -> list[tuple]:
    """当前画面里会盖住出牌的矩形：UI 按钮 + 自己手牌占据的范围。"""
    out = []
    if mine:
        sc = img_w / UI_BASE_W
        y1, y2 = UI_BAND_Y_1698[0] * sc, UI_BAND_Y_1698[1] * sc
        out += [(x1 * sc, y1, x2 * sc, y2) for x1, x2 in UI_BUTTON_BOXES_1698]
    for h in hand_dets:
        dx1, dy1, dx2, dy2 = _HAND_BOX
        out.append((h[1] + dx1, h[2] + dy1, h[1] + dx2, h[2] + dy2))
    return out


def tolerate_occlusion(dets: list[tuple], conf: float,
                       occl: list[tuple]) -> list[tuple]:
    """被遮挡物盖住的检出放低门槛，其余保持原门槛。

    调用方要用 `min(conf, CONF_UNDER_UI)` 去跑模型，否则低分框根本出不来。
    """
    if not occl:
        return [d for d in dets if d[3] >= conf]
    out = []
    for d in dets:
        if d[3] >= conf:
            out.append(d)
            continue
        if d[3] < CONF_UNDER_UI:
            continue
        mb = markup_box(d[1], d[2])
        if any(_overlaps(mb, o) for o in occl):
            out.append(d)
    return out


def tolerate_ui_occlusion(dets: list[tuple], img_w: int, mine: bool,
                          conf: float) -> list[tuple]:
    """旧接口（只考虑 UI 按钮），保留给不需要手牌信息的调用方。"""
    return tolerate_occlusion(dets, conf, occluders(img_w, mine, []))'''
assert old in s
s = s.replace(old, new)
Path('predict_cards.py').write_text(s, encoding='utf-8')
print('已推广成通用遮挡容忍')
