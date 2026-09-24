"""给 predict_cards.py 加「UI 按钮遮挡容忍」。

背景（用户实测报的 bug）：出牌区的牌面排版是「点数在上、花色在下」。
轮到我出牌时，「不出 / 提示 / 出牌」三个按钮横在屏幕中间，会盖住左右两家
伸过来的牌的**花色符号**。模型看到「一张没有花色的 6」就不敢认 ——
实测那 5 张 6 里被盖住的那张置信度只有 0.108，同组另外 4 张是 0.74~0.81，
于是 5 张 6 只报出 4 张。

修法：只在「轮到我」（按钮确实存在）时，只对**落在按钮矩形内**的检出
把置信度门槛降到 0.10。别的地方门槛不动，所以整体误检率不受影响。
"""
from pathlib import Path

p = Path('predict_cards.py')
s = p.read_text(encoding='utf-8')

anchor = "def split_hand_table("
block = '''# ---------------------------------------------------------------------------
# UI 按钮遮挡容忍
# ---------------------------------------------------------------------------
# 「不出 / 提示 / 出牌」三个按钮的外接框，按画面宽 1698 量出来的。
# 只有轮到我出牌时它们才出现，且会盖住从左/右两家伸进中间那块的牌。
UI_BUTTON_BOXES_1698 = ((388, 620), (813, 1054), (1082, 1300))   # 各自 (x1, x2)
UI_BAND_Y_1698 = (330, 470)          # 按钮的竖向范围（留了余量）
UI_BASE_W = 1698.0
CONF_UNDER_UI = 0.10                 # 被按钮盖住时放宽到这个门槛（实测够用）


def under_ui_button(cx: float, cy: float, img_w: int) -> bool:
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
    return out


def split_hand_table('''
assert anchor in s
s = s.replace(anchor, block, 1)
p.write_text(s, encoding='utf-8')
print('predict_cards.py 已加 UI 遮挡容忍')
