"""修 table_by_player 的归属逻辑：改成 x 优先。

用户实测报的 bug：机器人1 打了 8 张 9，游戏把它摆成两行
（上行 y≈205、下行 y≈312）。原来的归属只按「卡顶边 y」分三条带，
y=205 正好撞上「队友」区的位置（199），于是上行被整排判给了队友。

实测四家的 x 范围（用标注框中心的 x，四家互不重叠）：

    机器人1    222.5 ~  572.5   （左缘 204 起，向右排；最多 8 张）
    队友/我    636.5 ~  986.5   （中心 846；最多 8 张）
    机器人3   1047.5 ~ 1397.5   （右缘 1485 起，向左排；最多 8 张）

所以先用 x 把「左 / 中间 / 右」分开（三段之间还有 30px 余量），
只有落在中间那段的才需要用 y 区分队友和自己。
两行摆法只会上下错开、x 范围不变，所以这个改法天然不受影响。
"""
from pathlib import Path

p = Path('predict_cards.py')
s = p.read_text(encoding='utf-8')

old = """    out: dict[str, list[tuple]] = {k: [] for k in PLAYER_ZONES}
    for d in table:
        cy = d[2] - PLAY_MARK_CY + CARD_H * 0.804 / 2     # 卡的竖向中心
        # 三档卡心 y：对家 199+70=269、左右 312+70=382、自己 355+70=425
        # 分界取中点 326 和 404。后两档只差 43px，是全流程最紧的一处阈值，
        # 出牌区归一化时要留意（已在 generate 里把 y 抖动限制在 ±8 以内）。
        if cy < 326:
            who = "队友"
        elif cy > 404:
            who = "我"
        else:
            who = "机器人1" if d[1] < 846 else "机器人3"
        out[who].append(d)"""
new = """    out: dict[str, list[tuple]] = {k: [] for k in PLAYER_ZONES}
    for d in table:
        cx = d[1]                       # 标注框中心的 x
        if cx < X_LEFT_MAX:
            who = "机器人1"
        elif cx > X_RIGHT_MIN:
            who = "机器人3"
        else:
            # 只剩「队友 / 我」，这时 x 分不开了，用卡心 y 分
            cy = d[2] - PLAY_MARK_CY + CARD_H * 0.804 / 2
            who = "队友" if cy < 326 else "我"
        out[who].append(d)"""
assert old in s
s = s.replace(old, new)

old = """# 检出框中心 -> 卡的左上角：标注框是「点数+小花色」那块，相对卡左上角
# 是 (3, 2)~(43, 88)，取中心 (23, 45)；实测出牌缩放恒定在 0.804 左右。
PLAY_MARK_CX, PLAY_MARK_CY = 23 * 0.804, 45 * 0.804"""
new = """# 检出框中心 -> 卡的左上角：标注框是「点数+小花色」那块，相对卡左上角
# 是 (3, 2)~(43, 88)，取中心 (23, 45)；实测出牌缩放恒定在 0.804 左右。
PLAY_MARK_CX, PLAY_MARK_CY = 23 * 0.804, 45 * 0.804

# 归属判据的 x 分界（标注框中心的 x，按画面宽 1698 定）。
# 四家的 x 范围实测互不重叠：机器人1 222~572、队友/我 636~986、机器人3 1047~1397。
# 分界取两段之间的中点，两边各留约 30px 余量。
X_LEFT_MAX = 604.0
X_RIGHT_MIN = 1017.0"""
assert old in s
s = s.replace(old, new)
Path('predict_cards.py').write_text(s, encoding='utf-8')
print('table_by_player 已改成 x 优先')
