"""把截图喂给模型，输出识别出的牌列表。

这是整条链路最终的产出物：检测框 -> 牌列表。

用法:
    python predict_cards.py <权重.pt> <图片> [--imgsz 960] [--conf 0.25] [--save 输出图]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
from layout import CLASSES, CARD_W, CARD_H, PITCH_X, PITCH_Y  # noqa: E402

# 牌序（掼蛋：大王 > 小王 > 级牌 > A > K > ... > 3）
RANK_ORDER = "A23456789TJQK"
SUIT_CN = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
DEDUP_PX = 40          # 中心距小于此值视为同一张牌，只保留置信度最高的
BOTTOM_Y = 690         # 最底行牌的中心 y — 手牌按列底部对齐，都落在这里
COL_PX = 47            # x 相差小于此值算同一列


def card_value(cls: str, level: str) -> tuple:
    """排序键：越小越靠前。"""
    if cls == "JOKER_B":
        return (0, 0)
    if cls == "JOKER_S":
        return (0, 1)
    rank, suit = cls[1:], cls[0]
    if rank == level:
        return (1, "SCDH".index(suit))
    return (2 + "AKQJT98765432".index(rank), "SCDH".index(suit))


def display(cls: str) -> str:
    if cls == "JOKER_B":
        return "大王"
    if cls == "JOKER_S":
        return "小王"
    rank, suit = cls[1:], cls[0]
    return SUIT_CN[suit] + ("10" if rank == "T" else rank)


def dedup(dets: list[tuple]) -> list[tuple]:
    """同一位置保留置信度最高的框（类别不同的重复框很常见）。"""
    kept: list[tuple] = []
    for d in sorted(dets, key=lambda t: -t[3]):
        if all((d[1] - k[1]) ** 2 + (d[2] - k[2]) ** 2 > DEDUP_PX ** 2 for k in kept):
            kept.append(d)
    return kept


ON_LEVEL_TOL = 0.12     # 检出中心离手牌层级的相对容差

# 手牌和出牌的标注框形状完全不同，这是比位置可靠得多的判据：
#     手牌  93 x 66   宽高比 1.41（横的，因为手牌只露出最上面一条）
#     出牌  40 x 86   宽高比 0.47（竖的，因为出牌露出的是左边一条）
# 实测真实帧上：出牌框宽高比 0.44~0.66，手牌框 1.43~1.45，中间空得很开。
HAND_BOX_ASPECT = 1.0
# 元组里带框尺寸时才算形状；老调用方给 4 元组就跳过这条
_W_IDX, _H_IDX = 4, 5


def looks_like_hand_box(d: tuple) -> bool:
    """这个检出框像不像手牌（横的）。没带框尺寸就返回 True（不拦）。"""
    if len(d) <= _H_IDX or d[_H_IDX] <= 0:
        return True
    return d[_W_IDX] / d[_H_IDX] >= HAND_BOX_ASPECT
# 只可能是手牌的竖向范围：桌面四个区里最靠下的是「我」（卡顶边 355、
# 标注框中心 ≈391），所以 y 大于这个值的一定是手牌。
HAND_Y_MIN = 420
# 手牌最底行（标注框中心）在 747 左右，随窗口高度最多浮动几十像素。
# 比这个再往下的一定不是手牌 —— 实测低门槛跑模型时左下角头像区会冒出
# y≈920 的垃圾框，它会成为 max(low) 把锚点整个带跑（整手牌错位、判成桌面牌）。
HAND_Y_MAX = 810


def _fit_phase(vals: list[float], period: float, tol: float,
               center: float | None = None,
               half: float | None = None) -> float:
    """在**一个完整周期**内找栅格相位，目标：落在容差内的点最多，其次残差和最小。

    为什么要搜整个周期：早先的版本是在 `min(vals) ± 20` 里搜（周期是 62），
    只覆盖了 2/3 的相位空间。旧模型检不到桌面牌时没问题；guandan7 能检出出牌后，
    桌面牌（y≈349）把 `min(vals)` 拉偏、最优相位落到了搜索范围外，
    结果**整手牌都差 9px 落在线外**，手牌被整批判成了桌面牌。

    用「内点数优先」而不是「残差和」也很关键：残差和会让大量「差一点点」的点
    压过少量「完全对齐」的点 —— 这正是上面那次翻车的形式。
    """
    # 坐标一定要先转成 python float：命令行那条路径传进来的是 CUDA tensor，
    # np.arange(tensor, ...) 会直接抛 "can't convert cuda:0 device type tensor"。
    vals = [float(v) for v in vals]
    if not vals:
        return 0.0
    # center/half 给了就在指定范围内搜（已知大致位置时用，避免被离群值带跑）；
    # 没给就搜整个周期。
    c0 = float(center) if center is not None else min(vals)
    half = float(half) if half is not None else period / 2
    best, best_key = c0, None
    for cand in np.arange(c0 - half, c0 + half, 0.25):
        r = [abs((v - cand) / period - round((v - cand) / period)) for v in vals]
        key = (-sum(1 for x in r if x < tol), sum(r))
        if best_key is None or key < best_key:
            best_key, best = key, cand
    return best


# ---------------------------------------------------------------------------
# UI 按钮遮挡容忍
# ---------------------------------------------------------------------------
# 「不出 / 提示 / 出牌」三个按钮的外接框，按画面宽 1698 量出来的。
# 只有轮到我出牌时它们才出现，且会盖住从左/右两家伸进中间那块的牌。
UI_BUTTON_BOXES_1698 = (
    (388, 620),      # 不出
    (813, 1054),     # 提示
    (1082, 1300),    # 出牌
    (648, 810),      # 倒计时的小鸡（轮到我时也在，会盖住中间区）
)
UI_BAND_Y_1698 = (330, 470)          # 按钮的竖向范围（留了余量）
UI_BASE_W = 1698.0
CONF_UNDER_UI = 0.10                 # 被按钮盖住时放宽到这个门槛（实测够用）


# 手牌检出的标注框相对卡左上角是 (8,1)~(101,67)，中心 (54.5,34)；
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
    return tolerate_occlusion(dets, conf, occluders(img_w, mine, []))


def split_hand_table(dets: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """把手牌和桌面出牌区分开。

    手牌是严格的栅格：x 落在 x0 + 95k 上、y 落在固定的层级上（步长 62），
    而且每一列的底部都对齐到最底行。所以判据是三条同时满足：
        1. x 在栅格上
        2. y 在层级上
        3. 该列有牌落在最底行

    早先的版本只按 x 聚簇（47px 窗口）分列、再看该列有没有底牌，
    结果是桌面牌一旦 x 和某手牌列接近就被并进那一列、整列都算成了手牌
    （实测抽样 25 帧，桌面牌被识别成 0 个）。

    注意：y 阈值本身分不开 —— 手牌最上一层在 y=342，而桌面出牌区在 y=404~543，
    纵向是重叠的。
    """
    if not dets:
        return [], []

    PITCH_X, STEP = 95, 62
    BOTTOM_CY = 713 + 34          # 最底行牌的中心 y

    # x 栅格起点要拟合（手牌整手在画面里居中，起点随牌数变）。
    best_x0 = _fit_phase([d[1] - 54 for d in dets], PITCH_X, 0.12)

    # y 层级的锚点：**只拿下半部分的检出**去拟合。
    #
    # 为什么不能拿全部检出拟合：桌面出牌区的 y 都在 391 以下，一旦混进来就会
    # 把锚点带偏 —— 实测旧版就是这样，整手牌差 9px 落在线外、被整批判成桌面牌。
    # 为什么不能钉死成一个常数：游戏窗口大小不同，手牌整体位置会跟着变。
    # 实测 1009 高的窗口比 1000 高的低 9px，最下面几排就够偏出容差了
    # （`live/live_shot.png` 就是这么翻的车）。
    #
    # y >= 420 的区域里**只可能出现手牌**（桌面四个区最下面的是「我」区，
    # 检出框中心 y≈391），所以拿这部分定锚点既干净又能自适应窗口尺寸。
    #
    # 锚点 = 这堆里**最靠下**的那张。手牌是底部对齐的，底行一定是 y 最大的那行，
    # 取 max 而不是平均，才不会被某个偏高的误检带跑。然后在它附近微调相位。
    # 剩最后 1 张牌时这里只有 1 个点，也必须能工作（实测踩过：退回固定值就把
    # 那张手牌判成了桌面牌）。
    low = [d[2] for d in dets if HAND_Y_MIN <= d[2] <= HAND_Y_MAX]
    if low:
        seed = max(low)
        best_y0 = _fit_phase(low, STEP, ON_LEVEL_TOL, center=seed, half=12.0)
    else:
        best_y0 = BOTTOM_CY

    def col_of(d):
        return round((d[1] - 54 - best_x0) / PITCH_X)

    def on_level(d):
        return abs((d[2] - best_y0) / STEP - round((d[2] - best_y0) / STEP)) < ON_LEVEL_TOL

    # 3) 哪些列有底牌
    cols_with_bottom = {col_of(d) for d in dets if abs(d[2] - BOTTOM_CY) < 20}

    hand, table = [], []
    for d in dets:
        if looks_like_hand_box(d):
            # 框是横的（宽高比 >= 1）-> 只可能是手牌，不可能是出牌
            # （出牌的标注框是竖的）。落在手牌栅格上才算手牌；
            # 落不上的是头像旁边那种「级牌」小图标之类的干扰，直接丢掉，
            # 绝不能当出牌 —— 实测它们是误报的主要来源。
            if col_of(d) in cols_with_bottom and on_level(d):
                hand.append(d)
        else:
            table.append(d)
    return hand, table


# 出牌区的四个位置（画面坐标，牌**左上角**）。2026-09-23 在 337 帧真实画面上实测：
#
#     左家  左缘 204、  y 312   （从左边向右排）
#     右家  右缘 1485、 y 312   （从右边向左排）
#     对家  中心 846、  y 199
#     自己  中心 846、  y 355   （几乎不会和对家同时出现：117 帧里只有 2 帧）
#
# 归属判定用「卡顶边 y」分三条带（199 / 312 / 355，用中点 291 和 370 分开），
# 中间那条带再按 x 分左右。**不能用最近的区域中心**：一手 8 张的牌横跨 400px，
# 两边最外的牌离本区中心比离邻区中心还远，会被分错。
PLAYER_ZONES = {
    "队友":   (846, 199),
    "机器人1": (204, 312),
    "机器人3": (1485, 312),
    "我":     (846, 355),
}

# 检出框中心 -> 卡的左上角：标注框是「点数+小花色」那块，相对卡左上角
# 是 (3, 2)~(43, 88)，取中心 (23, 45)；实测出牌缩放恒定在 0.804 左右。
PLAY_MARK_CX, PLAY_MARK_CY = 23 * 0.804, 45 * 0.804

# 归属判据的 x 分界（标注框中心的 x，按画面宽 1698 定）。
# 四家的 x 范围实测互不重叠：机器人1 222~572、队友/我 636~986、机器人3 1047~1397。
# 分界取两段之间的中点，两边各留约 30px 余量。
X_LEFT_MAX = 604.0
X_RIGHT_MIN = 1017.0


def table_by_player(table: list[tuple]) -> dict[str, list[tuple]]:
    """把出牌区的牌按位置归到四家。

    出牌区在画面上的四个位置是固定的，按上面那套带划分归属。
    注意有动画帧会认不准，调用方应先确认画面稳定。

    检出框中心要减掉 Play 标记框相对卡左上角的偏移，换成「卡顶边 y」再分带，
    否则 312 和 355 这两条带只差 43px、会被框中心的偏移搞混。
    """
    out: dict[str, list[tuple]] = {k: [] for k in PLAYER_ZONES}
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
        out[who].append(d)
    for k in out:
        out[k].sort(key=lambda d: (d[2], d[1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("image")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--level", default="2", help="当前级牌点数，影响排序")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.model.names = {i: c for i, c in enumerate(CLASSES)}
    names = model.names

    res = model.predict(source=args.image, imgsz=args.imgsz, conf=args.conf,
                        verbose=False)[0]
    dets = [(names[int(b.cls)],
             float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
             float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
             float(b.conf),
             float(b.xyxy[0][2] - b.xyxy[0][0]),
             float(b.xyxy[0][3] - b.xyxy[0][1]))
            for b in res.boxes]
    dets = dedup(dets)

    hand_dets, table = split_hand_table(dets)
    hand = sorted(hand_dets, key=lambda d: card_value(d[0], args.level))

    print(f"图片: {args.image}   imgsz={args.imgsz}  conf={args.conf}")
    print()
    print(f"手牌 {len(hand)} 张（按掼蛋牌序）:")
    print("   " + "  ".join(f"{display(d[0])}({d[3]:.2f})" for d in hand))

    # 按点数分组，便于核对
    groups: dict[str, list[tuple]] = {}
    for d in hand:
        key = d[0] if d[0].startswith("JOKER") else d[0][1:]
        groups.setdefault(key, []).append(d)
    print()
    print("按点数分组:")
    order = sorted(groups, key=lambda k: card_value(groups[k][0][0], args.level))
    for k in order:
        cards = sorted(groups[k], key=lambda d: 0 if d[0].startswith("JOKER")
                       else "SCDH".index(d[0][0]))
        label = "王" if k.startswith("JOKER") else k
        print(f"   {label:<3} x{len(cards)}   " +
              " ".join(display(c[0]) for c in cards))

    if table:
        print()
        print(f"桌面出牌区 {len(table)} 张:")
        for d in sorted(table, key=lambda t: (t[2], t[1])):
            print(f"   {display(d[0]):<4} conf {d[3]:.2f}  @({d[1]:.0f},{d[2]:.0f})")

    if args.save:
        res.save(filename=args.save)
        print(f"\n标注图: {args.save}")


if __name__ == "__main__":
    main()
