"""掼蛋手牌布局模型。

实测参数（推导过程见 README「阶段 2」）:

    截图尺寸    1670x990 ~ 1688x999
    卡片尺寸    134 x 174 px
    水平步长    95 px
    竖向步长    62 px   —— 同一点数的牌竖向级联，让每张都露出左上角
    底部对齐    最底下那张牌的顶边 y = 714（卡底 888）
    水平居中    整手牌在画面里居中
    角标区域    相对卡片左上角 (8, 1)，尺寸 93 x 66   —— 这是 YOLO 的标注框

结构规则:
    1. 手牌按掼蛋牌序降序排列（大王 > 小王 > 级牌 > A > K > ... > 3）
    2. 同一点数的牌归为一组，占一个「列」
    3. 列内 n 张牌从顶部 y = 714 - 62*(n-1) 开始、每张下移 62 px，最后一张落在 714
    4. 绘制顺序：列从左到右；列内从上到下（后画的在下层牌之上）
"""
from __future__ import annotations

from dataclasses import dataclass

CARD_W, CARD_H = 132, 174
PITCH_X = 95
PITCH_Y = 62
Y_BOTTOM = 714

# YOLO 标注框：角标（点数 + 小花色符号）区域，相对卡片左上角
MARKER_DX, MARKER_DY = 8, 1
MARKER_W, MARKER_H = 93, 66

# 花色排列顺序 —— 实测证实是 ♠♣♦♥（先黑后红），不是 ♠♥♦♣。
# 顺序错会导致分级联里黑红牌位置对调，相邻牌的角标框内容颜色整个反过来。
SUITS = "SCDH"
# 掼蛋基础牌序（不含级牌和大小王），从大到小
BASE_RANKS = "AKQJT98765432"

# 本项目 54 类
CLASSES = (
    [f"S{r}" for r in "A23456789TJQK"]
    + [f"H{r}" for r in "A23456789TJQK"]
    + [f"D{r}" for r in "A23456789TJQK"]
    + [f"C{r}" for r in "A23456789TJQK"]
    + ["JOKER_S", "JOKER_B"]
)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


@dataclass(frozen=True)
class PlacedCard:
    """一张牌在画面中的位置（画布坐标，卡片左上角）。"""
    cls: str
    x: int
    y: int

    @property
    def marker_box(self) -> tuple[int, int, int, int]:
        """角标框 (x1, y1, x2, y2)。"""
        return (self.x + MARKER_DX, self.y + MARKER_DY,
                self.x + MARKER_DX + MARKER_W, self.y + MARKER_DY + MARKER_H)


def sort_key(cls: str, level: str = "2") -> tuple[int, int]:
    """掼蛋牌序的排序键（值越小越靠左）。

    大王 > 小王 > 级牌 > A > K > Q > ... > 3
    同点数内按花色 S H D C 排。
    """
    rank = cls[1:] if not cls.startswith("JOKER") else cls
    # 实测小王在上、大王在下（级联里上面的 y 更小、画的更早）
    if cls == "JOKER_S":
        return (0, 0)
    if cls == "JOKER_B":
        return (0, 1)
    suit = cls[0]
    if rank == level:
        return (1, SUITS.index(suit))
    # BASE_RANKS 里越靠前越大，转成越小越靠前
    return (2 + BASE_RANKS.index(rank), SUITS.index(suit))


def expand(hand_counts: dict[str, int]) -> list[str]:
    """把 {类别: 张数} 展开成类别列表。"""
    out: list[str] = []
    for cls, n in hand_counts.items():
        out.extend([cls] * n)
    return out


def layout(cards: list[str], image_w: int, level: str = "2") -> list[PlacedCard]:
    """给一手牌算出每张牌的位置。

    cards 是牌的类别列表（可含重复，掼蛋两副牌）。返回按绘制顺序排列的列表。
    """
    ordered = sorted(cards, key=lambda c: sort_key(c, level))

    # 按点数分组，每组占一列（大小王各占一列）
    groups: list[list[str]] = []
    for cls in ordered:
        gkey = _group_key(cls)
        if groups and _group_key(groups[-1][0]) == gkey:
            groups[-1].append(cls)
        else:
            groups.append([cls])

    n_cols = len(groups)
    hand_w = PITCH_X * (n_cols - 1) + CARD_W
    x0 = (image_w - hand_w) // 2

    placed: list[PlacedCard] = []
    for col, group in enumerate(groups):
        x = x0 + col * PITCH_X
        n = len(group)
        # group[0] 在最高处，group[-1] 落在 Y_BOTTOM
        for i, cls in enumerate(group):
            y = Y_BOTTOM - PITCH_Y * (n - 1 - i)
            placed.append(PlacedCard(cls, x, y))
    return placed


def _group_key(cls: str) -> str:
    """分组用的键：同一点数归一组。

    大小王共占一列（实测 img_3：小王在上、大王在下），都归到 "JOKER" 组。
    """
    if cls.startswith("JOKER"):
        return "JOKER"
    return cls[1:]


def draw_order(placed: list[PlacedCard]) -> list[PlacedCard]:
    """绘制顺序：列从左到右；列内从上到下（下方牌后画，压住上方牌的底边）。"""
    return sorted(placed, key=lambda p: (p.x, p.y))


def to_yolo(placed: list[PlacedCard], image_w: int, image_h: int) -> list[str]:
    """转成 YOLO 标注行：class_id cx cy w h（均归一化，clamp 到 0~1）。"""
    lines = []
    for p in placed:
        x1, y1, x2, y2 = p.marker_box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image_w, x2); y2 = min(image_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / image_w
        cy = (y1 + y2) / 2 / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h
        lines.append(f"{CLASS_TO_ID[p.cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ---------------------------------------------------------------------------
# 出牌区（桌面上的牌）
# ---------------------------------------------------------------------------
# 出牌区实测（2026-09-23 在 337 帧真实画面上重新量过）：
#   - 牌 106 x 140，正好是手牌 132x174 的 0.804 倍；**缩放基本恒定**
#     （275 个牌组，中位高 140，5%~95% 落在 139~140）
#   - 间距 50 —— 重度重叠，每张只露出左边 50px（最后一张全露）
#   - 出牌区有 **4 个固定位置**（左上角坐标）：
#         左家   左缘 204、  y 312   （向右排）
#         右家   右缘 1485、 y 312   （向左排）
#         对家   中心 846、  y 199
#         自己   中心 846、  y 355
#     注意「对家 y=199」和「自己 y=355」几乎不会同时出现（117 帧里只有 2 帧），
#     所以它们是两个不同玩家的区，不是同一玩家的两档高度。
#
# 之前写的 PLAYED_SCALE=0.85 / RANGE=(0.72,1.0) 是猜的，偏大且范围过宽。
PLAYED_SCALE = 0.804                       # 106/132，也 = 140/174
PLAYED_SCALE_RANGE = (0.78, 0.83)          # 只留一点抖动
PLAYED_PITCH = 50                          # 并排多张牌时的水平步长（重度重叠）
PLAYED_CARD_W = round(CARD_W * PLAYED_SCALE)   # 106
PLAYED_CARD_H = round(CARD_H * PLAYED_SCALE)   # 140

# 出牌区的标注框：出牌的花色符号在点数**正下方**，所以框是竖长条，
# 框住「点数 + 小花色」这一块（手牌是横的 93x66，两者形状不同）。
PLAY_MARKER_DX, PLAY_MARKER_DY = 3, 2
PLAY_MARKER_W, PLAY_MARKER_H = 40, 86

# 四个出牌区（锚点 + 一级对齐方式）
PLAYED_ZONES = {
    "left":   dict(x=204,  y=312, anchor="left"),    # 左家，从左缘向右排
    "right":  dict(x=1485, y=312, anchor="right"),   # 右家，右缘固定，向左排
    "top":    dict(x=846,  y=199, anchor="center"),  # 对家
    "bottom": dict(x=846,  y=355, anchor="center"),  # 自己
}


def played_marker_box(x: float, y: float, sc: float = 1.0):
    """出牌区牌的标注框 (x1, y1, x2, y2)。sc 是相对 132x174 的缩放。"""
    return (x + PLAY_MARKER_DX * sc, y + PLAY_MARKER_DY * sc,
            x + (PLAY_MARKER_DX + PLAY_MARKER_W) * sc,
            y + (PLAY_MARKER_DY + PLAY_MARKER_H) * sc)


def marker_box(x: float, y: float, scale: float = 1.0) -> tuple[float, float, float, float]:
    """角标框 (x1, y1, x2, y2)，支持缩放。"""
    return (x + MARKER_DX * scale, y + MARKER_DY * scale,
            x + (MARKER_DX + MARKER_W) * scale,
            y + (MARKER_DY + MARKER_H) * scale)


def yolo_line(cls: str, x1: float, y1: float, x2: float, y2: float,
              image_w: int, image_h: int) -> str | None:
    """单行 YOLO 标注；框超出画面或退化则返回 None。"""
    x1 = max(0.0, x1); y1 = max(0.0, y1)
    x2 = min(float(image_w), x2); y2 = min(float(image_h), y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    cx = (x1 + x2) / 2 / image_w
    cy = (y1 + y2) / 2 / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
