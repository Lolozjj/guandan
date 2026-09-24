"""在带标注的真实截图上评估模型 —— 这是唯一可信的指标。

val 集来自合成分布，分数必然虚高（实测过：合成 val mAP50 0.993，真实图上一塌糊涂）。
所以每次训练完都必须跑这个脚本。

ground truth 是人工逐张读出的，见 GROUND_TRUTH。

用法:
    python eval_real.py <权重.pt> [--imgsz 960] [--conf 0.25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
from layout import CLASSES, CARD_W, PITCH_X, Y_BOTTOM, PITCH_Y, marker_box  # noqa: E402

# ---------------------------------------------------------------------------
# 人工标注真值 —— 用「卡左边缘 x + 顶边 y」的绝对坐标，不依赖 layout 模型。
#
# 为什么用绝对坐标：实测 img_5 的列 7/8/9 各塞了两个不同点数的分组（3♦ 上面摞着
# 2♣2♥），不符合「一列一组」的模型。但每张牌的位置本身是准的，所以测试集直接记
# 位置，不去套布局假设。
#
# 读法：把每张牌的角标裁出来逐张看。img.png / img_1 / img_3 还额外用
# 「每列牌数 == 该列点数」交叉校验过。
#
# 未纳入：img_4 —— 手牌只有几张、且位置漂移，几何不可靠。
# ---------------------------------------------------------------------------
PITCH_X = 95

def _cols(x0, items):
    """(类别, 列号, 顶边y) -> (类别, 绝对x, 顶边y)。"""
    return [(c, x0 + PITCH_X * k, y) for c, k, y in items]


_IMG = [
    ("JOKER_S", 0, 714),
    ("C2", 1, 652), ("C2", 1, 714),
    ("SK", 2, 590), ("CK", 2, 652), ("DK", 2, 714),
    ("CQ", 3, 652), ("DQ", 3, 714),
    ("HJ", 4, 714),
    ("S9", 5, 590), ("C9", 5, 652), ("D9", 5, 714),
    ("H8", 6, 714),
    ("S7", 7, 590), ("C7", 7, 652), ("H7", 7, 714),
    ("C6", 8, 528), ("D6", 8, 590), ("D6", 8, 652), ("H6", 8, 714),
    ("S5", 9, 590), ("D5", 9, 652), ("D5", 9, 714),
    ("C4", 10, 652), ("H4", 10, 714),
    ("S3", 11, 652), ("D3", 11, 714),
]
# img_1 / img_2 是 img.png 的后续时刻：J♥ 已打出，其后各列左移一列
_IMG1 = [(c, k if k < 4 else k - 1, y) for c, k, y in _IMG if c != "HJ"]

GROUND_TRUTH: dict[str, list[tuple[str, int, int]]] = {
    "img": _cols(245, _IMG),
    "img_1": _cols(305, _IMG1),
    "img_2": _cols(305, _IMG1),
    "img_3": _cols(352, [
        # 6 组实际有 5 张，最上面的 6♠ 在 y=465 —— 和 img_5 一样，
        # 最初的列检测从 y=498 起扫，把它切掉了
        ("S6", 0, 465), ("C6", 0, 528), ("D6", 0, 590), ("D6", 0, 652),
        ("H6", 0, 714),
        ("S9", 1, 528), ("C9", 1, 590), ("D9", 1, 652), ("H9", 1, 714),
        ("JOKER_S", 2, 652), ("JOKER_B", 2, 714),
        ("S4", 3, 590), ("C4", 3, 652), ("D4", 3, 714),
        ("SK", 4, 590), ("CK", 4, 652), ("DK", 4, 714),
        ("SQ", 5, 652), ("CQ", 5, 714),
        ("CJ", 6, 590), ("DJ", 6, 652), ("HJ", 6, 714),
        ("DT", 7, 714),
        ("C8", 8, 652), ("D8", 8, 714),
        ("C3", 9, 652), ("H3", 9, 714),
    ]),
    # img_5：实测列位 351 + 95k，**卡片层级有 6 层**（403/465/527/589/651/713）。
    # 最初漏了最上面两层，导致少了 4 张牌（4♦ 4♥ 10♣ 9♠）——列检测的扫描起点
    # 定在 y=498，把 403 和 465 两层整个切掉了。
    # 另注：这手牌里游戏是按「牌型」分组的（三连对 2-3-4、三带二），同一个列里
    # 会出现不同点数，不符合「一列一点数」的模型。所以真值一律用绝对坐标。
    "img_5": _cols(351, [
        ("SJ", 0, 527), ("CJ", 0, 589), ("CJ", 0, 651), ("DJ", 0, 713),
        ("JOKER_B", 1, 713),
        ("JOKER_S", 2, 713),
        ("S6", 3, 713),
        ("SA", 4, 713),
        ("DK", 5, 713),
        ("SQ", 6, 651), ("CQ", 6, 713),
        ("D4", 7, 403), ("H4", 7, 465), ("D3", 7, 527), ("D3", 7, 589),
        ("C2", 7, 651), ("H2", 7, 713),
        ("CT", 8, 465), ("DT", 8, 527), ("S7", 8, 589), ("H7", 8, 651),
        ("H7", 8, 713),
        ("S9", 9, 465), ("H9", 9, 527), ("S5", 9, 589), ("D5", 9, 651),
        ("H5", 9, 713),
    ]),
}

MATCH_PX = 34     # 检出框中心与真值框中心的距离阈值
HAND_TOP = 515    # 手牌区上界（框中心 y 小于此值 = 桌面出牌区，不在真值范围内）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", default="runs/real_eval")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    # 权重文件里内置的 names 来自训练时的 data.yaml。若那份 data.yaml 的类别顺序
    # 与 synth/layout.py 的 CLASSES 不一致（历史上错过一次），读出来的名字会整体
    # 错位——模型本身没学错，只是被念错了。这里强制用权威顺序覆盖。
    model.model.names = {i: c for i, c in enumerate(CLASSES)}
    names = model.names
    print(f"权重: {args.weights}   imgsz={args.imgsz}  conf={args.conf}")
    tot_n = tot_hit = tot_correct = tot_fp = tot_table = 0

    for shot, truth in GROUND_TRUTH.items():
        path = f"shots/{shot}.png"
        img = cv2.imread(path)
        if img is None:
            print(f"[FAIL] 读不到 {path}")
            continue
        res = model.predict(source=path, imgsz=args.imgsz, conf=args.conf,
                            verbose=False)[0]

        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append((names[int(b.cls)], (x1 + x2) / 2, (y1 + y2) / 2,
                         float(b.conf)))

        gts = []
        for cls, cx, y in truth:
            x1, y1, x2, y2 = marker_box(cx, y)
            gts.append((cls, (x1 + x2) / 2, (y1 + y2) / 2))

        # 贪心匹配：按置信度从高到低处理检出，各自认领最近的真值。
        #
        # 反过来（每个真值找最近检出）会出错：同一张牌常有高低两个框（如 H6@0.77
        # 和 D9@0.31 只差 0.4px），最近的不一定是正确的那个，会把正确的挤成误检。
        # 按置信度排序能让正确的框优先认领。
        matched_gt: dict[int, int] = {}      # gt 索引 -> det 索引
        used: set[int] = set()
        for di in sorted(range(len(dets)), key=lambda i: -dets[i][3]):
            _dc, dx, dy, _c = dets[di]
            best, bi = MATCH_PX, -1
            for gi, (_gc, gx, gy) in enumerate(gts):
                if gi in matched_gt:
                    continue
                d = ((dx - gx) ** 2 + (dy - gy) ** 2) ** 0.5
                if d < best:
                    best, bi = d, gi
            if bi >= 0:
                matched_gt[bi] = di
                used.add(di)

        hit = correct = 0
        errors = []
        for gi, (cls, gx, gy) in enumerate(gts):
            di = matched_gt.get(gi)
            if di is None:
                continue
            hit += 1
            d = ((dets[di][1] - gx) ** 2 + (dets[di][2] - gy) ** 2) ** 0.5
            if dets[di][0] == cls:
                correct += 1
            else:
                errors.append((cls, dets[di][0], round(d, 1)))

        fp_all = [d for i, d in enumerate(dets) if i not in used]
        # 真值只标了手牌；模型检出桌面上的牌是正确的行为，不该算误检。
        fp = [d for d in fp_all if d[2] >= HAND_TOP]
        table = [d for d in fp_all if d[2] < HAND_TOP]
        n = len(gts)
        tot_n += n; tot_hit += hit; tot_correct += correct; tot_fp += len(fp)
        tot_table += len(table)

        print(f"  {shot}:  真值 {n}  检出 {len(dets)}  "
              f"检出率 {hit / n:.1%}  端到端 {correct / n:.1%}  误检 {len(fp)}")
        if errors:
            print(f"    分错 {len(errors)} 张: " +
                  "; ".join(f"{a}->{b}" for a, b, _ in errors[:12]))
        if fp:
            print(f"    误检: " +
                  ", ".join(f"{d[0]}@{d[3]:.2f}" for d in
                            sorted(fp, key=lambda t: -t[3])[:8]))

        Path(args.save).mkdir(parents=True, exist_ok=True)
        res.save(filename=str(Path(args.save) / f"{shot}.jpg"))

    print("=" * 62)
    print(f"  合计: 真值 {tot_n}   检出率 {tot_hit / tot_n:.1%}   "
          f"端到端准确率 {tot_correct / tot_n:.1%}   误检 {tot_fp}"
          f"   桌面检出 {tot_table}")
    print("  (合成 val 的分数不可信；这一行才是唯一有意义的指标)")
    print(f"  标注结果图: {args.save}/")


if __name__ == "__main__":
    main()
