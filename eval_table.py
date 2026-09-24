"""出牌区验收 —— 在人工核对过真值的 15 帧真实画面上评估。

两种「检测器」都可以跑这套评分，方便对照：
    python eval_table.py runs/detect/guandan7/weights/best.pt   # YOLO
    python eval_table.py --matcher                              # 模板匹配

这是出牌区唯一可信的指标（和 eval_real.py 之于手牌同一个道理）。
真值见 table_gt.py，是逐帧放大核对出来的。

用法:
    python eval_table.py <权重.pt> [--imgsz 960] [--conf 0.25] [--save]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'synth'))
sys.path.insert(0, str(ROOT / 'live'))
from layout import CLASSES, CARD_H                           # noqa: E402
from predict_cards import (dedup, split_hand_table, table_by_player,  # noqa: E402
                           tolerate_occlusion, occluders, CONF_UNDER_UI)
from table_gt import GROUND_TRUTH, LOW_CONFIDENCE             # noqa: E402
from turn import is_my_turn                                   # noqa: E402

FRAME_DIR = ROOT / 'live' / 'frames'
CARD_H_PLAYED = CARD_H * 0.804      # 出牌区卡的竖向尺寸（实测 140）


ZONE_OF = {"left": "机器人1", "right": "机器人3", "top": "队友", "bottom": "我"}


def make_detector(args):
    """返回 f(图像) -> [(类别, 框中心x, 框中心y, 置信度)]，统一成同一种输出。"""
    if args.matcher:
        from match_played import build_templates, detect
        from predict_cards import PLAY_MARK_CX, PLAY_MARK_CY
        tpl = build_templates()
        # 模板给的是「卡左上角」，要换成和 YOLO 标注一致的「标注框中心」。
        # 这个偏移量必须和 played_marker_box 一致，否则 table_by_player
        # 会二次偏移 -> 所有牌都被分到下一条带（踩过一次）。
        def f(img):
            # 模板 = 出牌标注框，尺寸是已知的常量
            return [(cls, x + PLAY_MARK_CX, y + PLAY_MARK_CY, score,
                     PLAY_MARK_CX * 2, PLAY_MARK_CY * 2)
                    for cls, _lv, score, x, y in detect(img, tpl, thr=args.conf)]

        return f

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.model.names = {i: c for i, c in enumerate(CLASSES)}

    def _predict(img, imgsz, conf):
        res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        return [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf),
                 float(b.xyxy[0][2] - b.xyxy[0][0]),
                 float(b.xyxy[0][3] - b.xyxy[0][1])) for b in res.boxes]

    def f(img):
        # 和面板保持一致：低门槛跑 + 高分辨率补一遍，再按遮挡规则过滤
        raw = dedup(_predict(img, args.imgsz, min(args.conf, CONF_UNDER_UI)))
        if args.imgsz2 and args.imgsz2 != args.imgsz:
            raw = dedup(raw + _predict(img, args.imgsz2, args.imgsz2_conf))
        mine, _ = is_my_turn(img)
        hand_hi, _ = split_hand_table([d for d in raw if d[3] >= args.conf])
        return tolerate_occlusion(raw, args.conf,
                                  occluders(img.shape[1], mine, hand_hi))

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="")
    ap.add_argument("--matcher", action="store_true", help="用模板匹配而不是 YOLO")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz2", type=int, default=1600,
                    help="第二遍（高分辨率）补漏用的尺寸；0 = 关掉")
    ap.add_argument("--imgsz2-conf", type=float, default=0.30)
    ap.add_argument("--save", action="store_true", help="输出标注图")
    args = ap.parse_args()
    if not args.matcher and not args.weights:
        ap.error("要么给权重路径，要么加 --matcher")

    detector = make_detector(args)

    n_gt = n_found = n_right = n_spur = 0
    n_zone_ok = n_zone_tot = 0
    print(f"检测器: {'模板匹配' if args.matcher else args.weights}"
          f"   imgsz={args.imgsz}  conf={args.conf}")
    print(f"（真值 {len(GROUND_TRUTH)} 帧，来自逐帧放大核对；标 * 的是读得不太确定的）\n")

    for name, gt in GROUND_TRUTH.items():
        path = FRAME_DIR / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"  {name}: 读不到，跳过")
            continue
        dets = dedup(detector(img))
        _, table = split_hand_table(dets)
        by_player = table_by_player(table)

        gt_n = sum(len(v) for v in gt.values())
        n_gt += gt_n
        got_n = len(table)
        mark = " *" if name in LOW_CONFIDENCE else "  "
        parts = []
        for zone in ("left", "right", "top", "bottom"):
            g = gt.get(zone, [])
            d = by_player.get(ZONE_OF[zone], [])
            if not g and not d:
                continue
            # 按类别配对（同类之间不分先后）
            left_g = list(g)
            right = 0
            for dd in sorted(d, key=lambda t: -t[3]):
                if dd[0] in left_g:
                    left_g.remove(dd[0])
                    right += 1
            n_found += len(d)
            n_right += right
            n_spur += len(d) - right
            n_zone_tot += 1
            n_zone_ok += (sorted(x[0] for x in d) == sorted(g))
            parts.append(f"{zone} 真{len(g)} 检{len(d)} 对{right}")
        print(f"{mark}{name:<20} 真值 {gt_n}  检出 {got_n}   " + " | ".join(parts))

        if args.save:
            vis = img.copy()
            for d in table:
                x, y = int(d[1]), int(d[2])
                cv2.circle(vis, (x, y), 6, (0, 0, 255), 2)
                cv2.putText(vis, d[0], (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            out = ROOT / 'runs' / 'table_eval'
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / name), vis)

    print("\n" + "=" * 62)
    print(f"  真值出牌 {n_gt} 张   模型报出 {n_found} 张")
    print(f"  其中类别正确 {n_right}  -> 端到端召回 {n_right / max(1, n_gt):.1%}")
    print(f"  类别错误/多余 {n_spur}  -> 误检率 {n_spur / max(1, n_found):.1%}")
    print(f"  按区整组完全正确 {n_zone_ok}/{n_zone_tot}")
    print("  （和手牌一样：这一行才是出牌区唯一有意义的指标）")


if __name__ == "__main__":
    main()
