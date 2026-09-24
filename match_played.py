"""模板匹配读出牌区的牌 —— 不依赖训练，直接可用。

出牌区的牌缩放恒定（106x140）、排版固定、字体就是游戏原生素材拼的，
所以「点数 + 点数下方的小花色」那一小块可以直接当模板做归一化互相关。
实测在真实画面上的 NCC 是 0.95~0.99，位置精确到 1~2px。

**不用全图滑窗**：出牌位置是固定的栅格（横向间距 50、四个锚点实测已知），
直接在候选栅格点上算相关系数就行 —— 每帧约 10ms，比滑窗快两个数量级。

它同时是合成牌面还原度最硬的验证：合成的牌只要有一点不对，NCC 上不去。

用法:
    python match_played.py shot.png          # 单张
    python match_played.py --eval            # 在 live/frames 上跑召回
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'synth'))
import layout as L                                       # noqa: E402
from compose import render_card, _paste, CARD_W, CARD_H   # noqa: E402

# 模板 = 卡左上角那块（点数 + 下方小花色）。宽度必须 < 间距 50，
# 否则会咬到右边相邻牌露出来的部分。
MARK_W, MARK_H = 48, 78
PITCH = L.PLAYED_PITCH
CW, CH = L.PLAYED_CARD_W, L.PLAYED_CARD_H
MAX_CARDS = 8                                     # 掼蛋两副牌同点数最多 8 张


def build_templates(levels="23456789TJQKA"):
    """模板表 {(cls, level_or_None): 归一化灰度模板}，已去均值/归一化。"""
    raw = {}
    for cls in L.CLASSES:
        base = None
        for lv in levels:
            c = np.full((CARD_H, CARD_W, 3), (120, 160, 130), np.uint8)
            _paste(c, render_card(cls, level=lv, style="played"), 0, 0, 1.0)
            small = cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA)
            t = small[:MARK_H, :MARK_W]
            if cls[0] == "H":
                raw[(cls, lv)] = t          # 红桃每级都可能是金色（逢人配）
            elif base is None:
                base = t
                raw[(cls, None)] = t        # 其他牌与级牌无关
    out = {}
    for k, t in raw.items():
        g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g -= g.mean()
        n = float(np.sqrt((g * g).sum()))
        out[k] = (g / n if n > 1e-6 else g).ravel()
    return out


def candidates(img_shape):
    """出牌区所有可能的「卡左上角」位置。位置是固定栅格，直接枚举。

    左家   左缘固定在 x=204，从左边向右排
    右家   右缘固定在 x=1485，最右那张在 x=1379，向左排
    对家   中心固定在 x=846，y=199
    自己   中心固定在 x=846，y=355
    """
    h, w = img_shape[:2]
    out = []
    for k, z in L.PLAYED_ZONES.items():
        y = z["y"]
        if z["anchor"] == "left":
            xs = [z["x"] + PITCH * i for i in range(MAX_CARDS)]
        elif z["anchor"] == "right":
            xs = [z["x"] - CW - PITCH * i for i in range(MAX_CARDS)]
        else:
            xs = []
            for n in range(1, MAX_CARDS + 1):
                x0 = z["x"] - (PITCH * (n - 1) + CW) / 2
                xs += [x0 + PITCH * i for i in range(n)]
            xs = sorted(set(round(v) for v in xs))
        for x in xs:
            if 0 <= x <= w - MARK_W and 0 <= y <= h - MARK_H:
                out.append((k, int(round(x)), int(y)))
    return out


def detect(img, tpl, thr=0.75):
    """返回 [(cls, level, score, x, y)]，x/y 是卡左上角。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    keys = list(tpl)
    T = np.stack([tpl[k] for k in keys])                 # (n_tpl, MARK_W*MARK_H)
    hits = []
    for zone, x, y in candidates(img.shape):
        patch = gray[y:y + MARK_H, x:x + MARK_W]
        if patch.shape != (MARK_H, MARK_W):
            continue
        p = patch.ravel().copy()
        p -= p.mean()
        n = float(np.sqrt((p * p).sum()))
        if n < 1e-6:
            continue
        scores = T @ (p / n)
        i = int(scores.argmax())
        if scores[i] >= thr:
            cls, lv = keys[i]
            hits.append((cls, lv, float(scores[i]), x, y))
    # 同一格子在多个 n 下会重复出现（居中区的候选集是并集），按位置去重
    kept = {}
    for h in hits:
        k = (h[3] // 8, h[4] // 8)
        if k not in kept or h[2] > kept[k][2]:
            kept[k] = h
    return sorted(kept.values(), key=lambda t: (t[4], t[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tpl = build_templates()
    print(f"模板 {len(tpl)} 个，阈值 {args.thr}", flush=True)

    if args.eval:
        import time
        from collections import Counter
        rows = json.load(open(ROOT / 'played_scan.json'))
        if args.limit:
            rows = rows[:args.limit]
        n_hit = n_fp = n_miss = n_group = 0
        det_frames = 0
        t0 = time.time()
        for ri, r in enumerate(rows):
            img = cv2.imread(r["frame"])
            if img is None:
                continue
            ref = [g[:4] for v in r["zones"].values() for g in v if g[3] >= 125]
            n_group += len(ref)
            hits = detect(img, tpl, thr=args.thr)
            det_frames += bool(hits)
            used = set()
            for h in hits:
                cx, cy = h[3] + MARK_W / 2, h[4] + MARK_H / 2
                m = [i for i, (bx, by, bw, bh) in enumerate(ref)
                     if bx - 6 <= cx <= bx + bw + 6 and by - 6 <= cy <= by + bh + 6]
                if m:
                    n_hit += 1
                    used.add(m[0])
                else:
                    n_fp += 1
            n_miss += len(ref) - len(used)
            if ri and ri % 50 == 0:
                print(f"  ...{ri}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        print(f"参照牌组 {n_group}（来自 played_scan.json 的白块扫描，"
              f"动画变暗的帧会漏，所以是下界）")
        print(f"  检出 {n_hit + n_fp} 个: 落在参照牌组内 {n_hit}，"
              f"不在任何一个里 {n_fp}  -> 精确率 {n_hit/max(1,n_hit+n_fp):.1%}")
        print(f"  参照牌组里一张都没碰到的 {n_miss}  -> 组召回 "
              f"{(n_group-n_miss)/max(1,n_group):.1%}")
        print(f"  有检出的帧 {det_frames}/{len(rows)} = {det_frames/len(rows):.1%}"
              f"   总耗时 {time.time()-t0:.0f}s")
        return

    for f in args.images:
        img = cv2.imread(f)
        if img is None:
            continue
        hits = detect(img, tpl, thr=args.thr)
        print(f"\n{f}")
        for cls, lv, sc, x, y in hits:
            print(f"   {cls:<8} {'级=' + lv if lv else '':<5} {sc:.3f}  @({x},{y})")
        if not hits:
            print("   （无匹配）")


if __name__ == "__main__":
    main()
