"""边玩边采集：按固定间隔抓游戏画面，存成图片供后续分析。

用途：拿到真实对局各阶段的画面（轮到我 / 别人出牌 / 结算…），
据实设计「级别识别」「轮次识别」和面板，而不是凭空猜。

只在画面**有变化**时保存，避免存几百张一样的图。

用法:
    python live/collect.py --seconds 300 --out live/frames
    # 打一局，Ctrl+C 提前结束
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import GameCapture  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="掼蛋")
    ap.add_argument("--out", default="live/frames")
    ap.add_argument("--seconds", type=float, default=300, help="采集总时长")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒）")
    ap.add_argument("--change", type=float, default=0.6,
                    help="与上一张的像素差超过这个值才保存（0 表示全存）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = GameCapture(args.title)

    if not cap.ensure_window():
        raise SystemExit("[FAIL] 没找到游戏窗口，先把掼蛋打开")

    print(f"开始采集：每 {args.interval}s 抓一次，共 {args.seconds:.0f}s，存到 {out}/")
    print("（画面变化超过阈值才保存；Ctrl+C 可提前结束）\n")

    t0 = time.time()
    n_saved = n_seen = 0
    prev = None
    try:
        while time.time() - t0 < args.seconds:
            img = cap.grab()
            if img is None:
                print("  窗口不可用，等待…")
                time.sleep(1.0)
                cap.hwnd = None
                continue
            n_seen += 1
            if prev is not None and prev.shape == img.shape:
                d = float(cv2.absdiff(img, prev).mean())
            else:
                d = 999.0
            if d >= args.change:
                n_saved += 1
                el = int(time.time() - t0)
                name = out / f"f{el:05d}_{n_saved:04d}.png"
                cv2.imwrite(str(name), img)
                print(f"  [+{el:>4}s] 存第 {n_saved} 张  (变化 {d:>6.2f})  {name.name}")
            prev = img
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  收到中断，提前结束")

    print(f"\n共抓取 {n_seen} 帧，保存 {n_saved} 张到 {out}/")
    if n_saved == 0:
        print("一张没存 —— 可能画面一直没变（没在牌局里），或阈值设太高")


if __name__ == "__main__":
    main()
