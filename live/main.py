"""实时面板：抓游戏画面 -> 认牌 -> 侧边置顶小窗显示。

架构（tkinter 必须在主线程）：
    主线程   tkinter 界面，定时从队列取结果刷新
    工作线程 抓图 -> 认牌 -> 压进队列

只在画面有变化时才重新识别，省算力。

用法:
    python live/main.py
    python live/main.py --weights runs/detect/guandan4/weights/best.pt
    python live/main.py --no-panel        # 只打印不弹窗，用于排查
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import GameCapture                    # noqa: E402
from level import read_level                       # noqa: E402
from phase import detect_phase                     # noqa: E402
from rules import classify as classify_play        # noqa: E402
from turn import countdown_side, is_my_turn        # noqa: E402
from layout import CLASSES                         # noqa: E402
from predict_cards import (card_value, dedup, split_hand_table,  # noqa: E402
                           table_by_player, tolerate_occlusion,
                           occluders, CONF_UNDER_UI)

SUIT_SHOW = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RANK_NAME = {"T": "10"}
WHO = {"left": "机器人1", "top": "队友", "right": "机器人3", "mine": "我"}
# 出牌区四家的显示顺序（按座位：左 -> 右 -> 对家 -> 自己）
TABLE_ORDER = ("机器人1", "机器人3", "队友", "我")


def fmt_card(cls: str) -> str:
    if cls == "JOKER_B":
        return "大王"
    if cls == "JOKER_S":
        return "小王"
    return SUIT_SHOW[cls[0]] + RANK_NAME.get(cls[1:], cls[1:])


def group_hand(hand, level):
    """按点数分组，按掼蛋牌序排列。hand 是 [(类别, 置信度)]。"""
    groups = defaultdict(list)
    for cls, _c in hand:
        groups[cls if cls.startswith("JOKER") else cls[1:]].append(cls)
    lv = level if level and level != "T" else "2"

    def key(k):
        if k.startswith("JOKER"):
            return (0, 0 if k == "JOKER_S" else 1)
        if k == lv:
            return (1, 0)
        return (2 + "AKQJT98765432".index(k), 0)

    out = []
    for k in sorted(groups, key=key):
        cards = sorted(groups[k],
                       key=lambda c: 0 if c.startswith("JOKER")
                       else "SCDH".index(c[0]))
        out.append((k, [fmt_card(c) for c in cards]))
    return out


class Worker(threading.Thread):
    def __init__(self, args, out_q):
        super().__init__(daemon=True)
        self.args = args
        self.q = out_q
        self.stop = threading.Event()
        self._cap = GameCapture(args.title)

    def run(self):
        from ultralytics import YOLO

        m = YOLO(self.args.weights)
        m.model.names = {i: c for i, c in enumerate(CLASSES)}
        names = m.names

        prev = None
        stable = 0
        last_level, last_conf = None, 0.0
        while not self.stop.is_set():
            t0 = time.time()
            img = self._cap.grab()
            if img is None:
                self.q.put({"status": "no-window"})
                time.sleep(1.0)
                self._cap.hwnd = None
                continue

            # 只在画面**连续稳定**之后识别一次。
            # 出牌/发牌有动画，中间态的牌被半透明层压着、特征本身就模糊，
            # 认了也是错的。等它稳定下来再认，既准又省算力。
            if prev is None or prev.shape != img.shape:
                prev, stable = img, 0
                time.sleep(self.args.interval)
                continue

            diff = float(cv2.absdiff(img, prev).mean())
            prev = img
            if diff >= self.args.change:
                stable = 0
                time.sleep(self.args.interval)
                continue

            stable += 1
            if stable != self.args.stable_frames:   # 稳定期只认一次
                time.sleep(0.15)
                continue

            # 换局/结算画面上的牌不是「出牌」，先判阶段，非正常阶段直接跳过识别
            phase, why = detect_phase(img)
            if phase != "normal":
                self.q.put({"status": "ok", "phase": phase, "phase_why": why,
                            "elapsed": time.time() - t0})
                time.sleep(self.args.interval)
                continue

            res = self._model_run(m, names, img)
            hand, table = res
            ch, sc, _info = read_level(img)
            if ch:
                last_level, last_conf = ch, sc
            mine, orange = is_my_turn(img)
            side, _s = countdown_side(img)

            self.q.put({"status": "ok", "phase": "normal", "level": last_level,
                        "level_conf": last_conf, "hand": hand, "table": table,
                        "mine": mine, "orange": orange, "turn_side": side,
                        "elapsed": time.time() - t0})
            time.sleep(self.args.interval)

    @staticmethod
    def _predict(m, names, img, imgsz, conf):
        res = m.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        return [(names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf),
                 float(b.xyxy[0][2] - b.xyxy[0][0]),
                 float(b.xyxy[0][3] - b.xyxy[0][1])) for b in res.boxes]

    def _model_run(self, m, names, img):
        # 第一遍用「低门槛」跑主力分辨率：轮到我时 UI 按钮会盖住左右两家牌的
        # 花色符号，那些牌置信度掉到 0.1 左右，用常态门槛根本不会输出。
        # 跑完再按 tolerate_occlusion 过滤回正常门槛。
        raw = dedup(self._predict(m, names, img, self.args.imgsz,
                                  min(self.args.conf, CONF_UNDER_UI)))
        # 第二遍高分辨率：补第一遍完全认不出的难牌（实测 5 张 8 只能认出 4 张、
        # 但 1600 下第 5 张能出来）。两遍合并后再统一过滤。
        if self.args.imgsz2 and self.args.imgsz2 != self.args.imgsz:
            raw = dedup(raw + self._predict(m, names, img, self.args.imgsz2,
                                            self.args.imgsz2_conf))
        # 先用常态门槛找出「确定是手牌」的那些，它们的位置就是遮挡物之一
        # （自己的手牌会盖住自己刚出的牌）。再据此过滤出最终检出。
        mine, _orange = is_my_turn(img)
        hand_hi, _ = split_hand_table([d for d in raw if d[3] >= self.args.conf])
        dets = tolerate_occlusion(raw, self.args.conf,
                                  occluders(img.shape[1], mine, hand_hi))
        hand_dets, table = split_hand_table(dets)
        hand = sorted([(d[0], d[3]) for d in hand_dets],
                      key=lambda t: card_value(t[0], "2"))
        # 按四家分区 —— 面板要的是「每家出了什么牌」，不是一坨
        by_player = table_by_player(table)
        return hand, {k: [(d[0], d[3]) for d in v] for k, v in by_player.items()}


def render_lines(d) -> list[tuple[str, str]]:
    """把一份识别结果转成面板要显示的 (文字, 样式) 列表。

    抽成模块级函数是为了能脱离 tkinter 测 —— 面板显示逻辑不该只能靠读代码验证。
    """
    if not d:
        return [("正在连接游戏窗口…\n", "h1")]
    if d.get("phase", "normal") != "normal":
        # 换局 / 结算 / 进贡：这时候画面上的牌不是「出牌」，一律不显示读数
        return [("换局中\n", "wait"),
                ("   %s\n" % d.get("phase_why", ""), "grp"),
                ("   等这一局开始再看\n", "dim")]
    if d.get("status") == "no-window":
        return [("没找到游戏窗口\n", "err"),
                ("请把掼蛋窗口打开并保持可见\n", "dim")]

    out: list[tuple[str, str]] = []
    lv = d["level"]
    out.append(("当前级牌\n", "title"))
    out.append(("   %s\n" % (RANK_NAME.get(lv, lv) if lv else "?"), "lvl"))

    if d["mine"]:
        out.append(("● 轮到我出牌\n", "mine"))
    else:
        out.append(("○ 等待中 (%s)\n" % WHO.get(d["turn_side"], "?"), "wait"))

    out.append(("我的手牌  %d 张\n" % len(d["hand"]), "h1"))
    if not d["hand"]:
        out.append(("  (未识别到手牌)\n", "dim"))
    for key, cards in group_hand(d["hand"], d["level"]):
        label = "王" if key.startswith("JOKER") else RANK_NAME.get(key, key)
        out.append(("  %s×%d   %s\n" % (label, len(cards), " ".join(cards)),
                    "grp"))

    table = d["table"]
    if any(table.values()):
        out.append(("桌面出牌\n", "h1"))
        for who in TABLE_ORDER:
            v = table.get(who) or []
            if not v:
                continue              # 没出牌的那家不占篇幅
            cards = [c for c, _ in sorted(v, key=lambda t: -t[1])]
            cols = [fmt_card(c) for c in cards]
            out.append(("   %s  %d 张   %s\n" % (who, len(v), " ".join(cols)),
                        "grp"))
            # 用掼蛋规则校验一下这个组合合不合法
            kind = classify_play(cards, d.get("level") or "2")
            if kind:
                out.append(("            %s\n" % kind, "dim"))
            else:
                out.append(("            ⚠ 不是合法牌型，可能有牌被挡住没认出来\n",
                            "warn"))

    out.append(("\n识别 %.0fms   级牌置信 %.2f   按钮橙色 %.2f\n"
                % (d["elapsed"] * 1000, d["level_conf"], d["orange"]), "dim"))
    return out


def safe_render_lines(d) -> list[tuple[str, str]]:
    """`render_lines` 的**不会打断刷新链**版本，给 tkinter 的 after 回调用。

    为什么必须是它：`render_lines` 在 `tick()` 里被调用，而那条路径**不在**
    `tick` 的 try 内 —— after 回调抛一次错，刷新链就**永久、静默地**停掉，
    面板定格在旧画面上，用户还以为是当前局面（本分支唯一的 Critical 就是这个
    机制：适配层词表对不上，真实着法 29% 抛错，面板从此不再更新）。

    所以异常在这里兜住 —— 但**不吞**：stderr 打完整栈（`print_exc`），面板上留
    一行错误标记。spec §6⑥「失败必须响」：宁可让用户看到「渲染出错」，
    也不要让面板静静地显示一个过期/错误的结论。
    """
    try:
        return render_lines(d)
    except Exception as e:          # noqa: BLE001 —— 这里就是要兜住一切
        traceback.print_exc()
        return [("⚠ 面板渲染出错：%s\n" % e, "err"),
                ("   上面一行是原因（stderr 里有完整栈）；面板会继续刷新\n", "dim")]


def run_panel(args, q):
    import tkinter as tk

    F = "C:/Windows/Fonts/msyh.ttc"
    root = tk.Tk()
    root.title("掼蛋助手")
    root.configure(bg="#1e1e1e")
    root.geometry("%dx%d+%d+%d" % (args.panel_width, args.panel_height,
                                   args.panel_x, args.panel_y))
    root.attributes("-topmost", bool(args.topmost))

    txt = tk.Text(root, bg="#1e1e1e", fg="#e8e8e8", font=(F, 11), wrap="word",
                  bd=0, highlightthickness=0, padx=14, pady=12)
    txt.pack(fill="both", expand=True)
    txt.tag_configure("title", font=(F, 10), foreground="#909090")
    txt.tag_configure("lvl", font=(F, 26, "bold"), foreground="#4ea3ff",
                      spacing3=8)
    txt.tag_configure("h1", font=(F, 13, "bold"), foreground="#ffffff",
                      spacing1=10, spacing3=4)
    txt.tag_configure("grp", font=(F, 11), foreground="#d8d8d8", spacing1=2)
    txt.tag_configure("mine", font=(F, 16, "bold"), foreground="#ffb020",
                      spacing1=6, spacing3=6)
    txt.tag_configure("wait", font=(F, 16, "bold"), foreground="#909090",
                      spacing1=6, spacing3=6)
    txt.tag_configure("dim", font=(F, 9), foreground="#707070", spacing1=8)
    txt.tag_configure("err", font=(F, 12), foreground="#ff6b6b")
    txt.tag_configure("warn", font=(F, 11, "bold"), foreground="#ffa020")

    state = {"last": None}

    def tick():
        try:
            while True:
                state["last"] = q.get_nowait()
        except queue.Empty:
            pass
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        # 用 safe_render_lines：渲染抛错**不能**打断刷新链（after 一抛就永久停），
        # 但错误要显示出来、不许吞掉 —— 见 safe_render_lines 的 docstring。
        for text, tag in safe_render_lines(state["last"]):
            txt.insert("end", text, tag)
        txt.configure(state="disabled")
        root.after(max(150, int(args.interval * 1000)), tick)

    tick()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/guandan7/weights/best.pt")
    ap.add_argument("--title", default="掼蛋")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--imgsz2", type=int, default=1600,
                    help="第二遍（高分辨率）补漏用的尺寸；0 = 关掉")
    ap.add_argument("--imgsz2-conf", type=float, default=0.30,
                    help="第二遍的置信度门槛")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--interval", type=float, default=0.6)
    ap.add_argument("--change", type=float, default=0.4,
                    help="与上一帧差异超过此值算画面在动")
    ap.add_argument("--stable-frames", type=int, default=3,
                    help="连续稳定多少帧后才识别一次")
    ap.add_argument("--panel-width", type=int, default=430)
    ap.add_argument("--panel-height", type=int, default=880)
    ap.add_argument("--panel-x", type=int, default=10)
    ap.add_argument("--panel-y", type=int, default=120)
    ap.add_argument("--topmost", type=int, default=1)
    ap.add_argument("--no-panel", action="store_true")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit("[FAIL] 权重不存在: %s" % args.weights)

    q = queue.Queue(maxsize=4)
    w = Worker(args, q)
    w.start()
    try:
        if args.no_panel:
            while True:
                d = q.get()
                if d.get("status") != "ok":
                    print("status:", d.get("status"))
                    continue
                hand = " ".join(fmt_card(c) for c, _ in d["hand"])
                print("[级牌%s] 轮到我=%s  手牌%d张: %s"
                      % (d["level"], d["mine"], len(d["hand"]), hand))
        else:
            run_panel(args, q)
    except KeyboardInterrupt:
        pass
    finally:
        w.stop.set()


if __name__ == "__main__":
    main()
