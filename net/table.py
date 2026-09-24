"""模拟牌桌的图形面板（tkinter）。

布局照着游戏里来：**队友在上、左对手在左、右对手在右、我在下**，
而且**每家出过的牌就摆在那一家的位置上**（不是旁边一栏文字）——
跟真的牌桌一样，牌打出来就留在自己面前。

几个坑，都不是随便选的：
- 微软雅黑**没有 ♠♥♣♦ 字形**（画出来是方框），所以花色单独用
  「Segoe UI Symbol」画。项目里 `make_table_fig.py` 注释里踩过同一个坑。
- **座位是运行时认出来的，不是写死的**：报文里的座位号每局都会变
  （同一台机器，16:01 那局我是 seat3、16:33 那局我是 seat1）。
  判据是 LeftCardList（字段 3.6.10）—— 只有自己的出牌带这个字段，
  见 `state.on_play`。方位只由相对座位决定（`state.seat_label`）。
- 牌面一律**按掼蛋大小排序**：大王 > 小王 > 级牌 > A > K > Q > J > 10 > … > 2。
  级牌跟着当前级别走（打 5 的时候四个 5 排在 A 前面），所以排序必须带上
  `state.level` —— 只按牌 ID 排是错的。

跑法：
    python -m net.table              # 实时（要 mitmproxy 在跑）
    python -m net.table --replay     # 用抓包文件回放，看长什么样
"""

import argparse
import json
import sys

from . import cards, protocol
from .addon import EVENTS
from .levelwatch import LevelWatcher
from .state import GameState

# 配色（照着游戏的绿桌布调）
BG = "#0d5f52"
BG_EDGE = "#0a4a40"
PANEL = "#0b5347"
TEXT = "#eaf6f2"
DIM = "#8fc4b8"
CARD_BG = "#fbfbf8"
CARD_EDGE = "#9aa0a6"
RED = "#c62828"
BLACK = "#22262b"
TURN = "#ffd54f"
FONT = "Microsoft YaHei"
FONT_SYM = "Segoe UI Symbol"

W, H = 1200, 830
CARD = (34, 46)             # 出过的牌
MY_CARD = (52, 74)          # 我的手牌
STEP = 21                   # 出过的牌横向步进（叠着放，露出左上角点数）
MY_STEP = 34


def parse_name(name: str):
    """牌面名 -> (点数, 花色, 是不是副牌)。'K♦' / '小王' / '5♥(二副)'。"""
    deck = "(二副)" in name
    core = name.replace("(二副)", "")
    if core in ("小王", "大王"):
        return core, "", deck
    return core[:-1], core[-1], deck


class TableWindow:
    """四个方位各自一块区域。每家出过的牌就铺在自己的区域里。"""

    # 每家的摆放区域 (x0, y0, x1, y1)，以及标签位置
    AREA = {
        "top": (600, 40, 300, 64, 900, 250),      # (标签x, 标签y, x0, y0, x1, y1)
        "left": (40, 250, 30, 276, 385, 600),
        "right": (1160, 250, 815, 276, 1170, 600),
        "me": (600, 560, 300, 580, 900, 690),
    }

    def __init__(self, root):
        import tkinter as tk
        self.tk = tk
        root.title("掼蛋牌桌 —— 数据直读网络，无截图")
        root.configure(bg=BG)
        self.cv = tk.Canvas(root, width=W, height=H, bg=BG,
                            highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.f_big = (FONT, 15, "bold")
        self.f_mid = (FONT, 12, "bold")
        self.f_small = (FONT, 10)
        self.f_rank = (FONT, 13, "bold")
        self.f_sym = (FONT_SYM, 15)
        self.f_rank_s = (FONT, 9, "bold")
        self.f_sym_s = (FONT_SYM, 10)

    # ------------------------------------------------------------ 画牌

    def _card(self, x, y, size, name, small=False):
        w, h = size
        rank, suit, deck = parse_name(name)
        color = RED if suit in "♥♦" else BLACK
        self.cv.create_rectangle(x, y, x + w, y + h, fill=CARD_BG,
                                 outline=CARD_EDGE, width=1)
        if deck:                       # 副牌右上角点一个金点，跟游戏一致
            self.cv.create_oval(x + w - 9, y + 4, x + w - 4, y + 9,
                                fill="#c9a227", outline="")
        fr = self.f_rank_s if small else self.f_rank
        fs = self.f_sym_s if small else self.f_sym
        if not suit:                   # 王
            self.cv.create_text(x + w / 2, y + h / 2, text=rank,
                                font=(FONT, 9 if small else 11),
                                fill=RED if rank == "大王" else BLACK)
            return
        self.cv.create_text(x + 4, y + 3, text=rank, anchor="nw",
                            font=fr, fill=color)
        self.cv.create_text(x + w / 2, y + h * 0.66, text=suit,
                            anchor="center", font=fs, fill=color)

    def _flow(self, names, x0, y0, x1, y1, step=STEP, size=CARD):
        """把一串牌从左到右铺开，铺不下就换行；行数超出就直接停。

        返回实际铺下的张数（不够铺时会小于 len(names)，调用方可以提示）。
        """
        if not names:
            return 0
        w, h = size
        per = max(1, int((x1 - x0 - w) / step) + 1)
        done = 0
        y = y0
        for i in range(0, len(names), per):
            if y + h > y1:
                break
            for j, name in enumerate(names[i:i + per]):
                self._card(x0 + j * step, y, size, name, small=(size is CARD))
                done += 1
            y += h + 4
        return done

    # ------------------------------------------------------------ 一家区域

    def _seat_area(self, st, seat, key):
        """画一家的区域：标签 + 剩几张 + 本局出过的所有牌。"""
        lx, ly, x0, y0, x1, y1 = self.AREA[key]
        hs = st.history.get(seat) or []
        left = st.remaining.get(seat)
        head = st.seat_label(seat)
        if left is not None:
            head += f"   剩 {left} 张"
        if hs:
            head += f"   已出 {sum(len(p.cards) for p in hs)} 张"
        color = TURN if st.turn == seat else TEXT
        anchor = "center" if key in ("top", "me") else ("w" if key == "left" else "e")
        self.cv.create_text(lx, ly, text=head, font=self.f_mid, fill=color,
                            anchor=anchor)
        if not hs:
            return
        # **整体排一次**，不是每手各排各的 ——
        # 每手内部排好、再按时间拼接，整体看还是乱的（用户一眼就看出来了）。
        # 合并之后按掼蛋大小重排，四个 K 自然挨在一起，跟记牌器一样。
        every = []
        for p in hs:
            every.extend(p.cards)
        self._flow(cards.names_sorted(every, st.level), x0, y0, x1, y1)

    # ------------------------------------------------------------ 主绘制

    def draw(self, st, hint, n_ev):
        cv = self.cv
        cv.delete("all")
        cv.create_rectangle(14, 14, W - 14, H - 14, outline=BG_EDGE, width=3)

        # 顶部信息条
        cv.create_text(34, 26, text=f"级别  打{st.level_name()}",
                       anchor="w", font=self.f_big, fill=TURN)
        turn = ("本局结束" if st.turn is None and st.plays
                else (st.seat_label(st.turn) if st.turn is not None
                      else "等待发牌"))
        cv.create_text(W / 2, 26, text=f"轮到：{turn}", font=self.f_big,
                       fill=TEXT)
        cv.create_text(W - 34, 26, text=f"已收 {n_ev} 个事件",
                       anchor="e", font=self.f_small, fill=DIM)

        # 桌子中央：当前待压的牌
        cx, cy = 600, 400
        cv.create_rectangle(cx - 175, cy - 68, cx + 175, cy + 68,
                            outline=BG_EDGE, width=2)
        cv.create_text(cx, cy - 52, text="桌面（待压）", font=self.f_small,
                       fill=DIM)
        if st.table and st.table.cards:
            names = st.table.names(st.level)
            w, h = 46, 62
            step = 30
            total = step * (len(names) - 1) + w
            x = cx - total / 2
            for n in names:
                self._card(x, cy - 30, (w, h), n)
                x += step
            cv.create_text(cx, cy + 52,
                           text=f"{st.seat_label(st.table.seat)} 出的",
                           font=self.f_small, fill=DIM)
        else:
            cv.create_text(cx, cy, text="—", font=self.f_big, fill=DIM)

        # 四家
        self._seat_area(st, (st.me + 2) % 4, "top")     # 队友 上
        self._seat_area(st, (st.me + 1) % 4, "left")    # 左
        self._seat_area(st, (st.me + 3) % 4, "right")   # 右
        self._seat_area(st, st.me, "me")                # 我

        # 我的手牌
        hx0, hy = 240, H - 128
        self._flow(st.hand_grouped(), hx0, hy, W - 240, H - 50,
                   step=MY_STEP, size=MY_CARD)
        cv.create_text(W / 2, H - 108,
                       text=f"我的手牌（{len(st.hand)} 张，按大小排序）",
                       font=self.f_small, fill=DIM)
        cv.create_text(W / 2, H - 24, text=hint, font=self.f_small, fill=DIM)

        if st.passes:
            cv.create_text(600, 496, text="要不起：" + "、".join(
                st.seat_label(s) for s in st.passes),
                font=self.f_small, fill=DIM)


# ------------------------------------------------------------------ 跑起来

def _replay_frames(capture):
    out = []
    with open(capture, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") == "ws_msg" and "hlxyxws" in str(r.get("host")) \
                    and r.get("dir") == "S→C":
                out.append(r)
    out.sort(key=lambda r: r["t"])
    return out


def run_live(st, events_path=EVENTS, seconds=0):
    import tkinter as tk
    from .panel import Tailer, apply_event
    root = tk.Tk()
    win = TableWindow(root)
    tail, lvl = Tailer(events_path), LevelWatcher()
    box = {"n": 0, "hint": "等游戏数据…（打开掼蛋打一局）"}

    def tick():
        for ev in tail.read():
            h = apply_event(st, ev)
            if h:
                box["hint"] = h
            box["n"] += 1
        lv = lvl.poll()
        if lv is not None:
            st.level = lv
            box["hint"] = f"级别更新：打{st.level_name()}"
        win.draw(st, box["hint"], box["n"])
        root.after(150, tick)

    if seconds:
        root.after(int(seconds * 1000), root.destroy)
    tick()
    root.mainloop()


def run_replay(st, capture, delay_ms=260, seconds=0):
    import tkinter as tk
    root = tk.Tk()
    win = TableWindow(root)
    it = iter(_replay_frames(capture))
    box = {"n": 0, "hint": "回放中…"}

    def apply(msg):
        if msg["msgid"] == 3005:
            p = protocol.decode_play(msg["fields"])
            if p:
                st.on_play(p["seat"], p["cards"], p["card_type"],
                           p["next"], p["left"], p.get("left_cards"))
                box["n"] += 1
                box["hint"] = (f"{st.seat_label(p['seat'])} 出 "
                               f"{' '.join(cards.decode_all(p['cards']))}")
        elif msg["msgid"] == 3019:
            h = protocol.decode_hand(msg["fields"])
            if h:
                st.on_hand(h["cards"])
                box["n"] += 1
                box["hint"] = f"手牌同步 {len(h['cards'])} 张"
        elif msg["msgid"] == 3006:
            q = protocol.decode_pass(msg["fields"])
            if q:
                st.on_pass(q["seat"], q["next"])
                box["hint"] = f"{st.seat_label(q['seat'])} 要不起"

    def tick():
        try:
            for _ in range(3):
                m = protocol.parse(bytes.fromhex(next(it)["hex"]))
                if m:
                    apply(m)
        except StopIteration:
            box["hint"] = "回放结束"
        win.draw(st, box["hint"], box["n"])
        root.after(delay_ms, tick)

    if seconds:
        root.after(int(seconds * 1000), root.destroy)
    tick()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--capture", default=r"C:\Users\17837\mitmtool\capture.jsonl")
    ap.add_argument("--seconds", type=float, default=0,
                    help="N 秒后自动关闭（自测用）")
    args = ap.parse_args()
    st = GameState()
    if args.replay:
        run_replay(st, args.capture, seconds=args.seconds)
    else:
        run_live(st, seconds=args.seconds)


if __name__ == "__main__":
    sys.exit(main())
