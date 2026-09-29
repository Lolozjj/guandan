"""自对弈回放的**图形版** —— 点「上一步 / 下一步」逐步看模型是怎么打的。

- **共用同一份重放**：`tools/show_game.py::replay_game`（文字版也是它）
- **共用同一个渲染器**：`net/table.py::TableWindow` —— 所以这里的牌桌跟**实机面板
  长得一模一样**，而且不存在「两份画牌代码」
- 与实机面板的唯一区别：候选**写出 Q 值**（用户 2026-09-28 定）。那是离线复盘，
  看的就是「当时每个候选打了多少分」；实机面板仍然不写数字（建议上屏会影响人打牌）

用法：
    .venv/Scripts/python.exe -m tools.game_viewer                 # 挑最新 best.pt
    .venv/Scripts/python.exe -m tools.game_viewer --seed 61
    .venv/Scripts/python.exe -m tools.game_viewer --opp-kind rule   # 模型 vs 规则式对手
    .venv/Scripts/python.exe -m tools.game_viewer --selftest       # 建窗口→翻到底→自关

⚠️ `--opp-kind` 时**对手那一队没有 Q 值**（空着）—— 拿模型的 Q 给规则式的牌打分
是编数据。底部提示条与右栏都会写明「这一手是谁出的」。

键盘：`←` 上一步、`→` 下一步、`Home`/`End` 跳首尾。
"""
from __future__ import annotations

import argparse
import sys

from net import table
from net.state import GameState
from net.sim import rules
from tools.accept_meld import _utf8_stdout
from tools.show_game import (TEAM_NAME, advice_of, frame_hint, replay_game,
                             seat_label)
from train.selfplay import OPP_KIND_CN, OPP_KINDS   # 对手类型唯一产地

BAR_H = 56                       # 底部按钮条的高度（画布往下拉这么多）


class Cursor:
    """翻页的位置 —— **纯逻辑，不开窗口就能测**（本仓库的规矩：
    面板的显示逻辑要能不打开窗口就测）。越界自动夹住，不许越到帧外面去。"""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("没有帧可以看")
        self.n, self.i = n, 0

    def goto(self, i: int) -> None:
        self.i = max(0, min(self.n - 1, i))

    def prev(self) -> None:
        self.goto(self.i - 1)

    def next(self) -> None:
        self.goto(self.i + 1)

    @property
    def at_start(self) -> bool:
        return self.i == 0

    @property
    def at_end(self) -> bool:
        return self.i == self.n - 1


class _View:
    """喂给 `TableWindow.draw` 的数据形状。

    ⚠️ **这不是复制 `GameState` 的逻辑**，只是把渲染器要的那十来个成员拼出来：
    能用真的 `GameState`（借它的 `level_name` / `hand_grouped`）、真的
    `net.state.Play`（装出牌与台面，借它的 `names`）。所以「牌按掼蛋大小排」
    这件事仍然只有一份实现。
    渲染器将来要是多要一个字段，这里会**当场 AttributeError** —— 不会静默画错。
    """

    def __init__(self, f, me: int = 0):
        self.me = me
        self.level = f.level
        self.turn = f.turn
        self.plays = [] if f.turn is None else [None] * f.step
        self.history = f.played
        self.remaining = {s: len(f.hands[s]) for s in rules.SEATS}
        #: 四家手牌（**回放器独有** —— 实机面板拿不到，所以 `GameState` 没这个方法）。
        #: 用户 2026-09-29 要的：一眼看全四家，而不是「走到谁才看得到谁的手牌」。
        self._hands = {s: set(f.hands[s]) for s in rules.SEATS}
        self.table = f.table
        self.passes = f.passes
        # 露给渲染器的手牌是**出手那个人**的（复盘要看的就是「他当时握着什么」），
        # 而四个方位固定以 `me` 为底 —— 不然每点一步整桌转一次，认不出谁是谁
        self._gs = GameState(level=f.level, hand=sorted(f.hands[_whose(f)]))

    def hand_of(self, seat):
        """这一家的手牌（`net/table.py::_seat_area` 会问它）。"""
        if seat not in self._hands:
            return None
        return sorted(self._hands[seat])

    @property
    def hand(self):
        return self._gs.hand

    def hand_grouped(self):
        return self._gs.hand_grouped()

    def level_name(self):
        return self._gs.level_name()

    def seat_label(self, s: int) -> str:
        return seat_label(s)


def _whose(f) -> int:
    """这一帧该露谁的手牌：出手那个人；终局帧就露 0 号位。"""
    return f.seat if f.turn is not None else 0


class Viewer:
    """牌桌 + 底部两个按钮。**表由 `TableWindow` 画，按钮条由这里画** ——
    所以新旧两块各管各的，谁都不会覆盖谁（`draw` 会 `delete("all")`，
    所以按钮条必须在它**之后**画）。"""

    def __init__(self, root, meta: dict, frames: list):
        self.root, self.meta, self.frames = root, meta, frames
        self.cur = Cursor(len(frames))
        self.win = table.TableWindow(root)
        self.win.cv.config(height=table.H + BAR_H)
        mode = (f"模型 vs {OPP_KIND_CN[meta['opp_kind']]}"
                if meta.get("opp_kind") else "自对弈")
        root.title(f"{mode} —— {meta['path']}（种子 {meta['seed']}）")
        for key, fn in (("<Left>", self.cur.prev), ("<Right>", self.cur.next),
                        ("<Home>", lambda: self.cur.goto(0)),
                        ("<End>", lambda: self.cur.goto(len(frames) - 1))):
            root.bind(key, lambda _e, fn=fn: (fn(), self.render()))
        self.render()

    def render(self) -> None:
        f = self.frames[self.cur.i]
        head = (f"终局（共 {self.meta['steps']} 手）" if f.over
                else f"第 {self.cur.i + 1} / {len(self.frames)} 手")
        self.win.draw(
            _View(f), frame_hint(f), 0, advice=advice_of(f), show_q=True,
            top_right=head,
            # 对手帧没有候选 —— 右栏得说清楚**为什么**空着，不能显示
            # 「轮到我时显示」（那句话在复盘里是错的，会让人以为这里是模型的决策点）
            advice_empty=(f"{OPP_KIND_CN[f.opp]}对手出的这一手（不打分）"
                          if f.opp else "（轮到我时显示）"),
            hand_label=f"{seat_label(_whose(f))} 的手牌"
                       f"（{len(f.hands[_whose(f)])} 张）")
        self._bar(head)

    # ---------------------------------------------------------- 底部按钮条

    def _bar(self, head: str) -> None:
        cv = self.win.cv
        y0 = table.H + 6
        cv.create_line(20, y0, table.W - 20, y0, fill=table.BG_EDGE, width=2)
        cy = y0 + 26
        self._button(40, cy, "◀  上一步", "prev", self.cur.prev,
                     not self.cur.at_start)
        self._button(225, cy, "下一步  ▶", "next", self.cur.next,
                     not self.cur.at_end)
        self._button(410, cy, "↺  从头看", "replay", lambda: self.cur.goto(0),
                     not self.cur.at_start)
        cv.create_text(620, cy, anchor="w", text=head,
                       font=self.win.f_mid, fill=table.TEXT)
        m = self.meta
        if self.frames[self.cur.i].over:
            cv.create_text(780, cy, anchor="w",
                           text=f"—— {TEAM_NAME[m['winner']]}队赢，得 {m['points']} 分"
                                f"（炸弹 {m['bombs']} 手，白炸 {m['waste']}/{m['chance']}）",
                           font=self.win.f_small, fill=table.DIM)
        cv.create_text(table.W - 40, cy, anchor="e",
                       text="← → 翻页 · Home / End 跳首尾",
                       font=self.win.f_small, fill=table.DIM)

    def _button(self, x, y, text, tag, cb, active: bool) -> None:
        cv, w, h = self.win.cv, 170, 32
        cv.create_rectangle(x, y - h / 2, x + w, y + h / 2, tags=tag,
                            outline=table.TURN if active else table.BG_EDGE,
                            width=2, fill=table.PANEL)
        cv.create_text(x + w / 2, y, text=text, tags=tag,
                       font=self.win.f_mid,
                       fill=table.TEXT if active else table.DIM)
        if active:      # `tag_bind` 是**替换**不是叠加，所以重复渲染不会点一下跳两格
            cv.tag_bind(tag, "<Button-1>", lambda _e: (cb(), self.render()))


def open_viewer(meta: dict, frames: list, autoclose_s: float = 0.0) -> None:
    """开窗口。`autoclose_s > 0` 时到点自己关 —— 给 `--selftest` 用
    （照 `smoke_panel.py` 的路子：把界面真建出来渲染一遍再关掉）。"""
    import tkinter as tk
    root = tk.Tk()
    view = Viewer(root, meta, frames)
    if autoclose_s:
        def sweep(n=0):
            if n <= len(frames):                 # 从头翻到尾，每帧都真画一遍
                view.cur.goto(n)
                view.render()
                root.after(4, sweep, n + 1)
            else:
                print(f"✓ 自检通过：{len(frames)} 帧全部渲染、按钮逻辑走通")
                root.destroy()
        root.after(50, sweep)
    root.mainloop()


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--selftest", action="store_true",
                    help="建窗口 → 从头翻到尾 → 自己关（不用手动点）")
    ap.add_argument("--opp-kind", choices=OPP_KINDS, default=None,
                    help="对手那一队走固定对手（默认 None = 自对弈）")
    ap.add_argument("--opp-team", type=int, choices=(0, 1), default=1,
                    help="哪一队当对手（默认乙队 = 座位 1、3）")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    what = (f"模型 vs {OPP_KIND_CN[a.opp_kind]}" if a.opp_kind else "一局自对弈")
    print(f"正在打{what}…（约 5 秒）")
    meta, frames = replay_game(a.weights, seed=a.seed, level=a.level, top=a.top,
                               opp_kind=a.opp_kind, opp_team=a.opp_team)
    print(f"打完了：{meta['steps']} 手，{TEAM_NAME[meta['winner']]}队赢。开窗口…")
    open_viewer(meta, frames, autoclose_s=8.0 if a.selftest else 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
