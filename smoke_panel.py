"""面板离线冒烟测试：把 tkinter 界面真的建出来渲染一遍，然后自动关闭。

不开游戏、不用手动关窗口。用「自动销毁的 Tk 子类」替掉 tkinter.Tk，
run_panel 里 `tk.Tk()` 建出来的就是它，1.5 秒后自己 destroy，mainloop 自然返回。
销毁前把 Text 控件里的内容打出来 —— 这就等于看到了用户实际会看到的面板。
"""
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'synth'))
sys.path.insert(0, str(ROOT / 'live'))

import tkinter                                              # noqa: E402
import main as M                                            # noqa: E402

_OrigTk = tkinter.Tk


class AutoCloseTk(_OrigTk):
    """建好 1.5 秒后自动销毁，并把 Text 里的内容打出来。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.after(1500, self._dump_and_close)

    def _dump_and_close(self):
        try:
            for w in self.winfo_children():
                if isinstance(w, tkinter.Text):
                    print('----- 面板实际显示 -----')
                    print(w.get('1.0', 'end').rstrip())
                    print('------------------------')
        finally:
            self.destroy()


class Args:
    panel_width, panel_height = 330, 880
    panel_x, panel_y = 10, 120
    topmost, interval = 0, 0.6


CASES = [
    ('正常·三家都有出牌', {
        'status': 'ok', 'phase': 'normal', 'level': 'J', 'level_conf': 0.98,
        'hand': [('S8', 0.9), ('H8', 0.9), ('C8', 0.9), ('D8', 0.9),
                 ('JOKER_S', 0.9), ('CJ', 0.9)],
        'table': {'机器人1': [('S5', 0.9), ('H5', 0.9), ('C7', 0.9),
                              ('H7', 0.9), ('H7', 0.9)],
                  '机器人3': [], '队友': [('S9', 0.9)],
                  '我': [('DQ', 0.9), ('CT', 0.9)]},
        'mine': False, 'orange': 0.0, 'turn_side': 'left', 'elapsed': 2.1}),
    ('正常·轮到我·桌上没牌', {
        'status': 'ok', 'phase': 'normal', 'level': 'T', 'level_conf': 0.97,
        'hand': [('DJ', 0.94)],
        'table': {'机器人1': [], '机器人3': [], '队友': [], '我': []},
        'mine': True, 'orange': 0.24, 'turn_side': 'mine', 'elapsed': 0.18}),
    ('换局横幅', {'status': 'ok', 'phase': 'banner',
                  'phase_why': '换局横幅（还贡/接风）', 'elapsed': 0.1}),
    ('结算画面', {'status': 'ok', 'phase': 'result',
                  'phase_why': '结算/升段（亮度 239）', 'elapsed': 0.1}),
    ('进贡画面', {'status': 'ok', 'phase': 'tribute',
                  'phase_why': '进贡/还贡', 'elapsed': 0.1}),
    ('没找到窗口', {'status': 'no-window'}),
    ('还没连上', None),
]


def main() -> None:
    tkinter.Tk = AutoCloseTk
    a = Args()
    for label, payload in CASES:
        print('\n=== %s ===' % label)
        q = queue.Queue()
        q.put(payload if payload is not None else None)
        if payload is None:
            q = queue.Queue()          # 空队列 = 还没连上
        M.run_panel(a, q)
    tkinter.Tk = _OrigTk
    print('\n全部 7 种情况渲染通过，无异常')


if __name__ == '__main__':
    main()
