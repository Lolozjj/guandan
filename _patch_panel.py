"""把 main.py 里 tick() 的拼文字逻辑抽成模块级函数 render_lines()。

为什么要抽：抽出来之后不用开游戏、不用开 tkinter 就能测面板显示什么，
不然面板的显示逻辑只能靠肉眼读代码。
"""
from pathlib import Path

p = Path('live/main.py')
s = p.read_text(encoding='utf-8')

start = s.index('    def tick():')
end = s.index('    tick()\n    root.mainloop()')
tick = s[start:end]

# 逗号分隔的 ("文字", "tag") 列表 -> tkinter 调用
func = '''def render_lines(d) -> list[tuple[str, str]]:
    """把一份识别结果转成面板要显示的 (文字, 样式) 列表。

    抽成模块级函数是为了能脱离 tkinter 测 —— 面板显示逻辑不该只能靠读代码验证。
    """
    if not d:
        return [("正在连接游戏窗口…\\n", "h1")]
    if d.get("phase", "normal") != "normal":
        # 换局 / 结算 / 进贡：这时候画面上的牌不是「出牌」，一律不显示读数
        return [("换局中\\n", "wait"),
                ("   %s\\n" % d.get("phase_why", ""), "grp"),
                ("   等这一局开始再看\\n", "dim")]
    if d.get("status") == "no-window":
        return [("没找到游戏窗口\\n", "err"),
                ("请把掼蛋窗口打开并保持可见\\n", "dim")]

    out: list[tuple[str, str]] = []
    lv = d["level"]
    out.append(("当前级牌\\n", "title"))
    out.append(("   %s\\n" % (RANK_NAME.get(lv, lv) if lv else "?"), "lvl"))

    if d["mine"]:
        out.append(("● 轮到我出牌\\n", "mine"))
    else:
        out.append(("○ 等待中 (%s)\\n" % WHO.get(d["turn_side"], "?"), "wait"))

    out.append(("我的手牌  %d 张\\n" % len(d["hand"]), "h1"))
    if not d["hand"]:
        out.append(("  (未识别到手牌)\\n", "dim"))
    for key, cards in group_hand(d["hand"], d["level"]):
        label = "王" if key.startswith("JOKER") else RANK_NAME.get(key, key)
        out.append(("  %-3s x%d   %s\\n" % (label, len(cards), " ".join(cards)),
                    "grp"))

    table = d["table"]
    if any(table.values()):
        out.append(("桌面出牌\\n", "h1"))
        for who in TABLE_ORDER:
            v = table.get(who) or []
            if not v:
                continue              # 没出牌的那家不占篇幅
            cols = [fmt_card(c) for c, _ in sorted(v, key=lambda t: -t[1])]
            out.append(("   %s  %d 张   %s\\n" % (who, len(v), " ".join(cols)),
                        "grp"))

    out.append(("\\n识别 %.0fms   级牌置信 %.2f   按钮橙色 %.2f\\n"
                % (d["elapsed"] * 1000, d["level_conf"], d["orange"]), "dim"))
    return out


'''

# 新的 tick：只负责把 render_lines 的结果画到 Text 上
new_tick = '''    def tick():
        try:
            while True:
                state["last"] = q.get_nowait()
        except queue.Empty:
            pass
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        for text, tag in render_lines(state["last"]):
            txt.insert("end", text, tag)
        txt.configure(state="disabled")
        root.after(max(150, int(args.interval * 1000)), tick)

'''

s = s[:start] + new_tick + s[end:]
# 把 render_lines 插到 run_panel 之前
anchor = 'def run_panel(args, q):'
s = s.replace(anchor, func + anchor, 1)
p.write_text(s, encoding='utf-8')
print('render_lines 已抽出')
