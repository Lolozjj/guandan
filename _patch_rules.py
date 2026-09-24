"""把牌型校验接进面板：每家出的牌后面标出牌型，不合法就报警。

用户提的建议：识别结果自己就能暴露错误（比如「两个 3 + 一个 10」在掼蛋里
根本不存在）。这种情况多半是牌被 UI 完全盖住、模型读不到 —— 那就该老实说
「不合法、可能有牌被挡住」，而不是把一个不存在的组合照直显示出来。
"""
from pathlib import Path

p = Path('live/main.py')
s = p.read_text(encoding='utf-8')

old = """from phase import detect_phase                     # noqa: E402"""
new = """from phase import detect_phase                     # noqa: E402
from rules import classify as classify_play        # noqa: E402"""
assert old in s
s = s.replace(old, new)

old = """    table = d["table"]
    if any(table.values()):
        out.append(("桌面出牌\\n", "h1"))
        for who in TABLE_ORDER:
            v = table.get(who) or []
            if not v:
                continue              # 没出牌的那家不占篇幅
            cols = [fmt_card(c) for c, _ in sorted(v, key=lambda t: -t[1])]
            out.append(("   %s  %d 张   %s\\n" % (who, len(v), " ".join(cols)),
                        "grp"))"""
new = """    table = d["table"]
    if any(table.values()):
        out.append(("桌面出牌\\n", "h1"))
        for who in TABLE_ORDER:
            v = table.get(who) or []
            if not v:
                continue              # 没出牌的那家不占篇幅
            cards = [c for c, _ in sorted(v, key=lambda t: -t[1])]
            cols = [fmt_card(c) for c in cards]
            out.append(("   %s  %d 张   %s\\n" % (who, len(v), " ".join(cols)),
                        "grp"))
            # 用掼蛋规则校验一下这个组合合不合法
            kind = classify_play(cards, d.get("level") or "2")
            if kind:
                out.append(("            %s\\n" % kind, "dim"))
            else:
                out.append(("            ⚠ 不是合法牌型，可能有牌被挡住没认出来\\n",
                            "warn"))"""
assert old in s
s = s.replace(old, new)

old = """    txt.tag_configure("err", font=(F, 12), foreground="#ff6b6b")"""
new = """    txt.tag_configure("err", font=(F, 12), foreground="#ff6b6b")
    txt.tag_configure("warn", font=(F, 11, "bold"), foreground="#ffa020")"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('面板已接入牌型校验')
