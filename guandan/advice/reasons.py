"""**解释型建议**：除了"出哪一手"，再给一行"**为什么**"。

为什么这是"打人"的关键一步（`plans/2026-10-03-human-play.md` §4.3 的结构性结论）：
量下来 **队友弱时胜负由队友决定**（我这边 +1~2pp 的强度改动在期望上被淹没 ✗），
⇒ 剩下的杠杆是**交互**：人不缺那一手，缺的是**为什么** ✓
（"喂你：你只剩一对" / "他只剩 1 张，压死他" / "别拿炸压这种牌"）

⚠️ **只解释、不改动作**：`explain(...)` 只看已经选定的那一手 ⇒
对强度**零影响**（不需要任何尺子复核），改的只是人能看懂多少 ✓
"""
from __future__ import annotations

from guandan.rl.eval import is_upgraded_wild, is_wasted_bomb, is_wasted_wild
from guandan.sim import meld, rules

#: 「快走完」的阈值（与 rule_policy 的 short_push 同口径）
SHORT_LEFT = 3


def explain(obs, acts, chosen_i: int, *, feed_left: int = 2) -> str:
    """给**已经选定的那一手**配一句人话理由（没有可说的地方就返回空串）。"""
    m = acts[chosen_i]
    mate = rules.PARTNER[obs.seat]
    opps = [s for s in rules.SEATS if rules.TEAM[s] != rules.TEAM[obs.seat]]
    short = min((obs.left[o] for o in opps), default=99)

    # ① 喂队友（人类队友最在意的一项）
    if not obs.table and 0 < obs.left[mate] <= feed_left and m is not None:
        need = meld.SINGLE if obs.left[mate] == 1 else meld.PAIR
        if m.kind == need:
            what = "一张" if obs.left[mate] == 1 else "一对"
            return f"喂队友：他只差 {what} 就走完了，给他一个接得住的出口"

    # ② 压死快走完的对手
    if short <= SHORT_LEFT and m is not None:
        return f"压死他：有对手只剩 {short} 张，出大牌别给他接的机会"

    # ③ 别浪费资源（与"擦浪费"同判定；只说人话，不改动作）
    if is_wasted_bomb(obs.table, acts, m):
        return "省着：有普通牌能压，别拿炸弹换这一手"
    if is_upgraded_wild(obs.table, acts, m):
        return "省着：天然牌就够，别用万能牌把这一手做大"
    if is_wasted_wild(obs.table, acts, m):
        return "省着：不用万能牌也能压，留着它凑炸/接牌"

    # ④ 抢回出牌权（我方能一把走完时）
    if m is not None and set(m.cards) == set(obs.hand):
        return "这一手能直接走完，抢"

    return ""
