"""掼蛋牌型校验 —— 适配层（给 live/ 那条截图识别线用）。

**牌型真源在 `net/sim/meld.py`**（网络直读 + RL 是主路径）。这里只做
「牌面名 <-> 牌 ID」的转换再转发。

老实现有 6 处已证实的错（见 spec §2.3）：`"10"` vs `"T"` 混用、A 不能当小牌、
两个王不算对子、二连对判合法、docstring 说有三带一、逢人配能力低估。
**不要再改回老实现**，那等于把 bug 固化。

老实现的判型主体（`RANK_SEQ` / `_rank` / `_idx` / `_legal_exact` /
老 `classify` / 老 `describe`）已整体删除。留在这里的只有三件：
  - `_to_ids`  牌名 -> ID（词表是生产者那一套，见下）
  - `_stronger` 同一组牌有多个解释时取最强的那条
  - `classify` / `describe`  对外的旧接口（签名与旧版完全一致）

⚠️ **词表是 `synth/layout.py` 的 `CLASSES`（54 类），不是我们编的**：
`ST` 是十、王是 `JOKER_S` / `JOKER_B`、打十的级别是 `'T'` —— 因为
`live/main.py:229` 传进来的就是模型类名本身。词表对不上的后果**不是少个功能**：
`render_lines` 在 tkinter 的 `after` 回调里、且不在 `tick` 的 try 内，
抛错会打断刷新链，面板从此不再更新。所以 `tests/test_rules_adapter.py`
里有一条测试**直接拿 `synth.layout.CLASSES` 当输入**（见该测试的 docstring）。
"""
from __future__ import annotations

from net.sim import meld


def _to_ids(names: list) -> list:
    """牌名 -> 牌 ID。词表 = `synth.layout.CLASSES`（生产那一套）。

    认不出的名字由 `meld.cid_from_name` **抛 ValueError**，这里不 try/except：
    把「认不出」悄悄变成「不合法」正是老实现那种「静默返回 None」的坑
    （识别错了却看起来像规则判的）。它会一路抛到 `live/main.py` 的
    `render_lines`，那里没兜住 —— 这是**有意的**（宁可响也不要面板显示假结论），
    代价是词表必须与生产者对齐，所以有测试从生产者取材。
    """
    return [meld.cid_from_name(name) for name in names]


def _stronger(a: meld.Meld, b: meld.Meld) -> bool:
    """同一组牌能解释成多个牌型时，a 是不是更强的那条。

    口径与 `tools/accept_meld.py` 的 `_stronger` 一致（那边是验收①
    「取最强解释」用的）：先比炸弹层级，再比牌型大小。

    **这一步不能省**：`melds_from` 对同一组牌会给出多条解释 —— 天然牌型与
    逢人配补出来的牌型、顺子与同花顺（枚举顺序里顺子在前面）。取第一条会把
    `9♣10♣J♣Q♣+♥2` 报成「顺子」，而它是**同花顺**（炸弹，压 5 炸）——
    旧实现在这一点上是对的（`return "同花顺" if len(set(suits)) == 1`），
    换成适配层不能把它弄丢。
    """
    ca, cb = meld.bomb_class(a), meld.bomb_class(b)
    if (ca is None) != (cb is None):
        return ca is not None                 # 炸弹类优先
    if ca is not None and cb is not None and ca != cb:
        return ca > cb
    return a.kind > b.kind


def classify(cards: list[str], level: str = "2") -> str | None:
    """判牌型。合法返回牌型名，不合法返回 None。

    level 是当前级牌（'2'..'10' / 'J' / 'Q' / 'K' / 'A'）。
    **认不出的牌名或级别会抛 ValueError**（不是返回 None）—— 见 `_to_ids`。
    """
    if not cards:
        return None
    lv = meld.level_idx(level)
    ids = _to_ids(cards)
    want = sorted(ids)
    hits = [m for m in meld.melds_from(ids, level=lv) if sorted(m.cards) == want]
    if not hits:
        return None
    best = hits[0]
    for m in hits[1:]:
        if _stronger(m, best):
            best = m
    return meld.describe_meld(best) + ("（含逢人配）" if best.wild_used else "")


def describe(cards: list[str], level: str = "2") -> str:
    """给面板显示用：合法就写牌型，不合法就明确说不合法。"""
    return classify(cards, level) or "不合法"
