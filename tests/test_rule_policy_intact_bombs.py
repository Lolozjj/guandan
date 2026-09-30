"""规则式**不许拆自己的炸** —— 用户 2026-09-29 报的那一手。

原话：「这个规则式，为啥第一手就把炸弹毫无意义的给拆了」。

查出两层毛病，**都在「取舍口径」这一层**（移植来的是「优先三带二」这条规则，
口径是我写的）：

1. `_key` 只看**三张**的点数 ⇒ `222+33` / `222+77` / `222+KK` 的键
   **完全一样**（11 个候选同分），`min` 退化成「按枚举顺序抽签」。
2. 没有任何一处看「这一手的牌是不是从我某个炸里挖出来的」——
   `is_fire(m)` 判的是**这一手本身**是不是炸，`222+77` 不是炸，就当普通牌放行。

实测代价（40 局，模型 1407 打对面）：规则式 **4.8% 的决策在拆自己的炸，
其中 4.7% 本可不拆**，波及 39/40 局；模型自己只有 2.3%。
⇒ 尺子被削弱了，`vs 规则式 88.2%` 是**虚高**的（该数已作废，重测见台账）。
"""
import pytest

from guandan.sim import meld
from tests.test_rule_policy import C, _acts, _follow_obs, _idx, _obs
from guandan.rl import rule_policy as rp


def d2(name: str) -> int:
    """二副的牌（牌 ID + 256）—— 战报里的 `(二副)`。"""
    return meld.cid_from_name(name, deck=2)


def _seat3_hand():
    """种子 61 第 1 手之前，座位3 的那 27 张（从战报原样抄，含二副）。

    四张 7 = 它整局唯一的**炸**；三张 2 = 领出时拿去配三带二的三张。
    """
    return C("JOKER_S", "H8", "DA", "HK", "DJ", "ST", "CT", "H9", "S7", "H7",
             "D5", "D3", "S2", "D2") + [
        d2("S8"), d2("D8"), d2("HK"), d2("HQ"), d2("CT"), d2("S9"), d2("D9"),
        d2("S7"), d2("C7"), d2("S4"), d2("D4"), d2("H3"), d2("D2")]


# ------------------------------------------------------------------ 判定本身

def test_four_of_a_kind_mined_for_one_to_three_cards_counts_as_breaking_it():
    hand = C("S7", "H7", "D7", "C7", "S3")
    assert rp.breaks_bomb(hand, meld.as_meld(C("S7", "H7"), 8))          # 一对
    assert rp.breaks_bomb(hand, meld.as_meld(C("S7", "H7", "D7"), 8))    # 三张
    assert not rp.breaks_bomb(hand, meld.as_meld(C("S7", "H7", "D7", "C7"), 8))  # 把炸打出去
    assert not rp.breaks_bomb(hand, meld.as_meld(C("S3"), 8))            # 跟炸无关的单张


def test_breaking_needs_four_in_hand_not_three():
    """手里只有三张 7 —— 出这一对/三张都**不是**拆炸（本来就不是炸）。"""
    hand = C("S7", "H7", "D7", "S3")
    assert not rp.breaks_bomb(hand, meld.as_meld(C("S7", "H7"), 8))
    assert not rp.breaks_bomb(hand, None)


def test_a_wild_used_as_the_pair_still_counts_as_breaking_the_bomb():
    """⚠️ **牌面点数会骗人**：`8♥` 在这一局是逢人配。

    `8♠8♥8♣8♦J♠`（8♥ 当 J 用去配那一对）里真被吃掉的 8 只有 **3 张** ——
    8 炸没了。但按牌面点数看，4 个 8 全在牌里，于是「四张全用掉 = 打的就是这个炸」
    那条豁免会**错放**它过。所以豁免必须**同时**要求 `is_bomb`。
    """
    hand = C("S8", "D8", "C8", "H8", "SJ")
    acts = meld.melds_from(sorted(hand), 8)
    m = next(x for x in acts if x.size == 5 and not x.is_bomb)
    assert m.wild_used == 1, "这条测试的前提是逢人配被当别的牌用了"
    assert rp.breaks_bomb(hand, m)


# ------------------------------------------------------------------ 用户报的那一手

def test_does_not_dig_into_its_own_bomb_when_another_pair_would_do():
    """**用户那一手**：四个 7 里抽两张去配三带二，而手里有 33/44 可以配。

    修之前选的是 `7♠ 7♥ 2♠2♦2♦`（枚举顺序里的第一个，纯抽签）。
    """
    hand = _seat3_hand()
    o = _obs(seat=3, hand=hand, level=8)
    acts = meld.melds_from(sorted(hand), 8)
    m = acts[rp.rule_choose(o, acts)]
    assert not rp.breaks_bomb(hand, m), f"拆了自己的炸：{m}"


def test_the_triple_pair_shortcut_is_what_refuses_to_break_the_bomb(monkeypatch):
    """把 **[源 2]**（按计划出牌）关掉，单独钉 **[源 4]** 的那道闸。

    ⚠️ 为什么要拆开：`SOURCE2_MAX_HANDS` 从 3 提到 12 之后（实测 +4pp），
    [源 2] 会**先于** [源 4] 触发，上面那条测试因此不再经过三带二这条捷径。
    原来把「必须出三带二」写进上面那条 —— 那是**过度指定**：
    它把 [源 4] 的行为当成了整体契约，而 [源 2] 合理地排在它前面。
    """
    monkeypatch.setattr(rp, "SOURCE2_MAX_HANDS", 0)      # 2 <= len(hands) <= 0 不可能
    hand = _seat3_hand()
    o = _obs(seat=3, hand=hand, level=8)
    acts = meld.melds_from(sorted(hand), 8)
    m = acts[rp.rule_choose(o, acts)]
    assert not rp.breaks_bomb(hand, m), f"拆了自己的炸：{m}"
    assert m.kind == meld.TRIPLE_PAIR, "关掉 [源 2] 之后，这一步该按 [源 4] 出三带二"


def test_triple_pair_tie_goes_to_the_smaller_pair():
    """`222+33` 与 `222+KK` 的 `_key` 完全同分 —— 细化键必须挑**小的那一对**。"""
    # ⚠️ 牌的**先后**是这条测试的关键：`melds_from` 按牌在排序后的手牌里
    # **首次出现**的顺序产出候选对子。`SK`(29) 排在 `H3`(35) 前面 ⇒ K 那对
    # 先被枚举 ⇒ 修之前抽签抽到的是 `222+KK`（错的）。修完必须挑 33。
    hand = C("S2", "H2", "D2", "SK", "HK", "H3", "D3")
    o = _obs(seat=3, hand=hand, level=8)
    acts = meld.melds_from(sorted(hand), 8)
    assert rp.rule_choose(o, acts) == _idx(acts, C("S2", "H2", "D2", "H3", "D3"))


def test_leading_skips_the_triple_pair_shortcut_when_it_would_break_a_bomb():
    """[源 4]「优先三带二（一次消 5 张）」**不能**盖过「不拆自己的炸」。

    四个 J + 一对 5：`JJJ+55` 是唯一的三带二，也就是**必须抽 3 张 J**。
    实测（40 局）剩下的拆炸里 37 例都是这个形状 —— 为了消 5 张把炸拆了。
    不拆的出路是有的（这个 J 炸本身、55、单张），所以该走后者。
    """
    hand = C("SJ", "HJ", "DJ", "CJ", "S5", "H5", "S9", "ST")
    o = _obs(seat=0, hand=hand, level=8)
    acts = meld.melds_from(sorted(hand), 8)
    m = acts[rp.rule_choose(o, acts)]
    assert not rp.breaks_bomb(hand, m), f"为凑三带二拆了 J 炸：{m}"


# ------------------------------------------------------------------ 偏好，不是禁令

def test_breaking_the_bomb_is_still_allowed_when_nothing_else_can_beat_it():
    """手里**唯一**能压的普通牌型就埋在炸里（跟三张）⇒ 该拆还得拆。

    ⚠️ 这条**不是**为「拆炸」辩护，是防一个很自然的错误实现：把偏好做成
    **全局过滤**（先在所有候选里剔掉拆炸的）。那样一剔，`_cheapest_plain`
    就只剩「整个炸」一条路，于是**跟一张三张会被逼着开一个炸** —— 比拆炸更蠢。
    """
    hand = C("S7", "H7", "D7", "C7", "S3")
    t = C("S5", "H5", "D5")
    o = _follow_obs(20, t, meld.TRIPLE, meld.point_value(5, 8), owner=1, hand=hand)
    acts = meld.melds_from(sorted(hand), 8)
    acts = [m for m in acts if meld.beats(m, meld.as_meld(t, 8))] + [None]
    picked = acts[rp.rule_choose(o, acts)]
    assert picked is not None and picked.kind == meld.TRIPLE
    assert picked.size == 3, f"被逼着开了整个炸：{picked}"
