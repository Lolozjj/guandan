"""缓存的正确性契约（不是「快不快」，是「会不会给旧答案」）。"""
from net.sim import meld


def test_enumeration_is_not_stale_after_the_hand_changes_in_place():
    """缓存必须按**牌面快照**键，不能按对象身份 —— 手牌是**就地改**的
    （`Hand.play()` 干的就是 `self.hands[seat] -= cs`）。

    实测：同一手牌里同一种牌面会被反复枚举（300 局里重复率 83%），所以要缓存；
    但键必须是「这一手是哪些牌」的快照，否则改完手牌还会拿到旧结果。

    ⚠️ **必须是同一个 list 对象就地改。** 换成新 list 的话，
    「按对象身份键」这个错误实现也能通过 —— 这条测试就等于没测。
    真实调用方就是这么干的：`Hand.play()` 改的是 `hands[seat]` 那个 set，
    下一次 `actions()` 再 `sorted(...)` 出来一个同内容的 list。
    """
    A = meld.cid_from_name
    hand = sorted([A("S3"), A("S4"), A("S5")])
    first = list(meld.melds_from(hand, 2))
    assert any(m.cards == (A("S3"),) for m in first), "第一手应当枚举出单张 3♠"

    hand.remove(A("S3"))                        # 模拟出牌：同一对象，内容变了
    second = list(meld.melds_from(hand, 2))
    assert not any(A("S3") in m.cards for m in second), (
        "同一个 hand 对象改过内容之后还枚举出了 3♠ —— 缓存是按键到**对象身份**了，"
        "给的是改之前的旧答案")

    # 反过来：内容改回去，必须又能拿到完整的（缓存不能把牌面绑死在对象上）
    hand.append(A("S3"))
    third = list(meld.melds_from(sorted(hand), 2))
    assert any(m.cards == (A("S3"),) for m in third), "内容改回来之后又少了东西"


def test_same_hand_same_order_gives_the_same_answers():
    """同输入必须给同答案（缓存不能引入不确定性）—— 而且**顺序也算输入**：
    `_by_idx` 保留插入序，代表牌是按顺序取的（`ids[0]`）。"""
    A = meld.cid_from_name
    h = [A("S3"), A("H3"), A("D3"), A("S9")]
    a = [(m.kind, m.size, m.rank, tuple(sorted(m.cards)))
         for m in meld.melds_from(h, 2)]
    b = [(m.kind, m.size, m.rank, tuple(sorted(m.cards)))
         for m in meld.melds_from(h, 2)]
    assert a == b
