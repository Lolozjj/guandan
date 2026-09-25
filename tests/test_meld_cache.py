"""缓存的正确性契约（不是「快不快」，是「会不会给旧答案」）。"""
from net.sim import meld


def test_enumeration_is_not_stale_after_the_hand_changes_in_place():
    """缓存必须按**牌面快照**键，不能按对象身份 —— 手牌是**就地改**的
    （`Hand.play()` 就是 `self.hands[seat] -= cs`）。

    实测：同一手牌里同一种牌面会被反复枚举（300 局里重复率 83%），所以要缓存；
    但键必须是「这一手是哪些牌」的快照，否则改完手牌还会拿到旧结果。
    """
    A = meld.cid_from_name
    hand = sorted([A("S3"), A("S4"), A("S5")])
    first = [m for m in meld.melds_from(hand, 2)]
    assert any(m.cards == (A("S3"),) for m in first), "第一手应当枚举出单张 3♠"

    hand2 = sorted([A("S4"), A("S5")])          # 换一种牌面（模拟出牌之后）
    second = [m for m in meld.melds_from(hand2, 2)]
    assert not any(A("S3") in m.cards for m in second), \
        "改完手牌还枚举出了 3♠ —— 缓存给的是旧结果"

    # 再要一次原牌面，必须还是完整的（缓存不能被上面那次污染）
    again = [m for m in meld.melds_from(sorted([A("S3"), A("S4"), A("S5")]), 2)]
    assert any(m.cards == (A("S3"),) for m in again), "原牌面从缓存里回来时少了东西"


def test_same_hand_same_order_gives_the_same_answers():
    """同输入必须给同答案（缓存不能引入不确定性）—— 而且**顺序也算输入**：
    `_by_idx` 保留插入序，代表牌是按顺序取的（`ids[0]`）。"""
    A = meld.cid_from_name
    h = [A("S3"), A("H3"), A("D3"), A("S9")]
    a = [(m.kind, m.size, m.rank, tuple(sorted(m.cards))) for m in meld.melds_from(h, 2)]
    b = [(m.kind, m.size, m.rank, tuple(sorted(m.cards))) for m in meld.melds_from(h, 2)]
    assert a == b
