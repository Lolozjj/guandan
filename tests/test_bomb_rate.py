"""用炸率：挂代价最典型的失败是「从此一刀切不炸」—— 那不比乱炸好。

`bomb_waste` 只回答「有得选的时候选错了吗」，回答不了「还炸不炸」。
两者一起看才完整（spec §5.3）。

⚠️ **`bomb_waste` 的签名与语义不许变** —— 全项目都在用它的 `(浪费数, 机会数)`。
所以「用炸率」是新函数，两者**共用同一次走局**（各写一份走局就是「副本会漂」）。
"""
from guandan.rl.eval import _bomb_stats, bomb_rate, bomb_waste
from guandan.rl.policies import greedy_policy


def test_bomb_rate_counts_bombs_and_games():
    b, g = bomb_rate(greedy_policy, games=6, seed=0)
    assert g == 6
    assert b >= 0


def test_bomb_waste_still_returns_its_old_pair():
    """加了共用走局之后，`bomb_waste` 的签名与语义**不许变**。"""
    w, c = bomb_waste(greedy_policy, games=6, seed=0)
    assert c > 0, "贪心局面里「能用普通牌压」的机会不该是 0"
    assert 0 <= w <= c


def test_both_views_come_from_the_same_walk():
    """两个数来自**同一次走局**（换种子必然一致），而且各返回各的那一对。"""
    assert _bomb_stats(greedy_policy, games=6, seed=0)[:2] == \
           bomb_waste(greedy_policy, games=6, seed=0)
    assert _bomb_stats(greedy_policy, games=6, seed=0)[2:] == \
           bomb_rate(greedy_policy, games=6, seed=0)


def test_a_different_seed_gives_a_different_walk():
    """钉住「它不是返回常量」—— 否则上面那条同源测试会假绿。"""
    assert bomb_rate(greedy_policy, games=6, seed=0) != \
           bomb_rate(greedy_policy, games=6, seed=99)


class _M:
    """假牌型：只要 `is_bomb`。"""
    def __init__(self, bomb):
        self.is_bomb = bomb


def test_waste_needs_a_plain_alternative():
    """**「白炸」的判定**（`bomb_opportunity` / `is_wasted_bomb`）—— 只此一份，
    `_bomb_stats` 的统计与战报里标出的那一手都走它。四种情形逐个钉。"""
    from guandan.rl.eval import bomb_opportunity, is_wasted_bomb
    plain, bomb = _M(False), _M(True)

    # 领出（桌上没牌要压）：炸不是浪费，是出牌
    assert not is_wasted_bomb(False, [plain, bomb], bomb)
    # 桌上要压，但手里**没有**非炸弹候选 —— 不得不炸，不算浪费
    assert not bomb_opportunity(True, [None, bomb])
    assert not is_wasted_bomb(True, [None, bomb], bomb)
    # 桌上要压、有普通牌能压、却炸了 —— **白炸**
    assert bomb_opportunity(True, [None, plain, bomb])
    assert is_wasted_bomb(True, [None, plain, bomb], bomb)
    # 同样是那个机会，选了普通牌 —— 不浪费
    assert not is_wasted_bomb(True, [None, plain, bomb], plain)
    # 「过」是 `None`，不是炸
    assert not is_wasted_bomb(True, [None, plain], None)
