"""策略与评测器的测试。

**这一组是「尺子本身准不准」的测试。** Plan 3 的判据是「打得过贪心基线」，
如果基线是个弱鸡（不比随机强多少），那条判据就毫无意义 —— 所以这里
除了测行为，还测**基线的强度**（实测 83.2% vs 随机）。

阈值都是**实测之后**定的，不是拍脑袋：随机 vs 随机 50.0%、贪心 vs 随机 83.2%、
贪心 vs 贪心 49.3%（各 600 局）。测试里用 200 局、留足噪声余量。
"""
import random

from guandan.capture import cards
from guandan.sim import env, meld
from guandan.rl.eval import match
from guandan.rl.policies import greedy_policy, random_policy

A = meld.cid_from_name


def _hand(**seats):
    hands = [set() for _ in range(4)]
    for name, ids in seats.items():
        hands[int(name[1:])] = set(ids)
    return hands


def _idxs(ids):
    return {cards.parts(c)[0] for c in ids}


# ---------------------------------------------------------------- 贪心的行为

def test_greedy_plays_the_smallest_meld_when_leading():
    """领出时出**最小**的那手。打 2 时 `3♠` 的 point_value=2、`9♠`=8，
    张数同为 1 —— 所以该出 3♠。"""
    e = env.GuandanEnv(seed=0)
    e.reset(level=2, hands=_hand(s0=[A("S3"), A("S9")], s1=[A("S4")],
                                s2=[A("S5")], s3=[A("S6")]), first=0)
    acts = e.legal()
    i = greedy_policy(e.observe(), acts, None)
    assert acts[i] is not None and _idxs(acts[i].cards) == {3}


def test_greedy_plays_the_smallest_meld_that_beats():
    """跟牌时出「最小的**能压的**那手」，不是手上最小的那手。

    桌面 6♠，其余三家手上都是「一张 4、一张 9」—— 4 压不过 6♠、9 压得过。

    ⚠️ **三家给一样的结构**，因为出牌顺序是 `0 → 3 → 2 → 1`（不是 `0 → 1 → 2 → 3`）：
    座位 0 领出之后轮到的是**座位 3**。只给座位 1 摆这个局面的话，
    测的其实是座位 3 的候选，断言会莫名其妙地红（我在这里红过一次）。
    """
    e = env.GuandanEnv(seed=0)
    e.reset(level=2, hands=_hand(s0=[A("S6")], s1=[A("S4"), A("S9")],
                                s2=[A("H4"), A("H9")], s3=[A("C4"), A("C9")]),
            first=0)
    e.step(0)                                   # 座位 0 领出 6♠
    acts = e.legal()
    i = greedy_policy(e.observe(), acts, None)
    assert acts[i] is not None, "压得过就不该过"
    assert _idxs(acts[i].cards) == {9}, \
        f"4 压不过 6♠，该出 9（最小的**能压的**那手），实际出的是 {acts[i].cards}"


def test_greedy_passes_only_when_it_cannot_beat():
    e = env.GuandanEnv(seed=0)
    e.reset(level=2, hands=_hand(s0=[A("SA")], s1=[A("S4")], s2=[A("S5")],
                                s3=[A("S6")]), first=0)
    e.step(0)                                   # 座位 0 领出 A♠，其余都压不过
    acts = e.legal()
    i = greedy_policy(e.observe(), acts, None)
    assert acts[i] is None, "压不过就该过"


# ---------------------------------------------------------------- 评测器 / 尺子

def test_random_vs_random_is_fair():
    """两队都随机 -> 胜率应当 ≈ 50%（评测无偏、赛制对称）。实测 50.0%。"""
    wr = match(random_policy(random.Random(1)), random_policy(random.Random(2)),
               games=200, seed=7)
    assert 0.42 <= wr <= 0.58, f"随机对随机只有 {wr:.1%}，评测或赛制有偏"


def test_greedy_beats_random_clearly():
    """**贪心基线必须明显强于随机** —— 否则「打得过贪心」这根尺子没有牙。

    实测 83.2%（600 局）。这里只要 ≥70% 就算过（200 局的噪声约 ±3%）。
    """
    wr = match(greedy_policy, random_policy(random.Random(3)), games=200, seed=11)
    assert wr >= 0.70, f"贪心对随机只有 {wr:.1%} —— 基线太弱，判据会失去意义"


def test_greedy_vs_itself_is_even():
    """同一个策略对打应当 ≈ 50%（座位对调的对称性自检）。实测 49.3%。"""
    wr = match(greedy_policy, greedy_policy, games=200, seed=13)
    assert 0.42 <= wr <= 0.58, f"贪心对自己只有 {wr:.1%}，座位对调或胜负判定有偏"


def test_batched_net_policy_matches_the_per_decision_one():
    """评测器的批量路径必须与逐个路径给出**同样的胜率** ——
    不然「胜率」这个数就取决于走哪条代码路径了（而且不报错）。"""
    import torch
    from guandan.rl.net import QNet
    from guandan.rl.policies import batch_net_policy, net_policy
    from guandan.rl.net import q_argmax_batch, q_values

    torch.manual_seed(0)
    net = QNet()
    one = net_policy(lambda o, a, h: q_values(net, o, a, h))
    batched = batch_net_policy(q_argmax_batch, net)
    wr_one = match(one, greedy_policy, games=100, seed=55)
    wr_bat = match(batched, greedy_policy, games=100, seed=55)
    assert wr_one == wr_bat, f"批量 {wr_bat:.0%} 与逐个 {wr_one:.0%} 不一致"


# ---------------------------------------------------------------- 炸弹浪费率
# 用户 2026-09-26 实测：模型「有普通牌可压却出炸」。这个指标就是那把尺子 ——
# 「vs 贪心」看不见它（贪心从不主动炸，不会惩罚浪费）。

def test_greedy_never_wastes_a_bomb():
    """对照：贪心基线在「能出普通牌」时**永远不炸**，所以它的浪费率必须是 0。

    这是这个指标的自检 —— 分母（能用普通牌压的局面）得真的出现过。
    """
    from guandan.rl.eval import bomb_waste
    from guandan.rl.policies import greedy_policy
    waste, chance = bomb_waste(greedy_policy, games=6, seed=0)
    assert chance > 0, "这些局里一次「能用普通牌压」的局面都没出现，指标没意义"
    assert waste == 0, f"贪心不该有浪费，实际 {waste}/{chance}"


def test_a_bomb_happy_policy_wastes_more_than_greedy():
    """反向对照：「能炸就炸」的浪费率**高于**贪心（贪心恒为 0）。

    ⚠️ 别断言它「很高」：炸弹很快就被打光，所以「能用普通牌压、且手里还有炸」
    的局面天然不多 —— 桩自己也只有 4%（2026-09-26 实测），模型 5~6%。
    这个指标是**相对**的尺子（同一批种子下比大小），不是「理想值接近 0」那种。
    """
    from guandan.rl.eval import bomb_waste

    def bombs_first(obs, acts, hist=None):
        bombs = [(i, m) for i, m in enumerate(acts) if m is not None and m.is_bomb]
        if bombs:
            return bombs[0][0]
        return 0

    waste, chance = bomb_waste(bombs_first, games=6, seed=0)
    assert chance > 0 and waste > 0, f"能炸就炸该有浪费，实际 {waste}/{chance}"
    w_g, c_g = bomb_waste(greedy_policy, games=6, seed=0)
    assert c_g > 0 and w_g == 0, "对照：贪心一次都不该浪费"
