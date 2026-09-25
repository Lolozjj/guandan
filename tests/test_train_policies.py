"""策略与评测器的测试。

**这一组是「尺子本身准不准」的测试。** Plan 3 的判据是「打得过贪心基线」，
如果基线是个弱鸡（不比随机强多少），那条判据就毫无意义 —— 所以这里
除了测行为，还测**基线的强度**（实测 83.2% vs 随机）。

阈值都是**实测之后**定的，不是拍脑袋：随机 vs 随机 50.0%、贪心 vs 随机 83.2%、
贪心 vs 贪心 49.3%（各 600 局）。测试里用 200 局、留足噪声余量。
"""
import random

from net import cards
from net.sim import env, meld
from train.eval import match
from train.policies import greedy_policy, random_policy

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
