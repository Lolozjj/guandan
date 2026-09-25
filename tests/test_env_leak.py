"""明牌泄漏的自动化测试（spec §6③ 最后一条 / §3）。

**为什么必须是测试而不是「看代码」**：泄漏不会报错，只会让训练分数变好看，
然后在真机上全废。看代码看不出来，只有构造对照局面才抓得住。
"""
import numpy as np
import pytest

from net import cards
from net.sim import env, meld, rules

A = meld.cid_from_name


def _obs_with_opponent_hand(opp_cards):
    """可观测部分**完全相同**，只有「对手手上还有什么」不同。"""
    h = rules.Hand(hands=[{A("S3"), A("S4")},          # 我
                          set(opp_cards),              # 下家
                          {A("S6")}, {A("S7")}],
                   level=2, turn=0)
    played = {s: set() for s in rules.SEATS}
    played[0] = {A("S8")}
    return env.observe(h, 0, played, None)


def test_two_games_differing_only_in_opponent_hands_encode_identically():
    """**对照局面必须张数相同、只有具体牌不同。**

    张数（`left`）与「出过的牌」都是公开信息，它们变了编码就该变 ——
    拿张数不同的两个局面做对照会假红，也会把「公开 vs 私有」的界线搅浑。
    """
    a = _obs_with_opponent_hand({A("S9"), A("ST")})
    b = _obs_with_opponent_hand({A("HJ"), A("HQ")})   # 同样 2 张，只换具体牌
    assert np.array_equal(env.encode_state(a), env.encode_state(b)), (
        "两个只差「对手手牌是哪些牌」的局面编出了不同的状态向量 —— **明牌泄漏**。\n"
        "最可能的原因：`encode_state` 或 `observe` 摸了 `Hand.hands` 里不属于自己的那几家。"
    )


def test_opponent_hand_size_difference_does_leak_by_design_and_that_is_fine():
    """**对手剩几张是公开信息，本来就该看得见**（记牌器也给人类这个信息，spec §4.1）。

    这条测试是拿来划清边界的：泄漏指的是「对手手上**具体是哪些牌**」，
    不是「对手还剩几张」。别把这一条误当成泄漏去「修」。"""
    a = _obs_with_opponent_hand({A("S9")})
    b = _obs_with_opponent_hand({A("S9"), A("ST")})
    assert not np.array_equal(env.encode_state(a), env.encode_state(b))


def test_encoding_never_reads_the_discarded_pile_of_an_opponent():
    """把对手**出过的牌**也换掉 —— 但那也是公开信息，所以**必须**导致编码不同。
    这条与上一条一起把「公开 / 私有」的界线钉在测试里。"""
    h1 = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    h2 = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    p1 = {s: set() for s in rules.SEATS}
    p2 = {s: set() for s in rules.SEATS}
    p1[1] = {A("S9")}
    p2[1] = {A("ST")}
    assert not np.array_equal(env.encode_state(env.observe(h1, 0, p1, None)),
                              env.encode_state(env.observe(h2, 0, p2, None)))


@pytest.mark.parametrize("seed", range(40))
def test_a_randomized_policy_scores_identically_on_paired_opponent_hands(seed):
    """**spec §6③ 的原话就是「若网络给两者的打分一样」** —— 所以这里真的接一个
    小网络，验打分而不是只验向量。向量那一层已经被上面钉住了，
    这一条防的是「以后有人给 env 加了个绕过 `Observation` 的入口」。"""
    import torch
    torch.manual_seed(seed)

    net = torch.nn.Sequential(torch.nn.Linear(env.STATE_DIM, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    for p in net.parameters():
        p.data.normal_(0, 0.1)

    def score(opp_cards):
        obs = _obs_with_opponent_hand(opp_cards)
        with torch.no_grad():
            return net(torch.from_numpy(env.encode_state(obs))).item()

    rng = np.random.default_rng(seed)
    free = [c for c in range(0, 334) if cards.is_card(c)
            and c not in {A("S3"), A("S4"), A("S6"), A("S7"), A("S8")}]
    picks = rng.choice(len(free), size=4, replace=False)
    # 两边都是 **2 张**，只有具体是哪两张不同
    x = score({free[picks[0]], free[picks[1]]})
    y = score({free[picks[2]], free[picks[3]]})
    assert x == y, f"网络给两个只差对手手牌的局面打出了不同的分：{x} vs {y} —— 泄漏"
