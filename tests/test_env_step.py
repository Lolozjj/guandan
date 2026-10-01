import numpy as np
import pytest

from guandan.capture import cards
from guandan.sim import env, meld, rules

A = meld.cid_from_name


def test_action_dim_matches_the_spec():
    # 基础 143（历史行用）+ 3 个**动作侧后果**特征 = 146（当前动作用）
    assert env.ACTION_BASE_DIM == 143     # 10 + 15 + 9 + 1 + 108
    assert env.ACTION_DIM == 146
    assert env.HISTORY_DIM == env.ACTION_BASE_DIM + 4


def test_encode_action_now_appends_the_consequence_block():
    """当前动作 = 基础 143 + 3；前 143 维必须与基础编码**逐位相同**，`None`（过）整条全 0。"""
    hand = [A("S3"), A("H3"), A("D3"), A("S5")]
    m = meld.as_meld([A("S3"), A("H3"), A("D3")], 2)
    now = env.encode_action_now(m, 2, hand)
    base = env.encode_action(m, 2)
    assert now.shape == (env.ACTION_DIM,)
    assert np.array_equal(now[:env.ACTION_BASE_DIM], base)
    # 三个 3 正好是这一手计划好的牌 ⇒ 「用掉一整手计划」那一维必须是 1
    assert now[env.ACTION_BASE_DIM] == 1.0
    assert np.all((now[env.ACTION_BASE_DIM:] >= 0) & (now[env.ACTION_BASE_DIM:] <= 1))
    z = env.encode_action_now(None, 2, hand)
    assert z.shape == (env.ACTION_DIM,) and z.sum() == 0


def test_pass_action_is_all_zeros_and_single_card_is_not():
    z = env.encode_action(None, level=2)
    assert z.shape == (env.ACTION_BASE_DIM,) and z.sum() == 0
    m = meld.as_meld([A("S3")], 2)
    v = env.encode_action(m, 2)
    assert v[:10].argmax() == meld.SINGLE - 1
    assert v[10:25].argmax() == 1              # 点数 3 -> _POINT[3]=2 -> 槽位 1
    assert v[25:34].argmax() == 0              # 1 张 -> 第 0 格
    assert v[34] == 0                          # 不含逢人配
    assert v[35 + cards.slot(A("S3"))] == 1


def test_joker_bomb_gets_its_own_size_slot():
    """天王炸 4 张，但张数那一格**必须与「4 张炸弹」区分开**
    （spec §4.2 把天王炸单列成一格）。"""
    from guandan.sim import meld as M
    jb = M.Meld(M.BOMB, 4, 0, (A("JOKER_B"), A("JOKER_B", deck=2),
                               A("JOKER_S"), A("JOKER_S", deck=2)))
    v = env.encode_action(jb, 2)
    assert v[25:34].argmax() == 8              # 第 8 格 = 天王炸


def test_env_step_advances_and_awards_zero_until_the_hand_ends():
    e = env.GuandanEnv(seed=7)
    e.reset(level=5, hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], first=0)
    obs, r, done, info = e.step(0)             # 座位 0 出单张
    assert r == 0 and not done and info["seat"] == 0
    assert obs.turn == 3                       # 下家是 3（出牌顺序 0→3→2→1，见 Task 1）
    assert obs.left == (0, 1, 1, 1)            # 座位 0 已经出完


def test_terminal_reward_is_awarded_exactly_once_and_equals_the_team_points():
    """**reward 只在终局那一步非零**，且等于「出牌人所在队的升级点」，符号跟着输赢。

    中间步恒为 0 是刻意的：牌类游戏中间没有即时反馈，DMC 的做法是拿终局 reward
    当整局所有决策点的回归目标 —— 那件事在训练循环里做，不在 env 里做。"""
    import random
    rng = random.Random(11)
    e = env.GuandanEnv(seed=11)
    e.reset(level=5)
    seen = []
    while not e.done:
        obs, r, done, info = e.step(rng.randrange(len(e.legal())))
        seen.append((info["seat"], r))
    nonzero = [(s, r) for s, r in seen if r != 0]
    assert len(nonzero) == 1, f"非零 reward 出现了 {len(nonzero)} 次，应该只有终局那次"
    seat, r = nonzero[0]
    ranks = e.ranks
    assert abs(r) == rules.points(ranks)
    assert (r > 0) == (rules.TEAM[seat] == rules.winner_team(ranks))
    assert any(x != 0 for x in seen), "终局那一步没有给 reward"


def test_legal_actions_include_pass_only_when_someone_has_played():
    e = env.GuandanEnv(seed=3)
    e.reset(level=2, hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], first=0)
    assert None not in e.legal()               # 领出
    e.step(0)
    assert None in e.legal()                   # 跟牌：能压也可以过


def test_a_full_random_rollout_terminates_and_conserves_108_cards():
    """随机自对弈必须**每局都收得了尾** —— 死循环是这类实现最常见的病。"""
    import random
    for seed in range(30):
        e = env.GuandanEnv(seed=seed)
        e.reset(level=random.Random(seed).randint(1, 13))
        n = 0
        while not e.done:
            acts = e.legal()
            e.step(random.Random(seed * 1000 + n).randrange(len(acts)))
            n += 1
            assert n < 1000, f"seed={seed} 这一局走了 {n} 步还没完 —— 死循环"
        assert sorted(e.ranks) == [1, 2, 3, 4]
        assert sum(len(h) for h in e.hand.hands) + sum(
            len(s.meld.cards) for s in e.hand.steps if s.meld) == 108


def test_the_crash_seed_rolls_out_to_the_end():
    """**回归**：`play()` 曾经只用 `as_meld` 重新判一次、丢掉调用方那一份读法。

    实测它们会对**同一套牌**给出不同的读法：seed=16 那局（打 6），
    `melds_from(手牌)` 给出一个 rank=13 的三带二，而 `as_meld(那 5 张)` 给出 rank=1。
    于是 `actions()` 把它当候选列出来、`play()` 却判「压不过桌面」而抛。

    这一条钉住那个具体的种子（通用的那条在上面 range(30) 的随机 rollout 里）。
    """
    import random
    e = env.GuandanEnv(seed=16)
    e.reset(level=random.Random(16).randint(1, 13))
    n = 0
    while not e.done:
        acts = e.legal()
        e.step(random.Random(16 * 1000 + n).randrange(len(acts)))
        n += 1
        assert n < 1000, "seed=16 走不完 —— play() 又开始丢调用方那份读法了"
    assert sorted(e.ranks) == [1, 2, 3, 4]
