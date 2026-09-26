def test_package_importable():
    from net import cards
    from net.sim import meld  # noqa: F401
    assert cards.decode(77) == "K♦"


# ---------------------------------------------------------------- 对手混合（2026-09-26）
# 用户实测发现模型「有普通牌可压却出炸」13 次、小炸够用却选大炸 1 次（88 个
# 级别正确的决策点里）。根因：**只跟自己打** —— 自对弈里对手也爱炸，
# 「不炸就被炸」成了均衡；而它 91% 那个尺子（贪心基线）从不主动炸，
# 所以「浪费炸弹」在训练里从来没被惩罚过。修法：混入非炸弹型对手。

def test_opp_mix_only_trains_on_the_learner_team():
    """混入固定对手时，**只记学习那一队的决策点** —— 固定对手的着法不是网络选的，
    记进去等于拿它当训练目标（把网络往贪心上带）。"""
    from train import selfplay
    import random as _r

    class Net:                     # 空壳：opp_mix=1.0 时轮不到它出手
        def parameters(self):
            raise AssertionError("固定对手那一局不该用网络")

    net_batches = []
    for _ in range(3):
        out = selfplay.generate_batch(Net(), _r.Random(0), eps=1.0, n_games=8,
                                      opp_mix=1.0, greedy_share=1.0)
        net_batches += out
    assert net_batches, "一批都没跑出来"
    per_game = [{s for (_o, _a, _i, s, _h) in caps} for _rec, caps, _y in net_batches]
    assert any(per_game), "一个决策点都没记到"
    for seats in per_game:                     # **每局**只记一个队（哪一队当对手是每局抽的）
        assert seats <= {0, 2} or seats <= {1, 3}, \
            f"混入固定对手时每局只该记学习那一队，实际记到 {sorted(seats)}"


def test_opp_mix_zero_is_still_pure_selfplay():
    """`opp_mix=0`（默认）必须与老行为一致：四家都记（两队都学到）。"""
    from train import selfplay
    from train.net import QNet
    import random as _r
    out = selfplay.generate_batch(QNet(), _r.Random(0), eps=1.0, n_games=32,
                                  opp_mix=0.0)
    seats = {s for _rec, caps, _y in out for (_o, _a, _i, s, _h) in caps}
    assert {0, 2} & seats and {1, 3} & seats, f"两队都该出现在训练目标里：{sorted(seats)}"
