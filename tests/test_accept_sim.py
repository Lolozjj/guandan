"""验收脚本自身的测试：**先证明它会失败，再证明它会通过。**

理由（本项目吃过一次亏）：验收脚本最容易「什么都没跑也报绿」。
所以这里既测「真数据全过」，也测「喂一条篡改过的对局必须红」。
"""
import copy

from guandan.capture import cards
from guandan.sim import meld, rules
from tools import accept_sim
from tools.game_log import load_corpus


def _one_settled_game():
    games = [g for g in load_corpus() if g.settle and len(g.plays) > 20]
    assert games, "语料里没有带结算的对局 —— 先生成 data/game_corpus.json"
    return games[0]


def test_shape_ignores_which_copy_of_a_card_was_used():
    """`melds_from` 的契约是「每个形状一条代表」—— 所以比对口径必须是**形状**，
    不是牌张集合。真人打出的可能是 ♥5♥5，而枚举给的是 ♠5♠5 那条代表。"""
    a = meld.as_meld([meld.cid_from_name("S5"), meld.cid_from_name("S5", deck=2)], 2)
    b = meld.as_meld([meld.cid_from_name("H5"), meld.cid_from_name("H5", deck=2)], 2)
    assert a.cards != b.cards
    assert accept_sim.shape(a) == accept_sim.shape(b) == (meld.PAIR, 2, 4)   # rank 是 meld.point_value(5,2)=4，不是 5


def test_replay_walks_a_real_game_to_the_end_without_raising():
    g = _one_settled_game()
    r = accept_sim.replay(g)
    assert r.steps == len(g.plays)
    assert r.hand.is_over()
    # 出完顺序必须与日志名次一致（第 1~3 名）
    logged = g.settle["Rank"]
    for i, seat in enumerate(r.hand.order):
        assert logged[seat] == i + 1, f"出完顺序与日志名次对不上：{r.hand.order} vs {logged}"


def test_replay_flags_a_tampered_game_instead_of_passing_it():
    """把某一手的牌换成手上没有的牌 —— 回放必须炸，不能「跳过这一手继续跑」。"""
    g = copy.deepcopy(_one_settled_game())
    g.plays[5].cards = [meld.cid_from_name("JOKER_B", deck=2)]   # 几乎不可能在他手上
    try:
        accept_sim.replay(g)
    except rules.IllegalPlay:
        return
    raise AssertionError("篡改过的对局居然走通了 —— 回放没有真的在验规则")


def test_report_is_red_when_nothing_was_checked():
    """`total == 0` 必须是**失败**（「一项都没检查到」不是「全过」）。"""
    res = accept_sim.check_replay([])
    assert not res.ok
    assert "0" in res.report() or "一项" in res.report()
