"""影子记录器：一行一个决策点，回填我实际出了什么，局末补一行结果。"""
import json

from net import shadow
from net.sim.meld import cid_from_name as A
from net.state import GameState
from train.net import QNet


def _reader(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _new(tmp_path, **kw):
    # 随机初始化的网络就够 —— 这里测的是记录逻辑，不是模型的水平
    return shadow.ShadowLog(net=QNet().eval(), out_path=str(tmp_path / "shadow.jsonl"),
                            weights="runs/rl/test/best.pt", **kw)


def _play(st, seat, nxt, played, rest, mine=False):
    st.on_play(seat, list(played), 0, nxt, len(rest), sorted(rest) if mine else None)


def test_a_decision_point_is_recorded_once_and_backfilled_with_what_i_played(tmp_path):
    log = _new(tmp_path)
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4"), A("H5")], mine=True)   # 我出 S3 -> 轮到 0
    assert st.me == 1
    _play(st, 0, 1, [A("S4")], [A("S9")])                       # 0 出 4 -> 轮到我
    assert st.turn == 1
    log.after_event(st, {"type": "play"})                        # 决策点（pos=2）
    assert log.n_decisions == 0, "还没回填，不该落盘"
    _play(st, 1, 0, [A("H5")], [A("S4")], mine=True)             # 我真出了 H5
    log.after_event(st, {"type": "play"})                        # 回填
    dec = [r for r in _reader(log.out_path) if r["type"] == "decision"]
    assert len(dec) == 1
    assert dec[0]["deal"] == 0 and dec[0]["pos"] == 2
    assert dec[0]["resolved"] is True
    assert dec[0]["actual"] == [A("H5")] and dec[0]["actual_is_me"] is True
    assert 0 <= dec[0]["actual_rank"] < dec[0]["n_cand"], \
        "我出的牌必须落在候选里（状态没错的话一定在）"
    assert dec[0]["level"] == 9 and dec[0]["seat"] == 1
    assert dec[0]["table"] == [A("S4")] and dec[0]["table_kind"] == 1
    # ⚠️ **别断言 top[0] 有牌面名** —— 候选里含「过」，而随机网络完全可能把它排第一，
    # 那时 names 天然是空的。断「按 Q 降序」+「有牌必有牌面名」才是确定性的不变式。
    tops = dec[0]["top"]
    assert [t["q"] for t in tops] == sorted((t["q"] for t in tops), reverse=True),         "前几名必须按 Q 降序"
    assert all(bool(t["names"]) == bool(t["cards"]) for t in tops),         "有牌就该有牌面名（「过」的两样都空）"
    sess = [r for r in _reader(log.out_path) if r["type"] == "session"]
    assert sess and sess[0]["weights"] == "runs/rl/test/best.pt"


def test_my_silent_pass_is_backfilled_as_an_empty_actual(tmp_path):
    """我要不起时没有出牌事件 —— 回填必须来自推断出来的那一步，且记成「过」。"""
    log = _new(tmp_path)
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)      # 我出牌 -> 轮到 0
    _play(st, 0, 3, [A("S9")], [A("SK")])                 # 0 压过 -> 轮到 3
    st.turn = 1                                           # 摆到「轮到我」这个决策点
    log.after_event(st, {"type": "play"})
    st.on_pass(0, 3)                                      # 0 要不起（我会收到）—— 轮次越过了我
    log.after_event(st, {"type": "pass"})                 # 这一步把「我要不起」推出来了
    _play(st, 3, 2, [A("SJ")], [A("SQ")])                 # 3 出牌 -> 轮到 2
    log.after_event(st, {"type": "play"})
    dec = [r for r in _reader(log.out_path) if r["type"] == "decision"]
    assert len(dec) == 1, "只该有这一个决策点（3 出牌之后轮到 2，不是我）"
    assert dec[0]["actual"] == [] and dec[0]["actual_is_me"] is True
    assert dec[0]["resolved"] is True


def test_a_new_deal_flushes_the_pending_record_as_unresolved(tmp_path):
    """退出/换局时还没回填的决策点也必须落盘 —— 丢掉的全是「局末那几个」，统计会有偏。"""
    log = _new(tmp_path)
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)
    _play(st, 0, 1, [A("S4")], [A("S9")])
    log.after_event(st, {"type": "play"})
    st.on_deal()                                          # 直接换局
    log.after_event(st, {"type": "deal"})
    recs = _reader(log.out_path)
    dec = [r for r in recs if r["type"] == "decision"]
    assert len(dec) == 1 and dec[0]["resolved"] is False and dec[0]["unresolved_reason"]
    ends = [r for r in recs if r["type"] == "deal_end"]
    assert len(ends) == 1 and ends[0]["deal"] == 0
    assert ends[0]["n_decisions"] == 0 and ends[0]["n_unresolved"] == 1


def test_close_flushes_what_is_still_pending(tmp_path):
    """面板退出时的收尾：和换局那条路走同一个函数。"""
    log = _new(tmp_path)
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)
    _play(st, 0, 1, [A("S4")], [A("S9")])
    log.after_event(st, {"type": "play"})
    log.close()
    recs = _reader(log.out_path)
    assert [r["type"] for r in recs].count("deal_end") == 1
    dec = [r for r in recs if r["type"] == "decision"][0]
    assert dec["resolved"] is False and dec["unresolved_reason"] == "面板退出"


def test_deal_end_carries_whether_my_team_won(tmp_path):
    """队友先出完 = 我这队赢（判据与 `rules.winner_team` 同源：名次最好的那个决定）。

    ⚠️ `after_event` 必须**每个事件都调**（面板就是这样）——
    它靠「上一次快照」算局末汇总，一次都不调的话换局时手里没有快照。
    """
    log = _new(tmp_path)
    st = GameState()
    st.level = 9
    plays = [
        (0, 3, [A("S2")], [A("S9")], False),      # 0 出牌
        (3, 2, [A("S5")], [A("SK")], False),      # 3（我的队友）出牌
        (2, 1, [A("S6")], [A("SQ")], False),      # 2 出牌 -> 轮到我
        (1, 0, [A("S7")], [A("S4")], True),       # 我出牌（座位就此认出来）
        (0, 3, [A("S8")], [A("S9")], False),      # 0 出牌
        (3, 2, [A("SK")], [], False),             # 队友出完 -> finish=[3]
        (2, 1, [A("SQ")], [], False),             # 2 出完 -> finish=[3,2]
    ]
    for seat, nxt, played, rest, mine in plays:
        _play(st, seat, nxt, played, rest, mine=mine)
        log.after_event(st, {"type": "play"})
    st.on_deal()
    log.after_event(st, {"type": "deal"})
    end = [r for r in _reader(log.out_path) if r["type"] == "deal_end"][0]
    assert end["finish"] == [3, 2]
    assert end["me_team_won"] is True, "座位 1 与 3 是同队（TEAM=(0,1,0,1)）"
    assert end["deal"] == 0 and end["n_decisions"] == 0 and end["n_unresolved"] == 1, \
        "最后一次轮到我时挂起的那个决策点，要在换局时落成 resolved:false"


def test_it_stays_silent_when_the_model_is_missing(tmp_path):
    """权重没有/坏了：不许崩，也不许假装在记 —— 面板会显示这句话。"""
    out = tmp_path / "s.jsonl"
    log = shadow.ShadowLog(net=None, out_path=str(out),
                           weights_note="找不到权重（runs/rl/*/best.pt）")
    st = GameState()
    st.level = 9
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)
    st.turn = 1
    log.after_event(st, {"type": "play"})
    log.close()
    assert not log.enabled and "找不到权重" in log.last_line
    assert not out.exists(), "降级之后不该产出任何文件"
