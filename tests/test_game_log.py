import json
import os

import pytest

from tools.game_log import LOG_DIR, load_games, conserved

# skipif 只挂在真正需要本机日志的测试上。
# 千万别提升成模块级 pytestmark：那会波及用 tmp_path 的那几个，日志 2 天轮转删掉后
# 整个文件变成全员 skip —— pytest 退出码 0、看着全绿，解析器的回归防线就此停摆。


@pytest.mark.skipif(not os.path.isdir(LOG_DIR), reason="本机没有游戏日志")
def test_loads_games():
    games = load_games()
    assert len(games) > 0, "日志目录在，却一局都没解出来 —— 不要静默通过"
    for g in games[:5]:
        assert len(g.my_cards) == 27
        # 日志偶尔用 14 表示 A（guandan/capture/cards.py 里也踩过这条），所以上限放到 14
        assert 1 <= g.trump <= 14


@pytest.mark.skipif(not os.path.isdir(LOG_DIR), reason="本机没有游戏日志")
def test_settled_games_are_conserved():
    """出过的牌 + 结算剩的牌 == 108。这是「对局记录完整」的硬证据。"""
    settled = [g for g in load_games() if g.settle]
    assert len(settled) >= 20, f"完整局太少（{len(settled)}），样本不足以下结论"
    bad = [g for g in settled if not conserved(g)]
    assert not bad, f"{len(bad)} 局不守恒：{[g.t0 for g in bad[:3]]}"


def test_missing_dir_raises(tmp_path):
    """日志被轮转删掉时必须明着报错，不能返回空列表当成功。"""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_games(log_dir=str(tmp_path / "nope"))


# --- 以下为实施期补充（brief 三例之外）：锁住实测踩到的两个边界 ---

_HDR = "2026-09-24|17:04:13:746|INFO|G|GameLogger|520|520|3222027089|"


def _log_dir(tmp_path, lines):
    d = tmp_path / "logs"
    d.mkdir(exist_ok=True)
    (d / "2026-09-24-17.log").write_text("".join(lines), encoding="utf-8")
    return str(d)


def test_companion_roundid_line_is_not_a_new_deal(tmp_path):
    """实测：发牌行同一毫秒还有一行 `SendCardsService set roundID:S…`。

    它与发牌行共享前缀，但只是伴随行、不是新的一局。必须显式跳过：
    漏掉它要么凭空多出一局（实测 63 局会被数成 126 局），要么直接抛错。
    """
    cards = list(range(1, 28))
    deal = _HDR + "SendCardsService set roundID:5,5,5,5 k : " + json.dumps(
        {"CardLen": 27, "Cards": cards, "Trump": 5}) + "\n"
    companion = _HDR + "SendCardsService set roundID:S7380R1T1669t6AB4E78ES0A\n"
    games = load_games(log_dir=_log_dir(tmp_path, [deal, companion]))
    assert len(games) == 1, f"伴随行被当成新的一局了（解出 {len(games)} 局）"
    assert games[0].my_cards == cards
    assert games[0].trump == 5


def test_unknown_deal_shaped_line_raises(tmp_path):
    """认不出的发牌行必须明着炸，不许静默跳过 —— 静默跳过等于凭空少一局。"""
    bad = _HDR + "SendCardsService set roundID:5,5,5,5 k : {\"Cards\": 坏掉的\n"
    with pytest.raises(RuntimeError):
        load_games(log_dir=_log_dir(tmp_path, [bad]))


def test_truncated_play_line_is_counted_not_dropped_silently(tmp_path):
    """出牌行被日志轮转截断（JSON 解不出）时不能凭空消失。

    硬要求是「不许静默」：丢了就记进 `unparsed` 让下游看得见，
    同时**这一局仍要正常返回**（不是整局丢弃）。
    反过来，「JSON 能解析但没有 CardList」（如「要不起」）不算丢失，不能计数 ——
    计进去会产生大量假计数。
    """
    cards = list(range(1, 28))
    deal = _HDR + "SendCardsService set roundID:5,5,5,5 k : " + json.dumps(
        {"CardLen": 27, "Cards": cards, "Trump": 5}) + "\n"
    # 被截断的出牌行：JSON 一定解不出
    truncated = _HDR + ('NotifyGiveCards 后台通知客户端出牌结果 info = '
                        '{"SeatID":2,"CardList":[55,29\n')
    # 能解析、但没有 CardList 的非出牌消息（不得计入）
    not_a_play = _HDR + ('NotifyGiveCards 后台通知客户端出牌结果 info = '
                         '{"SeatID":2,"Result":1}\n')

    games = load_games(log_dir=_log_dir(tmp_path, [deal, truncated, not_a_play]))
    assert len(games) == 1, "这一局不该被整局丢掉"
    assert games[0].my_cards == cards
    assert games[0].unparsed == 1, (
        f"截断行应计数 1（无 CardList 的消息不计），实际 {games[0].unparsed}")
    assert games[0].plays == [], "截断的行不该变成一手牌"


# ---------------------------------------------------------------- 语料快照


def test_snapshot_round_trips():
    """冻结再读回来，每一局的牌局内容必须完全一致。

    快照是 Plan 1 回归防线的替代语料（日志会被轮转删），它必须与实时解析等价 ——
    差一张牌，验收结论就不能信。
    """
    if not os.path.isdir(LOG_DIR):
        pytest.skip("本机没有游戏日志")
    from tools.game_log import to_jsonable, from_jsonable
    games = [g for g in load_games() if g.settle][:5]
    assert games, "没有已结算的局可供比对"
    for g in games:
        back = from_jsonable(to_jsonable(g))
        assert back.t0 == g.t0 and back.trump == g.trump
        assert back.my_cards == g.my_cards
        assert [(p.seat, p.cards, p.card_type, p.left, p.nxt)
                for p in back.plays] == \
               [(p.seat, p.cards, p.card_type, p.left, p.nxt)
                for p in g.plays]
        assert conserved(back) == conserved(g)


def test_load_corpus_reports_which_source_it_used():
    """用了快照还是实时日志，必须能报出来 —— 换源不能静默。"""
    from tools import game_log
    games = game_log.load_corpus()
    assert games
    assert game_log.LAST_SOURCE
    assert ("快照" in game_log.LAST_SOURCE) or ("实时日志" in game_log.LAST_SOURCE)


def test_missing_snapshot_raises_rather_than_returning_empty(tmp_path):
    """快照不存在要明着报错，并告诉人怎么生成。"""
    from tools.game_log import load_snapshot
    with pytest.raises(FileNotFoundError) as e:
        load_snapshot(str(tmp_path / "nope.json"))
    assert "snapshot_logs" in str(e.value)
