"""回放器要能演「模型 vs 固定对手」—— 看它在**像人的对手**面前怎么打。

2026-09-29 加。原来 `replay_game` 只能打自对弈（四家都是网络），
而用户在意的恰恰是「对面像人一样打时模型行不行」——
`vs 规则式` 是现在唯一没饱和的尺子（88.2%），另外两把早就打穿了。

⚠️ 两条纪律：
- **对手的着法不许打分** —— 拿模型的 Q 去给规则式出的牌打分是**编数据**，
  比不打分有害得多。
- **`opp_kind=None` 必须是老行为**（一行不变）：这个工具是看模型的窗口，
  口径漂了会把人骗得很惨（本仓库为此吃过亏）。
"""
import pytest

from net.cards import names_sorted
from net.sim import rules
from tools.game_viewer import _View
from tools.show_game import (OPP_KIND_CN, frame_hint, policy_name,
                             replay_game, seat_label)


def _key(frames):
    return [(f.seat, None if f.chosen is None else tuple(sorted(f.chosen.cards)))
            for f in frames]


def test_default_is_still_self_play():
    """老行为不变：不传 `opp_kind` 时四家都是网络，每一步都有打分。"""
    meta, frames = replay_game(seed=7, level=8)
    assert meta["opp_kind"] is None
    assert all(f.candidates for f in frames if not f.over)
    assert not any(f.opp is not None for f in frames)


def test_model_frames_score_and_opponent_frames_do_not():
    meta, frames = replay_game(seed=7, level=8, opp_kind="rule", opp_team=1)
    assert meta["opp_kind"] == "rule" and meta["opp_team"] == 1
    for f in frames:
        if f.over:
            continue
        is_opp = rules.TEAM[f.seat] == 1
        assert (f.opp is not None) is is_opp
        # 对手的着法**不打分** —— 用模型的 Q 给规则式的牌打分是假数据
        assert bool(f.candidates) is (not is_opp)


def test_switch_actually_takes_effect():
    """同一批牌、同一种子：rule 臂与 greedy 臂打出来的**必须是不同的局**。

    这条防的是「开关接上了但没生效」—— 那种错在日志和画面上都看不出来。
    """
    _, g = replay_game(seed=11, level=8, opp_kind="greedy", opp_team=1)
    _, r = replay_game(seed=11, level=8, opp_kind="rule", opp_team=1)
    assert _key(g) != _key(r)


def test_unknown_opp_kind_is_rejected():
    """认不出的类型必须炸 —— 静默退回贪心会让一整场复盘标错名字。"""
    with pytest.raises(ValueError):
        replay_game(seed=7, level=8, opp_kind="clever")


def test_frame_hint_marks_the_opponent_frame():
    """谁出的牌必须在画面上写得出来（换源必须可见）。"""
    _, frames = replay_game(seed=7, level=8, opp_kind="rule", opp_team=1)
    f = next(x for x in frames if not x.over and x.opp is not None)
    assert OPP_KIND_CN["rule"] in frame_hint(f)
    mine = next(x for x in frames if not x.over and x.opp is None)
    assert OPP_KIND_CN["rule"] not in frame_hint(mine)


# ------------------------------------------------ 每个座位「谁在打」

def test_self_play_labels_everyone_as_the_model():
    """默认（不给 `opp_kind`）= 四家全是网络。**别让人以为对面是规则式。**"""
    meta, _ = replay_game(seed=7, level=8)
    assert [policy_name(meta, s) for s in rules.SEATS] == ["模型"] * 4


def test_the_fixed_team_is_labelled_with_its_policy():
    for kind, cn in (("rule", "规则式"), ("greedy", "贪心")):
        meta, _ = replay_game(seed=7, level=8, opp_kind=kind, opp_team=1)
        got = [policy_name(meta, s) for s in rules.SEATS]
        assert got == ["模型", cn, "模型", cn], f"{kind}: {got}"


def test_the_viewer_tags_every_seat_label():
    """图形版每个座位名后面都跟着「谁在打」—— 渲染器写座位名的地方都走 `seat_label`，
    所以标一处就全标上了（四个方位 + 台面 + 要不起）。"""
    meta, frames = replay_game(seed=7, level=8, opp_kind="rule", opp_team=1)
    f = next(x for x in frames if not x.over)
    v = _View(f, meta=meta)
    assert "规则式" in v.seat_label(1) and "模型" in v.seat_label(0)
    assert v.seat_label(1).startswith(seat_label(1)), "原来的座位名要保留"


def test_the_green_marks_what_is_about_to_be_played_for_any_policy():
    """绿底 = **这一帧要出的那手牌**，所以**非模型的座位也能标**
    （规则式/贪心没有候选 —— 我们不给对手打分 —— 但它们的实际着法是知道的）。"""
    meta, frames = replay_game(seed=7, level=8, opp_kind="rule", opp_team=1)
    f = next(x for x in frames if not x.over and rules.TEAM[x.seat] == 1
             and x.chosen is not None)
    v = _View(f, meta=meta)
    assert set(v.advised(f.seat)) == set(
        names_sorted(f.chosen.cards, f.level))
    assert v.advised((f.seat + 1) % 4) == (), "只有出手那一家有"
