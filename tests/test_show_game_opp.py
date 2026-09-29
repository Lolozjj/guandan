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

from net.sim import rules
from tools.show_game import OPP_KIND_CN, frame_hint, replay_game


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
