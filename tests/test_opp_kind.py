"""`--opp-kind`：固定对手用哪种「强」。

2026-09-28 加：贪心**不像人**（100% 压自己队友、从不主动炸、不算剩牌），
所以多一个 `rule`（`guandan/rl/rule_policy.py`，实测 vs 贪心 67.0%）。

⚠️ **日志必须说实话**（本仓库纪律：换源必须可见）—— 两臂只差这一个开关，
如果日志都写「其中贪心 80%」，事后根本分不清哪一臂是谁。
"""
import random

import pytest

import guandan.rl.selfplay as sp
from guandan.sim import meld
from tests.test_rule_policy import C, _acts, _follow_obs
from guandan.rl.net import QNet
from guandan.rl.rule_policy import rule_choose
from guandan.rl.selfplay import _fixed_pick, generate_batch


def test_fixed_pick_rule_branch_calls_the_rule_policy():
    # ⚠️ 必须用**真牌 ID** —— 规则式要 `cards.parts`/`melds_from`，假 ID 会 KeyError
    o = _follow_obs(20, C("S6", "H6", "D6", "S4", "D4"), meld.TRIPLE_PAIR, 5, owner=1)
    acts = _acts(C("S7", "H7", "D7", "S5", "D5"), C("S3", "H3", "D3", "C3"),
                 with_pass=True)
    pending = (o, acts, None)
    assert _fixed_pick(("rule", 0), pending, random.Random(0)) == rule_choose(o, acts)


def test_opp_kind_actually_changes_the_fixed_teams_plays():
    """同一批牌、同一种子：贪心臂与规则式臂打出来的**必须是不同的局**。

    这条防的是「开关接上了但没生效」—— 那种错日志上看不出来。
    """
    net = QNet().eval()
    kw = dict(opp_mix=1.0, greedy_share=1.0)          # 混合局 100%、且都走「强」那一支
    g = generate_batch(net, random.Random(7), 0.0, 2, opp_kind="greedy", **kw)
    r = generate_batch(net, random.Random(7), 0.0, 2, opp_kind="rule", **kw)
    assert [rec.actions for rec, _p, _y in g] != [rec.actions for rec, _p, _y in r]


def test_unknown_opp_kind_is_rejected():
    """认不出的类型必须炸 —— 静默退回贪心会让一整晚的实验标错名字。"""
    with pytest.raises(ValueError):
        generate_batch(QNet().eval(), random.Random(0), 0.0, 1, opp_mix=1.0,
                       opp_kind="clever")


def test_cli_forwards_opp_kind(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0, "games": 0,
                "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--opp-kind", "rule"])
    assert seen["opp_kind"] == "rule"


def test_header_says_which_opponent(tmp_path):
    """日志头必须写出**实际用的是哪个**对手（换源必须可见）。"""
    lines = []
    sp.train(seconds=1, opp_kind="rule", log=lines.append,
             eval_games=1, eval_every=10 ** 9, out_dir=str(tmp_path))
    head = next(l for l in lines if "对手混合" in l)
    assert "规则式" in head, head
    assert "贪心" not in head, head
