"""面板接线：事件进来 -> 状态机 -> 影子记录器。**不开窗口**（只测接线本身）。"""
import json

from guandan.ui import panel

from guandan.advice import shadow
from guandan.sim.meld import cid_from_name as A
from guandan.capture.state import GameState
from guandan.rl.net import QNet


def test_apply_event_then_shadow_writes_a_decision(tmp_path):
    st = GameState()
    st.level = 9
    out = tmp_path / "s.jsonl"
    log = shadow.ShadowLog(net=QNet().eval(), out_path=str(out), weights="")
    events = [
        # 我出牌（带 LeftCardList，座位就此认出来）
        {"type": "play", "seat": 1, "cards": [A("S3")], "card_type": 0,
         "next": 0, "left": 1, "left_cards": [A("S4")]},
        # 0 出 S2 -> 轮到我
        {"type": "play", "seat": 0, "cards": [A("S2")], "card_type": 0,
         "next": 1, "left": 26},
        # 我出 S4（压得过 S2）
        {"type": "play", "seat": 1, "cards": [A("S4")], "card_type": 0,
         "next": 0, "left": 0, "left_cards": []},
    ]
    for ev in events:
        panel.apply_event(st, ev)          # 生产那份分发
        log.after_event(st, ev)
    log.close()
    recs = [json.loads(l) for l in open(out, encoding="utf-8")]
    dec = [r for r in recs if r["type"] == "decision"]
    assert len(dec) == 1, "应当正好算过一次"
    assert dec[0]["actual"] == [A("S4")] and dec[0]["actual_rank"] >= 0
