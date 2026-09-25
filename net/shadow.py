"""影子模式（spec §8.2）：每个决策点算一次建议、落盘，回填我实际出了什么。

**第一版不给画面上的建议**（用户 2026-09-25 定）—— 记录器只写文件，
面板只显示「算了几个」。理由：一旦上了屏，人就会被建议影响，
「模型与人的分歧」这份数据就废了。

落盘风格与 `net/events.jsonl` 一致：追加写、一行一个 JSON、面板可跟读。
一行一个决策点：

    {"type":"decision","t":…,"deal":0,"pos":7,"seat":2,"level":9,"level_age_s":12.4,
     "hand":[…],"table":[…],"table_kind":5,"left":[21,27,18,0],"passed":[0,3],
     "n_cand":37,"top":[{"cards":[…],"names":["♠5","♠5"],"kind":2,"q":1.23},…],
     "actual":[…],"actual_names":[…],"actual_is_me":true,"actual_rank":0,"resolved":true}

一局结束补一行：

    {"type":"deal_end","deal":0,"seat":2,"level":9,"finish":[2,3,1,0],
     "me_team_won":false,"n_decisions":41,"n_unresolved":1,"skips":{"座位未确认":2}}

⚠️ **红线（spec §8.4）：绝不替用户出牌、不往游戏发任何报文。** 这里只写文件。
"""
from __future__ import annotations

import json
import os
import time

from net import advise, cards
from net.sim import rules

SHADOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow.jsonl")
SCHEMA = 1


class ShadowLog:
    def __init__(self, net=None, out_path=SHADOW, weights="", weights_note="",
                 topk=3):
        self.net = net
        self.out_path = out_path
        self.weights = weights
        self.weights_note = weights_note
        self.topk = topk
        self.n_decisions = 0
        self.skips = {}
        self.last_line = weights_note or "影子模式已就绪"
        self._fh = None
        self._pending = None
        self._cands = []
        self._q = []
        self._order = []
        self._deal_seq = None
        self._last_key = None
        self._snap = {}
        self._n_unresolved = 0
        self._t_level = None
        if net is not None:
            rec = {"type": "session", "schema": SCHEMA, "weights": weights}
            if weights:
                rec.update(advise.weights_info(weights))
            self._emit(**rec)

    # ------------------------------------------------------------ 对外

    @property
    def enabled(self) -> bool:
        return self.net is not None

    def after_event(self, st, ev) -> None:
        """面板每处理完**一个网络事件**调一次（顺序：先喂状态机，再调这里）。"""
        if self.net is None:
            return
        if self._deal_seq is None:
            self._deal_seq = st.deal_seq
        elif st.deal_seq != self._deal_seq:
            # 换局：先把上一局收尾（未回填的决策点 + 汇总行），再认新局号。
            self._finish_deal(reason="新的一局")
            self._deal_seq = st.deal_seq
        self._resolve(st)
        self._snap = {"finish": list(st.finish_order), "level": st.level, "me": st.me}
        if self._pending is not None:
            return
        if st.turn != st.me or not st.me_confirmed:
            return
        key = (st.deal_seq, len(st.steps))
        if key == self._last_key:
            return                       # 同一个局面只算一次（手牌同步会反复触发）
        self._last_key = key
        got = advise.advise(st, self.net, topk=self.topk)
        if isinstance(got, advise.Skip):
            self.skips[got.reason] = self.skips.get(got.reason, 0) + 1
            return
        self._pending = {
            "type": "decision", "t": round(time.time(), 3),
            "deal": st.deal_seq, "pos": len(st.steps), "seat": st.me,
            "level": st.level,
            "level_age_s": (round(time.time() - self._t_level, 1)
                            if self._t_level else None),
            "hand": sorted(st.hand),
            "table": sorted(got.obs.table), "table_kind": got.obs.table_kind,
            "left": list(got.obs.left), "passed": sorted(st.passes),
            "n_cand": len(got.cands),
            "top": [{"cards": sorted(got.cands[i].cards) if got.cands[i] else [],
                     "names": (cards.names_sorted(got.cands[i].cards, st.level)
                               if got.cands[i] else []),
                     "kind": got.cands[i].kind if got.cands[i] else 0,
                     "q": round(got.q[i], 4)}
                    for i in got.order[:self.topk]],
            "actual": None, "actual_names": [], "actual_is_me": None,
            "actual_rank": None, "resolved": False,
        }
        self._cands, self._q, self._order = got.cands, got.q, got.order

    def note_level(self) -> None:
        """级别刚更新时调一次 —— 只为了在记录里留下「级别是什么时候变的」。"""
        self._t_level = time.time()

    def close(self) -> None:
        """收尾：没回填的决策点也要落盘（丢掉的全是局末那几个，统计会有偏）。"""
        if self.net is None:
            return
        self._finish_deal(reason="面板退出")
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------ 内部

    def _rank_of(self, cs) -> int:
        """我实际出的这一手在候选里的 Q 名次（0 = 就是头名）。`-1` = **不在候选里**。

        `-1` 要当成告警看：状态没错的话，我出的牌一定在合法候选里
        （`accept_meld` 的验收①就是这个口径 —— 1706 手真实着法全都枚举得出来）。
        """
        want = sorted(cs or [])
        for i, c in enumerate(self._cands):
            got = sorted(c.cards) if c is not None else []
            if got == want:
                return self._order.index(i)
        return -1

    def _resolve(self, st) -> None:
        if self._pending is None:
            return
        pos = self._pending["pos"]
        if len(st.steps) <= pos:
            return                       # 还没发生
        seat, cs = st.steps[pos]
        rec, self._pending = self._pending, None
        rec["resolved"] = True
        rec["actual"] = sorted(cs) if cs else []
        rec["actual_names"] = cards.names_sorted(cs, rec["level"]) if cs else []
        # 这一步**本该是我**（决策点就是「轮到我了」）—— 不是我的话，
        # 说明状态机把着法归错了人，这条记录要能被离线挑出来。
        rec["actual_is_me"] = (seat == rec["seat"])
        rec["actual_rank"] = self._rank_of(cs)
        self._emit(**rec)
        self.n_decisions += 1
        self.last_line = (f"影子：本局 {self.n_decisions} 个决策点"
                          + (f"，跳过 {sum(self.skips.values())}" if self.skips else ""))

    def _finish_deal(self, reason: str) -> None:
        if self._deal_seq is None:
            return                       # 一个事件都没来过
        if self._pending is not None:
            rec, self._pending = self._pending, None
            rec["resolved"] = False
            rec["unresolved_reason"] = reason
            self._emit(**rec)
            self._n_unresolved += 1
        finish = list(self._snap.get("finish") or [])
        me = self._snap.get("me")
        first = finish[0] if finish else None
        self._emit(type="deal_end", deal=self._deal_seq, seat=me,
                   level=self._snap.get("level"), finish=finish,
                   me_team_won=(None if (first is None or me is None)
                                else rules.TEAM[first] == rules.TEAM[me]),
                   n_decisions=self.n_decisions, n_unresolved=self._n_unresolved,
                   skips=dict(self.skips))
        self.n_decisions = 0
        self._n_unresolved = 0
        self.skips = {}
        self._last_key = None

    def _emit(self, **rec) -> None:
        if self._fh is None:
            d = os.path.dirname(self.out_path)
            if d:                       # 直接用文件名时 dirname 是空串，makedirs("") 会抛
                os.makedirs(d, exist_ok=True)
            self._fh = open(self.out_path, "a", encoding="utf-8", buffering=1)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
