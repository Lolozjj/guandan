"""影子模式（spec §8.2）：每个决策点算一次建议、落盘，回填我实际出了什么。

**第一版不给画面上的建议**（用户 2026-09-25 定）—— 记录器只写文件，
面板只显示「算了几个」。理由：一旦上了屏，人就会被建议影响，
「模型与人的分歧」这份数据就废了。

落盘风格与 `runtime/events.jsonl` 一致：追加写、一行一个 JSON、面板可跟读。
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

from guandan import paths
from guandan.advice import advise

from guandan.capture import cards
from guandan.sim import meld, rules

SHADOW = str(paths.SHADOW)
SCHEMA = 1


def open_shadow(out_path: str = SHADOW, topk: int = 3,
                show_advice: bool = True) -> "ShadowLog":
    """建一个记录器 —— **唯一的一处**（`guandan.launcher` 与两个面板都走它）。

    权重取 `GUANDAN_WEIGHTS` 或 `models/best.pt`（`advise.resolve_weights`），
    记进 session 行的也是**真正加载的那一份**。加载失败不抛异常：降级成「不记录」，
    把原因放在 `last_line` 上给面板显示。

    ⚠️ 这个函数存在的理由：原来建记录器的代码只写在两个面板的 `main()` 里，
    而**实机入口 `guandan.launcher` 是直接调 `run_live()` 的**（绕过 `main()`）——
    于是用户按台账打几十局，`runtime/shadow.jsonl` 一个字节都不会有。
    """
    from guandan.advice import advise
    path = advise.resolve_weights()
    net, err = advise.load_net(path)
    return ShadowLog(net=net, out_path=out_path, weights=path or "",
                     weights_note=err, topk=topk, show_advice=show_advice)


def model_banner(weights: str) -> str:
    """启动时那一行：**写清真正加载的是哪份权重**。

    面板挑权重有两级（`GUANDAN_WEIGHTS` → `models/best.pt`），
    设过环境变量之后从控制台**看不出加载了哪一份**，也就没法确认
    「试完之后有没有回到默认那份」—— 用户 2026-09-29 提的。

    带上训练局数是为了区分同一个目录里的不同文件（`best.pt` 与 `snap_160000.pt`
    是两个东西，光看文件名容易以为是同一份）。
    目录名也要写 —— `snap_160000.pt` 在好几轮训练里都叫这个名字。

    ⚠️ **只此一处**：`guandan.launcher` 与两个面板的 `main()` 打的都是
    `ShadowLog.last_line`，所以这一行改一次、三个入口同时生效（不会漂）。
    """
    if not weights:
        return "影子模式：没有模型（只记牌局，不给建议）"
    info = advise.weights_info(weights)
    games = info.get("games")
    # 取路径**末三段**：只取 `目录/文件名` 的话，快照全是 `pool/snap_*.pt` ——
    # 分不出是哪一次训练（`basename(dirname)` 给的是 `pool`）。这个是测试抓出来的。
    parts = [x for x in os.path.normpath(weights).replace("\\", "/").split("/") if x]
    where = "/".join(parts[-3:]) if parts else weights
    line = "影子模式就绪 —— 模型：" + where + (f"（{games:,} 局）" if games else "")
    # 「开关必须看得见」（本仓库的换源纪律）：`GUANDAN_TIDY` 开着就把它打出来。
    # 不打的话，用户根本分不清"面板没生效"还是"我设错了变量"（2026-10-02 用户问"怎么启动"）。
    mode = advise.tidy_mode()
    if mode != "off":
        line += f"  **擦浪费：{mode}**"
    return line


class ShadowLog:
    def __init__(self, net=None, out_path=SHADOW, weights="", weights_note="",
                 topk=3, show_advice=True):
        self.net = net
        self.out_path = out_path
        self.weights = weights
        self.weights_note = weights_note
        self.topk = topk
        #: 建议要不要给面板显示（用户 2026-09-26 要的；默认开）。
        #: **它同时决定记录里的 `advice_shown`** —— 上过屏的那段数据不再干净
        #: （人会被建议影响），离线分析「模型与人的分歧」时要把这段排除掉。
        self.show_advice = show_advice
        #: 给**终端**面板显示的一行建议（"建议：…（第 1/14，Q=0.183）"）；出手后清空
        self.last_advice = ""
        #: 给**图形**面板的结构化建议：`[{cards:[牌ID…], kind, q}, …]`，首选在前。
        #: 图形面板自己画牌面（用户 2026-09-26 要的：不要数字，用图片）；出手后清空。
        self.advice_top = []
        self.n_decisions = 0
        self.skips = {}
        # 加载失败时保留原因（那时用户更需要知道为什么），成功才报模型身份
        self.last_line = weights_note or model_banner(weights)
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
        self._closed = False
        if net is not None:
            rec = {"type": "session", "schema": SCHEMA, "weights": weights,
                   "show_advice": show_advice}
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
        if st.turn != st.me:
            return                       # 不是我的回合 —— 这不算决策点，不必计数
        key = (st.deal_seq, len(st.steps))
        if key == self._last_key:
            return                       # 同一个局面只算一次（手牌同步会反复触发）
        self._last_key = key
        if not st.me_confirmed:
            # **这一段必须计数。** 换局后到我第一次出牌之间，报文里的座位是陈旧的，
            # 这一手算不了。静默丢掉的话，台账那条判据（「跳过原因必须可解释，
            # 不能一片全是座位未确认」）永远触发不了 —— 座位认定真坏了也看不出来。
            self._count_skip(advise.SKIP_SEAT)
            return
        try:
            got = advise.advise(st, self.net, topk=self.topk)
        except Exception as exc:         # noqa: BLE001
            # 推理期的任何异常都只让**这一条**不记：面板挂了整晚就再也攒不到数据了
            # （红线是「不动游戏、不发报文」，所以这里只影响记录，不影响用户打牌）。
            self._count_skip(f"推理异常：{type(exc).__name__}")
            return
        if isinstance(got, advise.Skip):
            self._count_skip(got.reason)
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
            "actual_shape": None, "actual_rank": None, "resolved": False,
            # 这条建议有没有上过屏（上了屏 -> 这份数据不再干净，见 show_advice）
            "advice_shown": bool(self.show_advice),
            # 级别是从哪读来的（"结算行（下一局）" / "发牌行（本局）"）——
            # 3.2.2 之后有两个来源，可信度不同，离线要能分开看
            "level_src": getattr(st, "level_src", ""),
        }
        if self.show_advice:
            self.advice_top = self._pending["top"]
        if self.show_advice and got.order:
            i0 = got.order[0]
            m0 = got.cands[i0]
            who = cards.names_sorted(m0.cards, st.level) if m0 is not None else []
            self.last_advice = (f"建议：{' '.join(who) if who else '过'}"
                                f"（第 1/{len(got.cands)}，Q={got.q[i0]:.3f}）")
        self._cands, self._q, self._order = got.cands, got.q, got.order

    def panel_text(self) -> str:
        """面板底部那一行：影子进度 + 模型建议（**只有这一处定义**）。

        建议是用户 2026-09-26 要的（他要知道「模型和我想的一不一样」）。
        显示与记录**同源**：`last_advice` 就是这里算出来的那一份，面板不另算一遍。
        `--no-show-advice` 时它是空串，那一行就只剩进度。
        """
        return "    ".join(x for x in (self.last_line, self.last_advice) if x)

    def _count_skip(self, reason: str) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1
        # **跳过也要说得出来。** 面板只有这一行反馈：不写的话，级别读不到 /
        # 座位没认出来的那种整局跳过，用户看到的和「什么都没发生」一模一样
        # （2026-09-26 真踩过：一局 20 个决策点全被静默跳过，面板只写「已就绪」）。
        self.last_line = f"影子：跳过 {sum(self.skips.values())}（{reason}）"

    def note_level(self) -> None:
        """级别刚更新时调一次 —— 只为了在记录里留下「级别是什么时候变的」。"""
        self._t_level = time.time()

    def close(self) -> None:
        """收尾：没回填的决策点也要落盘（丢掉的全是局末那几个，统计会有偏）。

        **幂等** —— 回放那条路每个 tick 都会调它一次（`table.run_replay` 在回放喂完后
        窗口还开着，每个 tick 都进 `StopIteration` 分支），不幂等的话
        每 260 毫秒往日志追一行同样的 `deal_end`，放一晚几千行。
        """
        if self.net is None or self._closed:
            return
        self._closed = True
        self._finish_deal(reason="面板退出")
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------ 内部

    def _rank_of(self, cs, level) -> int:
        """我实际出的这一手在候选里的 Q 名次（0 = 就是头名）。`-1` = **不在候选里**。

        ⚠️ **按「形状」比，不按牌张比。** `melds_from` 的契约是「每个形状一条代表」
        （spec §13.2），候选里的牌常常是换了花色的同形状牌 —— 拿牌张比会把
        合法着法误判成「不在候选里」（实测：全量抓包两局里有 2 个决策点被误报成
        `actual_rank == -1`，其中一手是单张 ♣9。这也正是 `accept_meld` 验收① 的口径）。

        `-1` 才是告警：那说明状态重建有问题（我出的牌当时不是合法着法）。
        """
        want = self._shape_of(cs, level)
        for i, c in enumerate(self._cands):
            # 「过」在候选里是 `None`，它的形状与 `_shape_of(None)` 一样是 (0,0,0)
            got = (0, 0, 0) if c is None else (c.kind, c.size, c.rank)
            if got == want:
                return self._order.index(i)
        return -1

    @staticmethod
    def _shape_of(cs, level):
        """一手牌的「形状」`(牌型, 张数, 主点数)`；「过」是 `(0, 0, 0)`。

        判不出牌型时返回 `(-1, -1, -1)` —— 它不会与任何候选相等，
        于是 `actual_rank` 落成 `-1`，正是我们要的告警。
        """
        if not cs:
            return (0, 0, 0)
        m = meld.as_meld(list(cs), level)
        return (m.kind, m.size, m.rank) if m is not None else (-1, -1, -1)

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
        # 形状（牌型/张数/主点数）也记下来：离线看「模型与人的分歧」时，
        # 光有牌张还得再判一次牌型；而且候选是「每形状一条代表」，
        # 拿形状比才对得上（见 `_rank_of`）。
        rec["actual_shape"] = list(self._shape_of(cs, rec["level"]))
        rec["actual_rank"] = self._rank_of(cs, rec["level"])
        self.last_advice = ""            # 出手了 -> 清掉，别挂着上一手的建议误导人
        self.advice_top = []
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
        # **看全了没有**：完整一局的出完人数必然是 2（双上）或 3（三家出完）。
        # 少于 2 就是「我们接进来之前已经有人出完了」——那时 `finish[0]` 不是第 1 名，
        # 拿它判输赢会**判反**（实测能复现：真值第 1 名是对手，我们只看到队友出完，
        # 判出来是「我这队赢」）。宁可给 null，也不给一个错的真假值。
        seen_all = len(finish) >= 2
        self._emit(type="deal_end", deal=self._deal_seq, seat=me,
                   level=self._snap.get("level"), finish=finish,
                   me_team_won=(None if (not seen_all or first is None or me is None)
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
