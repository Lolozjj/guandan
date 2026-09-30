"""掼蛋牌局状态机 —— 把网络解出来的事件拼成一副看得懂的牌局。

刻意跟「在哪拿数据」解耦：喂给它的是 on_hand / on_play 这种事件，
至于事件是来自实时抓包、还是回放抓包文件，它不关心。
这样状态机可以**离线用抓包文件验证**，不用装证书。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import cards

# 级别值 -> 人读（与游戏日志的 Trump 字段一致）
_LEVEL_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K",
                **{i: str(i) for i in range(2, 11)}}

# 「我」的座位**只是个默认值，实际必须运行时认**。
#
# 报文里的座位号**每局都会变**：同一台机器，16:01 那局我是 seat3、
# 16:33 那局我是 seat1。原因是 4 个座位里哪个是你由服务器分配。
# 所以写死任何值都会错 —— 我在这上面反复搞反过两次。
#
# 真正的判据是 LeftCardList（字段 3.6.10）：**只有自己出牌时才有值**，
# 服务器只给自己的出牌附上「你还剩哪些牌」。在 on_play 里用它认座位 + 同步手牌。
#
# 另外别被日志的两套编号骗了：
#   - `玩家seatId = N出的牌` 是一套
#   - `NotifyGiveCards ... "SeatID":M` 是另一套（跟报文一致）
#   实测 M = (N + 2) % 4 之类的换算关系不稳定，**别做换算，要看带 LeftCardList 的那条**。
ME_SEAT = 1          # 仅供在认出来之前占位

# 判断「这手牌是不是我出的」时，至少要这么多张才有判别力 ——
# 1~2 张的牌太容易碰巧落在任何人手里。
_MIN_VOTE_CARDS = 4


@dataclass
class Play:
    """一条出牌记录（`GameState.plays` / `history` / `table` 的元素）。"""

    seat: int              # 谁出的（绝对座位号）
    cards: List[int]       # 出的牌（牌 ID）
    card_type: int = 0     # 服务器自己标的牌型编号 —— **只用于对账/报错信息**，判牌型走 `sim/meld.py`
    next_seat: int = 0     # 服务器给的下一手该谁；**-1 = 本局结束，不是座位号**（当座位会渲染成天文数字）
    left: int = 0          # 这一手之后该家还剩几张（服务器的 LeftCardLen）

    def names(self, level: int = None) -> List[str]:
        """按掼蛋大小排好的牌面（大的在前）。

        顺序是 大王 > 小王 > **级牌** > A > K > Q > J > 10 > … > 2。
        级牌跟着当前级别走，所以必须把 level 传进来 —— 只按牌 ID 排是错的。
        """
        return cards.names_sorted(self.cards, level)


@dataclass
class GameState:
    """桌面状态。字段都是「面板要显示什么」直接对应的。"""

    level: Optional[int] = None          # 打几
    hand: List[int] = field(default_factory=list)     # 我的手牌
    turn: Optional[int] = None           # 现在轮到谁
    plays: List[Play] = field(default_factory=list)   # 全部出牌（按时序）
    #: 按座位分的出牌记录（键 0..3）：面板上「每家出过什么牌」显示的就是它（记牌）。
    history: Dict[int, List[Play]] = field(
        default_factory=lambda: {0: [], 1: [], 2: [], 3: []})
    remaining: Dict[int, int] = field(default_factory=dict)   # 各家剩几张
    table: Optional[Play] = None         # 桌面上当前待压的牌
    passes: List[int] = field(default_factory=list)  # 本轮要不起的座位
    me: int = ME_SEAT                    # 我在哪个座位（见 ME_SEAT 的说明）
    deal_hand: set = field(default_factory=set)   # 本局起手牌，用来交叉验证座位
    #: 本局的动作流水（按时序）：`(座位, 出的牌)`，`None` = 该座位「要不起」。
    #: **和 `rules.Hand.steps` 同语义**：出完的座位不占步。影子模式的历史就取它。
    #: 注意里面**有推断出来的步** —— 我自己要不起时服务器不发事件（实测），
    #: 只能从轮转补；补的位置在下一个事件之前，见 `_advance_to`。
    steps: List[tuple] = field(default_factory=list)
    #: 局号。`on_deal` 时 +1；面板中途接进来时是 0。影子日志用它把决策点归到一局。
    deal_seq: int = 0
    #: 级别的来源（"结算行（下一局）" / "发牌行（本局）" / ""）。
    #: 只是给人看的溯源信息：游戏 3.2.2 之后级别有两个来源，可信度不同。
    level_src: str = ""
    #: 本局的座位认出来了没有。**换局时必须清零** —— 座位号每局都变，
    #: 认出来之前算出来的相对方位全是错的。
    me_confirmed: bool = False
    #: 本局出完的座位，按出完先后。影子日志的「那局赢没赢」用它推。
    finish_order: List[int] = field(default_factory=list)
    _votes: Dict[int, int] = field(default_factory=dict)  # 「我」座位判定的投票计数（只统计、**不自动改判**，事后排查用）
    _me_note: str = ""                                    # 座位是怎么认出来的（给人看的溯源，写进日志）

    # ------------------------------------------------------------- 事件入口

    def _sync_hand(self, ids: List[int]) -> None:
        """更新我的手牌，并顺带判断「是不是新一局」。

        判据是「**新手牌不是旧手牌的子集**」，不是「比现在多」——
        一局之内手牌只减不增，所以新手牌必然是旧手牌的子集；一旦出现
        「不包含」，只能是重新发了牌。用大小比较会漏：两局手牌都是 27 张时
        `27 > 27` 不成立，新一局就清不掉（这个坑我踩过）。

        手牌还空着时（面板中途接进来）不清 —— 那几手本来就是本局的，清了反而丢。
        """
        if self.hand and not set(ids) <= set(self.hand):
            self.on_deal()
        self.hand = sorted(ids)
        if len(ids) > len(self.deal_hand):
            self.deal_hand = set(ids)

    def _is_finished(self, seat: int) -> bool:
        """这个座位出完了吗。

        `remaining` 是服务器在每条出牌消息里直接给的（`LeftCardLen`），最可靠；
        没记到的时候退回「出过的张数 >= 27」—— 两副牌一个座位起手 27 张。
        """
        if seat in self.remaining:
            return self.remaining[seat] == 0
        played = sum(len(p.cards) for p in (self.history.get(seat) or []))
        return played >= 27

    def _advance_to(self, seat: int) -> None:
        """补上「从当前轮次走到 `seat`」中间那些**没有事件**的「要不起」。

        为什么必须有这一步：出牌顺序是座位号递减（`0→3→2→1`），而
        **我自己要不起时服务器不通知我**（实测）。轮次从别人跳到我、或从我
        跳到别人时，中间那一步只能自己推。
        `on_play` 与 `on_pass` **两条入口都要调它** —— 只挂在 `on_play` 上会漏：
        我过了之后，如果下一条事件是别人的「要不起」（3006），轮次照样越过了我。

        已出完的座位不补（`_is_finished`）—— 他没有「过」这个动作。
        """
        if self.turn is None or self.turn == seat:
            return
        s = self.turn
        for _ in range(4):
            if s == seat:
                return
            if s not in self.passes:
                self.passes.append(s)
                if not self._is_finished(s):
                    self.steps.append((s, None))
            s = (s - 1) % 4

    def _maybe_clear_table(self) -> None:
        """一圈扫不到人接手（或扫回主人自己）就清桌 —— 与 `rules.Hand._advance` 同一套判据。

        **不清的话「桌面待压」会一直挂着上一轮那手牌**：面板显示错，建议也错
        （以为必须压一手，其实是我领出）。实测（全量抓包两局）：4 个决策点的桌面
        与日志真值不符，连带那几手的建议全错。
        """
        if self.table is None or self.turn is None:
            return
        s = (self.table.seat - 1) % 4          # 出牌顺序是座位号递减
        for _ in range(3):                     # 另外三家
            if self._is_finished(s) or s in self.passes:
                s = (s - 1) % 4
                continue
            return                             # 有人还能接手 -> 这一轮没结束
        # 清桌要**连「谁要不起」一起清** —— `rules.Hand._advance` 就是两样一起清的：
        # 一轮结束、下一轮重新开始，上一轮的要不起名单不该留着
        # （留着的话喂给网络的那 4 维 `passed` 与训练时不一致，实测 4 个决策点因此不符）。
        self.table = None
        self.passes = []

    def on_leader(self, seat: int) -> None:
        """开桌那条报文说**这局谁领出**（msgid 3004 的 `3.5.1`）。

        为什么要它：`on_deal` 把 `turn` 清成 `None`，而**第一手出牌之前没有任何事件
        能说明轮到谁**。我领出那一局因此拿不到建议（影子模式的门是 `turn == me`）——
        用户 2026-09-29 报的「我的第一首出牌，模型不会有建议」。

        ⚠️ **只在轮次还是空的时候采用**：中局来一条陈旧的 3004 不许改掉已经建立的
        轮次 —— 改了会让 `_advance_to` 凭空补出几个不存在的「要不起」。
        万一这条给了错的座位，第一手出牌报文也会把它纠正回来（`on_play` 会设 `turn`）。
        """
        if not 0 <= seat <= 3:
            return
        if self.turn is None:
            self.turn = seat

    def on_seat(self, seat: int) -> None:
        """服务器直接说「我是哪个座位」（msgid 3019 的 `3.26.2`）。

        ⚠️ 这条**在出牌之前就能拿到**。修之前只能等自己出牌时的 `LeftCardList`，
        在那之前用的是写死的 `ME_SEAT` —— 用户 2026-09-29 报的
        「只有我出了一手之后座位才对」就是这件事。
        实测 8 段抓包里有 6 段的真实座位**不是**默认值，所以「偶尔对」纯属碰巧。
        """
        if not 0 <= seat <= 3:
            return
        if seat != self.me:
            self._me_note = (f"座位判定：同步报文说我是座位 {seat}"
                             f"（依据：3019 的 3.26.2）")
        self.me = seat
        self.me_confirmed = True

    def on_hand(self, ids: List[int], seat: Optional[int] = None) -> None:
        """服务器同步了我的手牌（msgid 3019，周期性重发）。

        `seat` 是同一条报文里的座位号（见 `on_seat`）—— 顺手一起认了。
        ⚠️ **顺序**：`_sync_hand` 可能触发 `on_deal`（会把 `me_confirmed` 清零），
        所以座位必须在它**之后**赋值。
        """
        self._sync_hand(ids)
        if seat is not None:
            self.on_seat(seat)

    def on_deal(self, level: Optional[int] = None) -> None:
        """新一局：清桌面状态，但保留级别（级别是跨局累积的）。

        `me_confirmed` 也要清 —— 报文里的座位号**每局都会变**，
        在认出本局的座位之前，任何相对方位都是猜的。
        """
        if level is not None:
            self.level = level
        self.deal_seq += 1
        self.hand = []
        self.plays = []
        self.steps = []
        self.finish_order = []
        self.me_confirmed = False
        # `turn` 也要清：新一局谁先出是由服务器重新分配的，**上一局的轮次是陈旧数据**。
        # 不清的话，换局后第一条事件会拿旧轮次去推「中间谁过了」，
        # 凭空补出几个不存在的手（`_advance_to` 是照 `turn` 走的）。
        # 清成 None 之后 `_advance_to` 会直接返回，直到本局第一条事件把轮次带回来。
        self.turn = None
        self.history = {0: [], 1: [], 2: [], 3: []}
        self.remaining = {}
        self.table = None
        self.passes = []
        self.deal_hand = set()
        self._votes = {}
        self._me_note = ""

    # --------------------------------------------------------- 座位与方位

    # 出牌顺序是座位号**递减**（日志实测：3→2→1→0→3），所以相对方位这样对应：
    # (座位 - 我) % 4 = 0 下（我） / 1 左 / 2 上（队友） / 3 右
    _ROLE = {0: "我", 1: "左对手", 2: "队友", 3: "右对手"}
    _ROLE_SHORT = {0: "我", 1: "左", 2: "队友", 3: "右"}

    def seat_label(self, seat: int) -> str:
        return self._ROLE.get((seat - self.me) % 4, f"座位{seat}")

    def seat_short(self, seat: int) -> str:
        """面板上位置窄，用短标签。"""
        return self._ROLE_SHORT.get((seat - self.me) % 4, str(seat))

    def on_play(self, seat: int, ids: List[int], card_type: int = 0,
                next_seat: int = 0, left: int = 0,
                left_cards: Optional[List[int]] = None) -> Play:
        """有人出牌了。

        next_seat 可能是 -1 —— 服务器用它表示「没有下一手了」（本局结束）。
        别当成座位号，否则会渲染成 18446744073709551615。
        """
        p = Play(seat=seat, cards=list(ids), card_type=card_type,
                 next_seat=next_seat, left=left)

        # LeftCardList 只有自己出牌时才有值 —— 一举解决两件事：
        #   1. 认出「我」是哪个座位（报文座位号每局都变，写死必错）
        #   2. 拿到自己精确的手牌，不用等 40 秒一次的 3019 同步
        # 放在最前面：万一是新一局的第一手，_sync_hand 会先把场面清干净，
        # 这一手再记到干净的状态里。
        if left_cards:
            if not self.deal_hand or set(left_cards) <= self.deal_hand:
                if seat != self.me:
                    self._me_note = (f"座位判定：报文座位 {seat} 是我"
                                     f"（依据：只有自己的出牌带 LeftCardList）")
                self.me = seat
                self._sync_hand(left_cards)
            else:
                # 换局的第一手：新手牌不是上一局的子集，`_sync_hand` 会清场。
                # **清完之后必须把座位也认下来** —— 报文里的座位号每局都会变，
                # 这里漏认的话整局的方位标签都会反（用户报过两次）。
                # 顺序不能反：`_sync_hand` 可能触发 `on_deal`（会把 me_confirmed 清零），
                # 所以赋值必须在它后面。
                self._sync_hand(left_cards)
                if seat != self.me:
                    self._me_note = f"换局重认座位：报文座位 {seat} 是我"
                self.me = seat
            self.me_confirmed = True

        # 出牌人跟预测的下一手对不上，说明中间那些座位行动过了（要不起）。
        # **服务器不给自己发「我要不起」的通知**（自己发起的不需要通知），
        # 所以这一步是必要的补记；不然轮次会一直停在我身上。顺带进动作流水。
        self._advance_to(seat)
        self.plays.append(p)
        self.history.setdefault(seat, []).append(p)
        if ids:
            # **一手打出来 = 重新开一轮**：其余三家重新获得机会（含之前「要不起」过的）。
            # 这条与 `rules.Hand.play` 完全一致（它在 55 局真实对局上验过），
            # 也是影子日志里 `passed` 那一维与训练时同源的前提。
            #
            # 原来写的是「只有领出的人再次出牌才清空」，理由是「不然轮次会算错」——
            # 那个理由站不住：有人压过一手之后，前面那些「要不起」还挂在名单上，
            # 于是 `_advance_to` 以为他们不用补步，**流水会少记**。
            # 实测（全量抓包那一局）：抓包 77 步 vs 日志真值 101 步，差的就是这个。
            self.passes = []
            self.table = p
            # left 是「这一手之后还剩几张」，0 是合法值（打完了），
            # 不能像别的字段那样把 0 当成"没这个信息"。
            self.remaining[seat] = max(0, left)
            self.steps.append((seat, list(ids)))
            if self.remaining[seat] == 0 and seat not in self.finish_order:
                self.finish_order.append(seat)
        self.turn = next_seat if 0 <= next_seat <= 3 else None
        self._maybe_clear_table()
        self._vote(seat, ids)
        return p

    def _vote(self, seat: int, ids: List[int]) -> None:
        """交叉验证「我在哪个座位」。

        我的起手牌 27 张，牌只会变少，**所以我出的牌一定是起手牌的子集**。
        4 张以上的牌凑巧落在别人手里概率很低（约 0.4%），信号够用。
        只在证据足够（≥3 票）且当前座位一票没有时才改判 —— 宁可保守，
        也不要在证据不足时乱跳。
        """
        if len(ids) < _MIN_VOTE_CARDS or not self.deal_hand:
            return
        if set(ids) <= self.deal_hand:
            self._votes[seat] = self._votes.get(seat, 0) + 1
            # 只记数、**不自动改判** —— 座位改判交给 LeftCardList 那个硬信号
            # （在 on_play 里）。这个统计只用于事后排查。

    def on_pass(self, seat: int, next_seat: Optional[int] = None) -> None:
        """有人要不起（msgid 3006）。

        服务器在 `3.7.3` 里**直接给了下一手是谁**，优先用它；拿不到时才自己推
        （出牌顺序是座位号递减，跳过本轮已经要不起的，转回本轮领出的人
        就说明这一轮结束、由他重新领出）。

        第一版漏了这件事：只看 3005 的 NextTurnSeatID，就会在
        「我出一手、别人都要不起、又轮回我」的时候一直停在别人身上 ——
        面板说「轮到右对手」，游戏里却已经亮着出牌按钮了。

        `_advance_to` 放在最前面：轮次可能已经越过了一些座位（包括我自己 ——
        我自己要不起时**收不到任何事件**，只能在这里补）。
        """
        self._advance_to(seat)
        if seat not in self.passes:
            self.passes.append(seat)
            if not self._is_finished(seat):
                self.steps.append((seat, None))
        if next_seat is not None:
            self.turn = next_seat
            self._maybe_clear_table()
            return
        leader = self.table.seat if self.table else None
        nxt = (seat - 1) % 4
        for _ in range(4):
            if nxt == leader or nxt not in self.passes:
                break
            nxt = (nxt - 1) % 4
        self.turn = nxt
        self._maybe_clear_table()

    # ------------------------------------------------------------- 显示用

    def level_name(self) -> str:
        return _LEVEL_NAMES.get(self.level, "?") if self.level else "?"

    def hand_grouped(self) -> List[str]:
        """我的手牌：按掼蛋大小排好（大的在前，级牌跟级别走）。"""
        return cards.names_sorted(self.hand, self.level)

    def render(self) -> str:
        """一个朴素的文本面板。先把数据打通，UI 后面再接。"""
        if self.turn is None:
            whose = "本局结束待发牌" if self.plays else "等待发牌"
        else:
            whose = self.seat_label(self.turn)
        out = [f"级别 打{self.level_name()}    轮到 {whose}", ""]
        out.append(f"我的手牌（{len(self.hand)} 张）:")
        out.append("  " + " ".join(self.hand_grouped()))
        out.append("")
        out.append("桌面:")
        if self.table and self.table.cards:
            t = self.table
            out.append(f"  {self.seat_label(t.seat)} 出 "
                       f"{' '.join(t.names(self.level))}")
        else:
            out.append("  （空）")
        if self.passes:
            out.append("  要不起: " + "、".join(
                self.seat_label(s) for s in self.passes))
        out.append("")
        out.append("各家出过的牌:")
        for seat in range(4):
            hs = self.history.get(seat) or []
            if not hs:
                continue
            left = self.remaining.get(seat)
            tail = f"（剩 {left} 张）" if left is not None else ""
            out.append(f"  {self.seat_label(seat)}{tail}:")
            for p in hs[-6:]:
                out.append(f"      {' '.join(p.names(self.level))}")
            if len(hs) > 6:
                out.append(f"      …共 {len(hs)} 手")
        return "\n".join(out)
