# 影子模式（出牌建议只记录、不上屏）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 轮到我出牌时，用已练成的 RL 模型算一次建议、写进 `net/shadow.jsonl`，同时记下我**实际出了什么**和**那局赢没赢** —— 不动游戏、不出牌、不给画面上的建议。

**Architecture:** 面板进程里多挂一个「影子记录器」：面板每收到一个网络事件，先把事件喂给现有状态机（`net/state.py`），再让记录器看一眼状态。轮到我了就算一次建议（`net/advise.py`：`GameState → Observation → meld.legal_moves → train.net 打分`），把决策点挂起；等我的着法真的发生（出牌事件，或从轮转**推断**出来的「要不起」）再回填落盘。一局结束时补一行结果行。**推理链只加载裁判（`net.sim.meld`）与权重，不加载模拟器。**

**Tech Stack:** Python 3（项目 venv `.venv`）、numpy、torch（已有，`train/net.py` 的 `QNet`）、pytest、tkinter（面板）。

**Spec:** `docs/superpowers/specs/2026-09-24-guandan-rl-advisor-design.md` §8（§8.1 推理链 / §8.2 影子模式 / §8.4 红线），配 `docs/superpowers/plans/2026-09-25-plan3-training-delivery.md` 的「五」（影子模式要做什么、四个会咬人的点）。

---

## 全局约束

- **红线（spec §8.4）**：绝不替用户出牌、**不往游戏发任何报文**。影子模式只写文件。
- **建议不上屏（用户 2026-09-25 决定）**：第一版面板只显示进度（算了几个决策点、跳过几个、上局赢没赢），**不显示建议内容** —— 否则用户会被建议影响，「模型与人的分歧」这份数据就废了。
- **状态只能含可观测信息**：一律走 `net/sim/env.py` 的 `Observation` + `encode_state`。那份类型里根本没有「四家手牌」字段，所以**不许绕开它自己拼向量**。
- **一律复用、不许复制第二份**：编码器 `env.encode_state/encode_action/encode_history`、裁判 `meld.legal_moves/as_meld/beats`、打分 `train.net.q_values`。本仓库为「副本会漂」吃过亏（Plan 1 有两处副本各修过一次同一条规则）。
- **座位一律相对化**（0 = 自己），四个座位共享一套权重。
- **级别必须 1..13 且是本局的**；拿不准就**跳过并计数**，不许猜（级别错 → 逢人配全错 → 建议全错）。
- 落盘风格与 `net/events.jsonl` 一致：追加写、`ensure_ascii=False`、`buffering=1`、一行一个 JSON。
- `net/shadow.jsonl` 是运行时产物，**加进 `.gitignore`**（`net/events.jsonl`、`net/raw.jsonl` 已在里面）。
- 面板上的文字给**不懂技术的人**看：中文、不用术语、别超过一行。
- 命令一律写成 `.venv/Scripts/python.exe -m ...`（用户机器上没有全局 python 环境）。

---

## Review Focus

规格是愿景文档，它没写的地方最容易出事。下面六条是这次最可能咬人的输入/场景，**每一条都在某个任务的测试里钉住**（括号里是钉它的地方）：

1. **级别是上一局的**：换局后日志要 ~20 秒才写出新的 `Trump`，这段窗口里 `st.level` 还是上一局的 → 建议是错的。**期望**：每条记录带 `level` / `level_age_s` / `deal`，离线能按日志的 `Trump` 逐局核对、数出「级别对不上」的条数，而不是悄悄当成正确（任务 3 落这几个字段 + 任务 5 的验收 ⑥）。
2. **座位还没认出来**：报文里的座位号**每局都会变**（实测同一段抓包里 11:11 那局我是 seat1、11:15 那局我是 seat2），而座位只能靠「我自己出牌时带的 `LeftCardList`」认出来 —— 所以**换局后到我第一次出牌之间是陈旧的**。**期望**：这段窗口一个决策点都不算，计入跳过原因「座位未确认」（任务 1 的 `me_confirmed` + 任务 3 的守卫）。
3. **我自己「要不起」在报文里是静默的**（实测：服务器只通知别人要不起）→ 动作流水里我那一行必须**从轮转推断**补上，且**已出完的人不能补**。**期望**：流水与真值 `rules.Hand.steps` 逐步一致（任务 1 的测试 + 任务 5 的验收 ②）。
4. **桌面牌解不出牌型**：`meld.as_meld` 返回 `None`（残留动画、或规则没覆盖）时**不能当成「桌上无牌」**（那会让候选全错）。**期望**：跳过并计数，原因「桌面牌解不出牌型」（任务 2 的守卫测试）。
5. **退出/换局时还没回填的决策点**：如果不落盘，「分歧点」的统计会有偏（丢掉的全是局末那几个）。**期望**：写成 `resolved:false` + 原因，并在局末汇总里单列 `n_unresolved`（任务 3 的测试）。
6. **权重文件缺失/损坏**：面板不许崩。**期望**：影子模式整体降级为「不记录」，面板上明说一行原因，游戏照常显示（任务 2 的 `load_net` + 任务 3/4 的测试）。

另有一条**不属于本计划**但要知道的：进贡/还贡期间的状态机行为已知不准（网络里没定位那两个字段）。影子模式不模拟、只读实时状态，所以**不受影响**；但进贡那几手如果被算成决策点，记录里会带着错的桌面 —— 验收里**不把进贡窗口计入分母**（任务 5 的说明）。

---

## 本次实测到的既有事实（执行者直接用，别重新推）

这些都是这次会话在真机素材上验过的，**重推一遍很费时**：

**素材**：`net/raw.jsonl`（全量载荷抓包，787 帧，2026-09-25 11:11:13 ~ 11:18:55，`GUANDAN_RAW=1` 那次留下的）里**正好两局**，本机日志 `2026-09-25-11.log` 覆盖同一时段，两局**都有结算**：

| 局 | 日志发牌时刻 | Trump | 日志 Rank（`Rank[seat]`） | 我先出牌（=座位被认出来）的时刻 |
|---|---|---|---|---|
| A | 11:11:14 | **9** | `[4, 3, 1, 2]` | 11:11:28（我是 **seat1**） |
| B | 11:15:03 | **10** | `[3, 2, 4, 1]` | 11:15:31（我是 **seat2**） |

判据与证据（都可复现）：

1. **我在哪个座位**：拿日志里发牌给我自己的 27 张（`SendCardsService` 的 `Cards`）去比报文里的 `3.6.10`（`LeftCardList`）—— 局 A 里 seat1 的 9 次带 `LeftCardList` 的出牌 **9/9** 都是我这 27 张的子集，seat2 的 7 次 **0/7**；局 B 正好反过来。**所以 `LeftCardList` 确实只在我自己出牌时出现，而且座位号每局都会变。**
2. **我自己「要不起」是静默的**：按轮转（`0→3→2→1`）从出牌序列反推出来的「该有的要不起」里，**属于我的那些没有对应的 3006**（例：11:16:02 seat3 要不起、`next=2`（我），紧跟着 11:16:05 就是 seat1 要不起 —— 中间我自己那次没有事件）。另外 3006 里 seat 字段被省略的恰好 30 条 = seat0 的 30 条（protobuf 省略 0），说明解码口径是对的。
3. **发牌报文里的级别不是「本局级别」**：msgid 3008（11:14:11 那一帧）的 `3.9.22.12` = `[10, 9, 10, 9]`，那是**两队各自的级别**（座位 0/2 一队打 10、1/3 一队打 9），而局 A 的实际 Trump 是 **9**（= 我那一队的级别，因为上一局我们输了）。**所以「面板从 3008 拿级别」这条在 Plan 3 台账里是错的 —— 本版级别仍从日志取（`LevelWatcher`），3008 先别接。**
4. 我自己的出牌在**日志**里也是全的（`NotifyGiveCards`，含 `SeatID`/`CardList`/`LeftCardLen`/`NextTurnSeatID`），两局 41 + 39 手，与抓包的 80 条出牌事件一一对应 —— 真值来源是现成的。
5. 出完顺序可以从报文直接读：`3005` 的 `3.6.5`（该家剩几张）掉到 0 的先后顺序，与日志 `Rank` **逐局一致**（局 A：seat2 → seat3 → seat1(我) → seat0 = `[4,3,1,2]` ✓）。

**跑测试要用的现成命令**（改动前先跑一遍当基线）：

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q                 # 期望 308 passed
.venv/Scripts/python.exe -m tools.accept_meld                # 四项全过
.venv/Scripts/python.exe -m tools.accept_sim                 # 53 局回放，退出码 0
```

---

## 文件结构

| 文件 | 新建/修改 | 职责（一个文件一件事） |
|---|---|---|
| `net/state.py` | 修改 | 状态机：**新增动作流水** `steps`（含推断出来的「我过了」）、局号 `deal_seq`、座位确认位 `me_confirmed`、出完顺序 `finish_order`；**修一个真 bug**：换局时座位没有重认 |
| `net/advise.py` | **新建** | 推理链：`GameState → Observation`（唯一的适配层）、从流水编历史、取候选、打分；所有「算不了」的原因都在这里定义 |
| `net/shadow.py` | **新建** | 影子记录器：决策点去重、落盘 `net/shadow.jsonl`、回填我实际出的牌、局末结果行、退出时收尾 |
| `net/panel.py` | 修改 | 文本面板：事件循环里挂影子记录器；显示一行进度 |
| `net/table.py` | 修改 | 图形牌桌面板：同上；`--replay` 加 `--level`（离线冒烟要用） |
| `tools/accept_sim.py` | 修改 | `replay()` 加一个可选 `record` 回调（**复用**已验证的回放循环，不另写一份） |
| `tools/accept_shadow.py` | **新建** | 离线验收：用抓包 + 日志真值验「流水 / 决策点状态 / 真实着法在候选里 / 两源同一建议 / 局末输赢 / 级别」 |
| `tests/test_state_steps.py` | **新建** | 任务 1 的测试 |
| `tests/test_advise.py` | **新建** | 任务 2 的测试（核心是**跨源一致性**：网络那条路与模拟器那条路编出同一个 700 维向量） |
| `tests/test_shadow.py` | **新建** | 任务 3 的测试 |
| `tests/test_panel_wiring.py` | **新建** | 任务 4 的接线测试（不开窗口） |
| `tests/test_accept_shadow.py` | **新建** | 任务 5 的「验收脚本自己不许假绿」 |
| `.gitignore` | 修改 | 加 `net/shadow.jsonl` |
| `HANDOFF.md` + `docs/superpowers/plans/2026-09-25-shadow-delivery.md` | 修改/新建 | 任务 6：怎么跑、怎么读、哪些地方还不准 |

**依赖方向**：`net/advise.py` 会 `import train.net`（拿 `QNet` 与 `q_values`）。**这是刻意的**：模型定义只能有一份，复制一份到 `net/` 就会漂。要是评审觉得方向别扭，正确的后续是把 `QNet` 挪到共用模块，**不是**复制。

---

### Task 1: 状态机补上「动作流水」并修掉换局不重认座位的洞

**Files:**
- Modify: `net/state.py`
- Test: `tests/test_state_steps.py`

**Interfaces:**
- Consumes: 无（只依赖现有的 `on_play` / `on_pass` / `on_deal` 事件入口）
- Produces（后面三个任务都靠这些）：
  - `GameState.steps: List[Tuple[int, Optional[List[int]]]]` —— 本局的动作流水，**按时序**，`(座位, 出的牌)`；`牌 is None` = 该座位「要不起」。与 `rules.Hand.steps` 同语义（**已出完的座位不占步**）。
  - `GameState.deal_seq: int` —— 局号，`on_deal` 时 +1；面板中途接进来时是 0。
  - `GameState.me_confirmed: bool` —— 本局的座位是否已经认出来（换局时清零）。
  - `GameState.finish_order: List[int]` —— 本局出完的座位，按出完先后。
  - 内部方法 `GameState._advance_to(seat)`、`GameState._is_finished(seat) -> bool`

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_state_steps.py`：

```python
"""动作流水（`GameState.steps`）—— 影子模式的历史就建在它上面。

⚠️ 这一份流水**不是**「把事件抄一遍」：我自己「要不起」时服务器**不发事件**
（实测，见计划文档「本次实测到的既有事实」第 2 条），只能从轮转推出来。
"""
from net.sim import meld
from net.state import GameState

A = meld.cid_from_name


def _play(st, seat, nxt, played, rest, mine=False):
    """一个出牌事件。`rest` = 这手之后该家还剩的牌（服务器报的 `LeftCardLen` = len(rest)）。
    `mine=True` 时同时带上 `LeftCardList`（只有我自己的出牌有）。
    """
    st.on_play(seat, list(played), 0, nxt, len(rest), sorted(rest) if mine else None)


def test_plays_and_notified_passes_are_recorded_in_order():
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4"), A("S5")], mine=True)   # 我出 S3 -> 轮到 0
    assert st.me == 1 and st.me_confirmed
    _play(st, 0, 3, [A("S9")], [A("SK")])                        # 0 出牌 -> 轮到 3
    st.on_pass(3, 2)                                             # 3 要不起（我会收到）
    assert [s for s, _c in st.steps] == [1, 0, 3]
    assert st.steps == [(1, [A("S3")]), (0, [A("S9")]), (3, None)]


def test_my_silent_pass_is_inferred_from_the_rotation():
    """我自己要不起时没有事件 —— 必须在下一个事件到来时从轮转补上，且顺序在它前面。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)             # 我出牌 -> 轮到 0
    _play(st, 0, 3, [A("S9")], [A("SK")])                        # 0 出牌 -> 轮到 3
    _play(st, 3, 2, [A("SJ")], [A("SQ")])                        # 3 出牌 -> 轮到 2
    st.on_pass(2, 1)                                             # 2 要不起 -> 轮到我
    st.on_pass(0, 3)                                             # 0 要不起（我会收到）
    assert [s for s, _c in st.steps] == [1, 0, 3, 2, 1, 0]
    assert st.steps[4] == (1, None), "我自己那一步是推断出来的，位置必须在 0 那一步之前"


def test_a_finished_seat_is_not_recorded_as_a_passer():
    """已出完的座位没有「过」这个动作（`rules.Hand._advance` 会跳过他不记步）。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)
    _play(st, 0, 3, [A("S9")], [A("SK")])
    _play(st, 3, 2, [A("SK")], [])                               # 座位 3 出完（left=0）
    assert st.finish_order == [3]
    st.turn = 1                                                  # 手工摆一个中间态：从 1 走到 2 会经过 0 和 3
    st.on_pass(2, 3)
    assert (3, None) not in st.steps, "出完的人不该被记成「过」"
    assert [s for s, _c in st.steps][-3:] == [1, 0, 2], \
        "1 是我（推断出来的过）、0 是真的过了、2 是自己要不起：中间的 3 被跳过"


def test_deal_change_resets_the_stream_and_reidentifies_my_seat():
    """换局：新手牌不是上一局的子集 -> 清场。**清完必须把座位也认下来** ——
    座位号每局都会变（实测 11:11 那局我是 seat1、11:15 那局我是 seat2），
    漏认的话整局方位都会反（用户报过两次）。"""
    st = GameState()
    _play(st, 1, 0, [A("S3")], [A("S4")], mine=True)             # 上一局我是 seat1
    assert st.me == 1 and st.deal_seq == 0
    _play(st, 2, 1, [A("H5")], [A("H6")], mine=True)             # 这一局我是 seat2
    assert st.deal_seq == 1
    assert st.me == 2 and st.me_confirmed, "换局后座位必须重认"
    assert st.steps == [(2, [A("H5")])], "新一局的流水要从零开始"
    assert st.finish_order == []
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_state_steps.py -q
```
Expected: FAIL —— `AttributeError: 'GameState' object has no attribute 'steps'`（以及 `me_confirmed` / `finish_order`）。

- [ ] **Step 3: 改 `net/state.py`**

在 `GameState` 的字段区加上四个字段（放在 `_votes` 前面，保持「公开字段在前、内部在后」的现有次序）：

```python
    #: 本局的动作流水（按时序）：`(座位, 出的牌)`，`None` = 该座位「要不起」。
    #: **和 `rules.Hand.steps` 同语义**：出完的座位不占步。影子模式的历史就取它。
    #: 注意里面**有推断出来的步** —— 我自己要不起时服务器不发事件（实测），
    #: 只能从轮转补；补的位置在下一个事件之前，见 `_advance_to`。
    steps: List[tuple] = field(default_factory=list)
    #: 局号。`on_deal` 时 +1；面板中途接进来时是 0。影子日志用它把决策点归到一局。
    deal_seq: int = 0
    #: 本局的座位认出来了没有。**换局时必须清零** —— 座位号每局都变，
    #: 认出来之前算出来的相对方位全是错的（见计划文档 Review Focus 第 2 条）。
    me_confirmed: bool = False
    #: 本局出完的座位，按出完先后。影子日志的「那局赢没赢」用它推。
    finish_order: List[int] = field(default_factory=list)
```

`on_deal` 里加四行：

```python
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
        self.history = {0: [], 1: [], 2: [], 3: []}
        self.remaining = {}
        self.table = None
        self.passes = []
        self.deal_hand = set()
        self._votes = {}
        self._me_note = ""
```

新增两个内部方法（放在 `_sync_hand` 后面）：

```python
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
```

把 `on_play` 里原来那段「补记要不起」的循环**换成** `self._advance_to(seat)`，并在追加出牌的那一段里补流水与出完记录：

```python
        self._advance_to(seat)
        self.plays.append(p)
        self.history.setdefault(seat, []).append(p)
        if ids:
            # 「要不起」只在新一轮（领出的人再次出牌）才清空 —— 同一轮里
            # 要不起的人是一直出局的，清早了轮次就会算错。
            if self.table is None or seat == self.table.seat:
                self.passes = []
            self.table = p
            # left 是「这一手之后还剩几张」，0 是合法值（打完了），
            # 不能像别的字段那样把 0 当成"没这个信息"。
            self.remaining[seat] = max(0, left)
            self.steps.append((seat, list(ids)))
            if self.remaining[seat] == 0 and seat not in self.finish_order:
                self.finish_order.append(seat)
        self.turn = next_seat if 0 <= next_seat <= 3 else None
        self._vote(seat, ids)
        return p
```

`on_pass` 开头加 `self._advance_to(seat)`，并在记 passes 时补一步：

```python
    def on_pass(self, seat: int, next_seat: Optional[int] = None) -> None:
        self._advance_to(seat)
        if seat not in self.passes:
            self.passes.append(seat)
            if not self._is_finished(seat):
                self.steps.append((seat, None))
        if next_seat is not None:
            self.turn = next_seat
            return
        ...（下面原有的「自己推下一手」逻辑不动）
```

`on_play` 里认座位的那个分支**修掉漏洞**（else 分支原来只同步手牌、不认座位）：

```python
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
```

- [ ] **Step 4: 跑测试与回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_state_steps.py -q     # 4 passed
.venv/Scripts/python.exe -m pytest tests/ -q                        # 原 308 + 新增，全过
.venv/Scripts/python.exe -m net.replay                              # 轮次一致性不许变差
```
Expected: 全过；`net.replay` 的「逐条对齐」与「轮次」两行数字**不下降**（原 28/28 与 24/27）。

- [ ] **Step 5: 提交**

```bash
git add net/state.py tests/test_state_steps.py
git commit -m "feat(state): 动作流水（含推断出来的「我过了」）+ 换局重认座位

- steps/deal_seq/me_confirmed/finish_order 四个字段，影子模式的历史建在 steps 上
- 我自己「要不起」服务器不发事件（实测），从轮转补；两条入口都要补
- 修 bug：换局第一手走的是 _sync_hand 的 else 分支，那条路原来不认座位
  （座位号每局都变，实测同一段抓包里 seat1 -> seat2）"
```

---

### Task 2: `net/advise.py` —— 推理链与守卫

**Files:**
- Create: `net/advise.py`
- Test: `tests/test_advise.py`

**Interfaces:**
- Consumes: `GameState.steps / deal_seq / me_confirmed / finish_order / history / remaining / passes / table / hand / turn / me / level`（Task 1）
- Produces:
  - 跳过原因常量 `SKIP_SEAT / SKIP_LEVEL / SKIP_TABLE / SKIP_HIST / SKIP_HAND / SKIP_NOCAND / SKIP_NOTURN`
  - `@dataclass Skip: reason: str`
  - `@dataclass Built: obs; hist; table_meld; reason`（`reason` 非空 = 跳过）
  - `build(st) -> Built`
  - `candidates(built) -> list`（含 `None`；`None` 只在跟牌时出现）
  - `@dataclass Advice: obs; hist; cands; q; order`（`order` = 按 Q 降序的下标列表）
  - `advise(st, net, topk=3) -> Union[Advice, Skip]`
  - `newest_weights(root="runs/rl") -> Optional[str]`、`load_net(path=None, device="cpu") -> Tuple[Optional[QNet], str]`、`weights_info(path) -> dict`

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_advise.py`。**核心是第一个测试的「跨源一致性」** —— 它是防止适配层漂移的唯一硬约束：

```python
"""推理链：`GameState -> Observation` 必须与模拟器那条路**编出同一个向量**。

白名单式的字段比对会漏（漏一个字段只是少一维，看不出来），只有
「700 维逐位相等」才能保证喂给网络的东西和训练时**完全一样**。
"""
import numpy as np
import pytest

from net import advise
from net.sim import env, meld, rules
from net.state import GameState

A = meld.cid_from_name
LEVEL = 9
MY_SEAT = 1


def _drive(actions):
    """同一局，两条路各走一遍 —— 一边是模拟器（明牌 `Hand`），一边是网络状态机（事件流）。

    `actions = [(座位, 牌列表 or None)]`（座位是绝对座位号，`None` = 要不起）。
    两边唯一的**故意差异**：我自己要不起时**不发事件**（服务器本来就不发，实测），
    这样「从轮转推断」那条路也被压进测试。
    返回 `(hand, st, played)`：走到同一位置的两边状态，以及各家出过的牌。
    """
    hands = [{A("S3"), A("S4")}, {A("H5"), A("H6"), A("D7")},
             {A("C8"), A("C9")}, {A("D8"), A("D9")}]
    hand = rules.Hand(hands=[set(h) for h in hands], level=LEVEL, turn=actions[0][0])
    st = GameState()
    st.level = LEVEL
    played = {s: set() for s in rules.SEATS}
    for seat, cs in actions:
        cs = list(cs) if cs else None
        # 「这一手之后还剩几张」要先算 —— 动手之后再算就多减了一手。
        rest = sorted(hand.hands[seat] - set(cs or ()))
        if cs is None:
            if seat != MY_SEAT:
                st.on_pass(seat)                    # 别人的要不起服务器会通知我
        else:
            st.on_play(seat, cs, 0, 0, len(rest),
                       rest if seat == MY_SEAT else None)
        m = None if cs is None else meld.as_meld(cs, LEVEL)
        assert cs is None or m is not None, f"测试素材本身有问题：{cs}"
        if m is None:
            hand.pass_turn(seat)
        else:
            hand.play(seat, m)
            played[seat] |= set(cs)
    return hand, st, played


def test_network_path_and_simulator_path_encode_the_same_vector():
    """`encode_state` 内部按出牌人相对化，所以两套座位编号只要差一个旋转就等价 ——
    两边的编号都遵守 `0→3→2→1` 的出牌顺序，因此可以直接逐位比。
    """
    acts = [(1, [A("S3")]), (0, [A("S4")]), (3, None), (2, None),
            (1, None), (0, [A("S9")])]      # 倒数第二步是**我要不起**（网络侧不发事件）
    hand, st, played = _drive(acts)
    assert st.steps[-2] == (MY_SEAT, None), "我自己那一步应当是推断出来的"
    b = advise.build(st)
    assert not b.reason, f"不该跳过：{b.reason}"
    truth = env.observe(hand, MY_SEAT, played, hand.table)
    assert np.array_equal(env.encode_state(b.obs), env.encode_state(truth))
    assert np.array_equal(b.hist, env.encode_history(hand, MY_SEAT))


def test_pass_is_a_candidate_only_when_there_is_a_table():
    hand, st, _p = _drive([(1, [A("S3")]), (0, [A("S4")])])     # 桌上一张 4，轮到我
    assert st.turn == MY_SEAT
    b = advise.build(st)
    assert None in advise.candidates(b), "跟牌时「能压也可以过」，None 永远在候选里"
    # 领出：桌上没牌时候选里**不许**有「过」
    st2 = GameState()
    st2.level = LEVEL
    st2.on_play(MY_SEAT, [A("S3")], 0, 0, 2, [A("S4"), A("S5")])  # 认座位
    st2.table = None
    st2.turn = MY_SEAT
    assert None not in advise.candidates(advise.build(st2))


def test_every_uncomputable_case_returns_a_reason_and_never_raises():
    st = GameState()
    st.level = LEVEL
    assert advise.build(st).reason == advise.SKIP_SEAT, "座位没认出来之前不许算"
    st.on_play(MY_SEAT, [A("S3")], 0, 0, 2, [A("S4"), A("S5")])
    assert st.me_confirmed
    st.turn = 0
    assert advise.build(st).reason == advise.SKIP_NOTURN
    st.turn = MY_SEAT
    st.table = None
    st.hand = []
    assert advise.build(st).reason == advise.SKIP_HAND
    st.hand = [A("S4")]
    st.level = None
    assert advise.build(st).reason == advise.SKIP_LEVEL, "级别未知必须跳过，不许猜"
    st.level = 14
    assert advise.build(st).reason == advise.SKIP_LEVEL, "14 是日志里 A 的另一种写法，越界"
    st.level = LEVEL
    # 桌面这两张牌组不成任何合法牌型（不是对子、不是顺子）—— 不许当成「桌上无牌」
    from net.state import Play
    st.table = Play(seat=0, cards=[A("S3"), A("S5")], card_type=0, next_seat=0, left=26)
    assert advise.build(st).reason == advise.SKIP_TABLE
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_advise.py -q
```
Expected: FAIL —— `ModuleNotFoundError: No module named 'net.advise'`。

- [ ] **Step 3: 写 `net/advise.py`**

```python
"""出牌建议的推理链（spec §8.1）：`GameState → Observation → 候选 → 打分 → 建议`。

**不加载模拟器**：这里不建 `GuandanEnv`、不建 `rules.Hand`、不跑对局循环。
但要**复用**它的编码器与裁判（`net/sim/env.py` 的 `encode_*`、`net/sim/meld.py` 的
`legal_moves/as_meld`、`train/net.py` 的 `q_values`）—— 本仓库为「副本会漂」吃过亏，
推理侧另写一份编码器，训练与上线就会悄悄不一致。

⚠️ **`import train.net` 是刻意的**：模型定义只能有一份。要改方向就把 `QNet`
挪到共用模块，**别复制**。

这个模块是**明牌与策略之间唯一的窄口**（和 `env.observe` 同一个角色）：
进来的是面板的状态机（里面有四家的出牌记录），出去的 `Observation` 里
只有公开信息 + 我的手牌 —— 那个类型的字段表里就没有「对手手牌」。
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from net.sim import env, meld, rules
from net.state import GameState

# 「算不了」的原因。**每一个都要能出现在影子日志里** —— 静默跳过等于数据有偏。
SKIP_SEAT = "座位未确认"
SKIP_LEVEL = "级别未知或越界"
SKIP_TABLE = "桌面牌解不出牌型"
SKIP_HIST = "流水里有解不出的历史着法"
SKIP_HAND = "手牌为空"
SKIP_NOCAND = "枚举不出任何着法"
SKIP_NOTURN = "没轮到我"

#: 一个座位起手 27 张（两副牌 108 / 4）。`remaining` 没记到时用它兜底。
DEAL_EACH = 27


@dataclass
class Skip:
    reason: str


@dataclass
class Built:
    obs: Optional[env.Observation] = None
    hist: Optional[np.ndarray] = None
    table_meld: Optional[meld.Meld] = None
    reason: str = ""


@dataclass
class Advice:
    obs: env.Observation
    hist: np.ndarray
    cands: list                 # 候选，含 None（过）
    q: List[float]              # 与 cands 等长
    order: List[int]            # 按 Q 降序的下标


class _HandLike:
    """`env.encode_history` 只要两样东西：`.steps`（每步有 `.meld` / `.seat`）与 `.level`。

    **故意复用它，不另写一份编码器** —— 训练与推理的历史必须逐位一致。
    """

    def __init__(self, steps, level):
        self.steps = steps
        self.level = level


def _left_of(st: GameState, seat: int) -> int:
    """该座位还剩几张。`remaining` 是服务器给的真值（每条出牌消息都带）。"""
    if seat in st.remaining:
        return st.remaining[seat]
    played = sum(len(p.cards) for p in (st.history.get(seat) or []))
    return max(0, DEAL_EACH - played)


def history_from_steps(steps, level: int, seat: int) -> Optional[np.ndarray]:
    """动作流水 -> `(15, 147)` 历史。**解不出的历史着法直接返回 None**（不猜）。

    ⚠️ 传进来的必须是**决策点当时**的流水（`st.steps` 的当前快照）——
    用打完之后的流水会把后面才发生的牌塞进历史，那是未来信息泄漏。
    """
    out = []
    for s, cs in steps[-env.HISTORY_LEN:]:
        if cs is None:
            out.append(rules.Step(s, None, 0))
            continue
        m = meld.as_meld(list(cs), level)
        if m is None:
            return None
        out.append(rules.Step(s, m, 0))
    return env.encode_history(_HandLike(out, level), seat)


def build(st: GameState) -> Built:
    """`GameState` -> 可观测状态 + 历史。算不了就返回 `reason`（**不抛异常**）。"""
    if not st.me_confirmed:
        return Built(reason=SKIP_SEAT)
    if st.turn != st.me:
        return Built(reason=SKIP_NOTURN)
    if not st.hand:
        return Built(reason=SKIP_HAND)
    if st.level is None or not 1 <= st.level <= 13:
        return Built(reason=SKIP_LEVEL)

    table, table_meld = (), None
    kind, rank = 0, -1
    if st.table and st.table.cards:
        # 桌面那串牌 ID 必须先解释成带 rank 的 Meld，`beats` 才用得上 ——
        # 这正是 `meld.as_meld` 存在的理由（推理链是它的生产调用方）。
        table_meld = meld.as_meld(list(st.table.cards), st.level)
        if table_meld is None:
            # **不能当成「桌上无牌」**：那会让候选变成「随便领出」，全错。
            return Built(reason=SKIP_TABLE)
        table, kind, rank = tuple(table_meld.cards), table_meld.kind, table_meld.rank

    hist = history_from_steps(st.steps, st.level, st.me)
    if hist is None:
        return Built(reason=SKIP_HIST)

    obs = env.Observation(
        seat=st.me,
        hand=frozenset(st.hand),
        played=tuple(frozenset(c for p in (st.history.get(s) or []) for c in p.cards)
                     for s in rules.SEATS),
        left=tuple(_left_of(st, s) for s in rules.SEATS),
        table=table, table_kind=kind, table_rank=rank,
        passed=tuple(s in st.passes for s in rules.SEATS),
        turn=st.turn,
        level=st.level,
    )
    return Built(obs=obs, hist=hist, table_meld=table_meld)


def candidates(b: Built) -> list:
    """当前候选。口径与 `rules.Hand.actions` **完全一致**（那是在 55 局真牌上验过的）：
    `melds_from(sorted(hand))` 再按 `beats` 过滤；跟牌时 `None` 永远在候选里。
    """
    moves = meld.legal_moves(sorted(b.obs.hand), b.table_meld, b.obs.level)
    if b.table_meld is None:
        return list(moves)
    return list(moves) + [None]


def advise(st: GameState, net, topk: int = 3) -> "Advice | Skip":
    """算一次建议。算不了返回 `Skip(原因)` —— 调用方负责计数落盘。"""
    from train.net import q_values

    b = build(st)
    if b.reason:
        return Skip(b.reason)
    cands = candidates(b)
    if not cands:
        return Skip(SKIP_NOCAND)
    q = [float(x) for x in q_values(net, b.obs, cands, b.hist)]
    order = sorted(range(len(cands)), key=lambda i: -q[i])
    return Advice(obs=b.obs, hist=b.hist, cands=cands, q=q, order=order)


def newest_weights(root: str = "runs/rl") -> Optional[str]:
    """最新的一版 `best.pt`（按修改时间）。没有就返回 None。"""
    found = glob.glob(os.path.join(root, "*", "best.pt"))
    return max(found, key=os.path.getmtime) if found else None


def load_net(path: str = None, device: str = "cpu") -> Tuple[Optional[object], str]:
    """加载权重。返回 `(net, 错误说明)`；**加载失败不抛异常**（面板不许因为这个崩）。

    默认放 **CPU**：一次决策点只做一次前向、候选不到 20 个，几十毫秒的量级；
    CPU 不跟游戏抢显存，结果也可复现（验收要拿它做逐位比对）。
    """
    import torch

    from train.net import QNet

    p = path or os.environ.get("GUANDAN_WEIGHTS") or newest_weights()
    if not p:
        return None, f"找不到权重（{os.environ.get('GUANDAN_WEIGHTS') or 'runs/rl/*/best.pt'}）"
    if not os.path.exists(p):
        return None, f"权重文件不存在：{p}"
    try:
        ck = torch.load(p, map_location="cpu")
        net = QNet()
        net.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)
        net.to(device).eval()
    except Exception as exc:                      # noqa: BLE001 - 面板不许崩
        return None, f"权重加载失败（{type(exc).__name__}: {exc}）：{p}"
    return net, ""


def weights_info(path: str) -> dict:
    """权重文件里的元信息（训练时存了局数与胜率）——写进影子日志的会话行做溯源。"""
    import torch
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception:                             # noqa: BLE001
        return {}
    if not isinstance(ck, dict):
        return {}
    return {k: ck[k] for k in ("games", "winrate_greedy", "winrate_random") if k in ck}
```

- [ ] **Step 4: 跑测试**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_advise.py -q     # 3 passed
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: PASS。第一个测试必须**真的过** —— 它红了就说明适配层与模拟器不一致，**先修适配层，别改测试**。

- [ ] **Step 5: 提交**

```bash
git add net/advise.py tests/test_advise.py
git commit -m "feat(advise): 推理链 GameState -> Observation -> 候选 -> 打分

- 编码/裁判/打分一律复用（env.encode_*/meld.legal_moves/q_values），不复制
- 跨源一致性测试：网络那条路与模拟器那条路编出同一个 700 维向量
- 七种算不了的原因都返回 reason，不抛异常（静默跳过 = 数据有偏）"
```

---

### Task 3: `net/shadow.py` —— 决策点记录器

**Files:**
- Create: `net/shadow.py`
- Test: `tests/test_shadow.py`

**Interfaces:**
- Consumes: `advise.advise/build/Skip/weights_info`（Task 2）、`GameState.steps/deal_seq/me_confirmed/finish_order/me/level/turn`（Task 1）、`net.cards`
- Produces:
  - `SHADOW = <项目根>/net/shadow.jsonl`、`SCHEMA = 1`
  - `class ShadowLog:` —— `__init__(self, net=None, out_path=SHADOW, weights="", weights_note="", topk=3)`、`after_event(self, st, ev) -> None`、`note_level(self) -> None`、`close(self) -> None`；属性 `enabled`、`out_path`、`n_decisions`、`skips`（`{原因: 次数}`）、`last_line`（给面板显示的一行中文）

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_shadow.py`：

```python
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
    assert "q" in dec[0]["top"][0] and dec[0]["top"][0]["names"]
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
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_shadow.py -q
```
Expected: FAIL —— `ModuleNotFoundError: No module named 'net.shadow'`。

- [ ] **Step 3: 写 `net/shadow.py`**

```python
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
```

⚠️ 三处容易写错，照着上面抄就行：`_rank_of` 用 `self._order`（**别自己再排一遍 Q**）；`_finish_deal` 开头必须判 `self._deal_seq is None`（否则 `close()` 会因为「一个事件都没来过」也写一行汇总）；`_emit` 里的 `os.makedirs` 在 `out_path` 没有目录时（比如直接用文件名）会抛 —— 判一下目录非空。

- [ ] **Step 4: 跑测试**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_shadow.py -q     # 6 passed
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -c "from net import shadow; print(shadow.SHADOW)"
```
Expected: 全过；最后一条打印 `...\net\shadow.jsonl`。**此刻还没有这个文件**（没跑过面板）。

- [ ] **Step 5: 提交**

```bash
git add net/shadow.py tests/test_shadow.py
git commit -m "feat(shadow): 影子记录器 —— 决策点落盘 + 回填实际着法 + 局末结果

- 一行一个决策点，一行一个 deal_end；追加写、与 events.jsonl 同风格
- 回填来自状态机的动作流水，所以我自己「要不起」也能记成「过」
- 退出/换局时未回填的也落盘（resolved:false + n_unresolved），避免统计有偏
- 权重缺失不抛异常，降级为「不记录」并在面板上说明"
```

---

### Task 4: 接进两个面板（只显示进度，不显示建议）

**Files:**
- Modify: `net/panel.py`（`draw` / `run_live` / `run_replay` / `main`）
- Modify: `net/table.py`（`TableWindow.draw` / `run_live` / `run_replay` / `main`）
- Modify: `.gitignore`
- Test: `tests/test_panel_wiring.py`

**Interfaces:**
- Consumes: `shadow.ShadowLog`（Task 3）、`advise.load_net/newest_weights`（Task 2）、`panel.apply_event`（已有）
- Produces: `panel.draw(st, hint, n_ev, live, shadow_line="")`、`panel.run_live(st, path, level=None, shadow_log=None)`、`panel.run_replay(st, capture, delay, level=None, shadow_log=None)`、`table.TableWindow.draw(self, st, hint, n_ev, shadow_line="")`、`table.run_live(st, events_path=EVENTS, seconds=0, level=None, shadow_log=None)`、`table.run_replay(st, capture, delay_ms=260, seconds=0, level=None, shadow_log=None)`；两个 `main()` 都支持 `--no-advice` 与 `--level N`

- [ ] **Step 1: 写接线测试**

新建 `tests/test_panel_wiring.py`：

```python
"""面板接线：事件进来 -> 状态机 -> 影子记录器。**不开窗口**（只测接线本身）。"""
import json

from net import panel, shadow
from net.sim.meld import cid_from_name as A
from net.state import GameState
from train.net import QNet


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
```

- [ ] **Step 2: 跑它**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_panel_wiring.py -q
```
Expected: **应当直接过** —— 它证明「事件 → 状态机 → 记录器」这条接线不用改 `apply_event` 就通。红了说明 Task 1/3 的接口对不上，回去修那两处，**别改这个测试**。

- [ ] **Step 3: 改 `net/panel.py`**

`draw` 加一行（`shadow_line` 默认空串，保持向后兼容）：

```python
def draw(st: GameState, hint: str, n_ev: int, live: bool,
         shadow_line: str = "") -> None:
    mode = "实时" if live else "回放"
    print(CLEAR, end="")
    print(f"{DIM}[{mode}] 已收 {n_ev} 个事件   {hint}{RESET}")
    if shadow_line:
        print(f"{DIM}{shadow_line}{RESET}")
    print(st.render())
```

`run_live` 加 `level` / `shadow_log` 参数，事件处理后挂钩，`finally` 里收尾：

```python
def run_live(st: GameState, path: str, level: int = None, shadow_log=None) -> None:
    tail = Tailer(path)
    n, hint = 0, "等游戏服数据…"
    if level:
        st.level = level
    line = shadow_log.last_line if shadow_log else ""
    draw(st, hint, n, True, line)
    try:
        while True:
            events = tail.read()
            if events:
                for ev in events:
                    h = apply_event(st, ev)
                    if h:
                        hint = h
                    n += 1
                    if shadow_log is not None:
                        shadow_log.after_event(st, ev)
                draw(st, hint, n, True, shadow_log.last_line if shadow_log else "")
            else:
                time.sleep(0.05)
    finally:
        if shadow_log is not None:
            shadow_log.close()
```

`run_replay` 同样加两个参数（它本来就是逐帧喂，把 `shadow_log.after_event` 放在 `apply_event` 旁边即可），`main()` 加参数并建好记录器：

```python
    ap.add_argument("--level", type=int, default=None,
                    help="回放/离线时直接给级别（网络里没有本局级别）")
    ap.add_argument("--no-advice", action="store_true", help="关掉影子模式")
    ...
    sh = None
    if not args.no_advice:
        from net import advise, shadow
        net, err = advise.load_net()
        sh = shadow.ShadowLog(net=net, weights=advise.newest_weights() or "",
                              weights_note=err)
        print(sh.last_line)
    try:
        if args.replay:
            run_replay(st, args.capture, args.delay, level=args.level, shadow_log=sh)
        else:
            run_live(st, args.events, level=args.level, shadow_log=sh)
    except KeyboardInterrupt:
        print("\n面板已退出。")
```

- [ ] **Step 4: 改 `net/table.py`**

`TableWindow.draw` 加 `shadow_line` 参数，并在底部提示行**上方**加一行（字色 `DIM`，与 `hint` 的 `H-24` 不冲突）：

```python
    def draw(self, st, hint, n_ev, shadow_line=""):
        ...
        if shadow_line:
            cv.create_text(W / 2, H - 66, text=shadow_line,
                           font=self.f_small, fill=DIM)
```

`run_live` 加 `level` / `shadow_log`，级别更新时通知记录器：

```python
def run_live(st, events_path=EVENTS, seconds=0, level=None, shadow_log=None):
    import tkinter as tk
    from .panel import Tailer, apply_event
    root = tk.Tk()
    win = TableWindow(root)
    tail, lvl = Tailer(events_path), LevelWatcher()
    box = {"n": 0, "hint": "等游戏数据…（打开掼蛋打一局）"}
    if level:
        st.level = level

    def tick():
        for ev in tail.read():
            h = apply_event(st, ev)
            if h:
                box["hint"] = h
            box["n"] += 1
            if shadow_log is not None:
                shadow_log.after_event(st, ev)
        lv = lvl.poll()
        if lv is not None:
            st.level = lv
            box["hint"] = f"级别更新：打{st.level_name()}"
            if shadow_log is not None:
                shadow_log.note_level()
        win.draw(st, box["hint"], box["n"],
                 shadow_log.last_line if shadow_log else "")
        root.after(150, tick)
    ...
    try:
        tick()
        root.mainloop()
    finally:
        if shadow_log is not None:
            shadow_log.close()
```

`run_replay` 同样加参数，并把事件分发给**合并到 `panel.apply_event`**（现在这里有一份自己的 `st.on_play(...)` 分发，是第二份真源）：

```python
def run_replay(st, capture, delay_ms=260, seconds=0, level=None, shadow_log=None):
    import tkinter as tk
    from .panel import apply_event
    root = tk.Tk()
    win = TableWindow(root)
    it = iter(_replay_frames(capture))
    box = {"n": 0, "hint": "回放中…"}
    if level:
        st.level = level

    def apply(msg):
        ev = None
        if msg["msgid"] == 3005:
            p = protocol.decode_play(msg["fields"])
            if p:
                ev = {"type": "play", "seat": p["seat"], "cards": p["cards"],
                      "card_type": p["card_type"], "next": p["next"],
                      "left": p["left"], "left_cards": p.get("left_cards")}
        elif msg["msgid"] == 3019:
            h = protocol.decode_hand(m["fields"])
            ...
```

⚠️ 上面这段是示意，**实现时按现有代码逐行改**：`decode_hand` 的变量名是 `m` 还是 `msg` 要跟原文件对上；要点只有两条 ——（a）每个分支只**构造** `ev` 字典，（b）统一走 `apply_event(st, ev)` + `shadow_log.after_event(st, ev)`。`run_replay` 的 `finally` 里也要 `shadow_log.close()`。

`main()` 加 `--level` / `--no-advice`，并在开窗前加载权重（打印一行状态，加载要几秒，别让人以为卡死）。

`.gitignore` 在「运行时产物」那一段加一行：

```
net/shadow.jsonl
```

- [ ] **Step 5: 跑离线冒烟（真的开窗口，但 12 秒自动关）**

```powershell
.venv/Scripts/python.exe -m net.table --replay --level 9 --seconds 12
```
Expected: 窗口开起来、放完回放、12 秒后自动退出，退出码 0。然后：

```powershell
.venv/Scripts/python.exe -c "
import json, collections
rs = [json.loads(l) for l in open('net/shadow.jsonl', encoding='utf-8') if l.strip()]
print('行数', len(rs), collections.Counter(r['type'] for r in rs))
print('跳过原因', [r['skips'] for r in rs if r['type'] == 'deal_end'][-3:])
"
```
Expected: 有若干 `decision` 行。**是 0 的话看 `deal_end` 的 `skips` 是什么原因** —— 如果全是「座位未确认」，那是回放素材里「我」从没出过牌（素材问题，记进台账）；如果是「级别未知或越界」，检查 `--level` 有没有传到。

- [ ] **Step 6: 提交**

```bash
git add net/panel.py net/table.py tests/test_panel_wiring.py .gitignore
git commit -m "feat(panel): 两个面板都挂上影子记录器（只显示进度，不上屏建议）

- run_live/run_replay 加 level 与 shadow_log；回放也能离线产出 shadow.jsonl
- table.run_replay 的事件分发合并到 panel.apply_event（原来有两份）
- shadow.jsonl 加进 .gitignore"
```

---

### Task 5: `tools/accept_shadow.py` —— 用真机素材做离线验收

**Files:**
- Modify: `tools/accept_sim.py`（`replay()` 加 `record` 回调）
- Create: `tools/accept_shadow.py`
- Test: `tests/test_accept_shadow.py`

**Interfaces:**
- Consumes: `accept_sim.replay(g, record=None)`、`game_log.load_games()`、`decision_points.initial_hands`、`addon.GuandanTap().handle(body)`、`panel.apply_event`、`shadow.ShadowLog`、`advise.advise/build/load_net/newest_weights`
- Produces: `python -m tools.accept_shadow`（退出码 0/1）、`accept_shadow.run(capture=None, log_dir=None) -> list[Result]`、`accept_shadow.LOG_DIR`

- [ ] **Step 1: 给 `accept_sim.replay` 加回调（向后兼容）**

```python
def replay(g, record=None) -> ReplayResult:
    """（原有 docstring 不动）

    `record(kind, hand)` 每**将要**走一步时调一次（`kind` 是 `"pass"` 或 `"play"`），
    传进来的是**动手之前**的 `hand` —— 影子模式的验收要拿它在每个决策点取真值
    （手牌、桌面、谁要不起、各家剩几张）。**回放循环只有这一份**（本仓库为
    「副本会漂」吃过亏），所以真值侧复用它、不另写一份。
    """
```
在 `hand.pass_turn(hand.turn)` 之前插 `if record is not None: record("pass", hand)`，在 `hand.play(rec.seat, m)` 之前插 `if record is not None: record("play", hand)`。

- [ ] **Step 2: 跑现有验收，确认没改坏**

```powershell
.venv/Scripts/python.exe -m tools.accept_sim        # 退出码 0
.venv/Scripts/python.exe -m pytest tests/test_accept_sim.py -q
```
Expected: 全过（回调默认 None，行为不变）。

- [ ] **Step 3: 写验收脚本**

`tools/accept_shadow.py` 的骨架如下。**真值侧与网络侧都用生产代码**（`accept_sim.replay` / `GuandanTap.handle` / `panel.apply_event`），脚本自己只负责对齐与比对：

```python
"""第三层验收：影子模式（spec §8.2）—— 拿**真机素材**离线验整条推理链。

素材：`net/raw.jsonl`（全量载荷抓包，2026-09-25 11:11:13~11:18:55，两局）
真值：同一时段的游戏日志（两局都有结算）—— 能一手不落地重建出明牌对局。

与另两个验收的分工：`accept_meld` 验牌型引擎，`accept_sim` 验牌局引擎；
**本脚本验「网络那条路能不能喂出和模拟器一模一样的输入」，以及建议本身。**

六项：
  ① 抓包事件流 -> 状态机的动作流水，与真值 `rules.Hand.steps` **逐手一致**
     （含推断出来的「我过了」；座位要对上，见下面的映射）
  ② 每个影子决策点的状态编码（700 维）与历史（15x147）与真值**逐位相等**
  ③ 真实着法必须在候选里（`accept_meld` 验收①的口径）
  ④ 同一个局面、两条独立重建（网络 / 日志）给出的**建议下标相同**
  ⑤ 每局的 `me_team_won` 与日志 `Rank` 一致
  ⑥ 每个决策点的 `level` 与日志里覆盖该时刻那一局的 `Trump` 一致（不等 = 级别是上一局的）

跑法：
    .venv/Scripts/python.exe -m tools.accept_shadow
"""
```

关键实现（每条都照做）：

```python
from net import addon, advise, cards as cardmod, panel, protocol, shadow
from net.sim import env, meld, rules
from net.state import GameState
from tools.accept_meld import Result, _utf8_stdout
from tools import accept_sim, decision_points, game_log as gl

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = gl.LOG_DIR
DEFAULT_CAPTURE = os.path.join(PROJ, "net", "raw.jsonl")
_MIN_DECISIONS = 30          # 地板：验到的决策点少于这个数，「全过」不足以称为结论
_MIN_DEALS = 2


def load_frames(path):
    """抓包 -> [(t, body)]（只要游戏服的服务器下发方向）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("k") == "frame" and r.get("dir") == "S→C":
            out.append((r["t"], bytes.fromhex(r["hex"])))
    out.sort(key=lambda x: x[0])
    return out


def wire_side(frames, net, out_path):
    """网络那条路：走**生产那份**解码与分发（只少了 mitmproxy）。

    `addon.GuandanTap` 只用它的 `handle()`（纯函数：帧 -> 事件字典），
    `_emit` 不碰，所以不会顺手建文件 —— 落盘只由 `ShadowLog` 干。
    """
    st = GameState()
    log = shadow.ShadowLog(net=net, out_path=out_path, weights="")
    tap = addon.GuandanTap()
    for t, body in frames:
        ev = tap.handle(body)
        if ev is None:
            continue
        panel.apply_event(st, ev)
        log.after_event(st, ev)
    log.close()
    return st, log, tap.stats


def truth_side(g):
    """真值那条路：日志 -> 明牌 Hand，逐点快照。

    返回 `points = [(座位, 动手前的手牌, 桌面 Meld, passed, 各家剩几张)]`
    与 `ann.hand.steps`（真值流水）。
    """
    points = []

    def record(kind, hand):
        points.append({"seat": hand.turn,
                       "hand": frozenset(hand.hands[hand.turn]),
                       "table": hand.table, "passed": tuple(sorted(hand.passed)),
                       "left": tuple(len(hand.hands[s]) for s in rules.SEATS)})

    ann = accept_sim.replay(g, record=record)
    return points, ann.hand.steps
```

比对要点（写进脚本注释）：

- **座位映射**：日志的座位号与报文的座位号不是一套。映射由**我的座位**定 ——
  日志侧我的座位 = `initial_hands(g)` 里与发牌 27 张重合最多的那个（进贡会换掉 1~2 张，
  所以判据是「重合 ≥ 25」而不是全等，达不到就报 FAIL：语料不足）；
  报文侧我的座位 = `st.me`。定完用 80 手出牌的牌面**逐手核对**（这正好是验收 ①）。
- **①**：按映射把两边座位对齐后，逐手比 `(座位, 牌的集合 or None)`。**长度也必须一样** ——
  长度差就是「推断补的步」错了。
- **②**：`env.encode_state` 内部按出牌人相对化，所以两套编号只要差一个**旋转**就等价
  （两边的出牌顺序都遵守 `0→3→2→1`）。因此可以直接逐位比，不需要知道映射细节 ——
  **但必须先过 ①**（否则可能是「错错相消」）。
  真值侧的 Observation 用 `env.observe(hand, 我的日志座位, played, hand.table)` 造，
  `played` 由脚本按真值维护；历史用 `env.encode_history(hand, 我的日志座位)`。
- **④**：两个 Observation、两份候选（都按 `advise.candidates` 的口径）、两次 `q_values`
  （**同一次进程、同一台设备**），比较 `argmax` 下标是否相同 —— 候选顺序两边都由
  `meld.legal_moves` 决定，所以下标可直接比。
- **⑥**：日志里每局的 `(t0, trump)` 是现成的；把每个决策点的 `t` 落到某一局的区间里就能比。
  对不上的条数**报出来**（这是「级别是上一局的」那个已知风险的量化），
  但**不算失败** —— 只要它不是压倒性多数；写成 `res.note` 附在报告里。

- [ ] **Step 4: 跑验收**

```powershell
.venv/Scripts/python.exe -m tools.accept_shadow
```
Expected: 六项 `[OK]`，退出码 0。**第一次跑几乎肯定会红一两项** —— 那正是这次工作的主要内容。按 ①②③④⑤⑥ 的顺序修（②红了先确认①是绿的）。

- [ ] **Step 5: 加一个「验收脚本自己不许假绿」的测试**

新建 `tests/test_accept_shadow.py`：

```python
"""验收脚本自己也要有地板 —— `total == 0` 必须算失败（假绿）。"""
from tools import accept_shadow


def test_missing_capture_reports_failure_not_silence(tmp_path):
    res = accept_shadow.run(capture=str(tmp_path / "nope.jsonl"))
    assert res, "找不到素材也必须给出结果，不能返回空"
    assert not any(r.ok for r in res), "缺素材时不许有任何一项报 OK"
```

（`run()` 找不到素材时要**报 FAIL**，不是抛异常、也不是静默返回空 —— 与 `game_log.load_games` 的「日志被轮转删了」同一条纪律。）

- [ ] **Step 6: 全量回归 + 提交**

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m tools.accept_meld
.venv/Scripts/python.exe -m tools.accept_sim
.venv/Scripts/python.exe -m tools.accept_tribute
.venv/Scripts/python.exe -m tools.accept_shadow
```

```bash
git add tools/accept_sim.py tools/accept_shadow.py tests/test_accept_shadow.py
git commit -m "feat(accept): 影子模式离线验收（抓包 + 日志双源逐位对齐）

- accept_sim.replay 加 record 回调，真值侧复用同一个回放循环
- 六项：动作流水逐手一致 / 700 维向量逐位相等 / 真实着法在候选里 /
  两源同一建议 / 局末输赢与日志 Rank 一致 / 级别与日志 Trump 一致
- 地板：决策点 < 30 或局数 < 2 报 FLOOR，不许假绿"
```

---

### Task 6: 文档 —— 怎么跑、怎么读、哪些地方还不准

**Files:**
- Modify: `HANDOFF.md`
- Create: `docs/superpowers/plans/2026-09-25-shadow-delivery.md`

- [ ] **Step 1: 写交付台账**

新建 `docs/superpowers/plans/2026-09-25-shadow-delivery.md`，照 Plan 1/2/3 台账的体例（一、一句话结论；二、怎么跑；三、这一增量做成了什么；四、实测与更正；五、没做的），至少包含：

- **三条实测更正**（写清「原来怎么写、实际是什么」，都有证据）：
  1. 「面板从 msgid 3008 拿级别」**不成立** —— 3008 的 `3.9.22.12` 是**两队各自的级别**（`[10,9,10,9]`），而那一局的 Trump 是 9（= 我那一队的级别，因为上一局我们输了）。级别仍走日志（`LevelWatcher`）。
  2. **座位每局都会变，而且是同一段抓包里就变**（11:11 那局 seat1 → 11:15 那局 seat2；用日志发牌的 27 张 × `LeftCardList` 子集关系证实为 9/9 与 7/7）；换局后到我第一次出牌之间，状态机的座位是**陈旧**的 —— 影子模式这段一律不算。
  3. **我自己「要不起」在报文里是静默的**（服务器只通知别人），所以动作流水里我那一行是**推断**出来的；已出完的座位没有这一步。
- **影子日志怎么读**：贴一行真实的 `decision` 与一行 `deal_end` 逐字段解释；给两条现成命令（统计「分歧率」「`actual_rank == -1` 的条数」「跳过原因分布」）。
- **已知限制**（照实写）：级别在换局后 ~20 秒可能是上一局的（记录里带 `level_age_s`，验收⑥的能量化）；进贡/还贡期间状态机不准（那几手别计入分母）；面板启动那一刻如果在一局中间，「本局」编号是 0。

- [ ] **Step 2: 更新 `HANDOFF.md`**

- 「二、当前状态」表加一行：`影子模式（spec §8.2）| ✅ 已做成，等实机积累 | net/shadow.py + net/advise.py + tools/accept_shadow.py`。
- 「三、下一步」第 1 条从「下一个会话做影子模式」改成「**下一个会话：看几十局的影子日志**」，指向新台账。
- 「二之二」的关键设计表补三行：**动作流水**（我自己要不起是推断的）、**座位每局都变**（换局后到首次出牌之间不算）、**级别只从日志来**（3008 是两队级别，不是本局级别）。
- 代码地图加 `net/advise.py`、`net/shadow.py`、`tools/accept_shadow.py`。
- 提交数那句**不许写死数字**（用 `git log --oneline <上一个台账的锚点>..HEAD`）。

- [ ] **Step 3: 提交**

```bash
git add HANDOFF.md docs/superpowers/plans/2026-09-25-shadow-delivery.md
git commit -m "docs: 影子模式交付台账 + HANDOFF 指向「看几十局日志」"
```

---

## 验收清单（全跑一遍再收工）

```powershell
cd C:\Users\17837\PycharmProjects\yolo

# 单元测试（原 308 + 本次新增）
.venv/Scripts/python.exe -m pytest tests/ -q

# 三个老验收不许退步
.venv/Scripts/python.exe -m tools.accept_meld        # ①1706 ②530 ③4 ④5 全过
.venv/Scripts/python.exe -m tools.accept_sim         # 53 局回放 -> 退出码 0
.venv/Scripts/python.exe -m tools.accept_tribute     # 还贡 ≤10 25/25 -> 退出码 0

# 本次的新验收（真机素材，两局）
.venv/Scripts/python.exe -m tools.accept_shadow      # 六项 [OK] -> 退出码 0

# 离线冒烟：面板真的能跑、回放能产出影子日志
.venv/Scripts/python.exe -m net.table --replay --level 9 --seconds 12
```

然后**实机跑一次**（这一步只有用户能做，别自己动他的机器）：

```powershell
python -m net.launcher
```
打开掼蛋打一到两局，**Ctrl-C 退出**（会自动卸证书、还原代理）。退出后：

```powershell
.venv/Scripts/python.exe -c "
import json, collections
rs = [json.loads(l) for l in open('net/shadow.jsonl', encoding='utf-8') if l.strip()]
print('总行数', len(rs), collections.Counter(r['type'] for r in rs))
print('跳过原因合计', {k: v for r in rs if r['type'] == 'deal_end' for k, v in r['skips'].items()})
d = [r for r in rs if r['type'] == 'decision' and r.get('resolved')]
print('实际着法不在候选里的条数', sum(1 for r in d if r.get('actual_rank') == -1))
print('与模型头名一致的比例', f\"{sum(1 for r in d if r.get('actual_rank') == 0)}/{len(d)}\" if d else '（没有决策点）')
"
```

**判据**：`decision` 行数 = 轮到我出牌的次数减去跳过数（跳过原因必须可解释，不能「一片全是座位未确认」）；`actual_rank == -1` 应当**接近 0**（不接近 0 说明状态重建有错，回任务 5 查）；`actual_rank == 0` 的比例 = 「模型与人不谋而合」的比例，这是下一个会话要看的数。

---

## 收尾时的自检（写完计划后我自己过一遍）

**规格覆盖**：spec §8.1 推理链 → 任务 2；§8.2 影子模式（决策点、实际着法、那局赢没赢、`net/shadow.jsonl`、与 `events.jsonl` 同风格、可离线复盘）→ 任务 3；§8.4 红线 → 全局约束 + 任务 3 的 docstring；Plan 3 台账「五」的四条咬人点 → 任务 1（历史流水 / 座位相对化）、任务 2（级别）、任务 3（绝不代打）；「两笔要记在前的账」（进贡在训练里关着、级别是均匀采样的）→ 任务 6 的台账。

**没做的（明确在计划外）**：反馈回收（spec §8.3，影子模式之后再谈）、vs 人类着法那条统计（下一个会话拿影子日志顺带算）、msgid 3008 接进面板（这次已证明它不是本局级别）、进贡/还贡字段定位、图形面板上的「复盘视图」。
