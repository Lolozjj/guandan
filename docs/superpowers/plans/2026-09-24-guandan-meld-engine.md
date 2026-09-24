# 掼蛋牌型裁判（Plan 1/4）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做出掼蛋牌型真源 `net/sim/meld.py`（合法着法枚举 + 大小比较 + 逢人配），并用 55 局真实对局离线验证它是对的。

**Architecture:** 先把游戏日志解析成结构化对局（含每家手牌与每个决策点的桌面），再用这些真实局面验收牌型引擎。牌型引擎是纯函数、只依赖 `net/cards.py` 的牌 ID 编码，不依赖任何牌局状态。它是后面模拟器与 RL 的共同地基 —— 模型只会从它给的候选里挑，所以它错了整个系统就错了。

**Tech Stack:** Python 3.12.10（`.venv`），pytest，标准库。本阶段**不需要 torch**。

**Spec:** `docs/superpowers/specs/2026-09-24-guandan-rl-advisor-design.md`（本计划是它的 §3/§6 落地，实施时两份一起读）

## Global Constraints

- **Python 一律用 `.venv/Scripts/python.exe`**，不要用系统 python。
- **牌型真源只有一个：`net/sim/meld.py`。** 它按**牌 ID**（int）工作，**该文件里不允许出现牌名字符串**（如 `"S3"`、`"5♥"`）。`live/rules.py` 的 6 个 bug 有一半来自字符串处理。
- **不复制 `live/rules.py`**（它有 6 处已证实的错，见 spec §2.3）。它只作为「牌名 → 牌 ID」的适配层保留。
- **级别口径与游戏日志一致**：`A=1`、`2..10`、`J=11`、`Q=12`、`K=13`；小王 `14`、大王 `15`。这与 `net/cards.parts()` 的 idx 完全一致。
- **炸弹顺序（用户口述 + 数据 0 矛盾）**：`4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 天王炸`。
- **牌型表**：单张/对子/三张/顺子(5)/三带二(5)/三连对(6)/钢板(6)/炸弹(4~8)/同花顺(5)/天王炸。**没有三带一**（4 张只有炸弹）。
- **连对 = 恰好 3 对**；**钢板 = 恰好 2 个连续三张**（数据里只见过这两个长度，更长的一律不产生）。
- **遇认不出的局面必须明着 `raise`**，不许静默返回「合法」或「过」。宁可崩掉，也不要拿错规则偷偷训出一版废模型。
- **数据路径**：游戏日志 `C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users\67b5ef56e08ab757e0cd7cac86e2366d\applet\local\wx2f60a7b40f3828a9\usr\HappySDKLogFiles`（**只保留 2 天**）；抓包事件 `net/events.jsonl`。
- **提交信息用中文**，结尾加 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。

## Review Focus

这五类输入 spec 隐含要求必须处理、但没有哪个任务的测试天然覆盖 —— 每一类都在下面某个任务里配了测试：

1. **日志被轮转删除**（只留 2 天）：解析器遇到缺文件、半截文件、字段缺失，必须**明着报错**，绝不能当成「0 局」静默通过 —— 那会让验收变成空跑还报绿。
2. **逢人配不是 2 张时**：一局里可能只摸到 1 张，或两张都在别人手上。枚举**不能假设总有 2 张**。
3. **两副牌的同名牌**（如同两张 `5♦`，ID 不同）：分组与去重不能把它们当成一张。
4. **领出局面**（桌面为空）：`legal_moves` 必须返回**非空**，且**不含「过」**。
5. **级牌正好是 2 或 A 时的边界算术**：`point_value` 的级牌提升与顺子的 A 可高可低，都会在这里出错。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `tools/game_log.py` | 解析游戏日志 → 结构化对局（发牌/出牌/结算） |
| `tools/decision_points.py` | 从对局重建每个决策点：谁、手上什么、桌上什么、实际出了什么 |
| `net/sim/__init__.py` | 包标记 |
| `net/sim/meld.py` | **牌型真源**：`Meld` / `melds_from` / `legal_moves` / `beats` |
| `tools/accept_meld.py` | 第一层验收脚本（spec §6 ①~④） |
| `tests/test_game_log.py` | 日志解析与守恒 |
| `tests/test_decision_points.py` | 决策点重建 |
| `tests/test_meld_basic.py` | 基础牌型 + 炸弹层级 |
| `tests/test_meld_seq.py` | 顺子类 |
| `tests/test_meld_wild.py` | 逢人配 |
| `tests/test_accept.py` | 验收①（真实着法必在合法集合内） |
| `tests/test_rules_adapter.py` | 旧接口适配层 |

`live/rules.py` 在 Task 8 改成适配层，**不动** `live/main.py` 的调用方式。

---

## Task 1: 环境骨架

**Files:**
- Create: `pytest.ini`
- Create: `net/sim/__init__.py`
- Create: `tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `net/sim/meld.py`（暂时只有 docstring）

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 `pytest`；`net.sim` 可 import

- [ ] **Step 1: 装 pytest**

```bash
.venv/Scripts/python.exe -m pip install pytest
```

- [ ] **Step 2: 让包能被 import**

`pytest.ini`：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

`net/sim/__init__.py`、`tools/__init__.py`、`tests/__init__.py`：三个空文件。

- [ ] **Step 3: 写冒烟测试**

`tests/test_smoke.py`：

```python
def test_package_importable():
    from net import cards
    from net.sim import meld  # noqa: F401
    assert cards.decode(77) == "K♦"
```

`net/sim/meld.py` 暂时只有一行 docstring：

```python
"""掼蛋牌型真源。见 docs/superpowers/plans/2026-09-24-guandan-meld-engine.md。"""
```

- [ ] **Step 4: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `1 passed`

- [ ] **Step 5: 提交**

```bash
git add pytest.ini net/sim tools tests
git commit -m "chore: 建 pytest 骨架与 net/sim 包"
```

---

## Task 2: 日志解析器

**Files:**
- Create: `tools/game_log.py`
- Test: `tests/test_game_log.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `LOG_DIR: str`
  - `@dataclass PlayRec: seat:int, cards:list[int], card_type:int, left:int, nxt:int`
  - `@dataclass GameLog: t0:datetime, trump:int, my_cards:list[int], plays:list[PlayRec], settle:Optional[dict]`
  - `load_games(log_dir:str=LOG_DIR) -> list[GameLog]`
  - `conserved(g:GameLog) -> bool`

**背景**（实施者需要知道的事实，别重新推）：

- 发牌行：`SendCardsService set roundID:... k : {json}`，JSON 里 `Cards`（27 张）与 `Trump`（级别）。**只有本机玩家的 27 张**，不是四家的。
- 出牌行：`NotifyGiveCards 后台通知客户端出牌结果 info = {json}`，字段 `SeatID` `NextTurnSeatID` `CardType` `LeftCardLen` `CardLen` `CardList`。
- 结算行：`EVA1B001结算协议 = {json}`，字段 `Rank`（**`Rank[i]` = 座位 i 的名次**，i 为 0-based）、`UpgradeInfo`、`LeftCards`（每家剩的牌）。
- 时间戳格式：`2026-09-22|16:33:00:239|INFO|...`（毫秒用**冒号**分隔，不是点）。
- 实测：**63 个真发牌段**，其中 **55 局有结算**，且 **55/55 局「出过的牌 + 结算剩的牌 = 108」分毫不差**。
- ⚠️ **`SendCardsService set roundID` 前缀共命中 126 行，其中一半是不带 JSON 载荷的伴随行**
  （同一毫秒、恒定相隔 8 行，形如 `roundID:S7380R1T1669t6AB4E78ES0A`）。
  **必须显式跳过伴随行**，否则会对它抛 `RuntimeError`；跳过判定要放在「收上一局」之前，
  否则会把当前局截断成两局。

- [ ] **Step 1: 写失败的测试**

`tests/test_game_log.py`：

```python
import os
import pytest
from tools.game_log import LOG_DIR, load_games, conserved

pytestmark = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                reason="本机没有游戏日志")


def test_loads_games():
    games = load_games()
    assert len(games) > 0, "日志目录在，却一局都没解出来 —— 不要静默通过"
    for g in games[:5]:
        assert len(g.my_cards) == 27
        # 日志偶尔用 14 表示 A（net/cards.py 里也踩过这条），所以上限放到 14
        assert 1 <= g.trump <= 14


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
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_game_log.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.game_log'`

- [ ] **Step 3: 实现**

`tools/game_log.py`：

```python
"""把游戏日志解析成结构化对局。

日志只保留 2 天，且按小时切文件，所以解析要按文件名排序、跨文件连续扫。
**解析不出来必须报错，不能返回空列表** —— 否则验收会空跑还报绿
（本项目吃过一次「静默出错」的亏，见 HANDOFF 验收协议）。
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

LOG_DIR = (r"C:\Users\17837\AppData\Roaming\Tencent\xwechat\radium\users"
           r"\67b5ef56e08ab757e0cd7cac86e2366d\applet\local"
           r"\wx2f60a7b40f3828a9\usr\HappySDKLogFiles")

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(\d{2}:\d{2}:\d{2}):(\d{3})")
_DEAL = re.compile(r"SendCardsService set roundID")
_DEAL_JSON = re.compile(r"k : (\{.*)")
_PLAY = re.compile(r"NotifyGiveCards 后台通知客户端出牌结果 info = (\{.*)")
_SETTLE = re.compile(r"EVA1B001结算协议 = (\{.*)")


@dataclass
class PlayRec:
    seat: int
    cards: list
    card_type: int
    left: int        # 这一手之后该家还剩几张
    nxt: int         # 服务器给的下一手座位；-1 表示本局结束


@dataclass
class GameLog:
    t0: datetime
    trump: int                 # 级别（A=1, 2..10, J=11, Q=12, K=13）
    my_cards: list             # 发牌给我自己的 27 张
    plays: list = field(default_factory=list)
    settle: Optional[dict] = None


def _ts(line):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}.{m.group(3)}",
                             "%Y-%m-%d %H:%M:%S.%f")


def _json(pattern, line):
    m = pattern.search(line)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def load_games(log_dir: str = LOG_DIR) -> list:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(
            f"日志目录不存在：{log_dir}\n"
            f"日志只保留 2 天，可能被轮转删了 —— 别把它当成「0 局」继续跑。")

    files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not files:
        raise RuntimeError(f"目录在但一个 .log 都没有：{log_dir}")

    games, cur = [], None
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _DEAL.search(line):
                    if cur is not None:
                        games.append(cur)      # 上一局没有结算，也收着
                    d = _json(_DEAL_JSON, line)
                    if not d:
                        raise RuntimeError(f"发牌行解不出 JSON：{line[:200]}")
                    cur = GameLog(t0=_ts(line) or datetime.min,
                                  trump=int(d["Trump"]),
                                  my_cards=list(d["Cards"]))
                    continue
                if cur is None:
                    continue
                if _PLAY.search(line):
                    d = _json(_PLAY, line)
                    if d and d.get("CardList"):
                        cur.plays.append(PlayRec(
                            seat=int(d["SeatID"]),
                            cards=list(d["CardList"]),
                            card_type=int(d.get("CardType", 0)),
                            left=int(d.get("LeftCardLen", 0)),
                            nxt=int(d.get("NextTurnSeatID", -1))))
                    continue
                if _SETTLE.search(line):
                    d = _json(_SETTLE, line)
                    if d:
                        cur.settle = d
                        games.append(cur)
                        cur = None
    if cur is not None:
        games.append(cur)
    return games


def conserved(g: GameLog) -> bool:
    """出过的牌 + 结算剩的牌 == 108（两副牌）。"""
    if not g.settle:
        return False
    played = sum(len(p.cards) for p in g.plays)
    left = sum(len(e.get("Cards") or []) for e in (g.settle.get("LeftCards") or []))
    return played + left == 108
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_game_log.py -q`
Expected: `3 passed`

若出现 skip（本机无日志），**停下来问用户**，不要继续往下做。

- [ ] **Step 5: 提交**

```bash
git add tools/game_log.py tests/test_game_log.py
git commit -m "feat: 游戏日志解析器（含 108 张守恒校验）"
```

---

## Task 3: 决策点重建

**Files:**
- Create: `tools/decision_points.py`
- Test: `tests/test_decision_points.py`

**Interfaces:**
- Consumes: `tools.game_log.GameLog` / `PlayRec`
- Produces:
  - `@dataclass Snapshot: seat:int, hand:list[int], table:Optional[list[int]], actual:list[int], level:int, t:datetime`
  - `initial_hands(g:GameLog) -> dict[int,set[int]]`
  - `decision_points(g:GameLog) -> list[Snapshot]`

**原理**（关键，别换算法）：

每家起手牌 = **他出过的所有牌 ∪ 他结算时剩的牌**。进贡/还贡发生在任何出牌之前，所以这样重建出来的正好是「进贡之后、第一手之前」的手牌，之后只减不增 —— 因此对局中每个时刻的手牌 = 该重建集合减去他此刻已出过的牌。

桌面的判定**不看牌型**（否则就循环依赖了，比较函数 Task 4 才实现）。
「新领出」只有两种情况：

1. **同一座位又出牌了** —— 其余三家都要不起，他重新领出；
2. **队友接风** —— 桌面主人上一手把牌打完了（`left == 0`），队友接着领出。

⚠️ **不能用服务器的 `NextTurnSeatID`（`PlayRec.nxt`）判领出 —— 已用 55 局实测证伪：**

- `nxt == 桌面主人` 全量 **0 次**（`nxt` 是「我这一手之后轮到谁」，不会绕回自己；
  其余三家要不起走的是 3006 报文，不会更新这一手记录里的 `nxt`）
- `nxt == 队友` 命中 44 次，但**成因是服务器算下一手时跳过已出完的座位**，不是接风
- 按 `nxt` 判的后果：全 55 局出现 **27 手「轮内不同型且非炸弹」的非法响应**（该清的桌没清）

改成「同座位 或（队友 且 主人已出完）」后，同一口径下非法响应 **0 手**。
反方向也验过：队友紧接着出牌的 114 手里，**主人没出完的 63 手中 0 手需要清桌**
（56 手与桌面同型、本就是合法响应），**主人已出完的 51 手中 33 手必须清桌** ——
所以「主人已出完」这个附加条件不是可选项，是必需的。

本局第一手时桌面为空。

- [ ] **Step 1: 写失败的测试**

`tests/test_decision_points.py`：

```python
import os
import pytest
from tools.game_log import LOG_DIR, load_games
from tools.decision_points import decision_points

pytestmark = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                reason="本机没有游戏日志")


def _settled():
    return [g for g in load_games() if g.settle]


def test_last_snapshot_hand_matches_settlement():
    """每个座位最后一次出牌后剩下的牌，必须等于结算里的 LeftCards。

    这是重建正确性最直接的证据 —— 手牌少算或多算都会在这里露出来。
    """
    for g in _settled()[:20]:
        snaps = decision_points(g)
        assert snaps, f"{g.t0} 一个决策点都没有"
        left = {i: set(e.get("Cards") or [])
                for i, e in enumerate(g.settle["LeftCards"])}
        for seat in range(4):
            seat_snaps = [s for s in snaps if s.seat == seat]
            if not seat_snaps:
                continue
            remain = set(seat_snaps[-1].hand) - set(seat_snaps[-1].actual)
            assert remain == left[seat], (
                f"{g.t0} 座位{seat} 重建剩 {sorted(remain)} "
                f"!= 结算 {sorted(left[seat])}")


def test_hand_never_grows():
    """一局之内手牌只减不增。"""
    for g in _settled()[:20]:
        prev = {}
        for s in decision_points(g):
            if s.seat in prev:
                assert len(s.hand) <= prev[s.seat], \
                    f"{g.t0} 座位{s.seat} 手牌变多了"
            prev[s.seat] = len(s.hand)


def test_same_seat_playing_again_is_a_new_lead():
    """同一座位连续出两手 = 其余三家都要不起 = 他重新领出，桌面必须清空。

    漏了这条，会把「他重新领出」误当成「他压自己」，验收①就会报假红。
    """
    for g in _settled()[:20]:
        snaps = decision_points(g)
        prev = None
        for s in snaps:
            if prev is not None and s.seat == prev.seat:
                assert s.table is None,                     f"{g.t0} 座位{s.seat} 连续出牌，桌面却没清空"
            prev = s


def test_first_snapshot_is_a_lead():
    """本局第一手的桌面必须是空的。"""
    for g in _settled()[:10]:
        assert decision_points(g)[0].table is None


def test_actual_cards_are_in_hand():
    """真实出的牌必须在他当时的手里 —— 否则重建错了。"""
    for g in _settled()[:20]:
        for s in decision_points(g):
            missing = set(s.actual) - set(s.hand)
            assert not missing, \
                f"{g.t0} 座位{s.seat} 出了手上没有的牌 {sorted(missing)}"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_decision_points.py -q`
Expected: FAIL — `No module named 'tools.decision_points'`

- [ ] **Step 3: 实现**

`tools/decision_points.py`：

```python
"""从结构化对局重建每个决策点：谁、手上什么、桌上什么、实际出了什么。

手牌重建原理：每家起手牌 = 出过的 ∪ 结算剩的。进贡/还贡在第一手之前完成，
所以重建出来的就是「第一手之前」的手牌，之后只减不增。

桌面判定刻意**不看牌型**（那是 net/sim/meld.py 的事，这里用了就循环依赖）。
只看服务器给的 NextTurnSeatID：转回桌面主人（或他的队友接风）= 这一轮结束。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tools.game_log import GameLog


@dataclass
class Snapshot:
    seat: int
    hand: list                 # 该座位此刻手上的牌（含他即将打出的）
    table: Optional[list]      # 桌面待压的牌；None = 他领出
    actual: list               # 他实际打出的牌（真值）
    level: int
    t: datetime


def initial_hands(g: GameLog) -> dict:
    """每家起手牌 = 他出过的所有牌 ∪ 他结算时剩的牌。"""
    hands = {i: set() for i in range(4)}
    for p in g.plays:
        hands.setdefault(p.seat, set())
        hands[p.seat] |= set(p.cards)
    if g.settle:
        for i, e in enumerate(g.settle.get("LeftCards") or []):
            hands.setdefault(i, set())
            hands[i] |= set(e.get("Cards") or [])
    return hands


def decision_points(g: GameLog) -> list:
    remaining = {i: set(v) for i, v in initial_hands(g).items()}
    snaps = []
    table = None
    table_seat = None
    prev_left = None          # 上一手 PlayRec.left；判队友接风要用

    for p in g.plays:
        if table is not None:
            partner = (table_seat + 2) % 4
            # 新领出只有两种情况：
            #   1) 同一座位又出牌了 —— 其余三家都要不起，他重新领出
            #   2) 队友接风 —— 桌面主人上一手把牌打完了（left == 0），队友接着领出
            #
            # **不要用 prev_nxt 判**：服务器算 nxt 时会跳过已出完的座位，所以 nxt 指到
            # 队友既可能是接风、也可能只是跳过了一个出完的座位（那时队友其实在压牌）。
            # 实测按 nxt 判会有 27 手非法响应；按下面这个判法 0 手。
            if p.seat == table_seat or (p.seat == partner and prev_left == 0):
                table = None

        snaps.append(Snapshot(seat=p.seat,
                              hand=sorted(remaining.get(p.seat, set())),
                              table=list(table) if table else None,
                              actual=list(p.cards),
                              level=g.trump,
                              t=g.t0))

        remaining.setdefault(p.seat, set())
        missing = set(p.cards) - remaining[p.seat]
        if missing:
            raise RuntimeError(
                f"{g.t0} 座位{p.seat} 打出了手上没有的牌 {sorted(missing)} —— "
                f"重建错了，不要静默跳过")
        remaining[p.seat] -= set(p.cards)
        if p.left == 0:
            remaining[p.seat] = set()

        table = list(p.cards)
        table_seat = p.seat
        prev_left = p.left
    return snaps
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_decision_points.py -q`
Expected: `5 passed`

若 `test_last_snapshot_hand_matches_settlement` 失败：**不要改测试去迁就实现。**
那说明领出/桌面的判定有问题。先打印失败局的 `plays` 与 `LeftCards` 人工核对，
再改实现。

- [ ] **Step 5: 提交**

```bash
git add tools/decision_points.py tests/test_decision_points.py
git commit -m "feat: 从日志重建决策点（手牌 + 桌面 + 真值着法）"
```

---

## Task 4: 牌型模型与基础枚举 + 炸弹比较

**Files:**
- Modify: `net/sim/meld.py`（替换掉 Task 1 的占位 docstring）
- Test: `tests/test_meld_basic.py`

**Interfaces:**
- Consumes: `net.cards.parts(cid) -> (idx, suit, deck)`
- Produces:
  - 牌型常量：`SINGLE=1 PAIR=2 TRIPLE=3 STRAIGHT=4 TRIPLE_PAIR=5 PAIR_RUN=6 PLATE=7 BOMB=8 STRAIGHT_FLUSH=9 BOMB6=10`，`JOKER_SMALL=14 JOKER_BIG=15`
  - `point_value(idx:int, level:Optional[int]) -> int`
  - `is_wild(cid:int, level:Optional[int]) -> bool`
  - `@dataclass(frozen=True) Meld: kind:int, size:int, rank:int, cards:tuple, wild_used:int=0`，属性 `is_bomb`
  - `bomb_class(m:Meld) -> Optional[int]`
  - `beats(a:Meld, b:Meld) -> bool`
  - `melds_from(hand, level=None) -> list[Meld]`（本任务只做单/对/三/三带二/炸弹）
  - `legal_moves(hand, table:Optional[Meld], level=None) -> list[Meld]`

- [ ] **Step 1: 写失败的测试**

`tests/test_meld_basic.py`：

```python
from net.sim import meld
from net import cards

_TABLE = {}
for _cid in range(1, 334):
    if cards.is_card(_cid):
        _TABLE[cards.decode(_cid)] = _cid
_TABLE.update({"小王": 14, "大王": 15, "小王(二副)": 270, "大王(二副)": 271})


def C(*names):
    """牌面名 -> 牌 ID。只在本测试文件里用，meld.py 里不许有字符串。"""
    return [_TABLE[n] for n in names]


def test_point_value_level_card_beats_ace():
    assert meld.point_value(5, 5) > meld.point_value(1, 5)     # 打5，5 比 A 大
    assert meld.point_value(1, 5) > meld.point_value(13, 5)    # A 比 K 大


def test_norm_level_ace_written_as_14():
    """日志里 A 可能写成 14，必须归一成 1，否则会被当成小王。"""
    assert meld.norm_level(14) == 1
    assert meld.norm_level(5) == 5
    assert meld.point_value(1, 14) == meld.point_value(1, 1)
    assert meld.point_value(14, 14) == meld.point_value(14, 1)   # 小王不变


def test_point_value_level_two_and_ace():
    """边界：级牌正好是 2 或 A。"""
    assert meld.point_value(2, 2) > meld.point_value(1, 2)     # 打2，2 最大
    assert meld.point_value(1, 1) > meld.point_value(13, 1)    # 打A，A 最大
    assert meld.point_value(15, None) > meld.point_value(14, None)


def test_single_pair_triple():
    hand = C("5♦", "5♣", "5♠", "7♥")
    kinds = {}
    for m in meld.melds_from(hand, level=2):
        kinds.setdefault(m.kind, []).append(m.size)
    assert kinds[meld.SINGLE] == [1, 1]
    assert kinds[meld.PAIR] == [2]
    assert kinds[meld.TRIPLE] == [3]


def test_two_decks_are_two_cards():
    """两副牌的同名牌是两张，不能当一张。"""
    hand = C("5♦", "5♦(二副)")
    pairs = [m for m in meld.melds_from(hand, level=2) if m.kind == meld.PAIR]
    assert len(pairs) == 1
    assert len(pairs[0].cards) == 2 and len(set(pairs[0].cards)) == 2


def test_triple_pair_allows_joker_pair():
    """王可以当三带二里的对子。

    真实数据里有一手 card_type=5：2♦ 2♦(二副) 2♣ + 小王 小王(二副)。
    排除王的话这手枚举不出来，Task 7 的验收①会直接报红。
    """
    hand = C("2♦", "2♦(二副)", "2♣", "小王", "小王(二副)")
    tp = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.TRIPLE_PAIR]
    assert len(tp) == 1
    assert sorted(tp[0].cards) == sorted(hand)
    assert tp[0].rank == meld.point_value(2, 9)      # 主键是三张的点数，不是王的


def test_two_jokers_make_a_pair_but_not_a_triple():
    """两张王成对；但凑不出三张，也凑不出普通炸弹。"""
    pair = C("小王", "小王(二副)")
    kinds = {m.kind for m in meld.melds_from(pair, level=9)}
    assert kinds == {meld.SINGLE, meld.PAIR}

    mixed = C("小王", "大王")
    kinds = {m.kind for m in meld.melds_from(mixed, level=9)}
    assert kinds == {meld.SINGLE}                     # 小王+大王 不成对


def test_bomb_sizes():
    hand = C("5♦", "5♦(二副)", "5♣", "5♣(二副)", "5♠", "5♠(二副)", "5♥")
    sizes = sorted(m.size for m in meld.melds_from(hand, level=9)
                   if m.kind == meld.BOMB)
    assert sizes == [4, 5, 6, 7]


def test_triple_pair():
    hand = C("5♦", "5♣", "5♠", "3♥", "3♣")
    tp = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.TRIPLE_PAIR]
    assert len(tp) == 1 and tp[0].size == 5


def _mk(kind, size, rank, ids):
    return meld.Meld(kind=kind, size=size, rank=rank, cards=tuple(ids))


def test_bomb_order_matches_user_spec():
    """4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 天王炸"""
    four = _mk(meld.BOMB, 4, 5, (1, 2, 3, 4))
    five = _mk(meld.BOMB, 5, 5, (1, 2, 3, 4, 5))
    six = _mk(meld.BOMB, 6, 5, (1, 2, 3, 4, 5, 6))
    seven = _mk(meld.BOMB, 7, 5, (1, 2, 3, 4, 5, 6, 7))
    eight = _mk(meld.BOMB, 8, 5, tuple(range(1, 9)))
    flush = _mk(meld.STRAIGHT_FLUSH, 5, 10, (1, 2, 3, 4, 5))
    joker = _mk(meld.BOMB, 4, 0, (14, 15, 270, 271))
    assert meld.beats(five, four)
    assert meld.beats(flush, five)
    assert meld.beats(six, flush)
    assert meld.beats(seven, six)
    assert meld.beats(eight, seven)
    assert meld.beats(joker, eight)
    assert not meld.beats(four, five)


def test_four_jokers_is_the_top_bomb():
    """四大天王必须能枚举出来，且是最高层级。

    漏了这条，用户永远拿不到「出天王炸」的建议 —— 数据里 465 手没出现过，
    所以只有这条测试能挡住它。
    """
    hand = C("小王", "小王(二副)", "大王", "大王(二副)")
    top = [m for m in meld.melds_from(hand, level=9)
           if meld.bomb_class(m) == 7]
    assert len(top) == 1
    assert len(top[0].cards) == 4
    assert meld.beats(top[0], _mk(meld.BOMB, 8, 5, tuple(range(1, 9))))


def test_bomb_beats_normal_and_not_reverse():
    pair = _mk(meld.PAIR, 2, 13, (1, 2))
    bomb = _mk(meld.BOMB, 4, 2, (3, 4, 5, 6))
    assert meld.beats(bomb, pair)
    assert not meld.beats(pair, bomb)


def test_same_kind_compares_rank_only():
    low = _mk(meld.PAIR, 2, 5, (1, 2))
    high = _mk(meld.PAIR, 2, 6, (3, 4))
    assert meld.beats(high, low)
    assert not meld.beats(low, high)
    assert not meld.beats(low, low)          # 一样大不能压


def test_different_kind_does_not_beat():
    pair = _mk(meld.PAIR, 2, 5, (1, 2))
    single = _mk(meld.SINGLE, 1, 13, (1,))
    assert not meld.beats(single, pair)


def test_unknown_bomb_size_raises():
    """认不出的炸弹张数必须报错，不许静默。"""
    import pytest
    weird = _mk(meld.BOMB, 3, 5, (1, 2, 3))
    with pytest.raises(ValueError):
        meld.bomb_class(weird)


def test_legal_moves_when_leading_is_not_empty():
    """领出时合法着法非空。"""
    hand = C("5♦", "7♥")
    moves = meld.legal_moves(hand, table=None, level=2)
    assert moves, "领出必须能出牌"
    assert all(m.size >= 1 for m in moves)


def test_legal_moves_against_table():
    """桌面是对 9，只有更大的对子和炸弹能出。"""
    hand = C("5♦", "5♣", "J♦", "J♣", "K♠")
    table = _mk(meld.PAIR, 2, meld.point_value(9, 2), (16 + 9, 32 + 9))
    moves = meld.legal_moves(hand, table=table, level=2)
    assert moves
    for m in moves:
        assert meld.beats(m, table)
    assert not any(m.kind == meld.SINGLE for m in moves)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_meld_basic.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.meld' has no attribute 'point_value'`

- [ ] **Step 3: 实现**

`net/sim/meld.py`（**本文件内不得出现牌名字符串**）：

```python
"""掼蛋牌型真源 —— 合法着法枚举与大小比较。

按**牌 ID**（int）工作，复用 net/cards.py 的编码：
    parts(cid) -> (idx, suit, deck)
    idx 口径 A=1、2..10、J=11、Q=12、K=13、小王=14、大王=15

为什么不用牌名字符串：live/rules.py 的 6 个 bug 有一半来自字符串处理
（"10" vs "T"），见 spec §2.3。这里一律用整数。

牌型表从游戏协议的 card_type 字段统计得到（465 手真牌，spec §2.2）。
炸弹顺序由用户口述 + 24 对真实证据交叉验证（0 条矛盾）：
    4炸 < 5炸 < 同花顺 < 6炸 < 7炸 < 8炸 < 天王炸
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from net import cards

SINGLE, PAIR, TRIPLE = 1, 2, 3
STRAIGHT, TRIPLE_PAIR = 4, 5
PAIR_RUN, PLATE = 6, 7
BOMB, STRAIGHT_FLUSH, BOMB6 = 8, 9, 10

JOKER_SMALL, JOKER_BIG = 14, 15

# 非序列牌型的点数比较值：级牌 > A > K > ... > 2
_POINT = {**{i: i - 1 for i in range(2, 11)}, 11: 10, 12: 11, 13: 12, 1: 13}
POINT_LEVEL, POINT_SMALL, POINT_BIG = 14, 15, 16

_MIN_BOMB = 4
# 两副牌一个点数最多 8 张，**再加最多 2 张逢人配 = 10**。
# 真实数据里就有一手 9 张炸：J♠J♠(二副) J♥J♥(二副) J♣J♣(二副) J♦J♦(二副) + 3♥
# （打 3 时 ♥3 是逢人配，card_type=10）。定成 8 会让它枚举不出来。
_MAX_BOMB = 10

# 炸弹阶层。4炸<5炸<同花顺<6炸<7炸<8炸 是**用户口述 + 24 对真实证据**（0 矛盾）；
# **9炸 / 10炸 的位置是自然延伸，数据未验** —— 数据里有 9 张炸的实例，但没有
# 「9炸与别的炸对压」的证据。遇到反例从这里查。
_BOMB_CLASS_BY_SIZE = {4: 1, 5: 2, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
CLASS_FLUSH = 3
CLASS_JOKER_BOMB = 9


def norm_level(level: Optional[int]) -> Optional[int]:
    """级别归一。**日志里偶尔用 14 表示 A**（net/cards.py 的 sort_key 也处理过这条），
    不归一的话 14 会被当成小王，级牌判定与顺子权重全错。
    """
    if level == 14:
        return 1
    return level


def point_value(idx: int, level: Optional[int]) -> int:
    level = norm_level(level)
    if idx == JOKER_BIG:
        return POINT_BIG
    if idx == JOKER_SMALL:
        return POINT_SMALL
    if level is not None and idx == level:
        return POINT_LEVEL
    return _POINT[idx]


def is_wild(cid: int, level: Optional[int]) -> bool:
    """级牌红桃 = 逢人配（万能牌）。"""
    level = norm_level(level)
    if level is None:
        return False
    idx, suit, _ = cards.parts(cid)
    return idx == level and suit == "♥"


@dataclass(frozen=True)
class Meld:
    kind: int
    size: int
    rank: int                       # 比较主键（同 kind 内可比）
    cards: tuple
    wild_used: int = 0

    @property
    def is_bomb(self) -> bool:
        return bomb_class(self) is not None


def _all_jokers(ids) -> bool:
    """四张牌全是王（大小王各两张，即天王炸）。

    先过 `cards.is_card` 再取 `parts`：单测里允许用占位整数当 cards（比较关系
    只取决于 kind/size/rank），不该为此炸 KeyError。真实牌局里 cards 全是合法牌 ID，
    这一层零影响。

    注意 `len(ids) == 4` 会**短路**，所以只有恰为 4 张的占位 Meld 会触发这条 ——
    brief 最初没加这层保护时，恰好是 2 个测试失败（test_bomb_order_matches_user_spec
    与 test_bomb_beats_normal_and_not_reverse）。
    """
    return (len(ids) == 4 and all(cards.is_card(c) for c in ids)
            and all(cards.parts(c)[0] in (JOKER_SMALL, JOKER_BIG) for c in ids))


def bomb_class(m: Meld) -> Optional[int]:
    """炸弹层级；不是炸弹返回 None。天王炸最高。"""
    if m.kind == STRAIGHT_FLUSH:
        return CLASS_FLUSH
    if _all_jokers(m.cards):
        return CLASS_JOKER_BOMB
    if m.kind in (BOMB, BOMB6):
        cls = _BOMB_CLASS_BY_SIZE.get(m.size)
        if cls is None:
            raise ValueError(
                f"不认识的炸弹张数 {m.size}（合法 {_MIN_BOMB}~{_MAX_BOMB}，牌 {m.cards}）")
        return cls
    return None


def beats(a: Meld, b: Meld) -> bool:
    """a 能不能压过 b。"""
    ca, cb = bomb_class(a), bomb_class(b)
    if ca is not None and cb is not None:
        return ca > cb if ca != cb else a.rank > b.rank
    if ca is not None:
        return True                 # 炸弹压普通牌型
    if cb is not None:
        return False                # 普通牌型压不了炸弹
    if a.kind != b.kind or a.size != b.size:
        return False                # 牌型不同不能压
    return a.rank > b.rank


def _by_idx(hand: Sequence[int]) -> dict:
    g = {}
    for c in hand:
        g.setdefault(cards.parts(c)[0], []).append(c)
    return g


def _melds_basic(hand: Sequence[int], level: Optional[int]) -> list:
    out = []
    for idx, ids in _by_idx(hand).items():
        pv = point_value(idx, level)
        out.append(Meld(SINGLE, 1, pv, (ids[0],)))
        if idx >= JOKER_SMALL:
            # 王不能当普通点数用：两个小王是一对，但不能凑三张/炸弹
            if len(ids) >= 2:
                out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
            continue
        if len(ids) >= 2:
            out.append(Meld(PAIR, 2, pv, tuple(ids[:2])))
        if len(ids) >= 3:
            out.append(Meld(TRIPLE, 3, pv, tuple(ids[:3])))
        for n in range(_MIN_BOMB, min(len(ids), _MAX_BOMB) + 1):
            out.append(Meld(BOMB, n, pv, tuple(ids[:n])))
    return out


def _melds_triple_pair(level, triples, pairs) -> list:
    out = []
    for t_idx, t_cards in triples:
        for p_idx, p_cards in pairs:
            if p_idx == t_idx:
                continue
            out.append(Meld(TRIPLE_PAIR, 5, point_value(t_idx, level),
                            tuple(t_cards) + tuple(p_cards)))
    return out


def _melds_joker_bomb(hand: Sequence[int]) -> list:
    """四大天王（大小王各两张）。最高层级，bomb_class 靠 _all_jokers 认。

    王在 _melds_basic 里是 continue 掉的（不能凑三张/普通炸弹），
    所以这里必须单独补一条 —— 漏了它，用户永远拿不到「出天王炸」的建议。
    """
    js = [c for c in hand if cards.parts(c)[0] in (JOKER_SMALL, JOKER_BIG)]
    if len(js) >= 4:
        return [Meld(BOMB, 4, 0, tuple(sorted(js[:4])))]
    return []


def melds_from(hand: Sequence[int], level: Optional[int] = None) -> list:
    """枚举手牌能组成的所有牌型（本任务：单/对/三/三带二/炸弹/天王炸）。

    逢人配在这里**当作它自己那张级牌**参与枚举（Task 6 再加替代能力）。
    """
    level = norm_level(level)
    g = _by_idx(hand)
    out = _melds_basic(hand, level)
    triples = [(i, v[:3]) for i, v in g.items()
               if i < JOKER_SMALL and len(v) >= 3]
    # 王**可以**当三带二里的对子（真实数据：2♦2♦2♣ + 小王 小王(二副)，card_type=5）。
    # 但王**不能**凑三张 —— 每种王只有两张，所以下面 triples 保持排除王。
    pairs = [(i, v[:2]) for i, v in g.items() if len(v) >= 2]
    out += _melds_triple_pair(level, triples, pairs)
    out += _melds_joker_bomb(hand)
    return out


def legal_moves(hand: Sequence[int], table: Optional[Meld],
                level: Optional[int] = None) -> list:
    """现在能出的所有牌。table 为 None 表示我领出（此时不产生「过」）。"""
    moves = melds_from(hand, level)
    if table is None:
        return moves
    return [m for m in moves if beats(m, table)]
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_meld_basic.py -q`
Expected: `15 passed`

⚠️ **如果照着上面 `_all_jokers` 的最初写法（不带 `is_card` 保护）跑，会看到
`13 passed, 2 failed`（`KeyError: 0`）—— 那不是你写错了，是 brief 原代码的真缺陷。**
测试用占位整数构造 Meld 来隔离比较逻辑，`cards.parts` 对非牌 ID 会崩。

- [ ] **Step 5: 提交**

```bash
git add net/sim/meld.py tests/test_meld_basic.py
git commit -m "feat: 牌型模型与基础枚举、炸弹层级比较"
```

---

## Task 5: 顺子类牌型

**Files:**
- Modify: `net/sim/meld.py`
- Test: `tests/test_meld_seq.py`

**Interfaces:**
- Consumes: Task 4 的 `Meld` / `point_value` / `_by_idx`
- Produces:
  - `nat_values(idx:int) -> tuple`
  - `melds_from` 增加 `STRAIGHT` / `STRAIGHT_FLUSH` / `PAIR_RUN` / `PLATE`

**规则**（来自 spec §2.2 的数据统计）：

- **顺子 = 恰好 5 张连续**，A 可作最小也可作最大：`A2345` … `10JQKA`。**王不能进顺子。**
- **同花顺 = 顺子且五张同花色。**
- **连对 = 恰好 3 个连续对子**（数据里 4 张只有炸弹，所以没有二连对）。
- **钢板 = 恰好 2 个连续三张。**
- 比较主键一律取**序列顶端**的自然值（`A2345` 顶端 5、`10JQKA` 顶端 14）。

- [ ] **Step 1: 写失败的测试**

`tests/test_meld_seq.py`：

```python
from net.sim import meld
from tests.test_meld_basic import C


def test_nat_values_ace_is_both_ends():
    assert set(meld.nat_values(1)) == {1, 14}
    assert meld.nat_values(10) == (10,)
    assert meld.nat_values(14) == ()       # 小王不进序列
    assert meld.nat_values(15) == ()       # 大王不进序列


def test_ace_low_straight_is_legal():
    """A2345 是合法顺子（数据里有 2 手），这是 live/rules.py 的 bug #2。"""
    hand = C("A♥", "2♦", "3♥", "4♠", "5♥")
    st = [m for m in meld.melds_from(hand, level=9) if m.kind == meld.STRAIGHT]
    assert len(st) == 1
    assert st[0].rank == 5              # 顶端是 5


def test_ace_high_straight():
    hand = C("10♦", "J♦", "Q♦", "K♠", "A♦")
    st = [m for m in meld.melds_from(hand, level=9) if m.kind == meld.STRAIGHT]
    assert len(st) == 1 and st[0].rank == 14


def test_joker_cannot_join_straight():
    hand = C("10♦", "J♦", "Q♦", "K♠", "大王")
    assert not [m for m in meld.melds_from(hand, level=9)
                if m.kind == meld.STRAIGHT]


def test_straight_flush_requires_same_suit():
    same = C("5♠", "6♠", "7♠", "8♠", "9♠")
    mixed = C("5♠", "6♠", "7♠", "8♥", "9♠")
    assert meld.STRAIGHT_FLUSH in [m.kind for m in meld.melds_from(same, level=2)]
    km = [m.kind for m in meld.melds_from(mixed, level=2)]
    assert meld.STRAIGHT_FLUSH not in km
    assert meld.STRAIGHT in km


def test_straight_flush_survives_card_order():
    """同花顺不能因为「同点数的杂色牌排在前面」而漏掉。

    两副牌下每个点数必有两张不同花色，所以这是常态而非边角。
    漏掉不只是少一个建议 —— Task 7 的验收①会把真实打出的同花顺报成枚举不出。
    """
    has_flush = C("5♥", "5♠", "6♠", "7♠", "8♠", "9♠")     # 5♠ 排在后面
    ordered = C("5♠", "5♥", "6♠", "7♠", "8♠", "9♠")
    for hand, name in ((has_flush, "杂色在前"), (ordered, "同花在前")):
        sf = [m for m in meld.melds_from(hand, level=2)
              if m.kind == meld.STRAIGHT_FLUSH]
        assert len(sf) == 1, f"{name} 的手牌漏了同花顺"
        assert sf[0].rank == 9
        assert {meld.cards.parts(c)[1] for c in sf[0].cards} == {"♠"}


def test_ace_low_straight_flush_survives_card_order():
    """A 低窗同样：A♦ 排在 A♥ 前面时不能漏掉 A♥2♥3♥4♥5♥。"""
    hand = C("A♦", "A♥", "2♥", "3♥", "4♥", "5♥")
    sf = [m for m in meld.melds_from(hand, level=9)
          if m.kind == meld.STRAIGHT_FLUSH]
    assert len(sf) == 1 and sf[0].rank == 5


def test_pair_run_is_exactly_three_pairs():
    three = C("4♦", "4♣", "5♦", "5♠", "6♣", "6♠")
    two = C("4♦", "4♣", "5♦", "5♠")
    assert [m for m in meld.melds_from(three, level=2)
            if m.kind == meld.PAIR_RUN]
    assert not [m for m in meld.melds_from(two, level=2)
                if m.kind == meld.PAIR_RUN]


def test_plate_is_two_consecutive_triples():
    plate = C("3♥", "3♦", "3♣", "4♦", "4♣", "4♠")
    assert [m for m in meld.melds_from(plate, level=2) if m.kind == meld.PLATE]
    not_plate = C("3♥", "3♦", "3♣", "5♦", "5♣", "5♠")
    assert not [m for m in meld.melds_from(not_plate, level=2)
                if m.kind == meld.PLATE]


def test_straight_rank_ordering_uses_top():
    low = C("A♥", "2♦", "3♥", "4♠", "5♥")
    high = C("6♦", "7♦", "8♦", "9♠", "10♥")
    a = [m for m in meld.melds_from(low, level=9) if m.kind == meld.STRAIGHT][0]
    b = [m for m in meld.melds_from(high, level=9) if m.kind == meld.STRAIGHT][0]
    assert meld.beats(b, a)


def test_ace_low_pair_run():
    """三连对也允许 A 当小牌：A-2-3。"""
    hand = C("A♥", "A♦", "2♥", "2♦", "3♠", "3♣")
    assert [m for m in meld.melds_from(hand, level=9) if m.kind == meld.PAIR_RUN]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_meld_seq.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.meld' has no attribute 'nat_values'`

- [ ] **Step 3: 实现**

在 `net/sim/meld.py` 里加：

```python
_SEQ_LEN = 5
_PAIR_RUN_LEN = 3
_PLATE_LEN = 2
_NAT_MAX = 14          # A 当大牌时的自然值


def nat_values(idx: int) -> tuple:
    """这个点数在序列里的取值。A 可作 1 或 14；王不参与序列。"""
    if idx == 1:
        return (1, _NAT_MAX)
    if 2 <= idx <= 13:
        return (idx,)
    return ()


def _idx_for_nat(nat: int) -> int:
    return 1 if nat == _NAT_MAX else nat


def _melds_straights(nat: dict, level) -> list:
    """`nat` = `_seq_lookup(g)`（自然值 -> 牌）。

    ⚠️ **同花顺必须逐花色试，不能只看每格第一张牌。**
    两副牌下每个点数必有两张不同花色，所以「第一张是杂色」是常态；
    只看第一张会漏掉真实存在的同花顺（同一手牌换个手序结果就不同）：
        5♥ 5♠ 6♠ 7♠ 8♠ 9♠  -> 漏（5♠6♠7♠8♠9♠ 明明在手上）
        5♠ 5♥ 6♠ 7♠ 8♠ 9♠  -> 命中
    漏掉的后果不只是少一个建议：Task 7 的验收①会把真实打出的同花顺报成「枚举不出」。
    """
    out = []
    for top in range(_SEQ_LEN, _NAT_MAX + 1):
        nats = list(range(top - _SEQ_LEN + 1, top + 1))
        if any(not nat.get(n) for n in nats):
            continue
        out.append(Meld(STRAIGHT, _SEQ_LEN, top,
                        tuple(nat[n][0] for n in nats)))
        # **每个花色都产出，不要 break。** 虽然同花顺比大小只看顶端，
        # 但手里的同花顺是**具体哪几张**会影响玩家实际能打出的牌：
        # 同时握着 ♠ 与 ♥ 两套同顶端同花顺时，只留一个代表会让真实打出另一套的
        # 局面「枚举不出」。上限 4 个/顶端，可忽略。
        for suit in "♠♥♣♦":
            pick = [next((x for x in nat.get(n, [])
                          if cards.parts(x)[1] == suit), None) for n in nats]
            if all(c is not None for c in pick):
                out.append(Meld(STRAIGHT_FLUSH, _SEQ_LEN, top, tuple(pick)))
    return out


def _melds_pair_run(g: dict, level) -> list:
    out = []
    for start in range(1, _NAT_MAX - _PAIR_RUN_LEN + 2):
        nats = list(range(start, start + _PAIR_RUN_LEN))
        if any(len(g.get(_idx_for_nat(n), [])) < 2 for n in nats):
            continue
        ids = [c for n in nats for c in g[_idx_for_nat(n)][:2]]
        out.append(Meld(PAIR_RUN, _PAIR_RUN_LEN * 2, nats[-1], tuple(ids)))
    return out


def _melds_plate(g: dict, level) -> list:
    out = []
    for start in range(1, _NAT_MAX - _PLATE_LEN + 2):
        nats = list(range(start, start + _PLATE_LEN))
        if any(len(g.get(_idx_for_nat(n), [])) < 3 for n in nats):
            continue
        ids = [c for n in nats for c in g[_idx_for_nat(n)][:3]]
        out.append(Meld(PLATE, _PLATE_LEN * 3, nats[-1], tuple(ids)))
    return out
```

并把 `melds_from` 改成：

```python
def melds_from(hand: Sequence[int], level: Optional[int] = None) -> list:
    g = _by_idx(hand)
    out = _melds_basic(hand, level)
    triples = [(i, v[:3]) for i, v in g.items()
               if i < JOKER_SMALL and len(v) >= 3]
    # 王**可以**当三带二里的对子（真实数据：2♦2♦2♣ + 小王 小王(二副)，card_type=5）。
    # 但王**不能**凑三张 —— 每种王只有两张，所以下面 triples 保持排除王。
    pairs = [(i, v[:2]) for i, v in g.items() if len(v) >= 2]
    out += _melds_triple_pair(level, triples, pairs)
    out += _melds_joker_bomb(hand)          # <-- 别漏！Task 4 加的，漏了天王炸就没了
    out += _melds_straights(g, level)
    out += _melds_pair_run(g, level)
    out += _melds_plate(g, level)
    return out
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_meld_seq.py tests/test_meld_basic.py -q`
Expected: `26 passed`（test_meld_basic 17 + test_meld_seq 9）

- [ ] **Step 5: 提交**

```bash
git add net/sim/meld.py tests/test_meld_seq.py
git commit -m "feat: 顺子/同花顺/连对/钢板（含 A 可作小牌）"
```

---

## Task 6: 逢人配

**Files:**
- Modify: `net/sim/meld.py`
- Test: `tests/test_meld_wild.py`

**Interfaces:**
- Consumes: Task 4/5 的全部枚举函数
- Produces: `melds_from` 与 `legal_moves` 支持逢人配；`Meld.wild_used` 被填上真实值

**规则**（spec §6④）：

- 级牌红桃（`is_wild`）能当**任意普通牌**用；**不能当王**。
- **先枚举天然牌型**，只有天然凑不成时才补逢人配 —— 天然的那些 `wild_used=0`。
- 逢人配当作**它自己那张级牌**的用法也要保留（打 5 时两张 ♥5 本身就是一对 5）。
- 结果**不重复**（同一组牌只出现一次）。
- **不能假设总有 2 张**：一张都没有、或只有 1 张，都要正常工作。

- [ ] **Step 1: 写失败的测试**

`tests/test_meld_wild.py`：

```python
from net.sim import meld
from tests.test_meld_basic import C


def test_wild_alone_is_its_own_card():
    """打 5 时 ♥5 本来就是一张 5。"""
    hand = C("5♥")
    assert [m for m in meld.melds_from(hand, level=5) if m.kind == meld.SINGLE]


def test_two_wilds_are_a_pair_of_level_cards():
    hand = C("5♥", "5♥(二副)")
    pairs = [m for m in meld.melds_from(hand, level=5) if m.kind == meld.PAIR]
    assert pairs and all(m.wild_used == 0 for m in pairs)


def test_wild_fills_a_bomb():
    """数据里 2♥2♥2♠10♥ 在打10时是四个2的炸。"""
    hand = C("2♥", "2♥(二副)", "2♠", "10♥")
    bombs = [m for m in meld.melds_from(hand, level=10) if m.kind == meld.BOMB]
    assert any(m.size == 4 and m.rank == meld.point_value(2, 10) for m in bombs)


def test_wild_fills_straight_flush():
    """数据里 9♣10♣J♣Q♣ + ♥2 在打2时是同花顺。"""
    hand = C("9♣", "10♣", "J♣", "Q♣", "2♥")
    sf = [m for m in meld.melds_from(hand, level=2)
          if m.kind == meld.STRAIGHT_FLUSH]
    assert sf, "逢人配应当能补成同花顺"


def test_wild_never_becomes_a_joker():
    hand = C("2♥", "大王", "大王(二副)", "小王")
    for m in meld.melds_from(hand, level=2):
        if m.kind == meld.BOMB and m.size == 4:
            assert all(meld.cards.parts(c)[0] in (14, 15) for c in m.cards), \
                "王炸里不能混进逢人配"


def test_no_wild_at_all_still_works():
    hand = C("5♦", "5♣", "5♠")
    assert [m for m in meld.melds_from(hand, level=9) if m.kind == meld.TRIPLE]


def test_only_one_wild_available():
    hand = C("5♦", "5♣", "2♥")
    triples = [m for m in meld.melds_from(hand, level=2) if m.kind == meld.TRIPLE]
    assert triples and triples[0].wild_used == 1


def test_results_are_not_duplicated():
    hand = C("5♦", "5♣", "5♠", "2♥")
    keys = [tuple(sorted(m.cards)) for m in meld.melds_from(hand, level=2)]
    assert len(keys) == len(set(keys)), "同一组牌重复出现了"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_meld_wild.py -q`
Expected: FAIL（`test_wild_fills_a_bomb` 等几条）

- [ ] **Step 3: 实现**

在 `net/sim/meld.py` 里把枚举重构为「天然 + 补牌」两段：

```python
def _split_wild(hand, level):
    """把逢人配拆出来。返回 (其余牌, 逢人配列表)。"""
    wilds = [c for c in hand if is_wild(c, level)]
    rest = [c for c in hand if not is_wild(c, level)]
    return rest, wilds


# 注意：这里的 _melds_natural 就是 Task 4/5 里那个 melds_from，
# **整体改名，函数体一行都不动**。不要写成 `return melds_from(...)`那样会无限递归。
# 改完之后本文件里应该只剩下一个 melds_from —— 就是下面 Task 6 新写的那个。

def _missing(nat: dict, nats, per: int) -> int:
    """要凑出 nats 这些自然值、每个 per 张，还缺几张。

    `nat` 是 `_seq_lookup(g)` 的结果（自然值 -> 牌），A 同时落在 1 与 14 两格。
    不要用 idx 直接查 `g` —— A 的两面性只在 `_seq_lookup` 里处理一次。
    """
    d = 0
    for n in nats:
        have = len(nat.get(n, []))
        if have < per:
            d += per - have
    return d


def _take(nat: dict, nats, per: int) -> tuple:
    out = []
    for n in nats:
        out.extend(nat.get(n, [])[:per])
    return tuple(out)


def _melds_wild(g: dict, level, n_wild: int) -> list:
    """用逢人配补出来的牌型。g 是**不含逢人配**的牌分组。

    只产出「需要补」的牌型 —— 天然的那些由 _melds_natural 负责，去重收口。
    """
    if n_wild <= 0:
        return []
    nat = _seq_lookup(g)          # 自然值 -> 牌（A 同时落 1 与 14）
    out = []
    ranks = [i for i in g if i < JOKER_SMALL]

    for i in ranks:
        have = len(g[i])
        pv = point_value(i, level)
        for need, kind in ((2, PAIR), (3, TRIPLE)):
            d = need - have
            if 0 < d <= n_wild:
                out.append(Meld(kind, need, pv, tuple(g[i]), wild_used=d))
        for n in range(_MIN_BOMB, _MAX_BOMB + 1):
            d = n - have
            if 0 < d <= n_wild:
                out.append(Meld(BOMB, n, pv, tuple(g[i]), wild_used=d))

    for t in ranks:
        for p in ranks:
            if p == t:
                continue
            d = (3 - len(g[t])) + (2 - len(g[p]))
            if 0 < d <= n_wild and len(g[t]) < 3 and len(g[p]) < 2:
                out.append(Meld(TRIPLE_PAIR, 5, point_value(t, level),
                                tuple(g[t]) + tuple(g[p]), wild_used=d))

    for top in range(_SEQ_LEN, _NAT_MAX + 1):
        nats = list(range(top - _SEQ_LEN + 1, top + 1))
        d = _missing(nat, nats, 1)
        if 0 < d <= n_wild:
            out.append(Meld(STRAIGHT, _SEQ_LEN, top, _take(nat, nats, 1),
                            wild_used=d))
        # 同花顺单独试，且**要在 `if 0 < d` 之外** —— 每个自然值都有牌、
        # 但都不是同一花色时，d == 0 而缺的全靠逢人配补。
        # 同样必须逐花色试（见 _melds_straights 的说明），不能只看 _take 那几张的花色。
        for suit in "♠♥♣♦":
            pick, miss = [], 0
            for n in nats:
                c = next((x for x in nat.get(n, [])
                          if cards.parts(x)[1] == suit), None)
                if c is None:
                    miss += 1
                else:
                    pick.append(c)
            if 0 < miss <= n_wild:      # miss == 0 是天然的，由 _melds_straights 负责
                out.append(Meld(STRAIGHT_FLUSH, _SEQ_LEN, top, tuple(pick),
                                wild_used=miss))    # 同样不 break，见 Task 5 的说明

    for start in range(1, _NAT_MAX - _PAIR_RUN_LEN + 2):
        nats = list(range(start, start + _PAIR_RUN_LEN))
        d = _missing(nat, nats, 2)
        if 0 < d <= n_wild:
            out.append(Meld(PAIR_RUN, _PAIR_RUN_LEN * 2, nats[-1],
                            _take(nat, nats, 2), wild_used=d))

    for start in range(1, _NAT_MAX - _PLATE_LEN + 2):
        nats = list(range(start, start + _PLATE_LEN))
        d = _missing(nat, nats, 3)
        if 0 < d <= n_wild:
            out.append(Meld(PLATE, _PLATE_LEN * 3, nats[-1],
                            _take(nat, nats, 3), wild_used=d))
    return out
```

把 Task 4/5 里那个 `melds_from` **改名为 `_melds_natural`**（删掉上面 `_melds_natural` 里的转发），
然后新的公开入口：

```python
def melds_from(hand: Sequence[int], level: Optional[int] = None) -> list:
    """枚举手牌能组成的所有牌型。逢人配当万能牌，但**先试天然的**。"""
    rest, wilds = _split_wild(hand, level)
    out = _melds_natural(hand, level)
    if wilds:
        out += _melds_wild(_by_idx(rest), level, len(wilds))
    seen, uniq = set(), []
    for m in out:
        key = (m.kind, tuple(sorted(m.cards)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq
```

同时 `legal_moves` 里对桌面的比较需要炸弹类可判 —— 不用改，`beats` 已能处理。

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add net/sim/meld.py tests/test_meld_wild.py
git commit -m "feat: 逢人配（受控补牌，先天然后补）"
```

---

## Task 7: 第一层验收脚本

**Files:**
- Create: `tools/accept_meld.py`
- Test: `tests/test_accept.py`

**Interfaces:**
- Consumes: Task 2~6 全部
- Produces:
  - `@dataclass Result: name, total, bad, ok, report(limit=5)`
  - `check_real_moves(games=None) -> Result`
  - `check_beats_from_records(games=None) -> Result`
  - `check_invariants() -> Result`
  - `check_wildcard() -> Result`
  - `main() -> int`（`python -m tools.accept_meld`，全绿返回 0）

**这是 spec §6 的落地。失败必须能定位到「哪一局、第几手」。**

- [ ] **Step 1: 写失败的测试**

`tests/test_accept.py`：

```python
import os
import pytest
from tools.game_log import LOG_DIR
from tools.accept_meld import (check_invariants, check_real_moves,
                               check_wildcard)

pytestmark = pytest.mark.skipif(not os.path.isdir(LOG_DIR),
                                reason="本机没有游戏日志")


def test_invariants_pass():
    assert check_invariants().ok


def test_wildcard_checks_pass():
    assert check_wildcard().ok


def test_real_moves_are_all_legal():
    """55 局真实记录里，每一手真实出的牌都必须在合法着法集合里。"""
    r = check_real_moves()
    assert r.ok, r.report()
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_accept.py -q`
Expected: FAIL — `No module named 'tools.accept_meld'`

- [ ] **Step 3: 实现**

`tools/accept_meld.py`：

```python
"""第一层验收（spec §6）：拿真实对局验证牌型引擎。

跑法：
    .venv/Scripts/python.exe -m tools.accept_meld

四项：
  ① 真实着法必须能枚举出来        ② 谁压谁逐条对齐
  ③ 不变量（不需要真值）          ④ 逢人配专项

失败会打印「哪一局、第几手、真实出的是什么」，便于定位。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from net.sim import meld
from tools.decision_points import decision_points
from tools.game_log import load_games


@dataclass
class Result:
    name: str
    total: int = 0
    bad: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bad

    def report(self, limit: int = 5) -> str:
        if self.ok:
            return f"[OK]   {self.name}  {self.total} 项全过"
        lines = [f"[FAIL] {self.name}  {len(self.bad)}/{self.total} 项不过"]
        lines += [f"       {b}" for b in self.bad[:limit]]
        if len(self.bad) > limit:
            lines.append(f"       …还有 {len(self.bad) - limit} 条")
        return "\n".join(lines)


def _stronger(a, b) -> bool:
    """同一组牌符合多个牌型时，a 是否比 b 更「强」。"""
    ca, cb = meld.bomb_class(a), meld.bomb_class(b)
    if (ca is None) != (cb is None):
        return ca is not None                 # 炸弹类优先
    if ca is not None and cb is not None and ca != cb:
        return ca > cb
    return a.kind > b.kind


def as_meld(ids, level):
    """把一组**具体**的牌判成一个 Meld；判不出返回 None。

    这里必须是精确集合 —— 参数就是那一手真实的牌，没有"代表牌"问题。

    ⚠️ **同一组牌可能符合多个牌型，必须取最强的那个：**
    5 张同花连续的牌**同时**是顺子(kind 4)与同花顺(kind 9)，
    `melds_from` 先产出顺子。取第一个的话：
      - 真实打出的同花顺会被当成顺子 -> 验收①对「漏枚举同花顺」完全视而不见
      - 桌面上的同花顺会被低估成顺子 -> legal_moves 会放进本该压不过的顺子
    游戏自己也把它叫同花顺（card_type 9）。
    """
    best = None
    for m in meld.melds_from(list(ids), level):
        if sorted(m.cards) != sorted(ids):
            continue
        if best is None or _stronger(m, best):
            best = m
    return best


def shape(m):
    """比较用的**形状**键：(牌型, 张数, 主点数)。**不含具体是哪几张牌。**

    为什么必须要形状键而不是精确牌组：`melds_from` 每个形状只给一个**代表**，
    而两手牌可能打出同一个形状的不同具体牌（手里同时有 ♠ 与 ♥ 两套同顶端的同花顺、
    同点数 5 张里挑哪 4 张做炸、两副牌的同名牌……）。用精确牌组比会把它们误报成
    「枚举不出」，而那正是本验收要抓的失败类的**假阳性**版本。

    炸弹归一：引擎用 `BOMB` 承载 4~8 张，而游戏协议用 card_type 8 表 4~5 张、
    10 表 6 张 —— 比较时必须先把这两种 kind 归一，否则 6 张炸会假红。
    """
    kind = meld.BOMB if m.kind in (meld.BOMB, meld.BOMB6) else m.kind
    return (kind, m.size, m.rank)


def check_real_moves(games=None) -> Result:
    """① 每一手真实出的牌，必须在 legal_moves 里。

    真实着法一定是合法的（游戏自己认过），所以不在里面 = 我们错了。
    **局限：55 局只能证伪，不能证明**（spec §6①）。
    """
    r = Result("① 真实着法可枚举")
    games = load_games() if games is None else games
    for g in games:
        if not g.settle:
            continue
        for i, s in enumerate(decision_points(g)):
            r.total += 1
            table = None
            if s.table:
                table = as_meld(s.table, s.level)
                if table is None:
                    r.bad.append(f"{g.t0:%m-%d %H:%M} 第{i}手 "
                                 f"桌面牌本身判不出牌型 {sorted(s.table)}")
                    continue
            # 两步：先看真实那一手**本身**能不能判出牌型（精确集合，能抓牌型缺口，
            # 例如「王当三带二的对子」那种）；再比**形状**是否在候选里
            # （形状比而非精确集合，因为枚举只给代表，见 shape() 的说明）。
            real = as_meld(s.actual, s.level)
            if real is None:
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 第{i}手 座位{s.seat} "
                    f"真实出的 {sorted(s.actual)} 本身判不出牌型"
                    f"（card_type={s.card_type if hasattr(s, 'card_type') else '?'}）")
                continue
            moves = meld.legal_moves(s.hand, table=table, level=s.level)
            if not any(shape(m) == shape(real) for m in moves):
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} 第{i}手 座位{s.seat} "
                    f"真实出的 {sorted(s.actual)}（{shape(real)}）不在候选里"
                    f"（手牌 {len(s.hand)} 张，"
                    f"桌面 {sorted(s.table) if s.table else '空'}，"
                    f"候选 {len(moves)} 个，形状集 {sorted({shape(m) for m in moves})}）")
    return r


def _bomb_pairs(g):
    """同一轮内「炸弹 A 之后又出了炸弹 B」的证据对。

    轮次边界判据与 tools/decision_points.py **完全一致**：
    同一座位又出牌、或队友接风（桌面主人上一手已出完）。

    ⚠️ **不要用 PlayRec.nxt 判** —— 服务器算下一手时会跳过已出完的座位，
    所以 nxt 指到队友既可能是接风、也可能只是在跳过。详见 Task 3 的原理说明。

    这条**独立于 legal_moves**：它只从出牌序列推「谁大」，所以 ① 全绿它仍可能红。
    """
    pairs = []
    table = None
    table_seat = None
    prev_left = None
    for p in g.plays:
        if table is not None:
            partner = (table_seat + 2) % 4
            # 与 tools/decision_points.py 同一判据。**不能用 nxt**：
            # 服务器算 nxt 时会跳过已出完的座位，会把它误判成接风。
            is_new_lead = (p.seat == table_seat
                           or (p.seat == partner and prev_left == 0))
            if not is_new_lead:
                pairs.append((table, p))
        table = p
        table_seat = p.seat
        prev_left = p.left
    return pairs


def check_beats_from_records(games=None) -> Result:
    """② 炸弹层级：真实对局里「炸弹 A 被炸弹 B 压掉」的证据必须逐条成立。

    专门验用户口述的炸弹顺序（4炸<5炸<同花顺<6炸<7炸<8炸<天王炸）。
    注意「同花顺夹在 5炸与 6炸之间」那半边**用户口述时数据没覆盖**（spec §2.1）
    —— 跑出来的条数要报出来，是 0 条就说明这段仍未验到。
    """
    r = Result("② 炸弹层级（谁压谁）")
    games = load_games() if games is None else games
    for g in games:
        if not g.settle:
            continue
        for a_ids, b_ids in _bomb_pairs(g):
            a = as_meld(a_ids, g.trump)
            b = as_meld(b_ids, g.trump)
            if a is None or b is None:
                continue
            if meld.bomb_class(a) is None and meld.bomb_class(b) is None:
                continue                      # 不是炸弹对，本检查不管
            r.total += 1
            if not meld.beats(b, a):
                r.bad.append(
                    f"{g.t0:%m-%d %H:%M} {sorted(a_ids)} -> {sorted(b_ids)} "
                    f"但 beats() 说压不过")
    return r


def check_invariants() -> Result:
    """③ 不变量：纯逻辑，不需要真值。"""
    r = Result("③ 不变量")

    r.total += 1
    if meld.legal_moves([], table=None, level=2):
        r.bad.append("空手牌不该有候选")

    r.total += 1
    hand = [16 + 5, 32 + 5]                     # 5♠ 5♥
    moves = meld.legal_moves(hand, table=None, level=9)
    if not moves:
        r.bad.append("领出时必须至少有一个候选")
    if any(m.size == 0 for m in moves):
        r.bad.append("候选里不该有「过」")

    r.total += 1
    for m in moves:
        if len(set(m.cards)) != len(m.cards):
            r.bad.append(f"候选 {m.cards} 里有重复牌")

    r.total += 1
    big = [16 + 5, 16 + 5 + 256, 32 + 5, 32 + 5 + 256]
    for m in meld.melds_from(big, level=9):      # 四张 5 的炸
        if m.kind == meld.BOMB and m.size == 4:
            if len(set(m.cards)) != 4:
                r.bad.append("四张炸里有重复牌 ID")
    return r


def check_wildcard() -> Result:
    """④ 逢人配专项。"""
    r = Result("④ 逢人配")
    level = 5
    wilds = [32 + 5, 32 + 5 + 256]               # ♥5 / ♥5(二副)
    # ⚠️ 必须用**非级牌**的点数（6，不是 5）。用 5 的话两张 ♥5 本身就是两张 5，
    # n=2 时天然就是四炸、wild_used == 0，而「天然优先」要求它必须是 0 ——
    # 断言 wild_used > 0 会与本任务的硬规格直接冲突。
    naturals = [64 + 6, 48 + 6, 16 + 6]          # ♦6 ♣6 ♠6

    for n in (0, 1, 2):
        r.total += 1
        hand = naturals + wilds[:n]
        bombs = [m for m in meld.melds_from(hand, level=level)
                 if m.kind == meld.BOMB]
        if n == 0:
            if bombs:
                r.bad.append("没有逢人配时，三张 6 不该有炸弹")
        elif not any(m.wild_used > 0 for m in bombs):
            r.bad.append(f"{n} 张逢人配时应当能补出炸弹")

    r.total += 1
    hand = [32 + 5, 15, 271, 14]                 # ♥5 + 大王 + 大王(二副) + 小王
    for m in meld.melds_from(hand, level=level):
        if m.kind == meld.BOMB and m.size == 4:
            if not all(meld.cards.parts(c)[0] in (14, 15) for c in m.cards):
                r.bad.append("王炸里混进了逢人配")
    return r


CHECKS = [check_real_moves, check_beats_from_records, check_invariants,
          check_wildcard]


def main() -> int:
    games = load_games()
    print(f"载入对局 {len(games)} 局，其中有结算的 "
          f"{sum(1 for g in games if g.settle)} 局\n")
    failed = 0
    for fn in CHECKS:
        res = fn(games)
        print(res.report())
        failed += 0 if res.ok else 1
    print()
    if failed:
        print(f"验收不通过：{failed} 项红")
        return 1
    print("验收全绿 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_accept.py -q`
Expected: `3 passed`

再跑完整脚本：

Run: `.venv/Scripts/python.exe -m tools.accept_meld`
Expected: 四项 `[OK]`，退出码 0

**这一条一定会先红，红了才有价值。** 按 `report()` 打印的「哪一局哪一手」去查：
是枚举漏了牌型，还是桌面重建错了。**不要为了让验收变绿而放宽
`beats` / `legal_moves`** —— 那等于把 bug 固化。

- [ ] **Step 5: 提交**

```bash
git add tools/accept_meld.py tests/test_accept.py
git commit -m "feat: 第一层验收脚本（真实着法 / 谁压谁 / 不变量 / 逢人配）"
```

---

## Task 8: `live/rules.py` 改成适配层

**Files:**
- Modify: `net/sim/meld.py`（补 `cid_from_name` / `name_from_cid` / `describe_meld`）
- Modify: `live/rules.py`（整体替换）
- Test: `tests/test_rules_adapter.py`

**Interfaces:**
- Consumes: `net.sim.meld` 全部
- Produces:
  - `meld.cid_from_name(name:str, deck:int=1) -> int`
  - `meld.name_from_cid(cid:int) -> str`
  - `meld.describe_meld(m:Meld) -> str`
  - `live.rules.classify(cards:list[str], level:str="2") -> str | None`（行为与旧版一致）

**目的**：牌型真源只有一个。旧实现里的 6 个 bug **不在旧代码里修**，而是删掉旧实现、转成适配层。

- [ ] **Step 1: 写失败的测试**

`tests/test_rules_adapter.py`：

```python
from live import rules


def test_ten_is_handled():
    """旧实现遇到 10 会直接崩（bug #1）。"""
    assert rules.classify(["S10", "H10", "D10"], "2") == "三张"


def test_ace_low_straight():
    """旧实现判非法（bug #2）。"""
    assert rules.classify(["SA", "H2", "D3", "C4", "S5"], "9") == "顺子"


def test_two_small_jokers_are_a_pair():
    """旧实现说「两个王不是对子」（bug #3）。"""
    got = rules.classify(["JOKER_SMALL", "JOKER_SMALL"], "9")
    assert got is not None and "对" in got


def test_two_pair_run_is_illegal():
    """旧实现把 4 张二连对判合法（bug #4），游戏里 4 张只有炸弹。"""
    assert rules.classify(["S4", "H4", "S5", "H5"], "2") is None


def test_illegal_returns_none():
    assert rules.classify(["S5", "H7", "D9"], "2") is None


def test_describe_falls_back():
    assert rules.describe(["S5", "H7", "D9"], "2") == "不合法"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_adapter.py -q`
Expected: FAIL（`test_ten_is_handled` 报 `ValueError: substring not found`）

- [ ] **Step 3: 实现**

在 `net/sim/meld.py` 末尾加：

```python
# ------------------------------------------------- 适配层用的名字转换
# 只在 live/rules.py 这个适配层里用；本模块内部一律用 ID。

_SUIT_LETTER = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}
_SUIT_BASE = {"♠": 16, "♥": 32, "♣": 48, "♦": 64}
_RANK_LETTER = {"A": 1, "J": 11, "Q": 12, "K": 13}


def cid_from_name(name: str, deck: int = 1) -> int:
    """牌面名 -> 牌 ID。'S3' / 'H10' / 'JOKER_BIG'。"""
    if name.startswith("JOKER"):
        big = name.endswith("BIG")
        if deck == 2:
            return 271 if big else 270
        return 15 if big else 14
    letter, rank = name[0], name[1:]
    idx = _RANK_LETTER.get(rank)
    if idx is None:
        if not rank.isdigit():
            raise ValueError(f"认不出的点数：{name}")
        idx = int(rank)
    return _SUIT_BASE[_SUIT_LETTER[letter]] + idx + (256 if deck == 2 else 0)


def name_from_cid(cid: int) -> str:
    return cards.decode(cid)


_KIND_NAMES = {
    SINGLE: "单张", PAIR: "对子", TRIPLE: "三张", STRAIGHT: "顺子",
    TRIPLE_PAIR: "三带二", PAIR_RUN: "连对", PLATE: "钢板",
    STRAIGHT_FLUSH: "同花顺",
}


def describe_meld(m: Meld) -> str:
    if _all_jokers(m.cards):
        return "四大天王"
    if m.kind in (BOMB, BOMB6):
        return f"{m.size} 张炸"
    return _KIND_NAMES[m.kind]


def level_idx(level) -> Optional[int]:
    """级别 -> 点数索引。接受 1~13 或 'A'/'J'/'Q'/'K'/'2'..'10'。"""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    text = str(level).strip().upper()
    if text in _RANK_LETTER:
        return _RANK_LETTER[text]
    if text.isdigit():
        return int(text)
    raise ValueError(f"认不出的级别：{level}")
```

`live/rules.py` 整体替换为：

```python
"""掼蛋牌型校验 —— 适配层（给 live/ 那条截图识别线用）。

**牌型真源在 `net/sim/meld.py`**（网络直读 + RL 是主路径）。这里只做
「牌面名 <-> 牌 ID」的转换再转发。

老实现有 6 处已证实的错（见 spec §2.3）："10" vs "T" 直接崩、A 不能当小牌、
两个王不算对子、二连对判合法、docstring 说有三带一、逢人配能力低估。
**不要再改回老实现**，那等于把 bug 固化。
"""
from __future__ import annotations

from net.sim import meld


def _to_ids(names: list) -> list:
    out = []
    for name in names:
        deck = 2 if "(二副)" in name else 1
        core = name.replace("(二副)", "")
        out.append(meld.cid_from_name(core, deck))
    return out


def classify(cards: list, level: str = "2") -> str | None:
    """判牌型。合法返回牌型名，不合法返回 None。

    level 是当前级牌（'2'..'10' / 'J' / 'Q' / 'K' / 'A'）。
    """
    if not cards:
        return None
    lv = meld.level_idx(level)
    ids = _to_ids(cards)
    for m in meld.melds_from(ids, level=lv):
        if sorted(m.cards) == sorted(ids):
            got = meld.describe_meld(m)
            return got + ("（含逢人配）" if m.wild_used else "")
    return None


def describe(cards: list, level: str = "2") -> str:
    """给面板显示用：合法就写牌型，不合法就明确说不合法。"""
    return classify(cards, level) or "不合法"
```

- [ ] **Step 4: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全部通过

确认 YOLO 面板没被弄坏：

Run: `.venv/Scripts/python.exe -c "import live.main; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: 提交**

```bash
git add net/sim/meld.py live/rules.py tests/test_rules_adapter.py
git commit -m "refactor: live/rules.py 改为 net/sim/meld.py 的适配层"
```

---

## 完成标准

- [ ] `.venv/Scripts/python.exe -m pytest tests/ -q` 全绿
- [ ] `.venv/Scripts/python.exe -m tools.accept_meld` 四项 `[OK]`，退出码 0
- [ ] `net/sim/meld.py` 的**代码路径**里没有牌名字符串字面量
      （docstring 里为说明 `live/rules.py` 历史 bug 而提到的 `"10"` / `"T"` 属散文，不算）
- [ ] `git log --oneline` 有 8 个任务提交

## 下一步（不在本计划内）

Plan 2 模拟器（`net/sim/rules.py` 进贡/接风/结算 + `net/sim/env.py`）。
**本计划验收①的失败清单就是 Plan 2 的输入** —— 它暴露的规则空白正是模拟器要补的。

**Plan 2 必须带上的两件（本计划不覆盖，别忘）**：

1. spec §6③ 最后一条 —— **「喂给策略的状态不含对手手牌」的自动化测试**。
   做法：构造「对手手牌不同、其余相同」的两个局面，若网络给两者的打分**一样**
   → 状态里确实无对手信息；不一样 → 泄漏。**手动看代码看不出来，必须写成测试。**
2. spec §6⑤ —— 进贡/接风规则从日志解（2135 行进贡记录），并拿数据逐条验。
