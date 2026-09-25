# 掼蛋模拟器（Plan 2）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个能自对弈的掼蛋**一手牌**模拟器（`net/sim/rules.py`）、它的自对弈环境（`net/sim/env.py`），用 55 局真实记录验收规则、用最小训练冒烟证明 env 真能训。

**Architecture:** `meld.py` 已是牌型真源（Plan 1 交付，不要改）。`rules.py` 在它之上做**一手牌**的循环：发牌 → 进贡/还贡 → 出牌（含接风）→ 结算名次。到名次出来即终止。`env.py` 把它包成可自对弈的环境：内部是明牌，**对外只暴露可观测信息**（编码器是唯一出口，明牌泄漏靠自动化测试钉死）。级别由 env 采样，不实现升级/过 A。

**Tech Stack:** Python 3.12（venv 已装 numpy 2.5.2 / torch 2.11.0+cu128 / pytest 9.1.1）。`rules.py` 只用标准库；`env.py` 用 numpy；冒烟训练用 torch。

**Spec:** `docs/superpowers/specs/2026-09-24-guandan-rl-advisor-design.md`（**含 §13 补充裁定，必须先读**）

## Global Constraints

- **不要改 `net/sim/meld.py` 的枚举。** 它的契约是「每个 `(kind, size, rank)` 只返回一个代表」（见 `melds_from` 的 docstring）。`tools/accept_meld.py` 的验收①依赖这条口径，改了当场失效。Plan 2 照契约用。
- **不要改 `tools/accept_meld.py` 的四项验收口径。** 它是 Plan 1 的回归防线，每次动 `net/sim/` 必须仍然全绿。
- **牌 ID 一律用整数**（`net/cards.py`：`id = 花色基址 + 点数`，♠16 ♥32 ♣48 ♦64，A=1..K=13，小王 14 大王 15，第二副 +256）。**不出现牌名字符串**。
- **级别归一必须走 `meld.norm_level`**（日志里 A 有时写 14）。env 采样级别用 1..13，**不要**产生 14。
- **★ 座位轮转是 `(s - 1) % 4`，不是 `(s + 1) % 4`** —— 出牌顺序是 `0 → 3 → 2 → 1`；且**每出一手 `passed` 要清空**（一手打出 = 重新开一轮）。这两条都是 55 局实测定的（spec §13.7），按直觉写会让模拟器在真对局上大面积判「压不过桌面」。详见 Task 1 开头那段。
- **认不出的局面一律明着 raise，不许静默返回「合法」或「过」**（spec §6⑥）。
- **零点五奖励**：四个座位共享一套策略，每个决策点的回归目标是**该座位所在队**的收益，赢为正、输为负。
- 测试命令一律 `.venv/Scripts/python.exe -m pytest tests/ -q`。
- 提交信息用中文，格式跟现有历史一致（`feat: ...` / `fix: ...` / `test: ...`）。

## Review Focus

这五类输入/失败模式 spec 隐含要求处理，但没有哪个任务的测试天然覆盖 —— **最容易在真人用起来之后咬人**，每一条都在下面某个任务里配了测试：

1. **级别写成 14（= A）** —— `meld.norm_level` 只在 `meld.py` 内部调；`rules.py` / `env.py` 若自己比较级别（判逢人配、判级牌大小）就会把 A 当成小王，**整手牌的逢人配全错**，而且不报错。
   → `tests/test_rules_deal.py::test_level_14_is_rejected_not_silently_treated_as_a_joker`（Task 3）、`tests/test_rules_tribute.py::test_level_card_outranks_ace_when_deciding_the_biggest_card`（Task 4）
2. **手上同时有 2 张逢人配** —— 两张万能牌的组合空间是枚举最容易漏/重复的地方；漏了 = 模型永远看不到某个合法着法，重复了 = 动作空间被同一个着法占两格。
   → `tests/test_rules_turn.py::test_two_wilds_complete_a_triple_and_never_duplicate_a_candidate`（Task 1）
3. **某个座位只剩 1 张牌时领出 / 出完** —— 接风与清桌的分支只在这时候走到；判错就是「轮到已经出完的人」或死循环。
   → `tests/test_rules_turn.py::test_partner_inherits_the_lead_when_the_owner_goes_out`（Task 1）
4. **一局打到四家牌全出完** —— 局终条件若只写「3 家出完」，双上那 25/55 局会被多打几手；若写错方向则会提前终局、名次错。
   → `tests/test_rules_turn.py::test_double_win_ends_the_hand_with_two_seats_still_holding_cards` 与 `::test_three_seats_out_ends_the_hand_when_the_winners_are_not_first_and_second`（Task 1）
5. **状态编码若直接吃了内部 `hands`** —— 明牌泄漏，**最危险**：训练分数漂亮、真机全废，而且静默失效（同「合成 val 骗过一次」的教训）。必须靠自动化测试钉死，看代码看不出来。
   → 整个 `tests/test_env_leak.py`（Task 9）

## 明确不在本计划内（写下来免得执行时顺手扩大）

| 东西 | 为什么不在 |
|---|---|
| **replay buffer 50 万局**（spec §5.3） | 那是训练循环的组件。Task 11 的冒烟是**在线 batch**、只为了证 env 能训，**故意不做 buffer** —— 做了它就要写采样/淘汰逻辑，而那条逻辑本身不该在冒烟里被信任 |
| **§7 分水岭「胜率 vs 贪心」** | spec 明说是 Plan 3 的第一道关（要千万局量级） |
| **§6② 「谁压谁」逐条对齐、§6④ 逢人配专项** | **Plan 1 已交付**（`tools/accept_meld.py` 的 ② 与 ④，530 项 / 5 项全过）。本计划只保证每次都把它们跑绿，不重做 |
| **升级 / 过 A / 局级循环** | spec §13.1 已裁定不做 |
| **影子模式、推理链、面板接推荐**（spec §8） | Plan 3/4 |
| **把规则引擎改成 C++/Rust/numba** | Task 10 **只测量**。要不要换、换什么，是拿到数字之后的决定 |


---

## 文件结构

| 文件 | 责任 | 依赖 |
|---|---|---|
| `net/cards.py`（**改**） | 加一个 `slot(cid) -> 0..107`：牌 ID 到固定编码位的映射。**只加这一个函数**，别动现有的 | 无 |
| `net/sim/rules.py`（**新建**） | 一手牌的全部规则：牌堆 / 发牌 / 进贡还贡 / 出牌轮转（含接风）/ 结算名次 / 升级点 | `net/cards.py`、`net/sim/meld.py` |
| `net/sim/env.py`（**新建**） | 自对弈环境：状态编码（只含可观测）/ 动作编码 / `step` / reward / 自对弈 rollout | `rules.py`、`meld.py`、`numpy` |
| `tools/accept_sim.py`（**新建**） | 第二层验收：55 局真实记录回放 + 进贡 25 条逐条核对 | `rules.py`、`tools/game_log.py`、`tools/decision_points.py` |
| `tools/bench_sim.py`（**新建**） | 自对弈吞吐实测（spec §9 风险 1） | `env.py` |
| `train/smoke.py`（**新建**） | 最小 DMC 冒烟：训几分钟，验「胜率 vs 随机」 | `env.py`、`torch` |
| `tests/test_rules_*.py` / `tests/test_env_*.py` | 见各任务 | — |

**为什么 `rules.py` 和 `env.py` 要分开**：`rules.py` 是纯规则、明牌、可单独测；`env.py` 是「规则的什么部分能被策略看见」的策略层。混在一起时，「明牌泄漏」就变成一句口头约定而不是一条边界。

---

## Task 1: `net/cards.py` 加编码位 + `rules.py` 的轮转骨架

> ### ⚠️ 先读这一段：座位的「下家」是 `(s - 1) % 4`，不是 `(s + 1) % 4`
>
> **这是实测定出来的，不是推的。** 2026-09-25 拿 55 局真实对局逐手回放才定下来
> （见 Task 5 的验收①）。两件事与直觉相反：
>
> 1. **出牌顺序是 `0 → 3 → 2 → 1 → 0`**。报文的座位号不是按顺时针递增给的。
>    所以「下家」是 `NEXT[0] = 3`。**按 `(s+1)%4` 写，55 局一局都走不通。**
> 2. **每出一手，`passed` 要清空** —— 一手牌打出来等于**重新开一轮**，
>    其余三家重新获得机会（包括之前已经「要不起」过的）。不清空的话，
>    真人打过的牌会有一大半被判成「压不过桌面」。
>
> 这两条 `tools/decision_points.py` 一直没暴露出来，因为它只需要「谁出完了」这一个判据
> （用 `(table_seat + 2) % 4` 找队友，方向无关），从来没算过谁该接着出。

**Files:**
- Modify: `net/cards.py`（文件末尾追加）
- Create: `net/sim/rules.py`
- Test: `tests/test_cards_slot.py`, `tests/test_rules_turn.py`

**Interfaces:**
- Consumes: `net/cards.py` 的 `parts(cid)`、`is_card(cid)`、`decode(cid)`、`decode_all(ids)`；`net/sim/meld.py` 的 `melds_from(hand, level)`、`beats(a, b)`、`as_meld(ids, level)`、`describe_meld(m)`、`point_value(idx, level)`、`Meld`、`cid_from_name(name, deck)`
- Produces:
  - `cards.slot(cid) -> int`（0..107）、`cards.SLOTS = 108`
  - `rules.SEATS`、`rules.TEAM`、`rules.PARTNER`、`rules.NEXT`、`rules.IllegalPlay`
  - `rules.Step`（字段 `seat` / `meld` / `left`）
  - `rules.Hand`（字段 `hands: list[set]` / `level` / `turn` / `table` / `table_seat` / `passed` / `order` / `steps` / `over`）
  - `rules.Hand.actions(seat) -> list`、`rules.Hand.play(seat, meld) -> Step`、`rules.Hand.pass_turn(seat) -> Step`、`rules.Hand.is_over() -> bool`

- [ ] **Step 1: 写 `slot()` 的失败测试**

```python
# tests/test_cards_slot.py
from net import cards


def test_slot_is_a_bijection_over_the_108_cards():
    """两副牌正好 108 张，slot 必须一一对应 —— 编码位撞了就是静默丢牌。"""
    deck = [c for c in range(0, 334) if cards.is_card(c)]
    assert len(deck) == 108
    assert sorted(cards.slot(c) for c in deck) == list(range(108))


def test_slot_known_values():
    """钉住几张具体的牌，防止基址被改。"""
    from net.sim import meld
    A = meld.cid_from_name
    assert cards.slot(A("SA")) == 0             # 第一副 ♠A
    assert cards.slot(A("SK")) == 12            # 花色内按点数排
    assert cards.slot(A("HA")) == 13            # 花色是第一档分组
    assert cards.slot(A("DK", deck=2)) == 105   # 第二副整体 +54
    assert cards.slot(A("JOKER_S")) == 52       # 小王（第一副）
    assert cards.slot(A("JOKER_B", deck=2)) == 107
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cards_slot.py -q`
Expected: FAIL — `AttributeError: module 'net.cards' has no attribute 'slot'`

- [ ] **Step 3: 在 `net/cards.py` 末尾实现 `slot`**

```python
# ---------------------------------------------------------------- 编码位
#
# 给「把一手牌编码成定长向量」用的：两副牌 108 张 -> 0..107。
# 布局：每副 54 张 = ♠A..♠K(0..12) + ♥(13..25) + ♣(26..38) + ♦(39..51) + 小王(52) + 大王(53)，
# 第二副整体 +54。**花色是第一档分组**，这样同花色的牌落在连续区间里。

_SUIT_ORDER_4 = ("♠", "♥", "♣", "♦")
SLOTS = 108


def slot(cid: int) -> int:
    """牌 ID -> 0..107 的编码位。非法的牌 ID **直接炸**（不静默给个默认位）。"""
    if cid in _JOKER:
        name, deck = _JOKER[cid]
        return (deck - 1) * 54 + (52 if name == "小王" else 53)
    if not is_card(cid):
        raise ValueError(f"不是合法牌 ID：{cid}")
    deck = cid // 256
    low = cid % 256
    base = low // 16 * 16
    return deck * 54 + _SUIT_ORDER_4.index(_SUIT[base]) * 13 + (low - base - 1)
```

> ⚠️ `net/cards.py` 里现在是 `_JOKER` 字典（`{14: ("小王", 1), ...}`）。
> `slot` 用到的就是它，别新造一个。

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cards_slot.py -q`
Expected: `2 passed`

- [ ] **Step 5: 写轮转的失败测试**

```python
# tests/test_rules_turn.py
import pytest

from net import cards
from net.sim import meld, rules

A = meld.cid_from_name


def _hand(level=2, first=0, **seat_cards):
    """用手写牌面搭局面。`rules.Hand(hands=[...], level=..., turn=...)` 是唯一入口。"""
    hands = [set() for _ in range(4)]
    for name, ids in seat_cards.items():
        hands[int(name[1:])] = set(ids)
    return rules.Hand(hands=hands, level=level, turn=first)


def test_next_seat_is_the_previous_number_not_the_next_one():
    """**出牌顺序是 0 → 3 → 2 → 1**（55 局实测，见本任务开头）。

    这一条单独钉住，因为它是整个轮转的地基：写成 `(s+1)%4` 的话
    55 局真实对局**一局都走不通**。"""
    assert rules.NEXT == (3, 0, 1, 2)
    for s in rules.SEATS:
        assert rules.NEXT[rules.NEXT[rules.NEXT[rules.NEXT[s]]]] == s   # 四步回环


def test_leader_must_play_and_cannot_pass():
    h = _hand(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    assert None not in h.actions(0)                 # 领出不含「过」
    with pytest.raises(rules.IllegalPlay):
        h.pass_turn(0)


def test_follower_may_pass_even_when_able_to_beat():
    """**「能压也可以过」** —— HANDOFF 的开工前第 2 条只写了「压不过 = 只能过」，漏了这一半。
    掼蛋里跟牌的人永远可以过，哪怕手上压得过。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S5")], s3=[A("S4"), A("H4")])
    h.play(0, meld.as_meld([A("S3")], 2))
    assert h.turn == 3                              # 下家是 3，不是 1
    acts = h.actions(3)
    assert None in acts                             # 「过」永远在
    assert any(m is not None and set(m.cards) == {A("S4")} for m in acts)   # 4♠ 单张压得过 3♠
    # 对子压不过单张（牌型不同），所以它不该在候选里
    assert not any(m is not None and set(m.cards) == {A("S4"), A("H4")} for m in acts)


def test_when_nothing_beats_the_table_actions_is_exactly_pass():
    """`meld.legal_moves` 压不过时返回 `[]` —— env 必须读成「只能过」。
    **这是 HANDOFF「Plan 2 开工前第 2 条」要钉住的那件事。**

    注意别拿 2 当小牌：**打 2 的时候 2 是级牌，比 A 还大**。所以这里用 4♠（非级牌）
    去面对 A♠，才是真的压不过。"""
    h = _hand(s0=[A("SA")], s1=[A("S6")], s2=[A("S5")], s3=[A("S4")])
    h.play(0, meld.as_meld([A("SA")], 2))
    assert h.turn == 3
    assert h.actions(3) == [None]


def test_play_rejects_illegal_shape_and_missing_cards():
    h = _hand(s0=[A("S3"), A("S4")], s1=[A("S6")], s2=[A("S5")], s3=[A("H7")])
    with pytest.raises(rules.IllegalPlay):
        h.play(0, meld.as_meld([A("H9")], 2))       # 手上没有这张
    with pytest.raises(rules.IllegalPlay):
        h.play(0, meld.Meld(kind=meld.STRAIGHT, size=5, rank=3,
                            cards=(A("S3"), A("S4"))))   # 牌面根本不是顺子


def test_wrong_seat_cannot_act():
    h = _hand(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    with pytest.raises(rules.IllegalPlay):
        h.play(3, meld.as_meld([A("S6")], 2))


def test_a_new_play_reopens_the_round_for_everyone():
    """**每出一手，`passed` 清空** —— 这一手等于重新开一轮，三家重新有机会。

    实测依据：不清空的话，55 局里只有 4 局能走通（38 局会在中途被判成「压不过桌面」）。
    """
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S5")], s3=[A("S4")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.pass_turn(3)
    assert h.passed == {3}
    h.play(2, meld.as_meld([A("S5")], 2))           # 座位 2 压过 —— 新一轮
    assert h.passed == set(), "出了一手之后 passed 必须清空"
    assert h.actions(3) is not None                 # 座位 3 重新获得出手机会


# ---- 下面四条对应 Review Focus 的第 2、3、4 条
#      （spec 隐含要求，但没有哪一处的测试天然覆盖到）

def test_two_wilds_complete_a_triple_and_never_duplicate_a_candidate():
    """**Review Focus #2：手上同时有 2 张逢人配。**

    两件事必须成立：① 候选里**没有完全重复**的着法（重复 = 同一个着法占两格，
    它的 Q 值被两个样本分别训练）；② 两张逢人配确实能凑出「三个 9」
    （漏了 = 模型永远看不到某个合法着法）。"""
    wild = [A("H2"), A("H2", deck=2)]          # 打 2 时红桃 2 就是逢人配
    h = _hand(s0=wild + [A("S9"), A("S4"), A("S5")],
              s1=[A("S6")], s2=[A("S7")], s3=[A("S8")])
    acts = h.actions(0)
    keys = [(m.kind, tuple(sorted(m.cards))) for m in acts]
    assert len(keys) == len(set(keys)), "候选里出现了完全重复的着法"
    triples = [m for m in acts
               if m.kind == meld.TRIPLE and m.rank == meld.point_value(9, 2)]
    assert triples, "两张逢人配 + 一张 9♠ 应该能凑出「三个 9」"
    assert triples[0].wild_used == 2


def test_partner_inherits_the_lead_when_the_owner_goes_out():
    """**Review Focus #3：只剩 1 张牌的座位出完后，接风给队友。**

    判据是「**主人已经出完**」，不是服务器给的 `NextTurnSeatID` ——
    后者会跳过出完的座位，实测按它判会误清 34 手、漏清 27 手
    （`tools/decision_points.py` 的 docstring 记了这次实测）。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S5")], s3=[A("S2")])
    h.play(0, meld.as_meld([A("S3")], 2))       # 座位 0 出完
    assert h.order == [0]
    assert h.turn == 3                          # 下家是 3
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)
    assert h.table is None, "三家都要不起 -> 必须清桌"
    assert h.turn == 2, "**接风**：主人出完了，领出权给他的队友（座位 2）"


def test_double_win_ends_the_hand_with_two_seats_still_holding_cards():
    """**Review Focus #4：双上立即终局（实测 25/25 局：只出完 2 家）。**

    若把局终条件写成「3 家出完」，这 25 局会被多打几手、名次也可能错。"""
    h = _hand(s0=[A("S3")], s1=[A("S9")], s2=[A("S5")], s3=[A("ST")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)
    assert h.turn == 2
    h.play(2, meld.as_meld([A("S5")], 2))       # 队友也出完 -> 双上
    assert h.is_over(), "同一队包了前两名必须**立即**终局"
    assert h.order == [0, 2]                    # 名次怎么算在 Task 2 里验


def test_three_seats_out_ends_the_hand_when_the_winners_are_not_first_and_second():
    """**Review Focus #4 的另一半**：1、3 名 / 1、4 名时要打到 **3 家出完**（实测 30/30 局）。

    出完顺序按真实轮转（0 → 3 → 2），所以这里 0 是 1 名、3 是 2 名、2 是 3 名
    —— 1、3 名归座位 0 与 2，是「1、3 形态」。"""
    h = _hand(s0=[A("S3")], s1=[A("S6")], s2=[A("S2")], s3=[A("SA")])
    h.play(0, meld.as_meld([A("S3")], 2))
    h.play(3, meld.as_meld([A("SA")], 2))       # 轮转里座位 3 接着来
    assert not h.is_over(), "只出完 2 家、且分属两队 -> 还没完"
    h.play(2, meld.as_meld([A("S2")], 2))       # 打 2 时 2 是级牌，压得过 A
    assert h.is_over(), "出完 3 家必须终局"
    assert h.order == [0, 3, 2]                 # 名次与升级点在 Task 2 里验
```

- [ ] **Step 6: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_turn.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'net.sim.rules'`

- [ ] **Step 7: 实现 `net/sim/rules.py`**

```python
"""掼蛋**一手牌**的规则引擎 —— 发牌 / 进贡还贡 / 出牌(含接风) / 结算名次。

**一手牌一个 episode**（spec §13.1）：名次出来即终止。**不实现升级 / 过 A**，
级别（打几）由调用方给，本模块不推 —— 一手一个 episode 时升级没有消费者。

内部是**明牌**（四家手牌都知道，否则判不了输赢）。**喂给策略的状态由
`env.py` 负责裁剪**（spec §3「不许明牌泄漏」）。本模块不知道策略存在，也不该知道。

## 两条由 55 局实测定下来的轮转规则（别按直觉改）

**①「下家」是 `NEXT[s] = (s - 1) % 4`，出牌顺序是 `0 → 3 → 2 → 1 → 0`。**
报文的座位号不是按顺时针递增给的。2026-09-25 拿 55 局真实对局逐手回放才定下来：
按 `(s+1)%4` 写，**55 局一局都走不通**；按 `NEXT` 写，55/55 全过。
（`tools/decision_points.py` 一直没暴露这条，因为它只需要 `(table_seat + 2) % 4`
找队友，方向无关。）

**② 每出一手，`passed` 清空 —— 一手牌打出来等于重新开一轮。**
其余三家重新获得机会，包括之前已经「要不起」过的。不清空的话 55 局里只有 4 局能走通。

**接风**：一圈扫不到人接手时清桌；清桌后如果桌面主人**已经出完**，领出权给他的队友
（队友也出完了就顺着 `NEXT` 找下一个还有牌的）。**不要用服务器的 `NextTurnSeatID` 推**
—— 它会跳过出完的座位，实测按它判会误清 34 手、漏清 27 手。

**局终条件**（2026-09-25 在 55 局上核出，spec §13.5）：
    同一队包了前两名（双上） -> **只出完 2 家就终局**   25/25 局
    否则                     -> 出完 3 家才终局        30/30 局
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from net import cards
from net.sim import meld

SEATS = (0, 1, 2, 3)
TEAM = (0, 1, 0, 1)          # TEAM[s]：座位 0、2 一队，1、3 一队
PARTNER = (2, 3, 0, 1)       # 队友（在 NEXT 环上隔一个：0↔2、1↔3）

#: **下家**。出牌顺序 0 → 3 → 2 → 1（55 局实测，见模块 docstring ①）。
NEXT = (3, 0, 1, 2)


class IllegalPlay(Exception):
    """不合法的着法。**明着炸**，不静默吞（spec §6⑥）。"""


@dataclass
class Step:
    seat: int
    meld: Optional[meld.Meld]      # None = 要不起（过）
    left: int                      # 这一步之后该家还剩几张


@dataclass
class Hand:
    hands: List[Set[int]]
    level: Optional[int] = None
    turn: int = 0
    table: Optional[meld.Meld] = None
    table_seat: Optional[int] = None
    passed: Set[int] = field(default_factory=set)
    order: List[int] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    over: bool = False

    # ------------------------------------------------------------ 查询

    def actions(self, seat: int) -> list:
        """该座位的可行动作。**`None` 表示「过」。**

        - **领出**（桌上无牌）：只有着法，**不含 `None`** —— 领出必须出牌
        - **跟牌**：压得过的着法 **∪ {None}**。
          `None` 在跟牌时**永远**在集合里（能压也可以过）；
          压不过时集合是 `[None]` 而不是空 —— 这是 HANDOFF「Plan 2 开工前第 2 条」要钉的语义。
        """
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        moves = meld.melds_from(sorted(self.hands[seat]), self.level)
        if self.table is None:
            if not moves:
                raise IllegalPlay(
                    f"座位{seat} 手上还有 {len(self.hands[seat])} 张牌，却枚举不出任何着法 —— "
                    f"枚举漏了（spec §6③「领出时合法着法集合非空」）")
            return moves
        return [m for m in moves if meld.beats(m, self.table)] + [None]

    # ------------------------------------------------------------ 行动

    def play(self, seat: int, m: Optional[meld.Meld]) -> Step:
        if m is None:
            return self.pass_turn(seat)
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        # 从牌面重新判一次，**不用调用方给的 Meld** —— 牌型必须由牌面自己说了算。
        # `as_meld` 取最强解释：一手同花连续的牌同时是顺子与同花顺，游戏叫它同花顺
        # （`meld.strongest` 的 docstring 记了这条为什么只能有一份实现）。
        judged = meld.as_meld(list(m.cards), self.level)
        if judged is None:
            raise IllegalPlay(f"不是合法牌型：{cards.decode_all(list(m.cards))}")
        cs = set(judged.cards)
        if not cs <= self.hands[seat]:
            raise IllegalPlay("座位%d 打出了手上没有的牌 %s"
                              % (seat, cards.decode_all(sorted(cs - self.hands[seat]))))
        if self.table is not None and not meld.beats(judged, self.table):
            raise IllegalPlay("压不过桌面：%s vs %s"
                              % (meld.describe_meld(judged), meld.describe_meld(self.table)))

        self.hands[seat] -= cs
        if not self.hands[seat] and seat not in self.order:
            self.order.append(seat)
        # **一手打出来 = 重新开一轮**：其余三家重新获得机会（含之前「要不起」过的）。
        # 不清空的话 55 局真实对局里只有 4 局能走通 —— 见模块 docstring ②。
        self.passed = set()
        self.table = judged
        self.table_seat = seat
        st = Step(seat, judged, len(self.hands[seat]))
        self.steps.append(st)
        self._advance()
        return st

    def pass_turn(self, seat: int) -> Step:
        if self.over:
            raise IllegalPlay("这一手已经结束了")
        if self.turn != seat:
            raise IllegalPlay(f"现在轮到 {self.turn}，不是 {seat}")
        if self.table is None:
            raise IllegalPlay(f"座位{seat} 是领出，不能过（桌上没牌，必须出牌）")
        self.passed.add(seat)
        st = Step(seat, None, len(self.hands[seat]))
        self.steps.append(st)
        self._advance()
        return st

    # ------------------------------------------------------------ 轮转

    def _advance(self) -> None:
        """一步之后定下一个该谁。清桌与接风都在这里发生。"""
        if self.is_over():
            self.over = True
            return
        nxt = None
        s = NEXT[self.table_seat]
        for _k in range(4):
            if self.hands[s] and s not in self.passed:
                nxt = s
                break
            s = NEXT[s]
        if nxt is not None and nxt != self.table_seat:
            self.turn = nxt
            return
        # 一圈扫不到人接手（或扫回主人自己）-> 清桌、重新领出
        self.table = None
        self.passed = set()
        if self.hands[self.table_seat]:
            self.turn = self.table_seat                       # 他自己重新领出
        else:
            self.turn = self._lead_after(self.table_seat)     # 接风
        self.table_seat = None

    def _lead_after(self, seat: int) -> int:
        """`seat` 出完了，领出权给谁：**队友优先（接风）**，队友也出完了就顺着 `NEXT` 找。"""
        p = PARTNER[seat]
        if self.hands[p]:
            return p
        s = NEXT[seat]
        for _k in range(4):
            if self.hands[s]:
                return s
            s = NEXT[s]
        return -1   # 全出完了；is_over() 会先拦下，走到这里说明局终判错了

    def is_over(self) -> bool:
        """**局终条件（spec §13.5）**：同一队包了前两名立即终局，否则等 3 家出完。"""
        if len(self.order) == 2 and TEAM[self.order[0]] == TEAM[self.order[1]]:
            return True
        return len(self.order) == 3
```

- [ ] **Step 8: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_turn.py tests/test_cards_slot.py -q`
Expected: 全部 passed（`test_rules_turn.py` 10 条 + `test_cards_slot.py` 2 条）

- [ ] **Step 9: 跑 Plan 1 的回归防线，确认没碰坏**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`，再 `.venv/Scripts/python.exe -m tools.accept_meld`
Expected: `183 passed`（171 + 12）、`验收全绿 ✓`

- [ ] **Step 10: 提交**

```bash
git add net/cards.py net/sim/rules.py tests/test_cards_slot.py tests/test_rules_turn.py
git commit -m "feat: rules.py 轮转骨架（下家=(s-1)%4、出手重开一轮、接风）+ cards.slot"
```

---

## Task 2: 结算名次、升级点、零点五奖励

**Files:**
- Modify: `net/sim/rules.py`（追加）
- Test: `tests/test_rules_settle.py`

**Interfaces:**
- Consumes: Task 1 的 `rules.Hand` / `rules.TEAM`
- Produces:
  - `rules.Hand.ranks() -> list[int]`（`ranks[seat] = 1..4`）
  - `rules.winner_team(ranks) -> int`
  - `rules.POINTS_BY_WORST_RANK = {2: 3, 3: 2, 4: 1}`
  - `rules.points(ranks) -> int`
  - `rules.reward(ranks, seat) -> float`
  - `rules.MULTIPLIER = 1.0`（模块级开关，见下方注释）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rules_settle.py
import pytest

from net import cards
from net.sim import meld, rules

A = meld.cid_from_name

A = meld.cid_from_name


def _finish(order):
    """造一个已经打完的局面：order 是出完顺序，剩下的人按剩牌数排。"""
    h = rules.Hand(hands=[set() for _ in range(4)], level=2, turn=0)
    h.order = list(order)
    h.over = True
    return h


def test_ranks_follow_finish_order_then_leftover_cards():
    # 座位 2、0 出完（双上），1 剩 1 张、3 剩 5 张 -> 1 是第 3 名
    h = _finish([2, 0])
    h.hands[1] = {A("S3")}
    h.hands[3] = {A("S4"), A("S5"), A("S6"), A("S7"), A("S8")}
    assert h.ranks() == [2, 3, 1, 4]


def test_ranks_tiebreak_is_stable_by_seat():
    """剩牌同数时按座位小者排前 —— 只是为了让结果可复现。
    **这样定出来的 3/4 名对 reward 无影响**（双上时败方谁 3 谁 4 不改变形态），
    实测 25 局里有 2 局与「剩牌少者排前」不符，见 spec §13.5。"""
    h = _finish([2, 0])
    h.hands[1] = {A("S3"), A("S4")}
    h.hands[3] = {A("S5"), A("S6")}
    assert h.ranks() == [2, 3, 1, 4]


def test_ranks_after_a_real_double_win():
    """走完一整段真实轮转再算名次 —— 上面 `_finish` 那几条是手搭局面，
    这一条验的是「rules 自己走出来的局，名次算得对」。"""
    h = rules.Hand(hands=[{A("S3")}, {A("S9")}, {A("S5")}, {A("ST")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))       # 座位 0 出完
    h.pass_turn(3); h.pass_turn(2); h.pass_turn(1)   # 三家要不起 -> 接风给队友
    assert h.turn == 2
    h.play(2, meld.as_meld([A("S5")], 2))       # 队友也出完 -> 双上，立即终局
    assert h.is_over() and h.order == [0, 2]
    assert h.ranks() == [1, 3, 2, 4]
    assert rules.points(h.ranks()) == 3          # 座位 0、2 包了前两名 -> +3


def test_ranks_after_a_real_three_seat_finish():
    """出完 3 家、赢家是 1、3 名 -> 升级点 2（实测 30/30 局是这种 3 出完的形态）。"""
    h = rules.Hand(hands=[{A("S3")}, {A("S6")}, {A("S2")}, {A("SA")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))
    h.play(3, meld.as_meld([A("SA")], 2))       # 真实轮转里座位 3 接着来
    assert not h.is_over()
    h.play(2, meld.as_meld([A("S2")], 2))       # 打 2 时 2 是级牌，压得过 A
    assert h.is_over() and h.order == [0, 3, 2]
    assert h.ranks() == [1, 4, 3, 2]
    assert rules.points(h.ranks()) == 2


def test_ranks_requires_the_hand_to_be_over():
    h = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    with pytest.raises(rules.IllegalPlay):
        h.ranks()


@pytest.mark.parametrize("ranks,expect", [
    ([1, 3, 2, 4], 3),      # 座位 0、2 是 1、2 名 = 双上 -> +3
    ([1, 4, 2, 3], 3),      # 换一种写法，还是双上
    ([1, 2, 3, 4], 2),      # 座位 0、2 是 1、3 名 -> +2
    ([1, 2, 4, 3], 1),      # 座位 0、2 是 1、4 名 -> +1
    ([4, 1, 3, 2], 3),      # 座位 1、3 双上（名次 1、2）
    ([2, 1, 3, 4], 1),      # 座位 1、3 是 1、4 名
])
def test_points(ranks, expect):
    assert rules.winner_team(ranks) in (0, 1)
    assert rules.points(ranks) == expect


def test_reward_is_zero_sum_from_the_seat_perspective():
    """四个座位共享一套策略 -> 奖励必须从**出牌人所在队**的视角看，且零和。
    （斗地主式 DMC 的前提；视角搞错会让同一局给四个座位同一个目标。）"""
    ranks = [1, 3, 2, 4]                       # 0、2 队赢，双上
    assert rules.reward(ranks, 0) == 3 and rules.reward(ranks, 2) == 3
    assert rules.reward(ranks, 1) == -3 and rules.reward(ranks, 3) == -3
    assert sum(rules.reward(ranks, s) for s in rules.SEATS) == 0
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_settle.py -q`
Expected: FAIL — `AttributeError: type object 'Hand' has no attribute 'ranks'`

- [ ] **Step 3: 实现（追加到 `net/sim/rules.py`）**

```python
    # ------------------------------------------------------------ 结算

    def ranks(self) -> List[int]:
        """`ranks[seat] = 1..4`。出完的按出完顺序 1..k；没出完的按**剩牌少者排前**，
        同数按座位小者排前（只为可复现）。

        双上局里败方那两个人的 3/4 名是**服务器说了算的**：实测 25 局里 23 局
        「剩牌少者第 3」、2 局相反（spec §13.5）。**这两种定法对 reward 无影响**
        —— 双上时败方谁 3 谁 4 不改变形态，所以不为此加规则。
        """
        if not self.is_over():
            raise IllegalPlay("局还没终，名次还没定 —— 不要中途算名次")
        rest = [s for s in SEATS if s not in self.order]
        rest.sort(key=lambda s: (len(self.hands[s]), s))
        ranks = [0] * 4
        for i, s in enumerate(list(self.order) + rest, 1):
            ranks[s] = i
        return ranks
```

```python
# ---------------------------------------------------------------- 结算与奖励

#: 赢家那一队**较差的名次** -> 升级点。双上 = 前两名都被包 = 较差名次是 2。
#:
#: ⚠️ 「双上 -> 3」只在数据上对了 13/24 局，另 11 局实测是 **4**（spec §13.4）。
#:    同一 `Rank`、同一战前 `trump` 都能出 3 和 4，说明还依赖升级/过A 的字段 ——
#:    按 §13.1 不在 Plan 2 内。这里取 3（用户口述 + spec §5.4），
#:    **由 `tools/accept_sim.py` 把 3/4 的分布单独报出来**，不当成验收失败。
POINTS_BY_WORST_RANK = {2: 3, 3: 2, 4: 1}

#: reward 的倍数开关。**默认 1.0，即不带倍数。**
#:
#: spec §5.4 说「倍数必须带上 —— 打炸弹会抬倍数」，但 54 局实测**不支持这个前提**：
#: `TotalBombRatio` 在 49/54 局是 1（其中 36 局手上有 5~16 个炸弹/同花顺），
#: 真正的倍数是 `FinalDoubleRatio ∈ {1, 1.5, 2, 2.5}`，那是**发牌前选的加倍**，
#: 一手之内不变 ⇒ 对最优策略零影响（只有一手内变化的倍数才会改变打法）。
#: 详见 spec §13.4。Plan 3 想按 `FinalDoubleRatio` 分布采样时改这一个值即可。
MULTIPLIER = 1.0


def winner_team(ranks) -> int:
    """赢家队：两名队员里**较好的名次**更靠前的那一队。"""
    a = min(ranks[0], ranks[2])
    b = min(ranks[1], ranks[3])
    if a == b:
        raise IllegalPlay(f"两队最好名次相同（ranks={list(ranks)}），不可能是合法结算")
    return 0 if a < b else 1


def points(ranks) -> int:
    """赢家这一手的升级点：双上 3 / 1、3 名 2 / 1、4 名 1。"""
    t = winner_team(ranks)
    worst = max(ranks[0], ranks[2]) if t == 0 else max(ranks[1], ranks[3])
    return POINTS_BY_WORST_RANK[worst]


def reward(ranks, seat: int, multiplier: float = None) -> float:
    """**从 `seat` 视角**的零点五奖励 —— 四个座位共享一套策略，必须零和。

    ⚠️ 视角是**出牌人**，不是「我方固定座位」。喂给网络的状态必须先相对化
    （自己 / 下家 / 对家 / 上家），否则这一条会被悄悄用错。
    """
    m = MULTIPLIER if multiplier is None else multiplier
    p = points(ranks) * m
    return p if TEAM[seat] == winner_team(ranks) else -p
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_settle.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add net/sim/rules.py tests/test_rules_settle.py
git commit -m "feat: 结算名次 + 升级点 + 零点五奖励（倍数默认关，实测依据见 spec 13.4）"
```

---

## Task 3: 牌堆、发牌、`new_hand`

**Files:**
- Modify: `net/sim/rules.py`（追加）
- Test: `tests/test_rules_deal.py`

**Interfaces:**
- Consumes: Task 1/2 的 `rules.Hand`
- Produces:
  - `rules.FULL_DECK`（108 张牌 ID 的元组）
  - `rules.deal(rng, level, first=None) -> Hand`
  - `rules.new_hand(rng, level, hands=None, first=None, prev_ranks=None) -> Hand`（`hands` 给定时直接用，供回放真实对局）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rules_deal.py
import random
import collections

import pytest

from net import cards
from net.sim import rules


def test_full_deck_is_two_complete_decks():
    assert len(rules.FULL_DECK) == 108
    assert len(set(rules.FULL_DECK)) == 108                 # 没有重复 ID
    idx = collections.Counter(cards.parts(c)[0] for c in rules.FULL_DECK)
    assert all(n == 8 for k, n in idx.items() if k not in (14, 15))   # 每个点数两副共 8 张
    assert idx[14] == 2 and idx[15] == 2                     # 大小王各 2 张


@pytest.mark.parametrize("seed", range(20))
def test_deal_gives_everyone_27_distinct_cards_and_conserves_108(seed):
    rng = random.Random(seed)
    h = rules.deal(rng, level=2)
    sizes = sorted(len(h.hands[s]) for s in rules.SEATS)
    assert sizes == [27, 27, 27, 27]
    union = set().union(*[h.hands[s] for s in rules.SEATS])
    assert len(union) == 108                                 # 同一张牌不许出现在两家


def test_new_hand_with_explicit_hands_keeps_them_verbatim():
    """回放真实对局时要能手给四家牌 —— 这时**不许**洗牌。"""
    hands = [{1, 2, 3}, {4, 5}, set(), {6}]
    h = rules.new_hand(random.Random(0), level=7, hands=hands, first=2)
    assert [set(x) for x in h.hands] == [set(x) for x in hands]
    assert h.level == 7 and h.turn == 2


def test_level_14_is_rejected_not_silently_treated_as_a_joker():
    """日志里 A 有时写 14。**归一必须走 `meld.norm_level`**，
    绝不能让 14 顺着接口流进来 —— 那会被 `cards.parts` 当成小王。
    `new_hand` 是唯一入口，所以在这里拦。"""
    with pytest.raises(ValueError):
        rules.new_hand(random.Random(0), level=14, hands=[{1}] * 4)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_deal.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.rules' has no attribute 'FULL_DECK'`

- [ ] **Step 3: 实现（追加到 `net/sim/rules.py`）**

```python
# ---------------------------------------------------------------- 牌堆与发牌

#: 完整牌堆：两副牌 108 张的牌 ID。**顺序固定**（`range` 升序），
#: 这样给定 seed 的发牌结果永远一样 —— 训练要可复现。
FULL_DECK = tuple(c for c in range(0, 334) if cards.is_card(c))

_DEAL_EACH = 27                     # 两副牌 108 / 4 家


def _check_level(level) -> None:
    """**级别 14 不许顺着接口流进来**（spec Global Constraints）。

    日志里 A 有时写 14。`cards.parts(14)` 会把 14 当成**小王** —— 于是级牌判定、
    逢人配判定全错，而且一声不响。归一只有 `meld.norm_level` 一处，接口上拦住
    比在内部到处归一安全。
    """
    if level is None:
        return
    if not isinstance(level, int) or not 1 <= level <= 13:
        raise ValueError(
            f"级别必须在 1..13（A=1）。收到 {level!r} —— "
            f"14 是日志里 A 的另一种写法，请先过 meld.norm_level()")


def deal(rng, level=None, first=None) -> "Hand":
    """洗牌发牌，每家 27 张。`first` 不给就随机定领出者。"""
    _check_level(level)
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    hands = [set(deck[i * _DEAL_EACH:(i + 1) * _DEAL_EACH]) for i in SEATS]
    return Hand(hands=hands, level=level,
                turn=rng.randrange(4) if first is None else first)


def new_hand(rng, level=None, hands=None, first=None, prev_ranks=None) -> "Hand":
    """建一手牌。**这是唯一的入口** —— `Hand(...)` 直接构造只允许出现在测试里。

    - `hands` 给定：直接用（**不洗牌**），供回放真实对局
    - `hands=None`：洗牌发牌
    - `prev_ranks` 给定：按上一手的名次走**进贡/还贡**（Task 4 实现；本任务先只记着）
    """
    _check_level(level)
    if hands is None:
        h = deal(rng, level=level, first=first)
    else:
        h = Hand(hands=[set(x) for x in hands], level=level,
                 turn=rng.randrange(4) if first is None else first)
    return h
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_deal.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add net/sim/rules.py tests/test_rules_deal.py
git commit -m "feat: 牌堆 / 发牌 / new_hand（级别 14 在接口上拦住）"
```

---

## Task 4: 进贡 / 还贡 / 抗贡

> **这一节的每一格都要标清「有证据」还是「基线」。** 证据薄的格子不是不能实现，
> 是**不许当成已验证** —— `tools/accept_sim.py`（Task 6）会把它们逐条报出来。

**Files:**
- Modify: `net/sim/rules.py`（追加）
- Test: `tests/test_rules_tribute.py`

**Interfaces:**
- Consumes: Task 1~3 的 `rules.Hand`；`cards.sort_ids`；`meld.point_value(idx, level)`
- Produces:
  - `rules.Tribute`（`kind: str`、`gave: dict`、`returned: dict`、`leader: int`）
  - `rules.tributers(prev_ranks) -> list[int]`
  - `rules.apply_tribute(hand, prev_ranks) -> Tribute`（**就地**改 `hand.hands`，并把 `hand.turn` 设成先出者）
  - `rules.RETURN_MIN_RANK = 2`、`rules.RETURN_MAX_RANK = 10`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rules_tribute.py
import pytest

from net import cards
from net.sim import meld, rules

A = meld.cid_from_name
BIG1, BIG2 = A("JOKER_B"), A("JOKER_B", deck=2)          # 两张大王


def _h(level=2, **seat_cards):
    hands = [set() for _ in range(4)]
    for name, ids in seat_cards.items():
        hands[int(name[1:])] = set(ids)
    return rules.Hand(hands=hands, level=level, turn=0)


def test_tributers_single_when_the_two_losers_are_on_different_teams():
    # 名次：座位1=1、2=2、0=3、3=4 -> 第 3(座位0)、4(座位3) 名**不同队** -> 单贡
    assert rules.tributers([3, 1, 2, 4]) == [3]


def test_tributers_double_when_both_losers_are_one_team():
    # 名次：座位0=1、2=2、3=3、1=4 -> 第 3、4 名都是 1/3 队 -> 双贡
    assert sorted(rules.tributers([1, 4, 2, 3])) == [1, 3]


def test_single_tribute_gives_the_biggest_card_and_returns_the_smallest_ten_or_under():
    # 名次：座位2=1、1=2、0=3、3=4 -> 第 3、4 名不同队 -> 单贡，座位 3 贡给座位 2
    h = _h(s0=[A("S9")], s1=[A("S8")], s2=[A("S3"), A("SK"), A("H5")], s3=[A("SA"), A("D4")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert t.kind == "single"
    assert t.gave == {3: A("SA")}                  # 贡手上最大的牌
    assert t.returned == {2: A("S3")}              # 还 2..10 里最小的
    assert A("SA") in h.hands[2] and A("S3") in h.hands[3]
    assert t.leader == 2                           # 头游先出（基线，见实现里的注释）


def test_ace_is_not_a_valid_return_even_though_its_rank_index_is_1():
    """**「≤10」是点数 2..10，不是「索引 ≤10」。**

    牌 ID 体系里 A=1，所以 `idx <= 10` 会把 **A 也算成「≤10」** ——
    而 A 恰恰是除王与级牌之外最大的牌，等于白送。

    ⚠️ 这里**必须用 level=5**：打 2 的时候 2♦ 自己是级牌，`_smallest_returnable`
    会把它排到最后，测不出 A 的问题。打 5 时 2♦ 是普通小牌，才是干净的对照。
    """
    h = _h(level=5, s0=[A("S9")], s1=[A("S8")],
           s2=[A("SA"), A("SK"), A("HJ"), A("D2")], s3=[A("HA"), A("D4")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert cards.parts(t.returned[2])[0] == 2      # 只有 2♦ 合格
    assert cards.parts(t.returned[2])[0] != 1      # **A 绝不能被当成「≤10」还出去**


def test_resist_when_the_losing_team_holds_two_big_jokers():
    """抗贡判据：**输的那一队**手上合计 ≥2 张大王（不是只看进贡的那一个人）。

    证据：日志里 `TributeSectionStatrt seatId:1|大王| seatId:3|大王|` 出现 9 次，
    每次都是「同队两个座位各持一张大王」，**单贡时也只报这两个座位** ——
    说明判的是**队**而不是进贡的那个人。"""
    h = _h(s0=[A("S5")], s1=[BIG1, A("S3")], s2=[A("S6")], s3=[BIG2, A("S4")])
    t = rules.apply_tribute(h, prev_ranks=[1, 3, 2, 4])      # 1、3 是 3、4 名，同队
    assert t.kind == "resist" and t.gave == {} and t.returned == {}
    assert h.hands[1] == {BIG1, A("S3")} and h.hands[3] == {BIG2, A("S4")}   # 牌没动
    assert t.leader == 0


def test_level_card_outranks_ace_when_deciding_the_biggest_card():
    """**打 2 的时候 2 是级牌，比 A 大。** 自己比较点数会在这里栽跟头 ——
    所以取值必须走 `meld.point_value`。"""
    h = _h(s0=[A("S9")], s1=[A("S8")], s2=[A("S3")], s3=[A("SA"), A("D2")])
    t = rules.apply_tribute(h, prev_ranks=[3, 2, 1, 4])
    assert t.gave == {3: A("D2")}


def test_no_tribute_when_there_is_no_previous_hand():
    h = _h(s0=[A("S3")], s1=[A("S4")], s2=[A("S5")], s3=[A("S6")])
    t = rules.apply_tribute(h, prev_ranks=None)
    assert t.kind == "none" and t.gave == {} and t.returned == {}
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_tribute.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.rules' has no attribute 'tributers'`

- [ ] **Step 3: 实现（追加到 `net/sim/rules.py`）**

```python
# ---------------------------------------------------------------- 进贡 / 还贡
#
# 证据分级（**不许把「基线」当成「已验证」**）：
#
#   [硬证据 25/25]  还贡 ≤ 10 —— 25 条 `NotifyReturnTribute` / `TributeSectionEndService`
#                   记录里 `card=` 解码后点数**全部落在 2..10**（跨 4 花色、2 副牌）。spec §13.6
#   [证据 9 条]     抗贡判**队**不判人 —— `TributeSectionStatrt seatId:N|大王|` 出现 9 次，
#                   总是同队两个座位各一张大王，单贡时也只报这两个座位
#   [基线，待验收]  贡「最大的牌」、双贡怎么配对、进贡后谁先出 —— 只有 5 条直接记录，
#                   且我的对齐脚本不可靠（spec §13.6）。按通行规则实现，
#                   **由 `tools/accept_sim.py` 逐条报差分**。

#: 还贡允许的点数区间（闭区间）。**A=1 不在里面** —— 25/25 条硬证据。
RETURN_MIN_RANK, RETURN_MAX_RANK = 2, 10


@dataclass
class Tribute:
    kind: str                  # "none" / "single" / "double" / "resist"
    gave: dict                 # {进贡方座位: 牌}
    returned: dict             # {受贡方座位: 牌}
    leader: int                # 进贡阶段结束后先出的座位


def tributers(prev_ranks) -> List[int]:
    """谁要进贡：上一手的第 4 名；若第 3、4 名**同队**则两人都要（双贡）。"""
    last = [s for s in SEATS if prev_ranks[s] == 4]
    third = [s for s in SEATS if prev_ranks[s] == 3]
    if len(last) != 1 or len(third) != 1:
        raise IllegalPlay(f"上一手名次不合法：{list(prev_ranks)}")
    if TEAM[last[0]] == TEAM[third[0]]:
        return sorted([last[0], third[0]])
    return [last[0]]


def _value(cid: int, level) -> int:
    """这张牌的掼蛋大小（越大越强）。

    **必须走 `meld.point_value`**：打 2 的时候 2 是级牌、比 A 大，
    自己拿 `cards.parts(...)[0]` 比大小会在这里栽跟头（而且不报错）。
    """
    return meld.point_value(cards.parts(cid)[0], level)


def _biggest(hand, level) -> int:
    return max(sorted(hand), key=lambda c: (_value(c, level), c))


def _smallest_returnable(hand, level) -> int:
    """还贡：**2..10 里最小的那张**（硬证据只钉住「≤10」；给哪一张是受贡方的选择，
    25 条里不唯一 —— 这里定死成最小的，图可复现）。

    万一一张 2..10 都没有（27 张全是 J/Q/K/A/王，理论上可能），退化成「手上最小的」
    并**留下痕迹**，不静默。
    """
    pool = [c for c in hand if RETURN_MIN_RANK <= cards.parts(c)[0] <= RETURN_MAX_RANK]
    if not pool:
        pool = list(hand)
    return min(sorted(pool), key=lambda c: (_value(c, level), c))


def apply_tribute(hand: "Hand", prev_ranks=None) -> Tribute:
    """就地执行进贡阶段，并把 `hand.turn` 设成先出者。`prev_ranks=None`（第一手）则不动。

    **「先出者 = 头游」是基线**（无硬证据）—— `tools/accept_sim.py` 会拿发牌报文里的
    `nWhoIsFirstOut` 逐局对，对不上就在那里暴露出来。
    """
    if prev_ranks is None:
        return Tribute("none", {}, {}, hand.turn)
    level = hand.level
    givers = tributers(prev_ranks)
    leader = next(s for s in SEATS if prev_ranks[s] == 1)

    losing_team = TEAM[givers[0]]
    teammates = [s for s in SEATS if TEAM[s] == losing_team]
    if sum(1 for s in teammates for c in hand.hands[s]
           if cards.parts(c)[0] == 15) >= 2:
        hand.turn = leader
        return Tribute("resist", {}, {}, leader)

    # 双贡时：**贡牌大的那家给头游**，另一家给二游（基线）。
    receivers = sorted([s for s in SEATS if prev_ranks[s] in (1, 2)],
                       key=lambda s: prev_ranks[s])          # 头游在前
    ranked = sorted(givers,
                    key=lambda s: (_value(_biggest(hand.hands[s], level), level),
                                   _biggest(hand.hands[s], level)),
                    reverse=True)
    gave, returned = {}, {}
    for giver, receiver in zip(ranked, receivers):
        c = _biggest(hand.hands[giver], level)
        gave[giver] = c
        hand.hands[giver].discard(c)
        hand.hands[receiver].add(c)
        b = _smallest_returnable(hand.hands[receiver], level)
        returned[receiver] = b
        hand.hands[receiver].discard(b)
        hand.hands[giver].add(b)
    hand.turn = leader
    return Tribute("double" if len(givers) == 2 else "single", gave, returned, leader)
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rules_tribute.py -q`
Expected: 全部 passed

- [ ] **Step 5: 跑整套回归**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`，再 `.venv/Scripts/python.exe -m tools.accept_meld`
Expected: 全 passed、`验收全绿 ✓`

- [ ] **Step 6: 提交**

```bash
git add net/sim/rules.py tests/test_rules_tribute.py
git commit -m "feat: 进贡/还贡/抗贡（还贡 2..10 与抗贡判队有硬证据，其余标记为基线）"
```
## Task 5: `tools/accept_sim.py` —— 回放 55 局真实对局（spec §6①③）

**这是 Plan 2 的地基验收。** 它回答的是「我们的规则引擎，能不能把真人打过的每一手原样走一遍」。

**Files:**
- Create: `tools/accept_sim.py`
- Test: `tests/test_accept_sim.py`

**Interfaces:**
- Consumes: `rules.Hand`（Task 1~4）；`meld.as_meld(ids, level)`；`tools.game_log.load_corpus()` + `LAST_SOURCE`；`tools.decision_points.initial_hands(g)`
- Produces:
  - `accept_sim.shape(m) -> tuple`（`(kind, size, rank)`）
  - `accept_sim.replay(g) -> ReplayResult`（`steps`、`passes_inferred`、`ranks`）
  - `accept_sim.check_replay(games) -> Result`（复用 `tools/accept_meld.py` 的 `Result` 报告风格）
  - `accept_sim.main() -> int`（退出码）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_accept_sim.py
"""验收脚本自身的测试：**先证明它会失败，再证明它会通过。**

理由（本项目吃过一次亏）：验收脚本最容易「什么都没跑也报绿」。
所以这里既测「真数据全过」，也测「喂一条篡改过的对局必须红」。
"""
import copy

from net import cards
from net.sim import meld, rules
from tools import accept_sim
from tools.game_log import load_corpus


def _one_settled_game():
    games = [g for g in load_corpus() if g.settle and len(g.plays) > 20]
    assert games, "语料里没有带结算的对局 —— 先生成 data/game_corpus.json"
    return games[0]


def test_shape_ignores_which_copy_of_a_card_was_used():
    """`melds_from` 的契约是「每个形状一条代表」—— 所以比对口径必须是**形状**，
    不是牌张集合。真人打出的可能是 ♥5♥5，而枚举给的是 ♠5♠5 那条代表。"""
    a = meld.as_meld([meld.cid_from_name("S5"), meld.cid_from_name("S5", deck=2)], 2)
    b = meld.as_meld([meld.cid_from_name("H5"), meld.cid_from_name("H5", deck=2)], 2)
    assert a.cards != b.cards
    assert accept_sim.shape(a) == accept_sim.shape(b) == (meld.PAIR, 2, 4)   # rank 是 meld.point_value(5,2)=4，不是 5


def test_replay_walks_a_real_game_to_the_end_without_raising():
    g = _one_settled_game()
    r = accept_sim.replay(g)
    assert r.steps == len(g.plays)
    assert r.hand.is_over()
    # 出完顺序必须与日志名次一致（第 1~3 名）
    logged = g.settle["Rank"]
    for i, seat in enumerate(r.hand.order):
        assert logged[seat] == i + 1, f"出完顺序与日志名次对不上：{r.hand.order} vs {logged}"


def test_replay_flags_a_tampered_game_instead_of_passing_it():
    """把某一手的牌换成手上没有的牌 —— 回放必须炸，不能「跳过这一手继续跑」。"""
    g = copy.deepcopy(_one_settled_game())
    g.plays[5].cards = [meld.cid_from_name("JOKER_B", deck=2)]   # 几乎不可能在他手上
    try:
        accept_sim.replay(g)
    except rules.IllegalPlay:
        return
    raise AssertionError("篡改过的对局居然走通了 —— 回放没有真的在验规则")


def test_report_is_red_when_nothing_was_checked():
    """`total == 0` 必须是**失败**（「一项都没检查到」不是「全过」）。"""
    res = accept_sim.check_replay([])
    assert not res.ok
    assert "0" in res.report() or "一项" in res.report()
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_accept_sim.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.accept_sim'`

- [ ] **Step 3: 实现 `tools/accept_sim.py`**

```python
"""第二层验收（spec §6①③）：拿 55 局真实对局回放**整局**，验模拟器的出牌轮转。

与 Plan 1 的 `tools/accept_meld.py` 的分工：
  - `accept_meld` 验**牌型引擎**（这一手是不是合法牌型、谁压谁）
  - 本脚本验**牌局引擎**（轮转对不对、接风对不对、什么时候终局、名次对不对）

跑法：
    .venv/Scripts/python.exe -m tools.accept_sim

**回放的起点是「进贡之后」**：`decision_points.initial_hands` 用的是「出过的 ∪ 结算剩的」，
而进贡/还贡在**第一手之前**就完成了，所以重建出来的正是进贡后的手牌。
**因此本脚本不验进贡**（那在 Task 6 里单独验，用的另一份证据）。

比对口径是**形状**（`(kind, size, rank)`）而不是牌张集合 —— 因为 `melds_from` 的契约是
「每个形状一条代表」（见 `net/sim/meld.py` 的 docstring）。真人打出的可能是 ♥5♥5，
枚举给的代表可能是 ♠5♠5，形状相同就是对的。**这与 `accept_meld` 的验收①同口径。**
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from net.sim import meld, rules
# `_utf8_stdout` 直接复用 accept_meld 那份，**不复制** —— 它就是为「GBK 控制台下
# 打不出 ✓ 会让全绿的脚本返回 1」写的（Plan 1 踩过这个坑）。
from tools.accept_meld import Result, _utf8_stdout
from tools.decision_points import initial_hands
from tools.game_log import load_corpus

#: 语料地板：能回放的对局少于这个数，「全过」不足以称为结论。
#: 与 `tools/accept_meld.py` 的 `_MIN_SETTLED` 同口径。
_MIN_REPLAYABLE = 20

#: 「过」的推断上限。**不是 3。** 日志里没有「过」，所以一手真实出牌前面可能挂着
#: 好几次「过」：三家过完之后清桌，同一个人可能**先过了、又成了领出者**
#: （接风就是这么发生的：主人出完，队友轮到时先过、清桌后再由他领出）。
#: 实测 55 局里最长的连过是 5，这里留到 6 并在超限时报出上下文。
_MAX_INFERRED_PASSES = 6


def shape(m) -> tuple:
    """`(kind, size, rank)` —— 牌型的「形状」。比较候选时用这个口径，不用牌张。"""
    return (m.kind, m.size, m.rank)


@dataclass
class ReplayResult:
    steps: int
    passes_inferred: int
    hand: rules.Hand


def replay(g) -> ReplayResult:
    """把一局真实对局走一遍。**认不出的局面直接炸**，不跳过。

    ⚠️ **日志里没有「过」。** 所以不能写成「轮到日志的座位了就出手」——
    真人可能先过。判据必须是：**轮到日志的座位了，而且他这一手确实压得过桌面**
    （或桌面已清）。否则就把他记成「过」，继续往后走。

    这不是小事：**接风就是靠这个走出来的** —— 主人出完，队友轮到时先过（压不过），
    三家过完清桌，**再由队友领出**。只判「轮到谁」的话，接风那一手会被判成
    「压不过桌面」而炸掉（实测：只判轮到的版本在 55 局里挂 34 局）。
    """
    if not g.plays:
        raise ValueError(f"{g.t0} 这一局没有任何出牌记录")
    hands = initial_hands(g)
    hand = rules.Hand(hands=[set(hands[s]) for s in rules.SEATS],
                      level=g.trump, turn=g.plays[0].seat)
    passes = 0
    for i, rec in enumerate(g.plays):
        m = meld.as_meld(list(rec.cards), g.trump)
        if m is None:
            raise rules.IllegalPlay(
                f"{g.t0} 第 {i} 手：座位{rec.seat} 的真实着法判不出牌型 "
                f"card_type={rec.card_type} cards={rec.cards}")
        guard = 0
        while hand.turn != rec.seat or (hand.table is not None
                                        and not meld.beats(m, hand.table)):
            if guard > _MAX_INFERRED_PASSES:
                raise rules.IllegalPlay(
                    f"{g.t0} 第 {i} 手：连推了 {guard} 个「过」仍走不到座位{rec.seat} "
                    f"出手（我们停在 {hand.turn}，桌面"
                    f"{'清' if hand.table is None else meld.describe_meld(hand.table)}）")
            if hand.table is None and hand.turn != rec.seat:
                raise rules.IllegalPlay(
                    f"{g.t0} 第 {i} 手：轮到领出的是座位{hand.turn}，"
                    f"但日志说这手是座位{rec.seat}出的 —— 领出者判错了")
            hand.pass_turn(hand.turn)
            passes += 1
            guard += 1

        # ① 真实着法必须能在候选里找到（按形状比）
        cands = {shape(x) for x in hand.actions(rec.seat) if x is not None}
        if shape(m) not in cands:
            raise rules.IllegalPlay(
                f"{g.t0} 第 {i} 手：座位{rec.seat} 真实出了 {meld.describe_meld(m)}，"
                f"但候选里没有这个形状（候选 {len(cands)} 个）—— 枚举漏了")
        hand.play(rec.seat, m)

    if not hand.is_over():
        raise rules.IllegalPlay(
            f"{g.t0} 牌都出完了却没判成终局（出完 {len(hand.order)} 家）")
    return ReplayResult(steps=len(g.plays), passes_inferred=passes, hand=hand)


def check_replay(games) -> Result:
    """跑全部能回放的局：① 轮转能走到尾、③ 名次与日志一致。"""
    res = Result("① 真实对局回放（轮转 / 接风 / 终局 / 名次）")
    playable = [g for g in games if g.settle and g.plays]
    if len(playable) < _MIN_REPLAYABLE:
        res.floor = (f"能回放的对局只有 {len(playable)} 局 < 地板 {_MIN_REPLAYABLE} 局 —— "
                     f"「全过」不足以称为结论（语料被轮转删了？）")
        return res
    for g in playable:
        res.total += 1
        try:
            r = replay(g)
        except (rules.IllegalPlay, ValueError) as e:
            res.bad.append(str(e))
            continue
        logged = g.settle["Rank"]
        for i, seat in enumerate(r.hand.order):
            if logged[seat] != i + 1:
                res.bad.append(
                    f"{g.t0} 出完顺序 {r.hand.order} 与日志名次 {logged} 对不上")
                break
    return res


def main(argv=None) -> int:
    _utf8_stdout()          # ← 见下方说明，漏了这行 GBK 控制台下会抛 UnicodeEncodeError
    games = load_corpus()
    print(f"载入对局 {len(games)} 局，其中有结算的 "
          f"{sum(1 for g in games if g.settle)} 局\n")
    from tools import game_log
    print(f"语料来源：{game_log.LAST_SOURCE}\n")
    res = check_replay(games)
    print(res.report())
    ok = res.ok
    print("\n验收" + ("全绿 ✓" if ok else "**未通过** ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_accept_sim.py -q`
Expected: 4 passed

- [ ] **Step 5: 跑真实验收**

Run: `.venv/Scripts/python.exe -m tools.accept_sim`
Expected: `[OK] ① 真实对局回放 ... 55 项全过`，退出码 0

**如果这一步红了 —— 那才是这个任务的价值所在。** 失败信息里带着「哪一局、第几手、
日志说谁出的、我们说该谁出的」，照着查 `_advance` / `_lead_after` / `is_over`。
**不要为了让验收变绿去放宽 `replay` 的检查。**

- [ ] **Step 6: 跑 Plan 1 的回归防线**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`，再 `.venv/Scripts/python.exe -m tools.accept_meld`
Expected: 全 passed、`验收全绿 ✓`

- [ ] **Step 7: 提交**

```bash
git add tools/accept_sim.py tests/test_accept_sim.py
git commit -m "feat: tools/accept_sim.py —— 55 局真实对局回放验收（轮转/接风/终局/名次）"
```

---

## Task 6: 进贡记录的语料冻结 + ⑤ 核对

**为什么要单独冻结一份语料**：进贡记录只存在于**实时日志**里，而日志**只留 2 天、按小时轮转**
（Plan 1 已经因为这件事冻结过 `data/game_corpus.json`）。不冻结的话，进贡那 25 条证据
过两天就没了，验收会变成「什么都没查到却报绿」—— 正是本项目吃过一次亏的那种假绿。

**Files:**
- Modify: `tools/game_log.py`（加 `tributes` 字段 + 解析）、`tools/snapshot_logs.py`（带上它）
- Create: `tools/accept_tribute.py`
- Test: `tests/test_tribute_records.py`

**Interfaces:**
- Consumes: 实时日志目录 `tools.game_log.LOG_DIR`
- Produces:
  - `game_log.TributeRec`（`t: datetime`、`giver: int`、`card: int`、`kind: str`）
  - `game_log.load_tributes(log_dir=LOG_DIR) -> list[TributeRec]`
  - `GameLog` 不管进贡（进贡是**局间**的事，不属于任何一手）；`load_tributes` 独立返回
  - `accept_tribute.check_returns(records) -> Result`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tribute_records.py
"""进贡记录的解析测试。**用真日志跑**，但日志没了要明着说，不能静默返回空列表。"""
import os

import pytest

from tools import game_log


@pytest.mark.skipif(not os.path.isdir(game_log.LOG_DIR),
                    reason="实时日志已被轮转删除 —— 这不是失败，是跳过的理由")
def test_tribute_records_parse_and_every_return_is_at_most_10():
    """**25/25 条硬证据（spec §13.6）**：还贡的牌点数全部落在 2..10。

    A=1 不在里面 —— `idx <= 10` 会把 A 也算成「≤10」，那正是最难还出去的牌。
    """
    from net import cards
    recs = game_log.load_tributes()
    assert len(recs) >= 10, f"只解析出 {len(recs)} 条进贡记录，太少了（格式变了？）"
    bad = [(r.giver, cards.decode(r.card)) for r in recs
           if r.kind == "return" and not 2 <= cards.parts(r.card)[0] <= 10]
    assert not bad, f"这些还贡牌点数不在 2..10：{bad}"


@pytest.mark.skipif(not os.path.isdir(game_log.LOG_DIR), reason="日志不在")
def test_load_tributes_raises_when_the_directory_is_gone():
    """目录不在必须**炸**，不许返回空列表 —— 空列表会让验收「0 项全过」。"""
    with pytest.raises(FileNotFoundError):
        game_log.load_tributes(log_dir=os.path.join(game_log.LOG_DIR, "不存在"))
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tribute_records.py -q`
Expected: FAIL — `AttributeError: module 'tools.game_log' has no attribute 'load_tributes'`

- [ ] **Step 3: 实现（追加到 `tools/game_log.py`）**

```python
# ---------------------------------------------------------------- 进贡记录
#
# 进贡是**局间**的事（上一手结算之后、这一手发牌之前），不属于任何一手，
# 所以不塞进 GameLog，单独返回。
#
# 三行一起用才完整（2026-09-25 在 26 个日志文件 / 114 MB 上核出，spec §13.6）：
#   TributeService NotifyTribute localId=N Card=N                -> 我方贡出的牌（进贡）
#   GuanDanPlaySceneData NotifyReturnTribute SeatID=N Card=N ReturnSeatID=M
#                                                                -> 谁还给谁哪张（还贡）
#   TributeSectionEndService NotifyTributeSectionEnd fromLocalId=N destLocalId=M card=N
#                                                                -> 进贡阶段的最终确认（还贡）
# `card=` 是**牌 ID**，与 net/cards.py 的编码一致（>255 = 第二副），可直接喂 meld.py。

_TRIB_GIVE = re.compile(r"TributeService NotifyTribute localId=(\d+) Card=(\d+)")
_TRIB_RETURN = re.compile(
    r"TributeSectionEndService NotifyTributeSectionEnd "
    r"fromLocalId =(\d+) destLocalId =(\d+) card = (\d+)")


@dataclass
class TributeRec:
    t: Optional[datetime]
    kind: str        # "give" = 进贡 / "return" = 还贡
    giver: int       # 交出牌的人
    taker: Optional[int]   # 收到牌的人（"give" 时日志里没有，为 None）
    card: int


def load_tributes(log_dir: str = LOG_DIR) -> list[TributeRec]:
    """从实时日志里解出进贡/还贡记录。

    **目录不在就抛**（同 `load_games`）—— 返回空列表会让下游验收「0 项全过」。
    """
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(
            f"日志目录不存在：{log_dir}\n"
            f"日志只保留 2 天。进贡记录的**唯一**来源就是它 —— "
            f"别把它当成「0 条进贡」继续跑。")
    files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not files:
        raise RuntimeError(f"目录在但一个 .log 都没有：{log_dir}")
    out: list[TributeRec] = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                ts = _ts(line)
                m = _TRIB_GIVE.search(line)
                if m:
                    out.append(TributeRec(ts, "give", int(m.group(1)), None,
                                          int(m.group(2))))
                    continue
                m = _TRIB_RETURN.search(line)
                if m:
                    out.append(TributeRec(ts, "return", int(m.group(1)),
                                          int(m.group(2)), int(m.group(3))))
    return out
```

同时在 `tools/snapshot_logs.py` 里把 `load_tributes()` 的结果一起写进快照
（新键 `"tributes"`），并在 `load_snapshot` / `save_snapshot` 里透传 ——
**保持 `load_games` 的语义不变**（它照旧只读实时日志），「优先快照」这条策略留在
`load_corpus` 那一层，跟 Plan 1 的既有做法一致。快照里存成
`[t.isoformat(), kind, giver, taker, card]` 的数组，跟 `plays` 一样的紧凑写法。

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tribute_records.py -q`
Expected: 2 passed（或 1 passed + 1 skipped，取决于日志在不在）

- [ ] **Step 5: 重新冻结语料**

Run: `.venv/Scripts/python.exe -m tools.snapshot_logs`
Expected: 打印「先验 108 张守恒」通过，并报出**进贡记录条数**（应 ≥ 10）

- [ ] **Step 6: 写 `tools/accept_tribute.py`**

结构与 `tools/accept_sim.py` 相同（`main()` + `Result` 报告 + 退出码）。检查两项：

1. **还贡牌 ≤ 10**（25/25 硬证据）—— 违反就是规则实现错了
2. **进贡牌 = 贡方手上最大的牌**（基线）—— 拿「进贡后的手牌 + 被贡走的牌」重建进贡前的手牌，
   用 `meld.point_value` 判最大。**对不上就逐条报出来**，不要当成硬失败：
   先看是不是「级牌可不贡」之类的变体，把人读的牌面打出来给人判。

- [ ] **Step 7: 跑验收**

Run: `.venv/Scripts/python.exe -m tools.accept_tribute`
Expected: 第 1 项全过；第 2 项**打印出**一致性比例与不一致的样例（哪怕不一致也退出码 0，
因为它是基线不是硬证据；但报告里必须写清楚「未达 100%」）

- [ ] **Step 8: 提交**

```bash
git add tools/game_log.py tools/snapshot_logs.py tools/accept_tribute.py tests/test_tribute_records.py data/game_corpus.json
git commit -m "feat: 进贡记录的语料冻结 + accept_tribute（还贡≤10 硬证据，贡最大牌为基线）"
```
## Task 7: `net/sim/env.py` 的状态编码（**只含可观测信息**）

> **这是整个 Plan 2 里最危险的一块。** spec §3 的「不许明牌泄漏」在这里落地。
> 做法不是「记得别填对手手牌」，而是**类型上就没有那个字段** —— `Observation` 里
> 根本没有「四家手牌」这个东西，编码器也就无从泄漏。Task 9 会用自动化测试钉死它。

**Files:**
- Create: `net/sim/env.py`
- Test: `tests/test_env_state.py`

**Interfaces:**
- Consumes: `rules.Hand`、`rules.SEATS`、`meld.point_value`、`cards.slot`
- Produces:
  - `env.STATE_DIM = 700`
  - `env.Observation`（frozen dataclass，字段见下）
  - `env.observe(hand, played, seat) -> Observation`
  - `env.encode_state(obs) -> np.ndarray`（`float32`，形状 `(700,)`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_env_state.py
import numpy as np

from net import cards
from net.sim import env, meld, rules

A = meld.cid_from_name


def _obs(seat=0, hand=(A("S3"),), played=None, left=None, table=(),
         table_kind=0, table_rank=-1, passed=None, turn=0, level=2):
    n = 4
    return env.Observation(
        seat=seat, hand=frozenset(hand),
        played=tuple(frozenset(x) for x in (played or [()] * n)),
        left=tuple(left or [1] * n), table=tuple(table),
        table_kind=table_kind, table_rank=table_rank,
        passed=tuple(passed or [False] * n), turn=turn, level=level)


def test_state_dim_is_what_the_spec_says():
    assert env.STATE_DIM == 700          # spec §4.1：108+4*108+4+108+10+15+4+4+15
    assert env.encode_state(_obs()).shape == (700,)
    assert env.encode_state(_obs()).dtype == np.float32


def test_hand_and_table_land_in_the_right_slots():
    v = env.encode_state(_obs(hand=[A("S3"), A("DK", deck=2)],
                              table=[A("H5")]))
    assert v[cards.slot(A("S3"))] == 1 and v[cards.slot(A("DK", deck=2))] == 1
    assert v.sum() >= 3
    t0 = 108 + 4 * 108 + 4                     # 桌面牌段的起点
    assert v[t0 + cards.slot(A("H5"))] == 1


def test_seats_are_relativised_so_one_policy_can_play_all_four():
    """**座位必须相对化**（spec §3）：索引 0 = 自己、1 = 下家、2 = 对家、3 = 上家。

    同一个「我的牌 + 我的下家出过什么」的局面，无论坐在哪个绝对座位上，
    编码必须**一模一样**，否则四个座位就得有四套权重。"""
    played_abs = [(), (), (), ()]
    played_abs[2] = (A("S7"),)                 # 绝对座位 2 出过 7♠
    # a：我坐 0、轮到我、座位 2（我的**对家**）出过 7♠
    a = env.encode_state(_obs(seat=0, turn=0, played=played_abs))
    # c：我坐 1、轮到我、座位 3（此时也是我的**对家**）出过 7♠
    played_c = [(), (), (), (A("S7"),)]
    c = env.encode_state(_obs(seat=1, turn=1, played=played_c))
    assert np.array_equal(a, c)                # 可观测状态相同 -> 编码必须一模一样
    # b：我坐 1，出过 7♠ 的座位 2 是**下家**（不是对家）-> 与 a 不同
    b = env.encode_state(_obs(seat=1, turn=1, played=[(), (), (A("S7"),), ()]))
    assert not np.array_equal(a, b)


def test_level_card_uses_point_value_not_the_raw_index():
    """**打 2 的时候 2 是级牌、比 A 大。** 桌面主点数的编码必须走 `meld.point_value`，
    否则「桌面是一对 2」与「桌面是一对 3」在打 2 时会被编成相邻的两格，
    而它们实际差着整整一个层级。"""
    kind, size = meld.PAIR, 2
    v2 = env.encode_state(_obs(table=(A("S2"), A("H2")), table_kind=kind,
                               table_rank=meld.point_value(2, 2), level=2))
    v3 = env.encode_state(_obs(table=(A("S3"), A("H3")), table_kind=kind,
                               table_rank=meld.point_value(3, 2), level=2))
    seg = slice(108 + 4 * 108 + 4 + 108 + 10, 108 + 4 * 108 + 4 + 108 + 10 + 15)
    assert v2[seg].argmax() != v3[seg].argmax()
    assert v2[seg].argmax() == 13              # POINT_LEVEL -> 第 13 格
    assert v3[seg].argmax() == 1               # 点数 3 -> _POINT[3]=2 -> 槽位 1
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'net.sim.env'`

- [ ] **Step 3: 实现 `net/sim/env.py`**

```python
"""掼蛋自对弈环境（spec §4）。

**这个模块的唯一的、不可动摇的纪律：喂给策略的状态里不许有对手手牌。**

做法不是靠注释提醒，是靠**类型**：`Observation` 这个 dataclass 里根本没有
「四家手牌」这个字段 —— 它只有可观测的东西（我的手牌、各家出过的牌、各家剩几张、
桌面、轮到谁、级别）。编码器 `encode_state` 只吃 `Observation`，
所以**它在结构上就无法泄漏**。`tests/test_env_leak.py` 再用自动化测试钉死一遍。

泄漏的后果（spec §3）：会训出靠偷看才成立的打法 —— 训练分数漂亮、真机全废，
**而且静默失效**（同本项目「合成 val 骗过一次」的教训）。

座位一律**相对化**：索引 0 = 自己、1 = 下家、2 = 对家、3 = 上家。
四个座位共享一套权重，所以相对化不是可选项。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from net import cards
from net.sim import meld, rules

#: spec §4.1 的状态维度，逐项对齐：
#:     我的手牌 108 + 已出过的牌 4×108 + 各家剩几张 4 + 桌面待压 108
#:     + 桌面牌型 10 + 桌面主点数 15 + 谁要不起 4 + 轮到谁 4 + 级别 15
STATE_DIM = 108 + 4 * 108 + 4 + 108 + 10 + 15 + 4 + 4 + 15
assert STATE_DIM == 700

_OFF_HAND = 0
_OFF_PLAYED = _OFF_HAND + 108
_OFF_LEFT = _OFF_PLAYED + 4 * 108
_OFF_TABLE = _OFF_LEFT + 4
_OFF_KIND = _OFF_TABLE + 108
_OFF_RANK = _OFF_KIND + 10
_OFF_PASSED = _OFF_RANK + 15
_OFF_TURN = _OFF_PASSED + 4
_OFF_LEVEL = _OFF_TURN + 4

#: 桌面/动作里的「主点数」编码：点数 1..13 -> 0..12；
#: **级牌 -> 13**（`meld.POINT_LEVEL`）；王 -> 14（`POINT_SMALL` / `POINT_BIG` 合到一格，
#: 因为王当桌面牌型时只有「一对王」这一种，大小王不会再区分）。
_RANK_SLOTS = 15


def _rank_slot(point: int) -> int:
    if point >= meld.POINT_LEVEL:
        return _RANK_SLOTS - 1
    return min(max(point, 1), _RANK_SLOTS - 1) - 1


@dataclass(frozen=True)
class Observation:
    """策略能看见的东西。

    **故意没有「四家手牌」这个字段。** 这不是省略，是设计 ——
    只要它不在这个类型里，编码器就没有办法把它编进去。
    """
    seat: int                       # 出牌人（相对化的原点，**绝对**座位号）
    hand: frozenset                 # 我的手牌
    played: Tuple[frozenset, ...]   # 4 家各自出过的牌（公开信息）
    left: Tuple[int, ...]           # 4 家各剩几张（公开信息）
    table: Tuple[int, ...]          # 桌面待压的牌；空元组 = 我领出
    table_kind: int                 # 桌面牌型（0 = 无）
    table_rank: int                 # 桌面主点数（`meld.point_value` 口径；-1 = 无）
    passed: Tuple[bool, ...]        # 本轮谁「要不起」（公开信息）
    turn: int                       # 轮到谁（绝对座位）
    level: int


def _rel(seq, seat: int):
    """把按绝对座位排的序列转成相对座位（0 = 自己、1 = 下家 …）。"""
    return tuple(seq[(seat + i) % 4] for i in range(4))


def encode_state(obs: Observation) -> np.ndarray:
    """`Observation` -> 700 维 float32。**只吃 `Observation`**，别给它开别的入口。"""
    v = np.zeros(STATE_DIM, dtype=np.float32)

    for c in obs.hand:
        v[_OFF_HAND + cards.slot(c)] = 1.0

    for i, cards_i in enumerate(_rel(obs.played, obs.seat)):
        for c in cards_i:
            v[_OFF_PLAYED + i * 108 + cards.slot(c)] = 1.0

    for i, n in enumerate(_rel(obs.left, obs.seat)):
        v[_OFF_LEFT + i] = n / 27.0            # 归一，别让 27 这种量级和 0/1 混在一起

    for c in obs.table:
        v[_OFF_TABLE + cards.slot(c)] = 1.0

    if obs.table_kind:
        v[_OFF_KIND + obs.table_kind - 1] = 1.0
    if obs.table_rank >= 0:
        v[_OFF_RANK + _rank_slot(obs.table_rank)] = 1.0

    for i, p in enumerate(_rel(obs.passed, obs.seat)):
        v[_OFF_PASSED + i] = 1.0 if p else 0.0

    v[_OFF_TURN + (obs.turn - obs.seat) % 4] = 1.0
    v[_OFF_LEVEL + obs.level] = 1.0            # 1..13 用第 1..13 格；0 与 14 恒为 0

    return v


def observe(hand: "rules.Hand", seat: int, played, table_meld: Optional["meld.Meld"]):
    """从**明牌**的 `Hand` 里切出 `seat` 视角的可观测状态。

    **这是明牌与策略之间唯一的窄口**：进来的是整个 `Hand`（四家都看得见），
    出去的 `Observation` 里只有公开信息 + 自己的手牌。

    `played` 由调用方维护（`{座位: 出过的牌}`），因为 `Hand` 只记 `steps`，
    不按座位分桶 —— 分桶是面板/记牌器的口径，不是规则的口径。
    """
    table = ()
    kind, rank = 0, -1
    if table_meld is not None:
        table = tuple(table_meld.cards)
        kind = table_meld.kind
        rank = table_meld.rank
    return Observation(
        seat=seat,
        hand=frozenset(hand.hands[seat]),
        played=tuple(frozenset(played.get(s, ())) for s in rules.SEATS),
        left=tuple(len(hand.hands[s]) for s in rules.SEATS),
        table=table, table_kind=kind, table_rank=rank,
        passed=tuple(s in hand.passed for s in rules.SEATS),
        turn=hand.turn,
        level=hand.level,
    )
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_state.py -q`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add net/sim/env.py tests/test_env_state.py
git commit -m "feat: env 状态编码（Observation 类型上就没有对手手牌这个字段）"
```
## Task 8: `env.py` 的动作编码 + `step` / reward / 自对弈 rollout

**Files:**
- Modify: `net/sim/env.py`（追加）
- Test: `tests/test_env_step.py`

**Interfaces:**
- Consumes: Task 7 的 `Observation` / `encode_state` / `observe`；`rules.Hand` / `rules.reward` / `rules.SEATS`
- Produces:
  - `env.ACTION_DIM = 143`
  - `env.encode_action(m, level) -> np.ndarray`（`m=None` = 「过」→ 全 0）
  - `env.EnvConfig`（`tribute: bool = False`、`level: Optional[int] = None`、`seed: int = 0`）
  - `env.GuandanEnv`：`reset(...) -> Observation`、`legal() -> list`、`step(i) -> (obs, r, done, info)`、`rollout(policy) -> list[(obs, action_list, chosen_index, actor)]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_env_step.py
import numpy as np
import pytest

from net import cards
from net.sim import env, meld, rules

A = meld.cid_from_name


def test_action_dim_matches_the_spec():
    assert env.ACTION_DIM == 143          # 10 + 15 + 9 + 1 + 108


def test_pass_action_is_all_zeros_and_single_card_is_not():
    z = env.encode_action(None, level=2)
    assert z.shape == (143,) and z.sum() == 0
    m = meld.as_meld([A("S3")], 2)
    v = env.encode_action(m, 2)
    assert v[:10].argmax() == meld.SINGLE - 1
    assert v[10:25].argmax() == 1              # 点数 3 -> _POINT[3]=2 -> 槽位 1
    assert v[25:34].argmax() == 0              # 1 张 -> 第 0 格
    assert v[34] == 0                          # 不含逢人配
    assert v[35 + cards.slot(A("S3"))] == 1


def test_joker_bomb_gets_its_own_size_slot():
    """天王炸 4 张，但张数那一格**必须与「4 张炸弹」区分开**
    （spec §4.2 把天王炸单列成一格）。"""
    from net.sim import meld as M
    jb = M.Meld(M.BOMB, 4, 0, (A("JOKER_B"), A("JOKER_B", deck=2),
                               A("JOKER_S"), A("JOKER_S", deck=2)))
    v = env.encode_action(jb, 2)
    assert v[25:34].argmax() == 8              # 第 8 格 = 天王炸


def test_env_step_advances_and_awards_zero_until_the_hand_ends():
    e = env.GuandanEnv(seed=7)
    e.reset(level=5, hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], first=0)
    obs, r, done, info = e.step(0)             # 座位 0 出单张
    assert r == 0 and not done and info["seat"] == 0
    assert obs.turn == 1                       # 轮到下家
    assert obs.left == (0, 1, 1, 1)            # 座位 0 已经出完

def test_terminal_reward_is_awarded_exactly_once_and_equals_the_team_points():
    """**reward 只在终局那一步非零**，且等于「出牌人所在队的升级点」，符号跟着输赢。

    中间步恒为 0 是刻意的：牌类游戏中间没有即时反馈，DMC 的做法是拿终局 reward
    当整局所有决策点的回归目标 —— 那件事在训练循环里做，不在 env 里做。"""
    import random
    rng = random.Random(11)
    e = env.GuandanEnv(seed=11)
    e.reset(level=5)
    seen = []
    while not e.done:
        obs, r, done, info = e.step(rng.randrange(len(e.legal())))
        seen.append((info["seat"], r))
    nonzero = [(s, r) for s, r in seen if r != 0]
    assert len(nonzero) == 1, f"非零 reward 出现了 {len(nonzero)} 次，应该只有终局那次"
    seat, r = nonzero[0]
    ranks = e.ranks
    assert abs(r) == rules.points(ranks)
    assert (r > 0) == (rules.TEAM[seat] == rules.winner_team(ranks))
    assert any(x != 0 for x in seen), "终局那一步没有给 reward"


def test_legal_actions_include_pass_only_when_someone_has_played():
    e = env.GuandanEnv(seed=3)
    e.reset(level=2, hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], first=0)
    assert None not in e.legal()               # 领出
    e.step(0)
    assert None in e.legal()                   # 跟牌：能压也可以过


def test_a_full_random_rollout_terminates_and_conserves_108_cards():
    """随机自对弈必须**每局都收得了尾** —— 死循环是这类实现最常见的病。"""
    import random
    for seed in range(30):
        e = env.GuandanEnv(seed=seed)
        e.reset(level=random.Random(seed).randint(1, 13))
        n = 0
        while not e.done:
            acts = e.legal()
            e.step(random.Random(seed * 1000 + n).randrange(len(acts)))
            n += 1
            assert n < 1000, f"seed={seed} 这一局走了 {n} 步还没完 —— 死循环"
        assert sorted(e.ranks) == [1, 2, 3, 4]
        assert sum(len(h) for h in e.hand.hands) + sum(
            len(s.meld.cards) for s in e.hand.steps if s.meld) == 108
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_step.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.env' has no attribute 'ACTION_DIM'`

- [ ] **Step 3: 实现（追加到 `net/sim/env.py`）**

```python
# ---------------------------------------------------------------- 动作编码

#: spec §4.2：牌型 10 + 主点数 15 + 张数 9 + 逢人配 1 + 牌 108
ACTION_DIM = 10 + 15 + 9 + 1 + 108
assert ACTION_DIM == 143

_A_OFF_KIND = 0
_A_OFF_RANK = 10
_A_OFF_SIZE = 25
_A_OFF_WILD = 34
_A_OFF_CARDS = 35

#: 张数那一格：1..8 张 -> 0..7；**≥9 张或天王炸 -> 8**。
#: spec §4.2 写的是「1~8 张 + 天王炸」，但掼蛋里还有 9 炸 / 10 炸（8 张同点 + 逢人配），
#: 9 格装不下。这里把它们并进第 8 格 —— **不丢信息**，因为牌 multi-hot(108)
#: 已经把这手牌是什么完全写清楚了，张数格只是给网络的一个提示。
_SIZE_COMPOSITE = 8


def encode_action(m, level: int) -> np.ndarray:
    """一个候选着法 -> 143 维 float32。**`None`（过）编成全 0。**

    ⚠️ 主点数那一格的口径与 `Meld.rank` 一致，而 `rank` 在两种口径之间：
    序列类（顺子/连顺/钢板）存的是**自然值**（A 可作 1 或 14），
    其余存的是 `meld.point_value`（级牌 14、王 15/16）。
    天王炸的 `rank` 是 0。**所以这一格跨牌型不是单射** ——
    真正的判别力在牌 multi-hot 上，这一格是给网络的便捷特征。
    """
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    if m is None:
        return v                          # 「过」= 全 0（没有牌型、没有牌）
    v[_A_OFF_KIND + m.kind - 1] = 1.0
    v[_A_OFF_RANK + _rank_slot(m.rank)] = 1.0
    if m.size >= 9 or meld.bomb_class(m) == meld.CLASS_JOKER_BOMB:
        v[_A_OFF_SIZE + _SIZE_COMPOSITE] = 1.0
    else:
        v[_A_OFF_SIZE + m.size - 1] = 1.0
    if m.wild_used:
        v[_A_OFF_WILD] = 1.0
    for c in m.cards:
        v[_A_OFF_CARDS + cards.slot(c)] = 1.0
    return v


# ---------------------------------------------------------------- 环境

@dataclass
class EnvConfig:
    """一手牌怎么开局。

    `tribute=False`（默认）**不走进贡**：发牌后直接由 `first` 或随机座位领出。
    理由见 spec §13.1 —— 一手一个 episode 时，进贡要依赖**上一手的名次**，
    而训练时那个名次是采样的；`tribute=True` 配合 `prev_ranks` 才用得上。
    **进贡规则本身已经实现并验过（Task 4 / 6），这里只是训练时开不开。**
    """
    tribute: bool = False
    level: Optional[int] = None       # None -> 每局从 1..13 采一个
    seed: int = 0


class GuandanEnv:
    """一手牌的自对弈环境。

    **`self.hand` 是明牌的 `rules.Hand`，绝不能交给策略。** 策略只能拿到
    `observe()` 出来的 `Observation` 和 `encode_*` 的向量。
    """

    def __init__(self, cfg: EnvConfig = None, seed: int = None):
        self.cfg = cfg or EnvConfig()
        self.rng = random.Random(self.cfg.seed if seed is None else seed)
        self.hand: Optional[rules.Hand] = None
        self._played = {s: set() for s in rules.SEATS}
        self._last_actor = None

    # ------------------------------------------------------------ 开局

    def reset(self, level=None, hands=None, first=None, prev_ranks=None) -> Observation:
        lv = level if level is not None else self.cfg.level
        if lv is None:
            lv = self.rng.randint(1, 13)
        h = rules.new_hand(self.rng, level=lv, hands=hands, first=first)
        if self.cfg.tribute or prev_ranks is not None:
            rules.apply_tribute(h, prev_ranks)      # 会顺手把 turn 设成先出者
        self.hand = h
        self._played = {s: set() for s in rules.SEATS}
        self._last_actor = None
        return self.observe()

    # ------------------------------------------------------------ 查询

    @property
    def done(self) -> bool:
        return self.hand.over

    @property
    def ranks(self) -> list:
        return self.hand.ranks() if self.hand.over else None

    def legal(self) -> list:
        """当前该谁出，他的候选（含 `None` = 过）。"""
        return self.hand.actions(self.hand.turn)

    def observe(self, seat: int = None) -> Observation:
        a = self.hand.turn if seat is None else seat
        return observe(self.hand, a, self._played, self.hand.table)

    # ------------------------------------------------------------ 一步

    def step(self, index: int):
        """走第 `index` 个候选。返回 `(obs, reward, done, info)`。

        `reward` 只在**这一手结束的那一步**非零，且是**出牌人所在队**的收益
        （spec §5.4 的零点五口径）。中间步恒为 0 —— 牌类游戏中间没有即时反馈，
        DMC 的做法是拿终局 reward 当所有决策点的回归目标，那在训练循环里做。
        """
        seat = self.hand.turn
        acts = self.legal()
        if not 0 <= index < len(acts):
            raise rules.IllegalPlay(f"动作下标 {index} 越界（候选 {len(acts)} 个）")
        chosen = acts[index]
        self.hand.play(seat, chosen)
        if chosen is not None:
            self._played[seat] |= set(chosen.cards)
        self._last_actor = seat

        r = 0.0
        if self.hand.over:
            r = rules.reward(self.hand.ranks(), seat)
        info = {"seat": seat, "meld": chosen}
        return self.observe(), r, self.hand.over, info

    # ------------------------------------------------------------ 自对弈

    def rollout(self, policy) -> list:
        """跑完一手牌，返回每个决策点 `(obs, 候选, 选中下标, 出牌人)`。

        `policy(obs, actions) -> index`。`rules.reward` 的终局值由调用方回填 ——
        这里只负责把决策点如实记下来。
        """
        out = []
        obs = self.observe()
        while not self.done:
            acts = self.legal()
            i = policy(obs, acts)
            out.append((obs, acts, i, self.hand.turn))
            obs, _r, _done, _info = self.step(i)
        return out
```

**别忘了在文件头补 `import random`**（`GuandanEnv` 用到）。

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_step.py -q`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add net/sim/env.py tests/test_env_step.py
git commit -m "feat: env 动作编码 + step/reward + 自对弈 rollout"
```

---

## Task 9: 明牌泄漏自动化测试（spec §6③ 最后一条）

> spec 原文：「**最后一条必须自动化**：构造「对手手牌不同、其余相同」的两个局面，
> 若网络给两者的打分**一样** → 状态里确实无对手信息；不一样 → 泄漏。
> **手动看代码看不出来，必须写成测试。**」
>
> 这一条比其它所有验收都重要 —— 泄漏是**静默**的：训练分数会很漂亮。

**Files:**
- Create: `tests/test_env_leak.py`

**Interfaces:**
- Consumes: `env.GuandanEnv` / `env.encode_state` / `env.observe` / `env.encode_action`
- Produces: 无（纯测试）

- [ ] **Step 1: 写测试**

```python
# tests/test_env_leak.py
"""明牌泄漏的自动化测试（spec §6③ 最后一条 / §3）。

**为什么必须是测试而不是「看代码」**：泄漏不会报错，只会让训练分数变好看，
然后在真机上全废。看代码看不出来，只有构造对照局面才抓得住。
"""
import numpy as np
import pytest

from net import cards
from net.sim import env, meld, rules

A = meld.cid_from_name


def _obs_with_opponent_hand(opp_cards):
    """可观测部分**完全相同**，只有「对手手上还有什么」不同。"""
    h = rules.Hand(hands=[{A("S3"), A("S4")},          # 我
                          set(opp_cards),              # 下家
                          {A("S6")}, {A("S7")}],
                   level=2, turn=0)
    played = {s: set() for s in rules.SEATS}
    played[0] = {A("S8")}
    return env.observe(h, 0, played, None)


def test_two_games_differing_only_in_opponent_hands_encode_identically():
    b = _obs_with_opponent_hand({A("HJ"), A("HQ")})   # 同样 2 张，只换具体牌
    b = _obs_with_opponent_hand({A("HJ"), A("HQ"), A("HK")})
    assert np.array_equal(env.encode_state(a), env.encode_state(b)), (
        "两个只差「对手手牌」的局面编出了不同的状态向量 —— **明牌泄漏**。\n"
        "最可能的原因：`encode_state` 或 `observe` 摸了 `Hand.hands` 里不属于自己的那几家。"
    )


def test_opponent_hand_size_difference_does_leak_by_design_and_that_is_fine():
    """**对手剩几张是公开信息，本来就该看得见**（记牌器也给人类这个信息，spec §4.1）。

    这条测试是拿来划清边界的：泄漏指的是「对手手上**具体是哪些牌**」，
    不是「对手还剩几张」。别把这一条误当成泄漏去「修」。"""
    a = _obs_with_opponent_hand({A("S9")})
    b = _obs_with_opponent_hand({A("S9"), A("ST")})
    assert not np.array_equal(env.encode_state(a), env.encode_state(b))


def test_encoding_never_reads_the_discarded_pile_of_an_opponent():
    """把对手**出过的牌**也换掉 —— 但那也是公开信息，所以**必须**导致编码不同。
    这条与上一条一起把「公开 / 私有」的界线钉在测试里。"""
    h1 = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    h2 = rules.Hand(hands=[{A("S3")}] + [set() for _ in range(3)], level=2, turn=0)
    p1 = {s: set() for s in rules.SEATS}
    p2 = {s: set() for s in rules.SEATS}
    p1[1] = {A("S9")}
    p2[1] = {A("ST")}
    assert not np.array_equal(env.encode_state(env.observe(h1, 0, p1, None)),
                              env.encode_state(env.observe(h2, 0, p2, None)))


@pytest.mark.parametrize("seed", range(40))
def test_a_randomized_policy_scores_identically_on_paired_opponent_hands(seed):
    """**spec §6③ 的原话就是「若网络给两者的打分一样」** —— 所以这里真的接一个
    小网络，验打分而不是只验向量。向量那一层已经被上面钉住了，
    这一条防的是「以后有人给 env 加了个绕过 `Observation` 的入口」。"""
    import torch
    torch.manual_seed(seed)

    net = torch.nn.Sequential(torch.nn.Linear(env.STATE_DIM, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    for p in net.parameters():
        p.data.normal_(0, 0.1)

    def score(opp_cards):
        obs = _obs_with_opponent_hand(opp_cards)
        with torch.no_grad():
            return net(torch.from_numpy(env.encode_state(obs))).item()

    rng = np.random.default_rng(seed)
    free = [c for c in range(0, 334) if cards.is_card(c)
            and c not in {A("S3"), A("S4"), A("S6"), A("S7"), A("S8")}]
    picks = rng.choice(len(free), size=4, replace=False)
    # 两边都是 **2 张**，只有具体是哪两张不同
    x = score({free[picks[0]], free[picks[1]]})
    y = score({free[picks[2]], free[picks[3]]})
    y = score({deck[picks[2]], deck[picks[3]], deck[picks[4]], deck[picks[5]]})
    assert x == y, f"网络给两个只差对手手牌的局面打出了不同的分：{x} vs {y} —— 泄漏"
```

- [ ] **Step 2: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_leak.py -q`
Expected: 全部 passed

**如果第一条红了** —— 那就是真的抓到了泄漏，去查 `observe()` 有没有多填字段。
**不要改测试让它变绿。**

- [ ] **Step 3: 提交**

```bash
git add tests/test_env_leak.py
git commit -m "test: 明牌泄漏的自动化对照测试（spec §6③ 最后一条）"
```
## Task 10: `tools/bench_sim.py` —— 自对弈吞吐实测（spec §9 风险 1）

> spec §9 风险 1 原文：「**模拟器吞吐** —— 自对弈要每秒数千局，纯 Python 一秒几十局
> 都到不了 | 应对：裁判写成可批量形式，**先测吞吐再动训练**」。
>
> **这个任务的产出是一个数字，不是一段代码。** 它的作用是让 Plan 3（训练）
> 的规模决策有依据，而不是先写完训练再发现一秒跑不了几局。

**Files:**
- Create: `tools/bench_sim.py`
- Test: 无（它是一个测量脚本，本身没有可断言的正确性 —— 但它**必须**打印出可复现的命令）

**Interfaces:**
- Consumes: `env.GuandanEnv`、`env.EnvConfig`
- Produces: `bench_sim.run(seconds: float, seed: int) -> dict`（键 `games` / `decisions` / `games_per_sec` / `decisions_per_sec`）；`bench_sim.main(argv) -> int`

- [ ] **Step 1: 写 `tools/bench_sim.py`**

```python
"""自对弈吞吐实测（spec §9 风险 1）。

跑法：
    .venv/Scripts/python.exe -m tools.bench_sim            # 默认 10 秒
    .venv/Scripts/python.exe -m tools.bench_sim 60         # 跑 60 秒

**测的是「随机策略下的规则引擎 + 环境」的吞吐** —— 不含网络前向。
真实训练里每个决策点还要过一次网络，所以这里出来的数字是**上界**，
写进结论时要说明这一点，别拿它当训练速度。

输出里必须给出「一千万局要多久」—— spec §5.2 的量级是千万局，
不换算成小时的话这个数字没有决策价值。
"""
from __future__ import annotations

import random
import sys
import time

from net.sim import env


def run(seconds: float = 10.0, seed: int = 0) -> dict:
    e = env.GuandanEnv(seed=seed)
    rng = random.Random(seed)
    games = decisions = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        e.reset()
        while not e.done:
            e.step(rng.randrange(len(e.legal())))
            decisions += 1
        games += 1
    dt = time.perf_counter() - t0
    return {
        "seconds": dt,
        "games": games,
        "decisions": decisions,
        "games_per_sec": games / dt,
        "decisions_per_sec": decisions / dt,
        "decisions_per_game": decisions / max(games, 1),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    seconds = float(argv[0]) if argv else 10.0
    r = run(seconds)
    print(f"跑了 {r['seconds']:.1f} 秒：{r['games']} 局 / {r['decisions']} 个决策点")
    print(f"  局/秒        {r['games_per_sec']:,.1f}")
    print(f"  决策点/秒    {r['decisions_per_sec']:,.1f}")
    print(f"  每局决策点   {r['decisions_per_game']:.1f}")
    gps = r["games_per_sec"]
    if gps > 0:
        print(f"\n换算：一千万局（spec §5.2 的量级）" 
              f"≈ {1e7 / gps / 3600:,.1f} 小时 ≈ {1e7 / gps / 86400:,.1f} 天")
    print("\n**这是上界**：不含网络前向与反向。真实训练速度一定比它低，"
          "低多少取决于 batch 与设备。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 跑一次，把数字记下来**

Run: `.venv/Scripts/python.exe -m tools.bench_sim 20`

**把输出原样贴进 `docs/superpowers/plans/2026-09-25-plan2-delivery-and-rulings.md`
（Task 12 会建这个文件）**，并写一句判断：这个量级下 Plan 3 的规模该定多少、
要不要先把规则引擎搬到 C++/Rust/numba。**不要在这里动手优化** —— 这一任务只测量。

- [ ] **Step 3: 提交**

```bash
git add tools/bench_sim.py
git commit -m "feat: tools/bench_sim.py —— 自对弈吞吐实测（spec §9 风险 1）"
```

---

## Task 11: `train/smoke.py` —— 最小 DMC 冒烟（spec §7 第一行）

> **这个任务的唯一目的**：证明 `env.py` 真的能训 —— 跑起来、分数往上走。
> spec §7 第一行：「胜率 vs 随机 | 应很快到 90%+；**到不了说明状态/动作编码有问题**」。
>
> **不碰** §7 的分水岭「打得过贪心」—— 那是 Plan 3。

**Files:**
- Modify: `net/sim/env.py`（加 `encode_history`）
- Create: `train/__init__.py`、`train/smoke.py`
- Test: `tests/test_env_history.py`

**Interfaces:**
- Consumes: Task 8 的 `GuandanEnv` / `encode_action`；`rules.reward`
- Produces:
  - `env.HISTORY_LEN = 15`、`env.HISTORY_DIM = 147`、`env.encode_history(hand, seat) -> np.ndarray`（形状 `(15, 147)`）
  - `train.smoke.main(argv) -> int`

- [ ] **Step 1: 写 `encode_history` 的失败测试**

```python
# tests/test_env_history.py
import numpy as np

from net import cards
from net.sim import env, meld, rules

A = meld.cid_from_name


def test_history_shape_and_zero_padding_at_the_front():
    h = rules.Hand(hands=[{A("S3")}] * 4, level=2, turn=0)
    v = env.encode_history(h, 0)
    assert v.shape == (env.HISTORY_LEN, env.HISTORY_DIM)
    assert v.sum() == 0, "还没出过牌，历史必须全 0"


def test_recent_steps_are_right_aligned_and_carry_the_relative_seat():
    """最近一手必须在**最后一行**（右对齐），且带出牌人的**相对座位**。

    历史里不带「谁出的」会丢掉一半信息 —— 同样一张 9♠ 是下家出的还是对家出的，
    对判断局面完全不同。所以每行是 `encode_action(m) ⊕ 相对座位 one-hot(4)`。
    """
    h = rules.Hand(hands=[{A("S3")}, {A("S4")}, {A("S5")}, {A("S6")}], level=2, turn=0)
    h.play(0, meld.as_meld([A("S3")], 2))
    v = env.encode_history(h, 1)                      # 从座位 1 的视角看
    assert v[:env.HISTORY_LEN - 1].sum() == 0         # 只有一步，前面全是 0
    row = v[-1]
    assert row[:env.ACTION_DIM].sum() > 0             # 有动作
    assert row[env.ACTION_DIM:].argmax() == 3         # 座位 0 对座位 1 来说是**上家**（相对 3）


def test_history_is_truncated_to_the_last_15_steps():
    h = rules.Hand(hands=[{A("S3"), A("S4"), A("S5")},
                          {A("S6"), A("S7")}, {A("S8")}, {A("S9"), A("ST")}],
                   level=2, turn=0)
    for _ in range(3):
        if h.over:
            break
        s = h.turn
        acts = [a for a in h.actions(s) if a is not None]
        h.play(s, acts[0])
    v = env.encode_history(h, 0)
    assert v.shape == (env.HISTORY_LEN, env.HISTORY_DIM)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_history.py -q`
Expected: FAIL — `AttributeError: module 'net.sim.env' has no attribute 'encode_history'`

- [ ] **Step 3: 实现 `encode_history`（追加到 `net/sim/env.py`）**

```python
# ---------------------------------------------------------------- 动作历史

#: spec §4.1「最近 15 手的动作序列 → LSTM」。
HISTORY_LEN = 15

#: 每一行 = `encode_action(m)` ⊕ **出牌人的相对座位** one-hot(4)。
#:
#: ⚠️ 那 4 维是**在 spec §4.2 的动作编码之外加的**，加的理由是：
#: 同一个 9♠ 是下家出的还是对家出的，对判断局面完全不同 —— 不记「谁出的」，
#: 这 15 行能提供的信息会少一大半。动作编码本身仍严格按 §4.2（143 维）。
HISTORY_DIM = ACTION_DIM + 4
assert HISTORY_DIM == 147


def encode_history(hand: "rules.Hand", seat: int) -> np.ndarray:
    """最近 `HISTORY_LEN` 步，**右对齐**（最近的落在最后一行），新局前面补 0。

    历史里全是**公开信息**（谁出了什么、谁过了）—— 不构成明牌泄漏。
    """
    v = np.zeros((HISTORY_LEN, HISTORY_DIM), dtype=np.float32)
    steps = hand.steps[-HISTORY_LEN:]
    for i, st in enumerate(steps):
        row = v[HISTORY_LEN - len(steps) + i]
        row[:ACTION_DIM] = encode_action(st.meld, hand.level)   # 过 = 全 0
        row[ACTION_DIM + (st.seat - seat) % 4] = 1.0
    return v
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_env_history.py -q`
Expected: 3 passed

- [ ] **Step 5: 写 `train/smoke.py`**

```python
"""最小 DMC 冒烟（spec §7 第一行）—— **只证明 env 能训，不是训练实验**。

跑法：
    .venv/Scripts/python.exe -m train.smoke                 # 默认 3 分钟
    .venv/Scripts/python.exe -m train.smoke 600             # 10 分钟

判据（写死在退出码里）：
    末次评测的胜率 **≥ 70%**（随机基线 50%），且最后 3 次评测**单调不降**。
    spec §7 写的「很快到 90%+」是**目标**；几分钟 CPU 训练到不了 90% 不一定是编码错。
    但**打不过 70% 或曲线不涨，就是编码有问题** —— 先去查 `env`，别加大训练量。

网络形状照 spec §4.3（对齐 DouZero）：
    历史(15×147) → LSTM(128) ─┐
                              ├→ 拼接 → 6 层 MLP(512) → 一个 Q 值
    状态(700) + 单个候选动作(143) ┘
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from net.sim import env, rules

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_GAMES = 32              # spec §5.3：batch 32 局
LR = 1e-4
EPS_START, EPS_END = 1.0, 0.1
MLP_LAYERS = 6
MLP_HIDDEN = 512
LSTM_HIDDEN = 128


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(env.HISTORY_DIM, LSTM_HIDDEN, batch_first=True)
        layers, d = [], env.STATE_DIM + env.ACTION_DIM + LSTM_HIDDEN
        for _ in range(MLP_LAYERS):
            layers += [nn.Linear(d, MLP_HIDDEN), nn.ReLU()]
            d = MLP_HIDDEN
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, state, action, hist):
        _out, (h, _c) = self.lstm(hist)
        return self.mlp(torch.cat([state, action, h[-1]], dim=-1)).squeeze(-1)


def self_play_batch(net, rng, eps):
    """跑 `BATCH_GAMES` 局 ε-greedy 自对弈，返回 `(X, y)`。

    **DMC 的关键**：中间步没有即时反馈，所以每个决策点的回归目标是
    「这一局打完，他所在队拿了多少」—— 一次前向都不做，整局结束再回填。
    """
    st, ac, hi, y = [], [], [], []
    for _ in range(BATCH_GAMES):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        points, picks = [], []
        obs = e.observe()
        while not e.done:
            acts = e.legal()
            hist = env.encode_history(e.hand, e.hand.turn)
            if rng.random() < eps:
                i = rng.randrange(len(acts))
            else:
                with torch.no_grad():
                    q = net(torch.from_numpy(env.encode_state(obs)).unsqueeze(0).to(DEVICE),
                            torch.from_numpy(np.stack(
                                [env.encode_action(a, obs.level) for a in acts])).to(DEVICE),
                            torch.from_numpy(hist).unsqueeze(0).to(DEVICE))
                    i = int(q.argmax())
            # ⚠️ **历史必须在这里就取下来**（`hist` 是这一步之前的最近 15 手）。
            # 整局结束之后再 `encode_history(e.hand, seat)` 会把**后面才发生的牌**
            # 塞进历史的最后几行 —— 那是另一种泄漏（未来信息），而且同样静默。
            # `evaluate()` 里是在循环内取的，别把两处写成不一样。
            points.append((obs, acts, i, e.hand.turn, hist))
            obs, _r, _d, _info = e.step(i)
        ranks = e.ranks
        for o, a, i, seat, hist in points:
            st.append(env.encode_state(o))
            ac.append(env.encode_action(a[i], o.level))
            hi.append(hist)
            y.append(rules.reward(ranks, seat))          # 整局终局 reward，直接回归
    return (torch.from_numpy(np.stack(st)).to(DEVICE),
            torch.from_numpy(np.stack(ac)).to(DEVICE),
            torch.from_numpy(np.stack(hi)).to(DEVICE),
            torch.tensor(y, dtype=torch.float32).to(DEVICE))


def evaluate(net, games=200, seed=999):
    """座位 0、2 用网络（贪心），座位 1、3 随机 —— 报「我们的胜率」。"""
    rng = random.Random(seed)
    wins = 0
    for _ in range(games):
        e = env.GuandanEnv(seed=rng.randrange(1 << 30))
        e.reset()
        obs = e.observe()
        while not e.done:
            seat = e.hand.turn
            acts = e.legal()
            if rules.TEAM[seat] == 0:
                with torch.no_grad():
                    q = net(torch.from_numpy(env.encode_state(obs)).unsqueeze(0).to(DEVICE),
                            torch.from_numpy(np.stack(
                                [env.encode_action(a, obs.level) for a in acts])).to(DEVICE),
                            torch.from_numpy(env.encode_history(e.hand, seat))
                            .unsqueeze(0).to(DEVICE))
                    i = int(q.argmax())
            else:
                i = rng.randrange(len(acts))
            obs, _r, _d, _info = e.step(i)
        if rules.winner_team(e.ranks) == 0:
            wins += 1
    return wins / games


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    budget = float(argv[0]) if argv else 180.0

    torch.manual_seed(0)
    rng = random.Random(0)
    net = QNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)          # spec §5.3：Adam / 1e-4
    print(f"device={DEVICE}  预算 {budget:.0f} 秒  batch={BATCH_GAMES} 局\n")

    t0 = time.perf_counter()
    games = 0
    history = []
    while time.perf_counter() - t0 < budget:
        eps = EPS_START + (EPS_END - EPS_START) * min(
            1.0, (time.perf_counter() - t0) / budget)         # ε 线性退火
        st, ac, hi, y = self_play_batch(net, rng, eps)
        loss = nn.functional.mse_loss(net(st, ac, hi), y)
        opt.zero_grad(); loss.backward(); opt.step()
        games += BATCH_GAMES
        el = time.perf_counter() - t0
        print(f"  {el:6.1f}s  局数 {games:6d}  ε={eps:.2f}  loss={loss.item():.3f}  "
              f"{games / el:.1f} 局/秒")
        if games % (BATCH_GAMES * 20) == 0:
            wr = evaluate(net)
            history.append(wr)
            print(f"     >> 胜率 vs 随机 = {wr:.1%}")

    wr = evaluate(net)
    history.append(wr)
    print(f"\n末次胜率 vs 随机：{wr:.1%}   曲线 {['%.0f%%' % (h * 100) for h in history]}")

    rising = len(history) < 3 or all(b >= a - 0.02 for a, b in zip(history[-4:-1],
                                                                  history[-3:]))
    ok = wr >= 0.70 and rising
    print("冒烟" + ("通过 ✓  —— env 能训" if ok else
                    "**未通过** ✗ —— 先查 env 的状态/动作编码，别加大训练量"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 跑冒烟**

Run: `.venv/Scripts/python.exe -m train.smoke 180`
Expected: 胜率明显高于 50%，退出码 0

**红了先查这三处**（按可能性排序）：
1. `encode_state` 的座位相对化 —— 四个座位共享一套权重，相对化错了学习信号就是噪声
2. `rules.reward` 的视角 —— 必须是**出牌人所在队**
3. `y` 的回填 —— 是「这一局打完他拿了多少」，不是每一步的即时奖励

- [ ] **Step 7: 提交**

```bash
git add net/sim/env.py train/ tests/test_env_history.py
git commit -m "feat: encode_history + train/smoke.py 最小 DMC 冒烟（评：打得过随机）"
```

---

## Task 12: 收尾 —— 决策台账 + HANDOFF

**Files:**
- Create: `docs/superpowers/plans/2026-09-25-plan2-delivery-and-rulings.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: 写决策台账**

照着 Plan 1 的做法（`docs/superpowers/plans/2026-09-24-plan1-delivery-and-rulings.md`）
建一份，**执行期做的每一个判断都记进去**，尤其是这几条必须在内：

- 进贡的四格证据分级（哪格是硬证据、哪格是基线、验收报出来的比例是多少）
- 双上「3 还是 4」的实测分布（spec §13.4），以及为什么这一格在 Plan 2 内无解
- 倍数项为什么默认关（spec §13.4 的实测）
- `encode_history` 多加的那 4 维相对座位（对 spec §4.2 的有意偏离及理由）
- 吞吐实测的原始输出（Task 10 贴进来的那段）
- 冒烟的最后胜率曲线

- [ ] **Step 2: 更新 `HANDOFF.md`**

把「下一步」那一节改成 Plan 3（训练）的入口，并写清楚：

- Plan 2 交付了什么、`pytest` 多少 passed、两条验收命令（`tools/accept_sim`、`tools/accept_tribute`）
- **吞吐数字**（Plan 3 的规模决策依据）
- `tribute` 在训练里默认关（`EnvConfig.tribute=False`），原因写在 `EnvConfig` 的 docstring 里
- 还没做的：升级/过A（spec §13.1 裁定不做）、影子模式（spec §8）、推理链（spec §8.1）

- [ ] **Step 3: 提交**

```bash
git add docs/superpowers/plans/2026-09-25-plan2-delivery-and-rulings.md HANDOFF.md
git commit -m "docs: Plan 2 决策台账 + HANDOFF 指向 Plan 3"
```
