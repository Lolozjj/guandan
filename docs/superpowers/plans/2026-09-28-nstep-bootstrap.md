# 换目标：n 步自举 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 DMC 的回归标签从「整局的终局分」变成「往后走 n 步、按目标网络估的值
＋ 终局分的混合」，一次修好「一局 ~132 个决策点共享同一个标签」这件事。

**Architecture:** 标签仍然只有一个产地（`replay.mc_targets` → `y_mc`），自举项以
**第二个纯函数**（`replay.blend`）挂在它外面，只在训练回路里调用一次。
自举要的后继局面 `s_{t+n}` 由 `replay.expand` 在既有的一次重放里顺手产出
（多一个 `n` 参数与一个定长环），值 `V̄(s) = max_a Q̄(s,a)` 由**目标网络**
（`net` 的冻结副本，每 1000 局同步）算，worker 侧完全不动。

**Tech Stack:** Python 3 + PyTorch（CPU）、pytest、既有 `net/sim` 规则引擎与 `env` 编码器。

**Spec:** `docs/superpowers/specs/2026-09-28-nstep-bootstrap-design.md`（本计划从它论证；
执行者两份都要读）

## Global Constraints

- **标签唯一的产地不变**：`y_mc` 只能由 `replay.mc_targets` 产出；自举只许经
  `replay.blend` 进入，且只许在**一处**调用（`train/selfplay.py::_learn_step`）。
- **`β = 1` 时必须与现状逐点相等**（默认值就是 1.0）—— 有测试钉着。
- **目标项不许带梯度**：`q_max_batch` 返回 **Python float**（不是 tensor），
  `blend` 也返回 float，`y` 由 `torch.tensor(list[float])` 造 —— 结构上就带不了梯度。
  目标网络 `requires_grad=False`，算它时 `torch.no_grad()`。
- **worker 完全不动**：自举只在学习者那一侧（`train/worker.py` 一行都不改）。
- **γ = 1.0（不打折）**：一手牌是有限回合、无时间价值，保持标签尺度与现状一致。
- **初值**：`N_STEP = 3`、`MC_MIX = 1.0`（默认；处理臂用 `0.5`）、
  `TGT_SYNC_GAMES = 1000`、`Q_ABS_MAX = 30.0`。
- **发散必须响**：一个 batch 里 `|Q|` 或 `|标签|` 的最大值 `> 30` 就 `raise`，
  并打出当时的局数与 loss（本项目纪律：失败必须响，不许静默地训下去）。
- **测试基线**：改动前 `407 passed / 1 failed`，唯一那条红是**预存**的
  `tests/test_tribute_records.py`（日志轮转，见现用方案 §七.3）。每个 task 结束时
  必须是「除它以外全绿」。
- **命令一律**：`.venv/Scripts/python.exe`，训练前 `$env:GUANDAN_DEVICE="cpu"`。
- **`runs/rl/` 绝不写**（面板按修改时间加载 `runs/rl/*/best.pt`，写了就是**静默换源**）。
  A/B 与测量一律落 `runs/ab/` 或临时目录。
- **A/B 两臂串行**，不许并发（每臂约 1.2 GB×worker 数，2026-09-27 三条并发把机器打爆过）。
- **预登记的判据不许事后改**（spec §1.5）：判据 1 `vs 贪心` 不掉 ≥1 个标准差；
  判据 2 动作边际变宽；两者必须一起读。

## Review Focus

执行者做完每个 task 后，逐条确认这几件事**没有被自己踩到**（它们都是「不报错、
但结果悄悄错」的类型，本仓库为同类问题反复吃过亏）：

1. **一台机器上并发跑两臂会内存爆** —— 本计划的 A/B 必须一条一条起。
2. **`β < 1` 时那条「刚打完的这一批」快捷缓存必须让路** —— 缓存里没有自举值，
   用了它就会让一部分样本悄悄退回纯 MC，而 loss 曲线上看不出来（Task 4 专门钉）。
3. **`boot` 与 `points` 的下标必须同长同序** —— 错开一格就是「拿别人的未来当自己的标签」。
4. **目标网络被卷进梯度图** —— 忘了 `no_grad` / 返回了 tensor 就是「对目标求导」，
   静默地学歪；靠返回 float 在**类型上**断掉。
5. **越界、NaN、无穷** —— `t + n` 越过终局的点整项退回 MC（不许编一个 0）；
   发散守门要用 `not (x <= limit)` 写，否则 NaN 会悄悄溜过去。

---

### Task 1: `replay.blend` —— 自举进入标签的唯一一处

**Files:**
- Modify: `train/replay.py`（在 `mc_targets` 之后加一个纯函数）
- Test: `tests/test_blend.py`（新建）

**Interfaces:**
- Consumes: 无（纯标量运算，不 import torch —— `replay.py` 的纪律）
- Produces: `blend(y_mc: list[float], boot: list[float | None] | None, beta: float = 1.0) -> list[float]`

- [ ] **Step 1: 写失败测试**

```python
"""`blend` —— MC 标签与自举值的混合（spec §3.1）。"""
import pytest

from train import replay


def test_beta_one_is_exactly_the_old_labels():
    """β=1 = 现在的 DMC（默认）。**逐点相等**，不是「约等于」。"""
    y = [-3.0, 1.0, 2.0]
    assert replay.blend(y, [9.9, 9.9, 9.9], 1.0) == y


def test_beta_one_ignores_a_poisoned_bootstrap_value():
    """β=1 时**连看都不看**自举值 —— 结构上相等，不是浮点凑出来的。

    把自举值塞成 inf：若实现写成 (1-β)·v + β·y，会算出 0.0*inf = nan。"""
    assert replay.blend([1.0], [float("inf")], 1.0) == [1.0]


def test_beta_zero_is_pure_bootstrap():
    assert replay.blend([1.0, -3.0], [2.0, 4.0], 0.0) == [2.0, 4.0]


def test_half_mixes_halfway():
    assert replay.blend([0.0], [4.0], 0.5) == [2.0]


def test_none_falls_back_to_the_mc_label():
    """越过终局、或本轮不算自举的点，**整项退回 MC**（不许编一个 0）。"""
    assert replay.blend([1.0, -3.0], [None, 5.0], 0.0) == [1.0, 5.0]


def test_all_none_is_the_mc_labels():
    assert replay.blend([1.0, -3.0], [None, None], 0.5) == [1.0, -3.0]


def test_no_boot_at_all_is_the_mc_labels():
    """`boot=None` 的整条路（`n=0`）也要能用。"""
    assert replay.blend([1.0, -3.0], None, 0.5) == [1.0, -3.0]


def test_length_mismatch_raises():
    """错开一格就是**拿别人的未来当自己的标签** —— 必须炸，不许 zip 截断。"""
    with pytest.raises(ValueError):
        replay.blend([1.0, 2.0], [1.0], 0.5)


@pytest.mark.parametrize("beta", [-0.1, 1.5])
def test_beta_out_of_range_raises(beta):
    with pytest.raises(ValueError):
        replay.blend([1.0], [1.0], beta)
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blend.py -q`
Expected: FAIL —— `AttributeError: module 'train.replay' has no attribute 'blend'`

- [ ] **Step 3: 写实现**

```python
def blend(y_mc, boot, beta: float = 1.0) -> list:
    """把 MC 标签与自举值按 β 混合 —— **自举进入标签的唯一一处**（spec §3.1）。

        y_t = (1 − β) · V̄(s_{t+n})  +  β · y_mc

    - `beta = 1.0` 时**原样返回 `y_mc`**（默认 = 现在的 DMC，逐点相等）。
      这里刻意提前返回、而不是算 `(1-β)*v + β*y` —— 结构上相等，而且
      顺带不会去碰 `boot`（那条路上 `boot` 可能是空表或全是 None）。
    - `boot` 里某一项是 `None`（越过终局 / 本轮不算自举）→ 那一项整项退回 `y_mc`。
      **不许拿 0 或上一项顶上** —— 那是凭空造一个未来。
    - 长度必须对齐：错开一格就是「拿别人的未来当自己的标签」，
      而这种错在 loss 曲线上完全看不出来（本仓库纪律：失败必须响）。
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"β 必须在 [0, 1]：{beta}")
    if beta >= 1.0 or boot is None:
        return list(y_mc)
    if len(boot) != len(y_mc):
        raise ValueError(f"自举值与标签长度不一致：{len(boot)} vs {len(y_mc)}")
    return [y if v is None else (1.0 - beta) * v + beta * y
            for y, v in zip(y_mc, boot)]
```

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_blend.py tests/test_bomb_cost.py -q`
Expected: PASS（9 条 + 原有那批）

- [ ] **Step 5: 提交**

```bash
git add train/replay.py tests/test_blend.py
git commit -m "feat(replay): blend —— 自举进入标签的唯一一处（β=1 逐点等于现状）"
```

---

### Task 2: `replay.expand` 顺手产出 `s_{t+n}`

**Files:**
- Modify: `train/replay.py::expand`（签名与返回都变）
- Modify: `train/selfplay.py:469`、`train/selfplay.py:622`（两处调用点）
- Modify: `tests/test_bomb_cost.py:87,104`、`tests/test_expand_learn.py:34,49,57,64`、
  `tests/test_train_replay.py:35`（七处解包）
- Test: `tests/test_expand_boot.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: `expand(rec, bomb_cost: float = 0.0, n: int = 0) -> (points, y_mc, boot)`，
  其中 `boot[i]` 是 `(obs, acts, hist)` 或 `None`，**与 `points` 等长同序**；
  `n = 0` 时 `boot == []`

> **对 spec 的一处更正（执行时按这里做）**：spec §3.2 说调用方包括
> `replay.play_capturing` —— **它不调用 `expand`**（它自己走 `mc_targets`），
> 不需要改。真正的调用点是 `train/selfplay.py` 的两处 + `tests/` 的七处。

- [ ] **Step 1: 写失败测试**

```python
"""`expand(n=)` 产出的自举源必须是**真的** `s_{t+n}`，而且与 points 对齐。"""
import random

import numpy as np
import pytest

from net.sim import env
from train import replay
from train.policies import greedy_policy


def _rec(seed=0, level=5):
    rec, _pts, _y = replay.play_capturing(greedy_policy, random.Random(seed),
                                          level=level, capture=False)
    return rec


def _kept_mask(rec):
    """哪些步是「保留的决策点」（`learn` 为 None 时四家都保留）。"""
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    out = []
    for i in rec.actions:
        out.append(learn is None or e.hand.turn in learn)
        e.step(i)
    return out
```

def test_n_zero_returns_no_boot_and_the_old_points():
    """`n=0`（默认）与老行为逐点相等 —— `boot` 是空表。"""
    rec = _rec()
    pts, y, boot = replay.expand(rec)
    assert boot == []
    assert len(pts) == len(y) == len(rec.actions)


def test_boot_is_aligned_with_points():
    rec = _rec()
    pts, y, boot = replay.expand(rec, n=3)
    assert len(boot) == len(pts) == len(y)
    assert any(b is not None for b in boot), "至少有一个点该有自举源"
    assert all(b is None or len(b) == 3 for b in boot), "源是 (obs, acts, hist)"


def test_boot_is_really_the_state_n_steps_later():
    """**不靠自证**：另外手工走一遍 env，比对第 t+n 步的 obs 与历史。

    ⚠️ 这条是本次改动的核心正确性 —— 数错一步（t+n-1 或 t+n+1）在训练里
    只会表现为「学得慢一点」，任何 loss 曲线都看不出来。
    """
    rec = _rec(seed=3)
    # 手工重放，逐步记下「每一步开始前」的 (obs, hist)
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    steps = []
    for i in rec.actions:
        steps.append((e.observe(), env.encode_history(e.hand, e.hand.turn)))
        e.step(i)
    kept = [t for t, ok in enumerate(_kept_mask(rec)) if ok]
    for n in (1, 2, 3):
        _pts, _y, boot = replay.expand(rec, n=n)
        for j, b in enumerate(boot):
            t = kept[j]
            if t + n >= len(steps):
                assert b is None, f"越过终局那一步该退回 MC（t={t} n={n}）"
                continue
            obs, hist = steps[t + n]
            assert b is not None, f"该有自举源（t={t} n={n}）"
            assert np.array_equal(b[0], obs), f"obs 对不上（t={t} n={n}）"
            assert np.array_equal(b[2], hist), f"历史对不上（t={t} n={n}）"


def test_negative_n_raises():
    with pytest.raises(ValueError):
        replay.expand(_rec(), n=-1)
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_expand_boot.py -q`
Expected: FAIL —— `too many values to unpack` / `unexpected keyword argument 'n'`

- [ ] **Step 3: 改实现**

```python
def expand(rec: GameRecord, bomb_cost: float = 0.0, n: int = 0):
    """把记录重放成 `(决策点, MC 标签, 自举源)`。

    决策点是 `(obs, acts, 选中下标, 出牌人, hist)` —— 与 `env.rollout` 同形状，
    训练循环因此**不需要区分**「刚打的」和「从 buffer 里取的」。

    `rec.learn` 里的座位才产出决策点（`None` = 四家都产出，老行为）。
    ⚠️ **局面必须每一步都往前走** —— 过滤只发生在 `points.append` 那一行。
    提前 `continue` 会让重放错位（这是这个函数最容易被写错的地方）。

    `n > 0` 时额外产出 `boot[i]`：第 i 个决策点**往后数 n 步**那个局面的
    `(obs, acts, hist)` —— 自举要的 `s_{t+n}`（spec §3.2）。**越过终局的点是
    `None`**（那些点整项退回 MC）。`n = 0` 时 `boot` 是空表，与老行为逐点相等。

    ⚠️ **`boot` 与 `points` 必须等长同序** —— 错开一格就是拿别人的未来当自己的标签。
    对齐靠一个**定长环**（`deque(maxlen=n+1)`）：走到第 t 步时环首正好是第 `t-n` 步，
    此刻的局面就是它要的自举源。历史**必须现在取**（`encode_history`）——
    整局打完再取会把后面的牌塞进去，那是另一种泄漏（未来信息）。
    """
    if n < 0:
        raise ValueError(f"n 不能为负：{n}")
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    seq, points, boot = [], [], []
    ring = deque(maxlen=n + 1) if n else None     # 定长环：环首 = 第 t−n 步
    obs = e.observe()
    for i in rec.actions:
        acts = e.legal()
        seat = e.hand.turn
        seq.append((seat, acts[i]))       # **全量步骤**：B_t 要沿着一局往后数
        tag = None                        # 这一步在 points 里的下标（没保留就是 None）
        if learn is None or seat in learn:
            tag = len(points)
            # 历史必须**在这一步当时**取 —— 整局打完再取会把后面的牌塞进历史
            # （那是另一种泄漏：未来信息）。selfplay 那边也是这么取的。
            points.append((obs, acts, i, seat, env.encode_history(e.hand, seat)))
            boot.append(None)             # 自举源在 t+n 步，那时才补得上
        if ring is not None:
            ring.append(tag)
            # 环满（t >= n）时环首才是「第 t−n 步」；不满时那些点的源不存在
            if len(ring) == n + 1 and ring[0] is not None:
                boot[ring[0]] = (obs, acts, env.encode_history(e.hand, seat))
        obs, _r, _done, _info = e.step(i)
    # 标签**只有一个产地**（`mc_targets`）—— 过滤也在它里面做
    return (points, mc_targets(seq, e.ranks, learn=rec.learn, bomb_cost=bomb_cost),
            boot)
```

- [ ] **Step 4: 改七处测试解包 + 两处调用点**

`tests/` 里把 `pts, y = replay.expand(...)` 一律改成 `pts, y, _b = replay.expand(...)`
（`_pts2, y_expand = ...` → `_pts2, y_expand, _b2 = ...`）。

`train/selfplay.py` 的两处（`train()` 里的 `fresh.get(id(rec)) or replay.expand(rec)`、
`train_parallel()` 里的 `pts, y = replay.expand(rec)`）这一轮**先临时**改成
`pts, y, _b = replay.expand(rec)` —— Task 4 会把它们收进共享函数。

- [ ] **Step 5: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 除 `tests/test_tribute_records.py` 那条**预存**的红以外全绿

- [ ] **Step 6: 提交**

```bash
git add train/replay.py train/selfplay.py tests/
git commit -m "feat(replay): expand(n=) 顺手产出 s_{t+n}（定长环对齐，越过终局退回 MC）"
```

---

### Task 3: `q_max_batch`（值）与 `check_q_scale`（发散守门）

**Files:**
- Modify: `train/net.py`（把批量前向抽成 `_flat_scores`，`q_argmax_batch` 改用它）
- Test: `tests/test_q_max.py`（新建）

**Interfaces:**
- Consumes: `env.encode_state/encode_action`、`QNet`
- Produces:
  - `_flat_scores(net, pending) -> (tensor_q, counts)`（私有）
  - `q_argmax_batch(net, pending) -> list[int]`（**签名与行为不变**）
  - `q_max_batch(net, pending) -> list[float | None]`（`pending` 允许含 `None`）
  - `Q_ABS_MAX = 30.0`、`check_q_scale(q_abs_max, games, loss, what="Q", limit=Q_ABS_MAX) -> None`

- [ ] **Step 1: 写失败测试**

```python
"""`q_max_batch`：自举项 `V̄(s) = max_a Q̄(s,a)`。**返回 float，不是 tensor。**"""
import random
import pytest
import torch

from net.sim import env
from train import replay
from train.net import (QNet, check_q_scale, q_argmax_batch, q_max_batch,
                       q_values)
from train.policies import greedy_policy


def _pending(games=2, seed=0):
    """一批形状正确的 `(obs, acts, hist)`。"""
    out = []
    for k in range(games):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), capture=False)
        e = env.GuandanEnv(seed=0)
        e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
        out.append((e.observe(), e.legal(), env.encode_history(e.hand, e.hand.turn)))
    return out


def test_matches_the_single_state_path():
    """批量算出来的必须**逐个等于** `q_values(...).max()` —— 最强的等价性测试。"""
    net = QNet().eval()
    pend = _pending()
    for (obs, acts, hist), v in zip(pend, q_max_batch(net, pend)):
        assert v == pytest.approx(float(q_values(net, obs, acts, hist).max()), abs=1e-5)


def test_returns_floats_not_tensors():
    """**类型上就带不了梯度** —— 目标项不许 backprop 到任何东西。"""
    assert isinstance(q_max_batch(QNet().eval(), _pending(games=1))[0], float)


def test_none_entries_are_preserved():
    """`None` 的位置原样返回 `None`（那些点整项退回 MC），**不许挪位**。"""
    pend = _pending(games=1)
    got = q_max_batch(QNet().eval(), [None, pend[0], None])
    assert got[0] is None and got[2] is None and isinstance(got[1], float)


def test_all_none_does_not_touch_the_network():
    assert q_max_batch(QNet().eval(), [None, None]) == [None, None]


def test_two_different_nets_give_different_values():
    """防「拿错了网络」—— 目标网络接错成在线网络时，这条会红。"""
    torch.manual_seed(0)
    a, b = QNet().eval(), QNet().eval()
    pend = _pending(games=1)
    assert q_max_batch(a, pend)[0] != pytest.approx(q_max_batch(b, pend)[0])


def test_no_grad_leaks_into_the_network():
    net = QNet()
    q_max_batch(net, _pending(games=1))
    assert all(p.grad is None for p in net.parameters())


def test_argmax_batch_is_unchanged_by_the_refactor():
    """重构不许改变老行为 —— 与 `q_values` 的 argmax 逐个比。"""
    net = QNet().eval()
    pend = _pending()
    for (obs, acts, hist), i in zip(pend, q_argmax_batch(net, pend)):
        assert i == int(q_values(net, obs, acts, hist).argmax())


def test_check_q_scale_raises_above_the_limit():
    with pytest.raises(RuntimeError):
        check_q_scale(30.1, games=1000, loss=1.0)


def test_check_q_scale_passes_at_the_limit():
    check_q_scale(30.0, games=1000, loss=1.0)


def test_check_q_scale_raises_on_nan():
    """NaN 与任何数比较都是 False —— 写成 `x > limit` 会让它悄悄溜过去。"""
    with pytest.raises(RuntimeError):
        check_q_scale(float("nan"), games=1000, loss=float("nan"))
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_q_max.py -q`
Expected: FAIL —— `ImportError: cannot import name 'q_max_batch'`

- [ ] **Step 3: 写实现（含把 `q_argmax_batch` 重构到 `_flat_scores` 上）**

`q_argmax_batch` 的 docstring 原文保留（它解释了「为什么必须批量」），函数体换成：

```python
def _flat_scores(net, pending):
    """`pending = [(obs, acts, hist), ...]` -> `(q, counts)`：所有候选的 Q 首尾相接。

    不等长的候选**不补 padding**：直接首尾相接，用每组的下标区间取 max/argmax。
    补 padding 会白白多算一截，而且要把「无效候选」屏蔽掉，多一处出错的机会。
    状态与历史按**局面**存一次（`idx` 那一手），只有动作是「每个候选一行」。
    """
    dev = next(net.parameters()).device
    counts = [len(acts) for _o, acts, _h in pending]
    st = np.stack([env.encode_state(o) for o, _a, _h in pending])
    hi = np.stack([h for _o, _a, h in pending])
    ac = np.stack([env.encode_action(a, o.level)
                   for o, acts, _h in pending for a in acts])
    idx = np.repeat(np.arange(len(pending)), counts)
    with torch.no_grad():
        return net(torch.from_numpy(st[idx]).to(dev),
                   torch.from_numpy(ac).to(dev),
                   torch.from_numpy(hi[idx]).to(dev)), counts


def q_argmax_batch(net, pending):
    q, counts = _flat_scores(net, pending)
    out, off = [], 0
    for c in counts:
        out.append(int(q[off:off + c].argmax()))
        off += c
    return out


def q_max_batch(net, pending):
    """`pending = [(obs, acts, hist) | None, ...]` -> `list[float | None]`。

    每个局面的 `V̄(s) = max_a Q̄(s, a)` —— 自举项（spec §3.1）。
    `None` 的位置（越过终局的点）原样返回 `None`，**位置不许挪**。

    ⚠️ 返回的是 **Python float，不是 tensor** —— 这是故意的：
    目标项绝不能带梯度，而 float 根本没法 `backward`。类型上就断掉了，不靠注释
    （本仓库对「明牌泄漏」用的也是同一招：靠类型，不靠纪律）。
    """
    idxs = [i for i, p in enumerate(pending) if p is not None]
    out = [None] * len(pending)
    if not idxs:                       # 全 None 时 `np.stack([])` 会炸，先挡掉
        return out
    q, counts = _flat_scores(net, [pending[i] for i in idxs])
    off = 0
    for i, c in zip(idxs, counts):
        out[i] = float(q[off:off + c].max())
        off += c
    return out


#: 发散守门（spec §6）：|Q| 超过这个数就**响亮地炸**。
#: 标签尺度是 ±3，一个数量级以上的偏离只可能是自举发散了。
Q_ABS_MAX = 30.0


def check_q_scale(q_abs_max: float, games: int, loss: float, what: str = "Q",
                  limit: float = Q_ABS_MAX) -> None:
    """`|Q|` 超过 `limit` 就 raise（本项目纪律：失败必须响）。

    ⚠️ **必须用 `not (x <= limit)` 写。** NaN 与任何数比较都是 False，
    写成 `x > limit` 会让 NaN 悄悄溜过去 —— 而 NaN 正是发散最典型的形态。
    """
    if not (q_abs_max <= limit):
        raise RuntimeError(
            f"{what} 的量级炸了：{q_abs_max:.4g} > {limit:g}"
            f"（第 {games:,} 局，loss={loss:.3f}）—— 自举发散，停在这里。"
            f"（标签尺度是 ±3；确认要放宽就动 train/net.py::Q_ABS_MAX）")
```

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_q_max.py tests/test_train_device.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add train/net.py tests/test_q_max.py
git commit -m "feat(net): q_max_batch（返回 float）+ 发散守门 check_q_scale；批量前向收进 _flat_scores"
```

---

### Task 4: 训练步收成一份共享实现（**行为不变**的 REFACTOR）

**Files:**
- Modify: `train/selfplay.py`（新增 `build_samples`、`_learn_step`；`train()` 与
  `train_parallel()` 各删掉自己那段）
- Test: `tests/test_learn_step.py`（新建）

**Interfaces:**
- Consumes: `replay.expand`（Task 2）、`_tensors`（既有）
- Produces:
  - `build_samples(buf, rng, batch_games, *, fresh=None, bomb_cost=0.0, n_step=0) -> (samples, y_mc, boot)`
  - `_learn_step(net, net_tgt, buf, rng, opt, games, *, batch_games, bomb_cost, mc_mix, n_step, fresh=None) -> float`

> 这一轮 `_learn_step` **还不算自举**（`n_boot` 恒为 0、不建目标网络）——
> 目的是先把两条路线的重复代码收成一份，**行为逐字不变**，由既有测试守住。
> 这是 TDD 里的 REFACTOR 步：不写新行为，先跑绿再动，动完还绿。
> 现在那段代码在 `train()` 与 `train_parallel()` 里各有一份 —— 「副本会漂」
> 这个坑本仓库已经吃过（炸弹代价那一轮就得同时改两处）。

- [ ] **Step 1: 先记下重构前的基线**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 除那条预存红外全绿（**记下通过数**，Step 5 必须一样）

- [ ] **Step 2: 写新函数的测试（此时是红的）**

```python
"""训练步的共享实现 —— 采样/重放/拼张量这一段不许在两条路线里各写一份。"""
import random

import torch

from train import replay
from train.net import QNet
from train.policies import greedy_policy
from train.selfplay import _learn_step, build_samples


def _buffer(n=8, seed=0):
    buf = replay.ReplayBuffer(capacity_games=n)
    for k in range(n):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), capture=False)
        buf.add(rec)
    return buf


def test_build_samples_is_aligned():
    s, y, b = build_samples(_buffer(), random.Random(0), 4)
    assert len(s) == len(y) == len(b) > 0


def test_build_samples_uses_the_fresh_cache():
    """`fresh` 命中时不该再去重放（重放一局约 18ms，白花）。"""
    buf = _buffer(n=1)                 # 只有一局 -> 采样必然采到它
    rec = buf._games[0]
    pts = [("P", "A", 0, 0, "H")]
    s, y, b = build_samples(buf, random.Random(0), 1,
                            fresh={id(rec): (pts, [-3.0])})
    assert (s, y) == (pts, [-3.0]) and b == []


def test_build_samples_ignores_fresh_when_bootstrapping():
    """⚠️ **β<1 时快捷缓存必须让路** —— 缓存里没有自举值，
    用它就等于让这一部分样本悄悄退回纯 MC（loss 曲线上看不出来）。"""
    buf = _buffer()
    fresh = {id(r): ([("P", "A", 0, 0, "H")], [-3.0]) for r in buf._games}
    s, y, b = build_samples(buf, random.Random(0), 4, fresh=fresh, n_step=3)
    assert len(b) == len(s) > 4, "带了 n_step 就必须走 expand（boot 非空）"


def test_learn_step_returns_a_finite_loss():
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = _learn_step(net, None, _buffer(), random.Random(0), opt, 0,
                       batch_games=4, bomb_cost=0.0, mc_mix=1.0, n_step=3)
    assert isinstance(loss, float) and loss == loss


def test_learn_step_is_deterministic_for_one_seed():
    """同一批牌 + 同一种子 → 同一条轨迹（重放是确定性的）。"""
    def once():
        torch.manual_seed(0)
        net = QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        return _learn_step(net, None, _buffer(), random.Random(7), opt, 0,
                           batch_games=4, bomb_cost=0.0, mc_mix=1.0, n_step=3)
    assert once() == once()
```

- [ ] **Step 3: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_learn_step.py -q`
Expected: FAIL —— `ImportError: cannot import name 'build_samples'`

- [ ] **Step 4: 写实现并把两条路线接上**

```python
def build_samples(buf, rng, batch_games, *, fresh=None, bomb_cost=0.0, n_step=0):
    """从 buffer 采一批、重放成张量原料。返回 `(samples, y_mc, boot)`，**等长同序**。

    `fresh` 是单进程那条路的「刚打完的这一批」快捷缓存（省一次重放）。
    ⚠️ **要自举时它必须让路**（`n_step > 0`）—— 缓存里没有 `boot`，
    用它就等于让这一部分样本悄悄退回纯 MC，而 loss 曲线上一概看不出来。
    （多进程那条路本来就没有这个缓存，不受影响。）
    """
    if n_step:
        fresh = None
    samples, y_mc, boot = [], [], []
    for rec in buf.sample(batch_games, rng):
        got = fresh.get(id(rec)) if fresh else None
        if got is None:
            pts, y, b = replay.expand(rec, bomb_cost=bomb_cost, n=n_step)
        else:
            pts, y = got
            b = []
        samples += pts
        y_mc += y
        boot += b
    return samples, y_mc, boot


def _learn_step(net, net_tgt, buf, rng, opt, games, *, batch_games, bomb_cost,
                mc_mix, n_step, fresh=None):
    """从 buffer 采一批 → 重放 → 拼张量 → 一步 MSE。**两条训练路线共用这一份。**

    `mc_mix < 1` 时才多算一次前向（自举项，用目标网络、`no_grad`）——
    见 Task 5。`|Q|` 超限会**在这里 raise**（发散必须响）。
    """
    n_boot = 0 if mc_mix >= 1.0 else n_step
    samples, y_mc, boot = build_samples(buf, rng, batch_games, fresh=fresh,
                                        bomb_cost=bomb_cost, n_step=n_boot)
    st, ac, hi = _tensors(samples)
    dev = next(net.parameters()).device
    y_hat = net(st.to(dev), ac.to(dev), hi.to(dev))
    targets = replay.blend(y_mc, None if not n_boot else [], mc_mix)
    y = torch.tensor(targets, dtype=torch.float32).to(dev)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

把 `train()` 里那一段（采样 / 拼张量 / loss / backward，现在在 `for rec in
buf.sample(...)` 那一块）换成：

```python
        loss = _learn_step(net, None, buf, rng, opt, games,
                           batch_games=batch_games, bomb_cost=bomb_cost,
                           mc_mix=1.0, n_step=n_step, fresh=fresh)
        steps += 1
```

`train_parallel()` 里同样换成不带 `fresh` 的一次调用。

⚠️ **一处有意的小修，写进提交信息**：`train()` 原来是
`fresh.get(id(rec)) or replay.expand(rec)` —— 用的是**真值判断**，空表会掉下去；
新的 `build_samples` 用 `is None`。对 `capture=True` 的现状行为一致，且更正确。

- [ ] **Step 5: 跑全量测试确认与 Step 1 的基线一致**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 通过数与 Step 1 **完全相同**（外加 `test_learn_step.py` 的 5 条）

- [ ] **Step 6: 提交**

```bash
git add train/selfplay.py tests/test_learn_step.py
git commit -m "refactor(selfplay): 训练步收成 build_samples/_learn_step 一份（行为不变）"
```

---

### Task 5: 接上自举 —— 目标网络、混合、发散守门

**Files:**
- Modify: `train/selfplay.py`（常量、`sync_target`、`_targets`、`_learn_step`、
  `train`/`train_parallel` 建目标网络与同步、日志头）
- Test: `tests/test_learn_step.py`（追加）

**Interfaces:**
- Consumes: `replay.blend`（T1）、`replay.expand(n=)`（T2）、
  `q_max_batch`/`check_q_scale`（T3）
- Produces:
  - `N_STEP = 3`、`MC_MIX = 1.0`、`TGT_SYNC_GAMES = 1000`
  - `sync_target(net, net_tgt, games, last_sync, every=TGT_SYNC_GAMES) -> int`
  - `_targets(net, net_tgt, buf, rng, batch_games, bomb_cost, mc_mix, n_step, fresh=None) -> (samples, targets)`
  - `train(..., mc_mix=MC_MIX, n_step=N_STEP, tgt_sync=TGT_SYNC_GAMES)`（同上 `train_parallel`）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_learn_step.py`）**

```python
def test_beta_one_never_computes_a_bootstrap_value():
    """**默认行为不变**：β=1 时一次都不该去算 `V̄`（省掉那次昂贵的前向）。"""
    from train import net as net_mod
    calls = []
    orig = net_mod.q_max_batch
    net_mod.q_max_batch = lambda n, p: calls.append(p) or orig(n, p)
    try:
        torch.manual_seed(0)
        net = QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        _learn_step(net, None, _buffer(), random.Random(0), opt, 0,
                    batch_games=4, bomb_cost=0.0, mc_mix=1.0, n_step=3)
    finally:
        net_mod.q_max_batch = orig
    assert not calls, "β=1 是纯 MC，不该走自举那条路"


def test_beta_half_actually_bootstraps():
    """β<1 且给了目标网络时，**真的走了自举那一支**（boot 非空）。"""
    from train import net as net_mod
    calls = []
    orig = net_mod.q_max_batch
    net_mod.q_max_batch = lambda n, p: calls.append(p) or orig(n, p)
    try:
        torch.manual_seed(0)
        net, tgt = QNet(), QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        _learn_step(net, tgt, _buffer(), random.Random(0), opt, 0,
                    batch_games=4, bomb_cost=0.0, mc_mix=0.5, n_step=3)
    finally:
        net_mod.q_max_batch = orig
    assert calls and any(p is not None for p in calls[0]), "自举项没算"


def test_beta_half_differs_from_the_mc_targets():
    """同一个种子下，β=0.5 与 β=1 的目标**必须不同** —— 否则旋钮是假的。"""
    from train.selfplay import _targets
    torch.manual_seed(0)
    net, tgt = QNet(), QNet()
    _s1, y1 = _targets(net, tgt, _buffer(), random.Random(0), 4, 0.0, 1.0, 3)
    _s2, y2 = _targets(net, tgt, _buffer(), random.Random(0), 4, 0.0, 0.5, 3)
    assert y1 != y2


def test_sync_target_copies_only_at_the_interval():
    net, tgt = QNet(), QNet()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    assert sync_target(net, tgt, games=999, last_sync=0, every=1000) == 0
    assert not torch.equal(next(iter(net.parameters())),
                           next(iter(tgt.parameters())))
    assert sync_target(net, tgt, games=1000, last_sync=0, every=1000) == 1000
    assert torch.equal(next(iter(net.parameters())), next(iter(tgt.parameters())))


def test_sync_target_is_a_noop_without_a_target_net():
    assert sync_target(QNet(), None, games=10 ** 9, last_sync=0) == 0
```

（`sync_target` 记得加进 `tests/test_learn_step.py` 顶部的 import。）

- [ ] **Step 2: 跑测试确认它们红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_learn_step.py -q`
Expected: FAIL —— `ImportError: cannot import name 'sync_target'`

- [ ] **Step 3: 写实现**

```python
#: 自举的步数（spec §8）：一局每座位约 33 个决策点；n=1 最偏、n=∞ 就是 MC。
N_STEP = 3
#: β —— MC 与自举的混合比。**1.0 = 完全就是现在的 DMC（默认，行为不变）**。
MC_MIX = 1.0
#: 目标网络同步的间隔（局）—— 与权重广播同拍（那个节奏已经验过不拖死 learner）。
TGT_SYNC_GAMES = 1000


def sync_target(net, net_tgt, games, last_sync, every=TGT_SYNC_GAMES) -> int:
    """到点就把 `net` 复制进目标网络，返回新的 `last_sync`。

    ⚠️ 目标网络**永远不参与优化**，只用来算 `V̄`（spec §3.3）。它是为自举而存在的：
    没有它，目标就是「追自己的尾巴」，发散风险大增。`net_tgt=None`（β=1）时是空操作。
    """
    if net_tgt is None or every <= 0:
        return last_sync
    if games - last_sync < every:
        return last_sync
    net_tgt.load_state_dict(net.state_dict())
    return games


def _targets(net, net_tgt, buf, rng, batch_games, bomb_cost, mc_mix, n_step,
             fresh=None):
    """这一批训练样本的目标值。**自举在这里、且只在这里进入标签。**"""
    n_boot = 0 if mc_mix >= 1.0 else n_step
    samples, y_mc, boot = build_samples(buf, rng, batch_games, fresh=fresh,
                                        bomb_cost=bomb_cost, n_step=n_boot)
    vals = q_max_batch(net_tgt, boot) if n_boot else None
    return samples, replay.blend(y_mc, vals, mc_mix)
```

`_learn_step` 改成（**只留这一处算目标**）：

```python
def _learn_step(net, net_tgt, buf, rng, opt, games, *, batch_games, bomb_cost,
                mc_mix, n_step, fresh=None):
    samples, targets = _targets(net, net_tgt, buf, rng, batch_games, bomb_cost,
                                mc_mix, n_step, fresh=fresh)
    st, ac, hi = _tensors(samples)
    dev = next(net.parameters()).device
    y_hat = net(st.to(dev), ac.to(dev), hi.to(dev))
    y = torch.tensor(targets, dtype=torch.float32).to(dev)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    # 发散守门：**预测与标签都查**（自举跑飞时标签先炸）
    check_q_scale(float(y_hat.abs().max()), games, loss.item())
    check_q_scale(float(y.abs().max()), games, loss.item(), what="标签")
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

`train()` 里建目标网络（`net` 与 `load_init` 之后）：

```python
    # 目标网络：**deepcopy 而不是再 `QNet()`** —— 后者会多消耗一份 RNG，
    # 于是同一个 `--seed` 下两臂的网络初始化就不一样了（那是看不见的变量）。
    net_tgt = None
    if mc_mix < 1.0:
        net_tgt = copy.deepcopy(net).requires_grad_(False)
```
循环里 `loss = _learn_step(net, net_tgt, ...)` 之后加
`last_tgt = sync_target(net, net_tgt, games, last_tgt)`（`last_tgt = 0` 在循环外）。

`train_parallel()` 同样建 `net_tgt`，同步放在**已有的权重广播那一拍**：

```python
            if games - last_sync >= WEIGHT_SYNC_GAMES:
                sd = {k: v.cpu() for k, v in net.state_dict().items()}
                w = current_pfsp()
                for q in ctrls:
                    q.put(("weights", (sd, eps_for(games, eps_games, start=eps_start), w)))
                last_sync = games
                last_tgt = sync_target(net, net_tgt, games, last_tgt, tgt_sync)
```
并把采样那一段换成 `loss = _learn_step(net, net_tgt, buf, rng, opt, games, ...)`。

两条路线各加 `mc_mix: float = MC_MIX, n_step: int = N_STEP,
tgt_sync: int = TGT_SYNC_GAMES`，日志头加：

```python
        + (f"  自举 β={mc_mix:g} n={n_step}（目标网络每 {tgt_sync} 局同步）"
           if mc_mix < 1.0 else "  自举关（β=1）")
```

`import copy` 加到文件顶部（与 `glob`/`os` 一起）。

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 除预存红外全绿

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py tests/test_learn_step.py
git commit -m "feat(selfplay): 接上 n 步自举（目标网络 + β 混合 + 发散守门）"
```

---

### Task 6: CLI 开关 + 端到端真跑一次

**Files:**
- Modify: `train/selfplay.py::main`
- Test: `tests/test_mc_mix_cli.py`（新建）

**Interfaces:**
- Consumes: `train`/`train_parallel` 的新签名（T5）
- Produces: 命令行 `--mc-mix F`、`--n-step N`、`--tgt-sync G`

- [ ] **Step 1: 写失败测试**

```python
"""`--mc-mix` 等开关必须真的透传到训练函数（写错了在日志里看不出来）。"""
import inspect

import train.selfplay as sp


def test_cli_forwards_the_bootstrap_knobs(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--mc-mix", "0.5", "--n-step", "2", "--tgt-sync", "500"])
    assert (seen["mc_mix"], seen["n_step"], seen["tgt_sync"]) == (0.5, 2, 500)


def test_defaults_are_the_old_behaviour():
    sig = inspect.signature(sp.train)
    assert sig.parameters["mc_mix"].default == 1.0
    assert sig.parameters["n_step"].default == 3
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mc_mix_cli.py -q`
Expected: FAIL —— `KeyError: 'mc_mix'`

- [ ] **Step 3: 写实现**

`main()` 里按既有风格加三段（位置挨着 `--bomb-cost`）：

```python
    if "--mc-mix" in argv:
        # β：MC 与自举的混合比。1.0 = 现在的 DMC（默认）；处理臂用 0.5。
        kw["mc_mix"] = float(argv[argv.index("--mc-mix") + 1])
    if "--n-step" in argv:
        # 自举往后看几步（默认 3，spec §8）。β=1 时它不起作用。
        kw["n_step"] = int(argv[argv.index("--n-step") + 1])
    if "--tgt-sync" in argv:
        # 目标网络每多少局同步一次（默认 1000，与权重广播同拍）。
        kw["tgt_sync"] = int(argv[argv.index("--tgt-sync") + 1])
```

- [ ] **Step 4: 真跑一次（端到端，1 分钟）**

Run:
```bash
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -m train.selfplay 60 --workers 2 --batch 8 \
  --eval-games 20 --eval-every 1000000 --eps-games 100000 \
  --mc-mix 0.5 --n-step 3 --out-dir runs/ab/smoke_nstep
```
Expected: 头一行有 `自举 β=0.5 n=3（目标网络每 1000 局同步）`；
跑到结束**不报 `Q 的量级炸了`**；`loss=` 是有限数；
`grep -c "局/秒"` 的每一行都在。跑完把 `runs/ab/smoke_nstep` 删掉。

- [ ] **Step 5: 跑全量测试并提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 除预存红外全绿

```bash
git add train/selfplay.py tests/test_mc_mix_cli.py
git commit -m "feat(selfplay): --mc-mix/--n-step/--tgt-sync 三个开关"
```

---

### Task 7: 动作边际尺子落成 `tools/action_margin.py`

**Files:**
- Create: `tools/action_margin.py`
- Test: `tests/test_action_margin.py`（新建）

**Interfaces:**
- Consumes: `generate_batch`（既有）、`q_values`（既有）、`QNet`
- Produces: `margins_of(net, pending) -> list[float]`、
  `measure(net, games=32, seed=0) -> (n_points, stats)`，
  `stats` 含 `median_margin` / `p90_margin` / `tie_share` / `median_spread` / `n_margins`

> 判据 2（spec §1.2）要的就是这个数：`top1 − top2` 的中位，现在 **0.123**。
> 9-27 那次是在临时目录里算的，清理完数字就不可复现（评审 M8）——
> 所以这次**必须落在 `tools/` 里**。

- [ ] **Step 1: 写失败测试**

```python
"""动作边际：随机初始化的网络应当是 0（p90 也接近 0），训练过的应当明显大于 0。"""
import torch

from tools.action_margin import margins_of, measure
from train.net import QNet


def test_untrained_net_has_no_margin():
    """随机初始化的同结构网络 —— 记录在案的数字是 0.000（p90 0.001）。"""
    torch.manual_seed(0)
    _n, st = measure(QNet().eval(), games=2, seed=0)
    assert st["median_margin"] < 0.01


def test_measure_reports_the_fields_the_criterion_needs():
    torch.manual_seed(0)
    n, st = measure(QNet().eval(), games=2, seed=0)
    assert n > 0 and st["n_margins"] > 0
    for k in ("median_margin", "p90_margin", "tie_share", "median_spread"):
        assert k in st


def test_margins_are_never_negative():
    """`top1 >= top2` 是排序的性质，负的边际只可能是取错了下标。"""
    torch.manual_seed(0)
    _n, st = measure(QNet().eval(), games=2, seed=0)
    assert st["median_margin"] >= 0.0


def test_margin_definition_is_top1_minus_top2():
    """钉住口径：边际 = 排序后**前两名之差**（不是「首选与均值之差」）。"""
    import tools.action_margin as am

    def fake_q(_net, _obs, acts, _hist):
        return torch.arange(len(acts), dtype=torch.float32)

    orig, am.q_values = am.q_values, fake_q
    try:
        got = margins_of(object(), [("o", ["a", "b", "c"], "h")])
    finally:
        am.q_values = orig
    assert got == [1.0]
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_action_margin.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'tools.action_margin'`

- [ ] **Step 3: 写实现**

```python
"""量「动作边际」：同一局面下候选之间的 Q 差 —— 动作通道到底用了多少。

判据 2（spec §1.2）：现在是 **0.123**（标签尺度 ±3）。自举如果起作用，
Q 对动作的区分度就该变大。

⚠️ **必须和「vs 贪心」一起读**：边际变宽而胜率掉，那是数值发散，不是变好。

做法：`generate_batch(..., eps=0, opp_mix=0)` 自对弈若干局并抓决策点（四家都抓），
对**每个决策点**跑一遍 `q_values`（全部候选），取排序后前两名之差。
种子固定 → 结果可复现。

用法：
    .venv/Scripts/python.exe -m tools.action_margin runs/rl/20260926-1407/best.pt
"""
from __future__ import annotations

import random
import statistics
import sys

import torch

from train.net import QNet, q_values
from train.selfplay import generate_batch
from tools.accept_meld import _utf8_stdout


def margins_of(net, pending):
    """`pending = [(obs, acts, hist), ...]` -> 每个决策点的 `top1 − top2`。"""
    out = []
    for obs, acts, hist in pending:
        if len(acts) < 2:
            continue
        q = q_values(net, obs, acts, hist)
        top = torch.topk(q, k=2).values
        out.append(float(top[0] - top[1]))
    return out


def measure(net, games: int = 32, seed: int = 0):
    """返回 `(决策点数, 统计)`。"""
    rng = random.Random(seed)
    pending = []
    for _rec, caps, _y in generate_batch(net, rng, 0.0, games, capture=True,
                                         opp_mix=0.0):
        pending += [(o, a, h) for o, a, _i, _s, h in caps]
    m = margins_of(net, pending)
    # 候选间离散度（max−min）也报：与现用方案 §四.2 那张表同口径（0.401 / 0.123）
    spread = []
    for obs, acts, hist in pending:
        if len(acts) < 2:
            continue
        q = q_values(net, obs, acts, hist)
        spread.append(float(q.max() - q.min()))
    return len(pending), {
        "median_margin": statistics.median(m),
        "p90_margin": sorted(m)[int(0.9 * (len(m) - 1))],
        "tie_share": sum(1 for x in m if x < 0.01) / max(1, len(m)),
        "median_spread": statistics.median(spread) if spread else 0.0,
        "n_margins": len(m),
    }


def main(argv=None) -> int:
    _utf8_stdout()
    argv = sys.argv[1:] if argv is None else argv
    for p in (argv or ["runs/rl/20260926-1407/best.pt"]):
        d = torch.load(p, map_location="cpu", weights_only=False)
        net = QNet()
        net.load_state_dict(d["net"])
        net.eval()
        n, st = measure(net)
        print(f"== {p}（{d.get('games', 0):,} 局）")
        print(f"   决策点 {n} 个，参与统计 {st['n_margins']}")
        print(f"   首选次选之差 中位 {st['median_margin']:.3f}  "
              f"p90 {st['p90_margin']:.3f}  打平(<0.01) {st['tie_share']:.1%}")
        print(f"   候选间离散度 中位 {st['median_spread']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 量现役存档，把起点值记进台账**

Run: `.venv/Scripts/python.exe -m tools.action_margin runs/rl/20260926-1407/best.pt`
Expected: 中位在 **0.123** 附近（±0.05 内），离散度在 **0.4** 附近。
对不上就说明口径与 9-27 那次不同 —— **先查清楚再往下走**，别把两个口径的数混着比。

- [ ] **Step 5: 跑测试确认全绿并提交**

Run: `.venv/Scripts/python.exe -m pytest tests/test_action_margin.py -q`
Expected: PASS（4 条）

```bash
git add tools/action_margin.py tests/test_action_margin.py
git commit -m "feat(tools): action_margin —— 判据 2 的尺子落进 tools/（不再躺在临时目录）"
```

---

### Task 8: 先量成本（spec §5.5 的硬闸）—— **量完才准起 A/B**

**Files:**
- Modify: `tools/bench_train.py`（`main()` 透传 `--mc-mix` / `--n-step`）
- Create: `docs/superpowers/plans/2026-09-28-nstep-bootstrap-delivery.md`（台账）

**Interfaces:**
- Consumes: `bench_run(**kw)`（既有，已经会把 kw 转给 `train_parallel`）
- Produces: 台账里的一张两行表（`β=1` / `β=0.5` 的局/秒）与据此定下的预算

> 自举要多算一遍 `V̄(s_{t+n}) = max_a Q̄(s,a)` —— **每个候选一行**，
> 所以它的前向行数约是基础前向的 `平均候选数` 倍（不是「翻倍」）。而学习进程
> 本来就是瓶颈。这一条**必须实测**，不许估。

- [ ] **Step 1: 给 bench 加两个开关**

`main()` 里：

```python
    ap.add_argument("--mc-mix", type=float, default=1.0)
    ap.add_argument("--n-step", type=int, default=3)
```
调用处改成：

```python
        r = bench_run(w, seconds=args.seconds, log=log,
                      mc_mix=args.mc_mix, n_step=args.n_step)
```

- [ ] **Step 2: 两档各跑 3 分钟**

⚠️ **不许用 45 秒** —— 短跑会把 `spawn` + `import torch` 的启动成本摊进去，
这个坑踩过：bench 报 33.8 局/秒，实际长跑是 44.5。

Run:
```bash
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 180 --mc-mix 1.0
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 180 --mc-mix 0.5 --n-step 3
```
（`bench_run` 把权重写在 `tempfile.mkdtemp()`，**不会碰 `runs/rl/`** ✓）

- [ ] **Step 3: 按**预登记**的规则定预算，写进台账**

| 实测（β=0.5 相对 β=1 的局/秒） | 预算 |
|---|---|
| **≥ 70%** | 两臂各 **9000 秒**（spec §8 原计划），同秒数≈同局数 |
| **40% ~ 70%** | 两臂仍各 9000 秒，但**主判据改用同局数配对**（靠 `--snap-every 20000` 的快照），配对点取两臂**都够得着**的最大同局数（**取整到 2 万**） |
| **< 40%** | **停下来报告用户**，不要烧 5 小时 —— 那时该讨论的是「自举只施加在一部分决策点上」（等效剂量 = `(1−β) × 那一部分的比例`，成本 ∝ 那一部分的比例），那是 spec 之外的新决定 |

⚠️ 三条都不许事后改。**并且**：两臂的 `vs 贪心`、`bomb_waste`、动作边际
都必须落在**同一批牌 / 同一个局数**上比。

- [ ] **Step 4: 提交**

```bash
git add tools/bench_train.py docs/superpowers/plans/2026-09-28-nstep-bootstrap-delivery.md
git commit -m "feat(tools): bench_train 透传 --mc-mix/--n-step；路线②成本台账"
```

---

### Task 9: 起 A/B（两臂**串行**）

**Files:**
- Modify: `docs/superpowers/plans/2026-09-28-nstep-bootstrap-delivery.md`（结果续在里面）

- [ ] **Step 1: 记下起跑前的可用内存**

Run: `powershell -c "Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory"`
Expected: **≥ 10 GB**（单位 KB）。不到就先别起：每臂 2 个 worker 各约 1.2 GB，
学习进程还要一份。

- [ ] **Step 2: 起对照臂（β=1）**

```powershell
$env:GUANDAN_DEVICE="cpu"
$C = "--workers 2 --opp-mix 0.5 --init runs/rl/20260926-1407/best.pt --eps-games 200000 --snap-every 20000"
.venv/Scripts/python.exe -u -m train.selfplay 9000 $C --mc-mix 1.0 --out-dir runs/ab/R4_b10
```

- [ ] **Step 3: 它跑完再起处理臂（β=0.5，n=3）**

```powershell
.venv/Scripts/python.exe -u -m train.selfplay 9000 $C --mc-mix 0.5 --n-step 3 --out-dir runs/ab/R4_b05
```

⚠️ **必须一条一条起**。2026-09-27 三条并发把机器打爆过
（`numpy._ArrayMemoryError` + `RuntimeError: worker [1] 挂了`）。

- [ ] **Step 4: 同一把尺子**

```powershell
.venv/Scripts/python.exe -m tools.ab_compare runs/ab/R4_b10 runs/ab/R4_b05
.venv/Scripts/python.exe -m tools.action_margin runs/ab/R4_b10/last.pt runs/ab/R4_b05/last.pt runs/rl/20260926-1407/best.pt
```
外加**同局数快照**上的两把尺子（配对点见 Task 8 Step 3）：

```powershell
ls runs/ab/R4_b10/                       # 先看清快照的实际文件名
.venv/Scripts/python.exe -m tools.ab_compare runs/ab/R4_b10/<snap> runs/ab/R4_b05/<snap>
.venv/Scripts/python.exe -m tools.action_margin runs/ab/R4_b10/<snap> runs/ab/R4_b05/<snap>
```

- [ ] **Step 5: 逐条对预登记判据，写进台账**

| 判据 | 怎么看 | 结果 |
|---|---|---|
| 1. `vs 贪心` 不掉 ≥1 个标准差 | `ab_compare` 的 `vs 贪心`（400 局 `seed=1002`）对起点 95.2% 与对照臂 | |
| 2. 动作边际变宽 | `action_margin` 的中位对起点 0.123 | |
| 3. `bomb_waste` 两个数都降 + 用炸率不塌 | `ab_compare` 四列 | |
| 4. 真机影子日志同向 | **要用户打牌**（拿不到就如实写「没有真机证据」） | |
| 5. 杀停 | 判据 1 或 2 不满足 ⇒ **停路线②**，回头质疑「值函数 + argmax」框架 | |

⚠️ **判据 2 必须和判据 1 一起读**：边际变宽而胜率掉 = 数值发散，不是变好。
⚠️ 真机那条只有用户能结案 —— **不许把离线数字说成「已经治好」**。

---

## 自检记录

**Spec 覆盖**：§3.1 标签形状 → T1/T5；§3.2 数据流（`expand` 三元组、boot 对齐、
调用点）→ T2；§3.3 目标网络 + `q_max_batch` → T3/T5；§3.4 两条纪律 →
T5（β=1）+ T3（返回 float）；§4 取舍（γ=1、不做重要性采样、不盖路线①）→ Global
Constraints；§5.1 A/B → T9；§5.2 尺子（含动作边际落进 `tools/`）→ T7；
§5.3 五条单测 → T1/T2/T3/T5；§5.5 成本先量 → T8；§6 风险（发散守门、阈值 30）→
T3/T5。**§5.4 真机**是用户侧动作，只能如实标注（T9 Step 5）。

**对 spec 的三处更正**（执行时按本计划，不按 spec 那几句）：

1. `play_capturing` **不调用 `expand`**，不在改动面上（spec §3.2 写错了调用方）。
2. `expand` 返回三元组影响 **9 处**（`selfplay` 2 + `tests` 7）—— 不是「一行不变」。
3. `boot` 的每一项是 `(obs, acts, **hist**)`：网络输入必须有历史，
   spec §3.2 只写了 `(obs, acts)`；少了 hist 就得在训练步另算一遍（更贵）。
4. 「越过终局整项退回 **R**」在实现里是**退回 `y_mc`** —— λ=0 时两者相等；
   λ>0 时退回 `y_mc` 才自洽（`y_mc` 已含从 t 起的炸弹代价）。本轮两臂都 λ=0。

**Placeholder 扫描**：无 TBD / 「类似 Task N」/ 「加上适当的错误处理」；
每个 code step 都是可粘贴的整段代码；每个 Run 步骤都写了 Expected。

**类型一致性**（跨 task 逐个核过）：
`blend(y_mc, boot, beta) -> list[float]`、
`expand(rec, bomb_cost=0.0, n=0) -> (points, y_mc, boot)`、
`_flat_scores(net, pending) -> (q, counts)`、
`q_max_batch(net, pending) -> list[float | None]`、
`check_q_scale(q_abs_max, games, loss, what, limit)`、
`build_samples(buf, rng, batch_games, *, fresh, bomb_cost, n_step) -> (samples, y_mc, boot)`、
`_targets(net, net_tgt, buf, rng, batch_games, bomb_cost, mc_mix, n_step, fresh=None) -> (samples, targets)`、
`_learn_step(net, net_tgt, buf, rng, opt, games, *, batch_games, bomb_cost, mc_mix, n_step, fresh=None) -> float`、
`sync_target(net, net_tgt, games, last_sync, every) -> int`、
`margins_of(net, pending) -> list[float]`、`measure(net, games, seed) -> (int, dict)`。

**Review Focus 与它的测试**：五条分别钉在 —— ①T8 Step 2/T9 Step 1（串行 + 内存）；
②T4 `test_build_samples_ignores_fresh_when_bootstrapping`；③T2
`test_boot_is_really_the_state_n_steps_later`；④T3 `test_returns_floats_not_tensors`
+ `test_no_grad_leaks_into_the_network`；⑤T2 `test_negative_n_raises` + T3
`test_check_q_scale_raises_on_nan` + `test_all_none_does_not_touch_the_network`。

**已知的、故意留着的取舍**（不在本轮范围，写在这里免得被当成遗漏）：
不做重要性采样（spec §4）；γ 固定 1.0，不做成参数；`--boot-frac` 这种
「只对一部分决策点自举」的省钱法**没有实现** —— 它只在 T8 的第三档里作为
「停下来讨论」的选项出现，不预先造好（YAGNI）。
