# 策略式（REINFORCE）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把训练从「回归一个标量 Q + argmax」换成「在候选上做策略梯度」，
让更新天然只作用在**实际出的那一手**上 —— 用同一个现成的终局标签，绕开
「一局 132 个决策点共享同一个标签」这件事。

**Architecture:** 网络**不动**（现有标量当 logits，`π(a|s)=softmax_a z(s,a)`）。
新增：`log π(a_taken)` 与熵的批量计算（**要带梯度**）、从 π 采样的行为策略、
一个 PG 训练步（`loss = −log π(a)·(R−b) − β_ent·H`）、熵/logits 两处守门。
数据 **on-policy**：PG 模式不写 replay buffer，只用刚打完的那一批。
DMC 那条路**逐字保留**（`--algo dmc` 是默认）。

**Tech Stack:** Python 3 + PyTorch（CPU）、pytest、既有 `net/sim` 规则引擎与 `env` 编码器。

**Spec:** `docs/superpowers/specs/2026-09-29-policy-gradient-design.md`（本计划从它论证；
执行者两份都要读）

## Global Constraints

- **默认行为不许变**：`--algo` 默认 `dmc` ⇒ 走的就是现在的代码路径。有测试钉着。
- **网络结构与权重格式不许变**：标量输出当 logits 用 ⇒ `--init runs/rl/20260926-1407/best.pt`
  照旧能加载，而且**初始策略的众数 = 老的 argmax**（两臂同强度起步）。
- **决策路径不许变**：`argmax_a z` 就是策略众数 ⇒ 面板 / 影子模式 / 评测器一行不改。
- **`log π` 必须带梯度**。⚠️ 现有的 `_flat_scores` 是 `torch.no_grad()` 的
  （它服务于"算目标值"）—— **照抄它会让 loss 变成常数、梯度为 `None`、
  训练一步不动而日志上完全看不出来**。所以 `_flat_scores` 要加一个 `grad=False` 参数，
  PG 走 `grad=True`；而且**第一条测试就是「训一步之后参数必须变」**。
- **标签的唯一产地不变**：`R` 仍然只能由 `replay.mc_targets`（经 `replay.expand`）产出。
- **ε 在 PG 模式下失效**，探索由**熵系数**承担；**日志头必须写清这件事**
  （`ε 不适用`），否则日志写着 `ε 起点 0.30` 而实际没用 —— 违「换源必须可见」。
- **on-policy 结构上钉死**：PG 模式**不调用 `buf.add`**。
- **发散必须响**：`|z|` 超限、**熵塌到下限以下**都要 `raise`（本仓库纪律：失败必须响）。
- **固定对手的决策点不进 loss**（`learn` 过滤器），与现状一致。
- **测试基线**：改动前 **484 passed / 2 failed**，两条红都是**日志轮转**的环境红
  （`test_tribute_records` / `test_game_log`）。每个 task 结束时必须「除它俩以外全绿」。
- **命令一律**：`.venv/Scripts/python.exe`，训练前 `$env:GUANDAN_DEVICE="cpu"`。
- **`runs/rl/` 绝不写**；A/B 落 `runs/ab/`。
- **A/B 两臂串行**，一条臂约 4.8 GB。

## Review Focus

执行者做完每个 task 后逐条确认（都是「不报错但结果悄悄错」的类型）：

1. **梯度断了** —— `no_grad` / 返回 float / `detach()` 都会让 loss 变成常数、
   训练一步不动、日志上只看到 loss 平着不动。Task 1 专门钉。
2. **行为策略不是 π** —— 若 learner 仍然 argmax（或仍带 ε），数据就不是 on-policy 的 π
   ⇒ 策略梯度估的是**另一个分布**的梯度，静默学歪。Task 4 专门钉。
3. **ε 还在偷偷起作用 / 日志不承认** —— PG 模式下 ε 必须彻底不参与，且日志要直说。
4. **buffer 混进来** —— PG 只用刚打完的那批；混进旧策略的数据就是 off-policy（无修正）。
5. **熵塌 / logits 发散** —— 都要响亮地 raise，不许静默训完。

---

### Task 1: `log π(a_taken)` 与熵（**带梯度**）+ 熵守门

**Files:**
- Modify: `train/net.py`（`_flat_scores` 加 `grad` 参数；新增两个函数 + 一个常量）
- Test: `tests/test_pg_logprob.py`（新建）

**Interfaces:**
- Consumes: `_flat_scores`（既有，把「所有候选」拼成一批；现在多了 `grad`）
- Produces:
  - `_flat_scores(net, pending, grad: bool = False)` —— **默认仍然 `no_grad`**（老调用方不受影响）
  - `log_prob_and_entropy(net, samples) -> (lp, ent, zmax)`，
    `lp`/`ent` 是 **1-D tensor**（每个决策点一个元素），`zmax` 是这一批 logits 的
    **`|z|` 最大值**（float，给发散守门用）；`samples` 就是既有的
    `(obs, acts, 选中下标, 出牌人, hist)` 五元组（`replay.expand` 的产出形状）
  - `ENT_FLOOR_FRAC = 0.05`、`check_entropy(h, log_k, frac=ENT_FLOOR_FRAC) -> None`

- [ ] **Step 1: 写失败测试**

```python
"""`log π(a|s)` 与熵：**必须带梯度**（不带的话训练一步都不动，日志上看不出来）。"""
import random

import pytest
import torch

from net.sim import env
from train import replay
from train.net import (QNet, check_entropy, log_prob_and_entropy, q_values)
from train.policies import greedy_policy


def _samples(n=3, seed=0, level=5):
    """`replay.expand` 的产出形状：`(obs, acts, 选中下标, 出牌人, hist)`。"""
    out = []
    for k in range(n):
        rec, _pts, _y = replay.play_capturing(greedy_policy,
                                              random.Random(seed + k), level=level,
                                              capture=False)
        pts, _y2, _ = replay.expand(rec)
        out += pts[:4]
    return out


def test_log_prob_matches_a_hand_computed_softmax():
    """与「单局面单独算」逐点一致 —— 同 `q_max_batch` 那条测试的路子。"""
    net = QNet().eval()
    samples = _samples()
    lp, ent = log_prob_and_entropy(net, samples)
    for (obs, acts, i, _seat, hist), got in zip(samples, lp):
        z = q_values(net, obs, acts, hist)
        want = torch.log_softmax(z, dim=0)[i]
        assert float(got) == pytest.approx(float(want), abs=1e-5)


def test_entropy_is_the_softmax_entropy():
    net = QNet().eval()
    samples = _samples()
    _lp, ent = log_prob_and_entropy(net, samples)
    for (obs, acts, _i, _seat, hist), got in zip(samples, ent):
        z = q_values(net, obs, acts, hist)
        p = torch.softmax(z, dim=0)
        want = float(-(p * torch.log(p)).sum())
        assert float(got) == pytest.approx(want, abs=1e-5)


def test_log_prob_carries_gradient():
    """⚠️ **这一条是整个方案的命门**：`log π` 必须带梯度。

    现有 `_flat_scores` 是 `no_grad` 的（它服务于"算目标值"）。照抄它 ⇒
    loss 变成常数、梯度是 `None`、**训练一步不动而日志上完全看不出来**。
    """
    net = QNet()
    lp, ent = log_prob_and_entropy(net, _samples())
    assert lp.requires_grad and ent.requires_grad


def test_one_optimizer_step_actually_changes_the_parameters():
    net = QNet()
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lp, ent = log_prob_and_entropy(net, _samples())
    loss = -(lp * torch.tensor([2.0, 1.0, -1.0, 3.0])).mean() - 0.01 * ent.mean()
    opt.zero_grad(); loss.backward(); opt.step()
    changed = sum(1 for a, b in zip(before, net.parameters()) if not torch.equal(a, b))
    assert changed, "训一步参数没变 —— 梯度断在哪儿了"


def test_the_existing_helpers_still_run_under_no_grad():
    """`grad` 默认 False ⇒ 老调用方（`q_max_batch` 等）行为不变。"""
    from train.net import q_argmax_batch
    net = QNet().eval()
    first = _samples()[0]
    q_argmax_batch(net, [(first[0], first[1], first[4])])       # 不炸即通过


def test_entropy_guard_raises_when_collapsed():
    """熵塌到「候选数均匀熵」的 5% 以下 ⇒ 响亮地炸（等于偷偷退化成 argmax）。"""
    import math
    check_entropy(0.9 * math.log(10), math.log(10))              # 正常：不炸
    with pytest.raises(RuntimeError):
        check_entropy(0.001, math.log(10))
    with pytest.raises(RuntimeError):                            # NaN 也要拦住
        check_entropy(float("nan"), math.log(10))
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_logprob.py -q`
Expected: FAIL —— `ImportError: cannot import name 'log_prob_and_entropy'`

- [ ] **Step 3: 写实现**

`_flat_scores` 加参数（**默认不变**）：

```python
def _flat_scores(net, pending, grad: bool = False):
    """...（docstring 原文保留）...

    `grad=True` 时**不套 `no_grad`** —— 策略梯度要穿过 logits。
    ⚠️ 默认 `False` 是**故意的**：老的调用方（`q_argmax_batch` / `q_max_batch`）要的是
    "算出来的值"，带梯度进去只会白建图。
    """
    dev = next(net.parameters()).device
    counts = [len(acts) for _o, acts, _h in pending]
    st = np.stack([env.encode_state(o) for o, _a, _h in pending])
    hi = np.stack([h for _o, _a, h in pending])
    ac = np.stack([env.encode_action(a, o.level)
                   for o, acts, _h in pending for a in acts])
    idx = np.repeat(np.arange(len(pending)), counts)
    args = (torch.from_numpy(st[idx]).to(dev), torch.from_numpy(ac).to(dev),
            torch.from_numpy(hi[idx]).to(dev))
    if grad:
        return net(*args), counts
    with torch.no_grad():
        return net(*args), counts
```

新增：

```python
#: 熵的下限：低于「候选数对应的均匀熵」的这个比例，就认为策略塌成了 argmax。
ENT_FLOOR_FRAC = 0.05


def log_prob_and_entropy(net, samples):
    """`samples = [(obs, acts, 选中下标, 出牌人, hist), ...]`（`replay.expand` 的形状）

    返回 `(lp, ent, zmax)`：前两个是 **1-D tensor**（每个决策点一个元素），
    `zmax` 是这一批 logits 的 `|z|` 最大值（`check_q_scale` 的守门要用）。

    - `lp[i] = log π(a_i | s_i)`，`π = softmax(所有候选的 logits)`
    - `ent[i] = π 的香农熵`（自然对数）

    ⚠️ **返回张量而不是 float** —— 与 `q_max_batch` 正好相反，那里返回 float 是
    **故意的**（目标项不许带梯度）；这里要的就是梯度，所以必须是张量。
    """
    q, counts = _flat_scores(net, [(o, a, h) for o, a, _i, _s, h in samples], grad=True)
    lps, ents, off = [], [], 0
    for (_o, _a, i, _s, _h), c in zip(samples, counts):
        seg = torch.log_softmax(q[off:off + c], dim=0)
        off += c
        lps.append(seg[i])
        ents.append(-(seg.exp() * seg).sum())
    return torch.stack(lps), torch.stack(ents), float(q.detach().abs().max())


def check_entropy(h: float, log_k: float, frac: float = ENT_FLOOR_FRAC) -> None:
    """熵低于「均匀熵 `log_k` 的 `frac`」就 raise（本项目纪律：失败必须响）。

    ⚠️ 用 `not (h >= floor)` 写 —— NaN 与任何数比较都是 False，
    写成 `h < floor` 会让 NaN 悄悄溜过去（与 `check_q_scale` 同一个坑）。
    """
    floor = frac * log_k
    if not (h >= floor):
        raise RuntimeError(
            f"策略熵塌了：H={h:.4g} < {floor:.4g}（均匀熵 {log_k:.4g} 的 {frac:.0%}）"
            f"—— 已经退化成 argmax，等于白换框架。查 β_ent（`--beta-ent`）")
```

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_logprob.py tests/test_q_max.py -q`
Expected: PASS（新 7 条 + `q_max` 那批照旧）

- [ ] **Step 5: 提交**

```bash
git add train/net.py tests/test_pg_logprob.py
git commit -m "feat(net): log π(a|s) 与熵（带梯度）+ 熵守门；_flat_scores 加 grad 参数"
```

---

### Task 2: 行为策略 —— **从 π 采样**

**Files:**
- Modify: `train/net.py`（新增 `policy_sample_batch`）
- Test: `tests/test_pg_sample.py`（新建）

**Interfaces:**
- Consumes: `_flat_scores`（`grad=False`，采样不需要梯度）
- Produces: `policy_sample_batch(net, pending, rng) -> list[int]`
  —— `pending = [(obs, acts, hist), ...]`，返回每个决策点**采到的候选下标**

> 为什么必须有它：PG 的数据**必须来自 π 自己**。若行为策略仍是 argmax（或还带 ε），
> 那估的是**另一个分布**的梯度 —— 静默学歪，而且 loss 曲线看不出来。

- [ ] **Step 1: 写失败测试**

```python
"""行为策略：从 π 采样（不是 argmax、也不带 ε）。"""
import random

import torch

import train.net as net_mod
from train.net import QNet, policy_sample_batch


def _fake(monkeypatch, logits):
    monkeypatch.setattr(net_mod, "_flat_scores",
                        lambda net, pending, grad=False: (torch.tensor(logits),
                                                          [len(logits)]))


def test_sample_follows_pi(monkeypatch):
    """频率要贴近 π —— 固定 logits，采 4000 次。"""
    _fake(monkeypatch, [0.0, 1.0, 2.0])
    p = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), 0)
    rng = random.Random(0)
    cnt = [0, 0, 0]
    for _ in range(4000):
        cnt[policy_sample_batch(object(), [("o", ["a", "b", "c"], "h")], rng)[0]] += 1
    for got, want in zip(cnt, p):
        assert abs(got / 4000 - float(want)) < 0.03


def test_sample_is_reproducible_for_the_same_rng():
    net = QNet().eval()
    pend = [("o", ["a", "b"], "h")]
    a = policy_sample_batch(net, [("o", [1, 2], "h")], random.Random(7))
    b = policy_sample_batch(net, [("o", [1, 2], "h")], random.Random(7))
    assert a == b


def test_sample_returns_valid_indices():
    net = QNet().eval()
    from tests.test_pg_logprob import _samples
    pend = [(o, a, h) for o, a, _i, _s, h in _samples(2)]
    got = policy_sample_batch(net, pend, random.Random(0))
    assert len(got) == len(pend)
    assert all(0 <= j < len(a) for j, (_o, a, _h) in zip(got, pend))
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_sample.py -q`
Expected: FAIL —— `ImportError: cannot import name 'policy_sample_batch'`

- [ ] **Step 3: 写实现**

```python
def policy_sample_batch(net, pending, rng) -> list:
    """从 `π = softmax(候选的 logits)` **采样**，返回每个决策点的下标。

    这是 PG 的**行为策略** —— on-policy 的定义就落在这一个函数上。

    ⚠️ 用调用方的 `rng`（`random.Random`）而不是 torch 的生成器：
    整条链的可复现性都挂在同一个 `rng` 上（`generate_batch` 连「哪一队当对手」
    都用它抽）。走逆累积分布，不用 `torch.multinomial`。
    """
    q, counts = _flat_scores(net, pending)
    out, off = [], 0
    for c in counts:
        p = torch.softmax(q[off:off + c], dim=0)
        off += c
        r, acc, pick = rng.random(), 0.0, c - 1
        for j in range(c):
            acc += float(p[j])
            if r < acc:
                pick = j
                break
        out.append(pick)
    return out
```

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_sample.py -q`
Expected: PASS（3 条）

- [ ] **Step 5: 提交**

```bash
git add train/net.py tests/test_pg_sample.py
git commit -m "feat(net): policy_sample_batch —— 从 π 采样（PG 的行为策略）"
```

---

### Task 3: PG 训练步 + 基线 + 两处守门

**Files:**
- Modify: `train/selfplay.py`（新增 `_RunningMean`、`_pg_step`；常量 `BETA_ENT`）
- Test: `tests/test_pg_step.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `log_prob_and_entropy` / `check_entropy`；`check_q_scale`（既有）
- Produces:
  - `BETA_ENT = 0.01`
  - `_RunningMean(window=1000)`：`.update(x) -> float`、`.value -> float`
  - `_pg_step(net, opt, samples, rewards, base, games, beta_ent=BETA_ENT) -> dict`
    返回 `{"loss", "entropy", "adv"}`（都是 float，给日志用）

- [ ] **Step 1: 写失败测试**

```python
"""PG 训练步：基线的滑动均值 + 熵/logits 两处守门 + **一步真的动参数**。"""
import math
import random

import pytest
import torch

from train.net import QNet
from train.selfplay import BETA_ENT, _RunningMean, _pg_step
from tests.test_pg_logprob import _samples


def test_running_mean_moves_toward_the_data():
    b = _RunningMean(window=10)
    for _ in range(10):
        b.update(1.0)
    assert abs(b.value - 1.0) < 1e-9
    b.update(-1.0)
    assert b.value < 1.0


def test_pg_step_changes_the_parameters_and_reports():
    net = QNet()
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    samples = _samples(3)
    rewards = [2.0] * len(samples)
    out = _pg_step(net, opt, samples, rewards, _RunningMean(), games=32)
    assert set(out) == {"loss", "entropy", "adv"}
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters())), \
        "训一步参数没变 —— 梯度断了"


def test_pg_step_raises_when_the_policy_collapses():
    """人为把 logits 放大 1e4 倍 ⇒ 策略退化成 one-hot ⇒ 熵守门必须响。"""
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.0)
    samples = _samples(2)
    with torch.no_grad():                      # 直接把输出层放大
        for p in net.parameters():
            p.mul_(0.0)
        for p in net.mlp[-1].parameters():
            p.add_(1e4)
    with pytest.raises(RuntimeError):
        _pg_step(net, opt, samples, [1.0] * len(samples), _RunningMean(), games=32)


def test_beta_ent_default_is_small_and_positive():
    assert 0.0 < BETA_ENT <= 0.05
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_step.py -q`
Expected: FAIL —— `ImportError: cannot import name 'BETA_ENT'`

- [ ] **Step 3: 写实现**

```python
#: 熵系数（spec §8 的初值）。**这是 PG 的实验里第一个要动的数**：
#: 太小 ⇒ 策略提前塌成 argmax（有守门会炸）；太大 ⇒ 一直乱出。
BETA_ENT = 0.01


class _RunningMean:
    """`R` 的滑动均值 —— PG 的基线。

    为什么不学一个 V 网络（spec §3.2）：现在 95% 的局都赢 ⇒ `R` 的方差本来就小
    ⇒ 滑动均值够用，而学 V 要多一处能出错的地方。**这是刻意的简化。**
    """

    def __init__(self, window: int = 1000):
        self.window, self._buf, self._sum = window, [], 0.0

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        self._sum += float(x)
        while len(self._buf) > self.window:
            self._sum -= self._buf.pop(0)
        return self.value

    @property
    def value(self) -> float:
        return self._sum / len(self._buf) if self._buf else 0.0


def _pg_step(net, opt, samples, rewards, base, games, beta_ent: float = BETA_ENT):
    """一步策略梯度。**只作用于实际出的那一手**（这是它绕开"标签不区分动作"的全部理由）。

        L = − mean_i [ log π(a_i|s_i) · (R_i − b) ] − β_ent · mean_i H(π(·|s_i))

    `rewards` 是每个决策点的 `R`（同一局里每个点都一样 —— 标签仍由 `mc_targets` 产出）。
    """
    lp, ent, zmax = log_prob_and_entropy(net, samples)
    adv = torch.tensor(rewards, dtype=torch.float32) - base.value
    loss = -(lp * adv).mean() - beta_ent * ent.mean()
    # 两处守门：logits 发散 / 熵塌（都是"失败必须响"）
    check_q_scale(zmax, games, float(loss), what="logits")
    log_k = sum(math.log(len(a)) for _o, a, _i, _s, _h in samples) / len(samples)
    check_entropy(float(ent.mean()), log_k)
    opt.zero_grad(); loss.backward(); opt.step()
    return {"loss": float(loss), "entropy": float(ent.mean()), "adv": float(adv.mean())}
```

（`import math` 与 `from train.net import check_entropy, check_q_scale, log_prob_and_entropy`
加进 `train/selfplay.py` 顶部。）

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_step.py -q`
Expected: PASS（4 条）

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py tests/test_pg_step.py
git commit -m "feat(selfplay): PG 训练步 + R 的滑动均值基线 + 熵/logits 两处守门"
```

---

### Task 4: 接线 —— `--algo pg` 全通（两条路都用 π 采样、不写 buffer、日志说清）

**Files:**
- Modify: `train/selfplay.py`（`generate_batch(sample=)`、`train`/`train_parallel(algo=, beta_ent=, weight_sync_games=)`、CLI、日志头）
- Modify: `train/worker.py`（`worker_batch(sample=)`、`worker_cfg(sample=)`、`run_worker` 透传）
- Test: `tests/test_pg_wiring.py`（新建）

**Interfaces:**
- Consumes: Task 2 的 `policy_sample_batch`、Task 3 的 `_pg_step`/`_RunningMean`/`BETA_ENT`
- Produces:
  - `generate_batch(..., sample: bool = False)` —— `sample=True` 时**学习者改用 π 采样且完全忽略 ε**
  - `train(..., algo="dmc", beta_ent=BETA_ENT, weight_sync_games=WEIGHT_SYNC_GAMES)`（`train_parallel` 同）
  - CLI：`--algo {dmc,pg}`、`--beta-ent F`、`--weight-sync-games N`

> **两条路都要接**：多进程那条路里**是 worker 在打牌**，行为策略在 worker 手里。
> 只接单进程的话，`--workers 2`（A/B 就用的这个）会**偷偷仍用 argmax** ⇒ 数据不是 π 的 ⇒ 静默学歪。

- [ ] **Step 1: 写失败测试**

```python
"""`--algo pg` 的接线：行为策略必须是 π、ε 必须彻底失效、日志必须说清。"""
import random

import pytest
import torch

import train.selfplay as sp
from train.net import QNet
from train.selfplay import generate_batch


def test_pg_mode_completely_ignores_eps():
    """⚠️ **on-policy 的命门**：PG 模式下 ε 取什么值都必须不影响打出来的牌。

    行为策略是 π ⇒ 数据才是 π 的。若 ε 还在偷偷起作用，策略梯度估的是
    另一个分布的梯度 —— 静默学歪，loss 曲线看不出来。
    """
    def run(eps):
        return [rec.actions for rec, _p, _y in generate_batch(
            QNet().eval(), random.Random(3), eps, 4, capture=True, opp_mix=0.0,
            sample=True)]
    assert run(0.1) == run(0.9), "ε 在 PG 模式下还在起作用"


def test_pg_mode_actually_samples():
    """行为策略必须**不是** argmax（否则等于没换）：同一批牌，采样与 argmax 不能逐点全同。"""
    from train.net import q_argmax_batch
    net = QNet().eval()
    caps, chosen = [], []
    for _rec, pts, _y in generate_batch(net, random.Random(3), 0.0, 4, capture=True,
                                        opp_mix=0.0, sample=True):
        caps += pts
        chosen += [i for _o, _a, i, _s, _h in pts]
    picks = q_argmax_batch(net, [(o, a, h) for o, a, _i, _s, h in caps])
    assert chosen != picks, "采样与 argmax 完全一样 —— 行为策略没换成 π"


def test_pg_header_is_honest(tmp_path):
    """**换源必须可见**：PG 模式要说清 ε 不适用、β_ent 是多少、不用 buffer。"""
    lines = []
    sp.train(seconds=1, algo="pg", log=lines.append, eval_games=1,
             eval_every=10 ** 9, out_dir=str(tmp_path))
    head = "\n".join(lines[:4])
    assert "pg" in head.lower()
    assert "ε 不适用" in head
    assert "β_ent" in head
    assert "不用 replay buffer" in head


def test_dmc_is_still_the_default():
    import inspect
    for fn in (sp.train, sp.train_parallel):
        assert inspect.signature(fn).parameters["algo"].default == "dmc"


def test_cli_forwards_the_pg_knobs(monkeypatch):
    seen = {}

    def fake_train(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0, "games": 0,
                "curve": [], "best_greedy": 1.0, "elapsed": 0.0}

    monkeypatch.setattr(sp, "train", fake_train)
    sp.main(["1", "--algo", "pg", "--beta-ent", "0.02", "--weight-sync-games", "200"])
    assert (seen["algo"], seen["beta_ent"], seen["weight_sync_games"]) == ("pg", 0.02, 200)
```

- [ ] **Step 2: 跑测试确认它红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pg_wiring.py -q`
Expected: FAIL —— `TypeError: generate_batch() got an unexpected keyword argument 'sample'`

- [ ] **Step 3: 写实现**

`generate_batch` 里**学习者那一组**（只改这一处，插在 `eps >= 1.0` 那个分支**之前**）：

```python
            if who == "learner" and sample:
                # PG 的行为策略：**从 π 采样，且完全忽略 ε**（spec §3.3）
                for j, p in zip(js, policy_sample_batch(neti, [pending[j] for j in js], rng)):
                    picks[j] = p
                continue
```

`train()`：签名加 `algo="dmc", beta_ent=BETA_ENT`；主循环里分叉：

```python
        batch = generate_batch(net, rng, eps, batch_games, opp_mix=opp_mix,
                               greedy_share=greedy_share,
                               learn_all_seats=learn_all_seats, bomb_cost=bomb_cost,
                               opp_kind=opp_kind, sample=(algo == "pg"))
        if algo == "pg":
            # on-policy：**不写 buffer**，直接用刚打完那批的决策点与标签
            samples = [p for _rec, pts, _y in batch for p in pts]
            rewards = [r for _rec, _pts, y in batch for r in y]
            out = _pg_step(net, opt, samples, rewards, base, games, beta_ent)
            loss = out["loss"]
            games += len(batch)
        else:
            ...（原文：写 buffer + 从 buffer 采 + `_learn_step`）
```

`train_parallel()`：同一处分支，数据来自**收到的 `recs`**（learner 侧重放）：

```python
        if algo == "pg":
            samples, rewards = [], []
            for rec in recs:
                pts, y, _b = replay.expand(rec)
                samples += pts
                rewards += y
            out = _pg_step(net, opt, samples, rewards, base, games, beta_ent)
            loss = out["loss"]
        else:
            ...（原文：buf.add + 从 buffer 采 + `_learn_step`）
```

另外两处：`worker_cfg(..., sample=(algo == "pg"))` ✓；
权重广播用 `weight_sync_games` 取代常量 `WEIGHT_SYNC_GAMES` ✓。

日志头（**PG 模式**追加这几行；DMC 模式一个字不改）：

```
  算法 pg（REINFORCE）  β_ent 0.01   **ε 不适用**（采样本身就是探索）
  **不用 replay buffer**（on-policy，只用刚打完的那一批）
  权重广播每 200 局（PG 的 staleness 靠它压小）
```

- [ ] **Step 4: 跑测试确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 除那两条**预存的**环境红以外全绿（484 + 新增）

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py train/worker.py tests/test_pg_wiring.py
git commit -m "feat(selfplay): --algo pg 全通（两条路都用 π 采样、不写 buffer、日志说清 ε 不适用）"
```

---

### Task 5: 先量成本（spec §5 的闸）—— **量完才准起 A/B**

**Files:**
- Modify: `tools/bench_train.py`（`main()` 透传 `--algo` / `--beta-ent` / `--weight-sync-games`）
- Create: `docs/superpowers/plans/2026-09-29-policy-gradient-delivery.md`

- [ ] **Step 1: 加透传**

`ap.add_argument("--algo", default="dmc")`、`--beta-ent`、`--weight-sync-games`，
并在 `bench_run(...)` 调用里带上（它已经把 `**kw` 转给 `train_parallel` ✓）。

- [ ] **Step 2: 两档各跑 3 分钟**（**不许 45 秒**：短跑会把 spawn+import torch 摊进去，踩过）

```bash
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 180 --init runs/rl/20260926-1407/best.pt --algo dmc
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 180 --init runs/rl/20260926-1407/best.pt --algo pg --weight-sync-games 200
```

- [ ] **Step 3: 按**预登记**的规则定预算（与路线②同一张表）**

| 实测（pg 相对 dmc 的局/秒） | 预算 |
|---|---|
| **≥ 70%** | 各 9000 秒 |
| **40% ~ 70%** | 各 9000 秒，主判据用**同局数配对**（配对点取「两臂都够得着的最大 2 万倍数」） |
| **< 40%** | **停下来报告用户**，不要烧 5 小时 |

⚠️ 三条都不许事后改。**预期落在 40~70%**（前向行数约 10×，路线②同形态实测 −33%）。

- [ ] **Step 4: 提交**

```bash
git add tools/bench_train.py docs/superpowers/plans/2026-09-29-policy-gradient-delivery.md
git commit -m "feat(tools): bench 透传 --algo/--beta-ent/--weight-sync-games；PG 成本台账"
```

---

### Task 6: 起 A/B（**只跑 PG 一条臂**）

**Files:**
- Modify: `docs/superpowers/plans/2026-09-29-policy-gradient-delivery.md`

> **对照臂已经在盘上**：`runs/ab/R4_b10/`（老 DMC + `--init 1407` + 同种子 + 同 ε 曲线 +
> `--opp-mix 0.5` + `--snap-every 20000`），快照到 **28 万局** ✓（spec §7）

- [ ] **Step 1: 查可用内存 ≥ 10 GB**（用 ctypes；PowerShell 在拒绝清单里）

- [ ] **Step 2: 起 PG 臂**

```powershell
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -u -m train.selfplay 9000 --workers 2 --opp-mix 0.5 `
  --init runs/rl/20260926-1407/best.pt --eps-games 200000 --snap-every 20000 `
  --algo pg --weight-sync-games 200 --out-dir runs/ab/R6_pg
```

⚠️ 盯日志里的**熵**与**`|logits|`**：熵塌会 `raise`（Task 1/3 的守门）——
那是**响亮的**失败，不是静默变慢。

- [ ] **Step 3: 同一把尺子（两臂都跑，**3 个种子**）**

```powershell
.venv/Scripts/python.exe -m tools.ab_compare runs/ab/R4_b10 runs/ab/R6_pg --ckpt pool/snap_280000.pt
.venv/Scripts/python.exe -m tools.action_margin runs/ab/R4_b10/pool/snap_280000.pt runs/ab/R6_pg/pool/snap_280000.pt
```

⚠️ **`vs 规则式` 要跑 3 个种子取均值**（单种子 sd 约 2.1pp；2026-09-28 的教训：单种子会骗人）。

- [ ] **Step 4: 逐条对预登记判据，写进台账**

| # | 判据 | 结果 |
|---|---|---|
| **1** | **`vs 规则式` 比对照好 ≥1 sd**（3 种子均值、同局数） | |
| 2 | `vs 贪心` 不掉 ≥1 sd | |
| 3 | 熵不许塌（训练中没 raise） | |
| 4 | 白炸不升 + 用炸率不塌 | |
| 5 | 真机影子日志（**要用户打牌**） | |
| 6 | 杀停：判据 1 不满足 ⇒ 停策略式 | |

---

## 自检记录

**Spec 覆盖**：§3.1 策略参数化（网络不动）→ Global Constraints + T1/T2；§3.2 目标函数与基线 → T3；
§3.3 ε→熵（含日志说清）→ T4；§3.4 on-policy 不写 buffer → T4；§3.5 对手不变 → T4
（`sample` 只作用于 learner）；§4 取舍 → Global Constraints；§5 成本先量 → T5；
§6 风险与守门（熵 / logits）→ T1/T3；§7 A/B（复用对照臂）→ T6；
§8 五条单测 → T1（数值 / 梯度 / 守门）、T2（采样分布）、T3（一步动参数）、T4（ε 失效 + 众数一致）；
§9 不在范围内、§10 初值 → Global Constraints。

**自查发现并修掉的两处**（写进 spec 与这里）：

1. **`log π` 必须带梯度** —— 现有 `_flat_scores` 是 `no_grad` 的（它服务于"算目标值"），
   照抄会让 loss 变成常数、梯度为 `None`、**训练一步不动而日志上完全看不出来**。
   ⇒ 加 `grad` 参数（默认 `False`，老调用方不变）+ T1 那条「训一步参数必须变」的测试。
2. **`|z|` 守门需要 logits**，而 `log_prob_and_entropy` 内部才有 ⇒ 让它多返回一个 `zmax`。
   （Task 1 的接口就是三元组 `(lp, ent, zmax)`。）

**Placeholder 扫描**：无 TBD；每个 code step 都是可粘贴的整段；每个 Run 都写了 Expected。

**类型一致性**：`log_prob_and_entropy(net, samples) -> (lp, ent, zmax)`、
`policy_sample_batch(net, pending, rng) -> list[int]`、`check_entropy(h, log_k, frac)`、
`_RunningMean(window)`、`_pg_step(net, opt, samples, rewards, base, games, beta_ent) -> dict`、
`generate_batch(..., sample=False)`、
`train`/`train_parallel(..., algo, beta_ent, weight_sync_games)` —— 各 task 的 Interfaces 一致。

**Review Focus 与它的测试**：
① 梯度断 → T1 `test_one_optimizer_step_actually_changes_the_parameters` + `test_log_prob_carries_gradient`；
② 行为策略不是 π → T4 `test_pg_mode_actually_samples`；
③ ε 偷偷起作用 / 日志不承认 → T4 `test_pg_mode_completely_ignores_eps` + `test_pg_header_is_honest`；
④ buffer 混进来 → T4（PG 分支不调 `buf.add`）+ 日志那条；
⑤ 熵塌 / 发散 → T1/T3 的守门测试。
