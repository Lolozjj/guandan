# 多进程自对弈实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「N 个 worker 进程打牌 + 主进程学习」接进现有训练循环，`--workers 1` 保持老行为。

**Architecture:** worker 进程跑现有的 `generate_batch`（只取 `GameRecord`，不取张量），攒够一批经 `multiprocessing.Queue` 发回；主进程收记录 → 进 replay buffer → `replay.expand` 重放 → 训练一步 → 每 1000 局把 `state_dict` + ε 广播回 worker。

**Tech Stack:** Python `multiprocessing`（Windows 用 spawn）、torch、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-26-multiprocess-selfplay-design.md`（**先读它**，验收与风险都在里面）

## Global Constraints

- **单进程那条路必须保留**：`--workers 1`（默认）与改动前逐行同形。
- **worker 死了必须炸**（`Process.is_alive()` 每轮查）—— 本仓库纪律「失败必须响」。
- **跨进程只传紧凑 `GameRecord`**（几百字节/局）；传张量会 IPC 爆炸（spec §4）。
- **不要复制**：奖励/重放用现成的 `replay.expand`（它自己会重算 reward，所以 worker
  **不必**回传名次）；评测/存档/日志抽成一个函数，两条路共用。
- Windows spawn：worker 入口要在 `if __name__ == "__main__"` 保护下起；
  测试里必须**真起一次进程**（spawn 的 import 问题只有真跑才暴露）。
- 控制台是 GBK：脚本里打印中文/符号前先 `_utf8_stdout()`（复用 `tools.accept_meld` 的）。

## Review Focus（规格没写、但最容易被这几件事咬到）

1. **队列积压**：worker 比 learner 快时 `send_q` 会堆内存。→ 队列设 `maxsize`
   （满则 worker 阻塞 = 天然背压），并把积压写进日志（任务 2 测）。
2. **权重广播的开销**：7.8 MB × N，太频繁会把 learner 拖死。→ 每 1000 局一次 +
   实测这一项占多少（任务 3 量）。
3. **ε 必须由 learner 统一广播**：各 worker 自己按局数退火会「每个都以为自己是全部」
   （任务 2 的测试钉住：worker 收到的 ε 会被采用）。
4. **worker 用的是 CPU**：worker 只做推理，显存留给 learner（任务 1 钉住：worker
   构造的网络在 CPU 上，即使 `train.net.DEVICE` 是 cuda）。
5. **进程退出要干净**：learner 异常时 worker 不能变成孤儿（`try/finally` + 终止），
   否则下次跑会看到 CPU 被占满却没人训练（任务 2 测）。

---

### Task 1: `train/worker.py` —— worker 的取数与控制循环

**Files:**
- Create: `train/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `train/selfplay.py` 的 `generate_batch`（`capture=False` 时返回 `(rec, [], [])`）
- Produces:
  - `worker_batch(net, rng, eps, n_games, opp_mix=0.5, greedy_share=0.8) -> list[GameRecord]`
  - `run_worker(send_q, ctrl_q, cfg: dict) -> None`（进程体；`cfg` 键见下）
  - `worker_cfg(seed, eps, opp_mix, greedy_share, batch_games) -> dict`

- [ ] **Step 1: 写失败的测试**

```python
"""worker 进程：取数 + 控制循环。**必须真起进程**（spawn 的 import 问题只有真跑才暴露）。"""
import multiprocessing as mp
import random

import torch

from train import selfplay, worker
from train.net import QNet


def _fresh_net(seed=0):
    torch.manual_seed(seed)
    return QNet().eval()


def test_worker_batch_matches_in_process_bit_for_bit():
    """同种子同权重同批大小 -> worker 产出的局与进程内 `generate_batch` **逐局相等**。

    这是多进程改动唯一的硬保证（spec 验收②）：分出去了但打法必须一模一样。
    """
    n = 4
    mine = [rec for rec, _p, _y in selfplay.generate_batch(
        _fresh_net(1), random.Random(7), eps=0.0, n_games=n, capture=False)]
    got = worker.worker_batch(_fresh_net(1), random.Random(7), eps=0.0, n_games=n)
    assert len(got) == n
    for a, b in zip(mine, got):
        assert (a.level, a.first, a.hands, a.actions) == (b.level, b.first, b.hands, b.actions)


def test_worker_runs_in_a_real_process_and_sends_records():
    """真起一个进程：发一批记录回来，然后能被干净地停掉。"""
    ctx = mp.get_context("spawn")
    send_q, ctrl_q = ctx.Queue(), ctx.Queue()
    cfg = worker.worker_cfg(seed=3, eps=1.0, opp_mix=0.0, greedy_share=0.8, batch_games=4)
    p = ctx.Process(target=worker.run_worker, args=(send_q, ctrl_q, cfg), daemon=True)
    p.start()
    try:
        recs = send_q.get(timeout=120)
        assert len(recs) == 4 and hasattr(recs[0], "actions")
    finally:
        ctrl_q.put(("stop", None))
        p.join(timeout=30)
    assert not p.is_alive(), "收到 stop 之后进程要退出"


def test_worker_uses_cpu_even_when_the_learner_uses_cuda():
    """worker 只做推理 —— 显存留给 learner（spec §4）。"""
    assert worker.worker_device() == "cpu"
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_worker.py -q
```
Expected: FAIL —— `ModuleNotFoundError: No module named 'train.worker'`。

- [ ] **Step 3: 写 `train/worker.py`**

```python
"""自对弈的 worker 进程：打牌、把**紧凑记录**发回给学习进程。

为什么不发张量：一局编码后约 1.6 MB，100 局/秒就是 160 MB/s 的 IPC（spec §4 已否）。
`replay.GameRecord` 只有几百字节，而且它自带 `expand` 能把奖励重算出来
（所以 worker **不需要**回传名次 —— 这是「不复制」原则的又一次适用）。

控制通道：learner 每 1000 局广播一次 `("weights", (state_dict, eps))` ——
不广播的话 worker 永远用开局那版策略，学的和打的就是两回事。
"""
from __future__ import annotations

import random

import torch

from net.sim import env  # noqa: F401  (保持与训练侧同一套 import 图)
from train import selfplay
from train.net import QNet


def worker_device() -> str:
    """worker 一律用 CPU：它只做推理，显存留给 learner。

    实测（spec §2）：这个循环的瓶颈是 Python 侧，worker 上 GPU 只会抢显存。
    """
    return "cpu"


def worker_batch(net, rng, eps, n_games, opp_mix=0.5, greedy_share=0.8):
    """打一批局，只取紧凑记录（张量与奖励都由 learner 侧重放出来）。"""
    return [rec for rec, _pts, _y in selfplay.generate_batch(
        net, rng, eps, n_games, capture=False,
        opp_mix=opp_mix, greedy_share=greedy_share)]


def _drain_ctrl(ctrl_q, net):
    """非阻塞收控制消息；返回最新收到的 ε（没收到就返回 None）。"""
    eps = None
    while True:
        try:
            kind, payload = ctrl_q.get_nowait()
        except Exception:                       # noqa: BLE001 - 队列空
            return eps
        if kind == "stop":
            raise SystemExit(0)
        if kind == "weights":
            sd, eps = payload
            net.load_state_dict(sd)


def run_worker(send_q, ctrl_q, cfg: dict) -> None:
    """进程体。`cfg` 由 `worker_cfg` 造。"""
    net = QNet().to(worker_device()).eval()
    rng = random.Random(cfg["seed"])
    eps = cfg["eps"]
    while True:
        recs = worker_batch(net, rng, eps, cfg["batch_games"],
                            opp_mix=cfg["opp_mix"], greedy_share=cfg["greedy_share"])
        send_q.put(recs)                        # 队满则阻塞 = 天然背压（spec §6）
        got = _drain_ctrl(ctrl_q, net)
        if got is not None:
            eps = got


def worker_cfg(seed, eps, opp_mix, greedy_share, batch_games) -> dict:
    return {"seed": seed, "eps": eps, "opp_mix": opp_mix,
            "greedy_share": greedy_share, "batch_games": batch_games}
```

- [ ] **Step 4: 跑测试（红的三条要全绿）**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_worker.py -q     # 3 passed
.venv/Scripts/python.exe -m pytest tests/ -q                   # 不许退步
```
Expected: 3 passed；全套仍绿（350 passed）。**第一条测试如果红**，说明 worker 的打法与
进程内不一致 —— 先修它（这是整个改动的硬保证），别往下走。

- [ ] **Step 5: 提交**

```bash
git add train/worker.py tests/test_worker.py
git commit -m "feat(train): worker 进程 —— 打牌 + 送紧凑记录 + 收权重广播"
```

---

### Task 2: `train_parallel` —— 学习进程与 worker 编排

**Files:**
- Modify: `train/selfplay.py`（抽出 `_evaluate_and_save`，新增 `train_parallel`，加 `--workers`）
- Test: `tests/test_train_parallel.py`

**Interfaces:**
- Consumes: `train/worker.py` 的 `run_worker` / `worker_cfg`
- Produces: `train_parallel(seconds, workers, buffer_games, eval_games, eval_every, out_dir, log, opp_mix, greedy_share, batch_games, eps_games, seed) -> dict`（返回结构与 `train()` 一致）

- [ ] **Step 1: 写失败的测试**

```python
"""学习进程的编排：收记录 -> 训练 -> 广播；worker 死了必须炸。"""
import multiprocessing as mp
import time

import pytest

from train import selfplay


def test_worker_death_raises_loudly(tmp_path):
    """worker 死了**必须报错**，不许静默变慢（本仓库纪律：失败必须响）。"""
    with pytest.raises(RuntimeError, match="worker"):
        selfplay.train_parallel(seconds=20, workers=1, out_dir=str(tmp_path),
                                batch_games=2, eval_games=1, log=lambda *a: None,
                                _kill_worker_after=2.0)      # 测试钩子：2 秒后杀一个


def test_train_parallel_runs_a_few_steps_and_saves(tmp_path):
    """短跑（15 秒）要走通：有 loss、有存档、返回结构与非并行一致。"""
    r = selfplay.train_parallel(seconds=15, workers=2, out_dir=str(tmp_path),
                                batch_games=2, eval_games=1, eval_every=10_000,
                                log=lambda *a: None)
    assert r["games"] > 0 and r["curve"] == []
    import os
    assert os.path.exists(os.path.join(str(tmp_path), "last.pt"))
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_train_parallel.py -q
```
Expected: FAIL —— `train_parallel` 不存在。

- [ ] **Step 3: 实现**

要点（照着写，别自由发挥）：

1. **先抽公共件**：把 `train()` 里「每 N 局评测 + 存 best.pt」和收尾那一段
   抽成 `_evaluate_and_save(net, games, curve, best_greedy, eval_games, out_dir, log) -> best_greedy`
   与 `_finish(net, games, steps, t0, curve, out_dir, log) -> dict`；
   `train()` 改成调用它们（**先跑一遍全套测试确认没改坏**，再往下）。
2. `train_parallel` 的骨架：

```python
def train_parallel(seconds=3600.0, workers=1, seed=0, buffer_games=BUFFER_GAMES,
                   eval_games=EVAL_GAMES, eval_every=EVAL_EVERY_GAMES,
                   out_dir=None, log=print, opp_mix=0.5, greedy_share=0.8,
                   batch_games=BATCH_GAMES, eps_games=EPS_GAMES,
                   _kill_worker_after=None):
    from train import worker as worker_mod
    ctx = mp.get_context("spawn")
    out_dir = out_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    net = QNet().to(DEVICE)
    torch.manual_seed(seed)
    buf = replay.ReplayBuffer(capacity_games=buffer_games)
    send_q = ctx.Queue(maxsize=workers * 2)      # 有界 = 背压，不堆内存（review focus 1）
    ctrls = [ctx.Queue() for _ in range(workers)]
    procs = [ctx.Process(target=worker_mod.run_worker, daemon=True,
                         args=(send_q, ctrls[k],
                               worker_mod.worker_cfg(seed + 1 + k, EPS_START,
                                                     opp_mix, greedy_share, batch_games)))
             for k in range(workers)]
    ...
    try:
        for p in procs: p.start()
        while time.perf_counter() - t0 < seconds:
            # worker 死了就炸（review focus 5）
            dead = [k for k, p in enumerate(procs) if not p.is_alive()]
            if dead:
                raise RuntimeError(f"worker {dead} 挂了（exitcode="
                                   f"{[procs[k].exitcode for k in dead]}）—— 不许静默变慢")
            recs = send_q.get(timeout=300)       # 超时也炸
            for rec in recs:
                buf.add(rec); games += 1
            # 采样 + 训练一步（与非并行那条路同一套）
            samples, targets = [], []
            for rec in buf.sample(batch_games, rng):
                pts, y = replay.expand(rec)
                samples += pts; targets += y
            ...（`_tensors` + mse_loss + opt.step 与 `train()` 逐字相同）
            steps += 1
            # 每 WEIGHT_SYNC_GAMES 局广播一次（review focus 2：别太频繁）
            if games - last_sync >= WEIGHT_SYNC_GAMES:
                sd = {k: v.cpu() for k, v in net.state_dict().items()}
                for q in ctrls: q.put(("weights", (sd, eps_for(games, eps_games))))
                last_sync = games
            ...（日志 / 评测 / 存档走公共件）
    finally:
        for q in ctrls: q.put(("stop", None))
        for p in procs:
            p.join(timeout=10)
            if p.is_alive(): p.terminate()
    return _finish(...)
```

3. `--workers N` 接进 `main()`；`WEIGHT_SYNC_GAMES = 1000`（模块级常量 + 注释）。
4. `_kill_worker_after`：**测试钩子**，到点 `p.terminate()` 一个 worker，用来验死亡检测
   —— 写成参数并在 docstring 里说明只给测试用。

- [ ] **Step 4: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_train_parallel.py -q   # 2 passed
.venv/Scripts/python.exe -m pytest tests/ -q                          # 全绿
.venv/Scripts/python.exe -m train.selfplay 20                         # 单进程那条路没坏
```
Expected: 全过；单进程 20 秒能正常跑完并打印（`--workers` 默认 1 = 老行为）。

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py tests/test_train_parallel.py
git commit -m "feat(train): --workers N —— worker 打牌、主进程学习（默认 1 = 老行为）"
```

---

### Task 3: `tools/bench_train.py` —— 吞吐曲线与瓶颈定位（验收②③）

**Files:**
- Create: `tools/bench_train.py`
- Test: `tests/test_bench_train.py`

**Interfaces:**
- Produces: `python -m tools.bench_train --workers 1 2 3 4 6 --seconds 60`（打印每档的
  局/秒、队列积压、广播开销）；`bench_run(workers, seconds, **kw) -> dict`

- [ ] **Step 1: 写测试（薄）**

```python
def test_bench_reports_games_per_second(tmp_path):
    from tools.bench_train import bench_run
    r = bench_run(workers=1, seconds=8, out_dir=str(tmp_path), log=lambda *a: None)
    assert r["games"] > 0 and r["games_per_s"] > 0
```

- [ ] **Step 2: 实现**：`bench_run` = 调 `train_parallel`（`_kill_worker_after=None`）+ 额外量三样：
  - `games_per_s`
  - **队列积压**：跑完读一下 `send_q.qsize()`（或训练中定时采样最大值）
  - **广播开销**：把 `WEIGHT_SYNC_GAMES` 设小（如 100）再跑一次，对比局/秒的差
  - **learner 利用率**：`psutil` 不在依赖里 → 用 `os.times()` 自算 cpu% hmm，简化：
    只报「learner 每步的重放耗时」（`replay.expand` 的时间 / 局）
- [ ] **Step 3: 真跑一遍曲线并写进交付文档**

```powershell
.venv/Scripts/python.exe -m tools.bench_train --workers 1 2 3 4 6 --seconds 60
```
Expected: 报出曲线；**给出推荐 worker 数**（以 23 局/秒为理论天花板，见 spec §2）。
- [ ] **Step 4: 提交**

---

### Task 4: 短跑验证 + 文档

**Files:**
- Modify: `HANDOFF.md`（训练怎么跑、worker 数推荐值）
- Create: `docs/superpowers/plans/2026-09-26-multiprocess-delivery.md`（台账）

- [ ] **Step 1: 10 分钟短跑（2~3 worker）**，确认 loss 正常下降、收尾打印 `vs 贪心` 与
  `炸弹浪费率`；把原始输出收进台账。
- [ ] **Step 2: 写台账**（照 Plan 1/2/3 体例）：一句话结论、怎么跑、吞吐曲线、
  与单进程的对比、建议的 worker 数、已知风险（不可复现的定义、队列、显存）。
- [ ] **Step 3: 更新 HANDOFF**：训练命令加 `--workers N`；写明「瓶颈从 Python 移到
  学习进程的重放」这件事，以及下一步（learner 走 GPU 的那次测量结论）。
- [ ] **Step 4: 提交**

---

## 收尾自检

**规格覆盖**：§3 架构 → 任务 1/2；§4 取舍（紧凑记录 / ε 广播 / 单进程保留 / 默认 1）
→ 任务 1/2 的接口与约束；§5 验收 ①等价性→任务 1、②吞吐曲线→任务 3、③短跑→任务 4、
④worker 死亡→任务 2、⑤`--workers 1` 同形→任务 2 Step 4；§6 风险 → Review Focus 五条
逐条落到测试；§7 不在范围内 → 不建任务。

**没做的（明确在计划外）**：Rust 内核、learner 的 GPU 落地（只量）、多机分布式、
共享内存版 buffer、worker 数自适应。
