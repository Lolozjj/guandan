# 对手池实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把训练的固定对手从「贪心」换成「模型自己的历史版本池」，池内按 PFSP 采样，让「白炸一手」这类错误真的会输掉牌局。

**Architecture:** 分两期。**0 期**给 `GameRecord` 加 `learn` 座位字段，让 `expand` 重放时只产出学习那一队的决策点（修掉 17.3% 的标签污染；池子会放大它）。**1 期**加一个 `train/pool.py`（只做记账与采样数学，不碰进程）、定期权重快照、以及让 worker 常驻持有多份权重并**按权重分组做批量前向**。

**Tech Stack:** Python `multiprocessing`（Windows spawn）、torch、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-27-opponent-pool-design.md`（**先读它**，验收与风险都在里面）

## Global Constraints

- **`--workers 1` 与改动前逐行同形**（默认行为不变）。
- **判据不动**：退出码仍是 `vs 贪心 ≥ 55%`（保住与 0033/0940/1407 的可比性）。
- **不许复制**：`generate_batch` / `replay.expand` / `q_values` 各只有一份；
  新增的采样数学只写在 `train/pool.py`，训练与外层共用它。
- **失败必须响**：抽到不存在的池成员、权重加载失败、池子塌成一个成员 —— 都要报出来，
  不许静默退化成「少一个成员」。
- **跨进程只传紧凑记录**（几百字节/局）。**成员权重是唯一例外**，且**只在新增时传一次**。
- 控制台是 GBK：脚本里打印中文/符号前先 `_utf8_stdout()`（复用 `tools.accept_meld` 的）。
- 注释里出现的**实测数字必须带日期与口径**（本仓库靠这个防止「重新推导」）。

## Review Focus（规格没写、但最容易被这几件事咬到）

1. **`learn` 过滤会不会连「局面推进」也一起过滤掉** —— 重放要靠**全部**步骤才能走下去，
   过滤只能发生在产出训练点那一步。过滤早了，`e.step(i)` 就少走一步，局面直接错位。
2. **`learn_all_seats` 对照臂必须是「真·老行为」** —— 对手照旧换、四家照旧都学。
   只改一处却改了两处，A/B 就没有意义。
3. **`GameRecord` 加字段会不会破坏逐字段等价测试** —— `tests/test_train_replay.py`
   有一条「重放必须逐字段复现现场」，新字段进不进那条比对，要明确。
4. **禁用 ε 到对手身上** —— ε 只属于学习者；对手一律 argmax（spec §4）。
   手滑把 ε 也施加到池成员上，池子就变成了「一群会随机出牌的对手」。
5. **快照绝不能长得像 `best.pt`** —— `net/advise.py::newest_weights` 按修改时间
   挑 `runs/rl/*/best.pt`，误命中会把用户在面板上看到的建议换掉（静默换源）。
6. **`1/Σw²` 报警不能只在日志里** —— 池子塌了是「失败」，要走本仓库的「失败必须响」。

---

## 第 0 期

### Task 1: `GameRecord.learn` —— `expand` 只产出学习那一队

**Files:**
- Modify: `train/replay.py`（`GameRecord` 加字段；`of()` 加参数；`expand()` 加过滤）
- Modify: `train/selfplay.py`（`generate_batch` 传 `learn`；新参数 `learn_all_seats`；
  `train()` / `train_parallel()` / `main()` 接线）
- Test: `tests/test_expand_learn.py`（新建）

**Interfaces:**
- Consumes: 无（第一个任务）
- Produces:
  - `replay.GameRecord(level, first, hands, actions, learn=None)` ——
    `learn` 是**学习座位**的 tuple；`None` = 四家都学（老行为）
  - `replay.GameRecord.of(e, actions, hands0, learn=None)`
  - `selfplay.generate_batch(..., learn_all_seats=False)`

- [ ] **Step 1: 写失败的测试**

```python
"""`expand` 必须只产出**学习那一队**的决策点。

钉的是一个真 bug（2026-09-27 实测）：`generate_batch` 的现场抓取（`caps`）确实只记
学习那一队，但**单进程那条路几乎不用现场抓取** —— `buf.sample` 从 5 万局里有放回抽
32 局，撞上刚打那 32 局的期望是 **0.02 局/步**，所以实际训练数据 ~100% 来自 `expand`。
而 `GameRecord` 没有「哪一队是学习的」这个信息，`expand` 只能把四家全产出成训练点。

实测污染率：`opp_mix=0.5` 时 6008 个重放训练点里 **1040 个（17.3%）是固定对手的着法**，
标签却是那一局的胜负 —— 等于拿对手当老师。
"""
import random

import pytest
import torch

from train import replay, selfplay
from train.net import QNet


def _fresh_net(seed=0):
    torch.manual_seed(seed)
    return QNet().eval()


def _play(n_games=8, **kw):
    return selfplay.generate_batch(_fresh_net(1), random.Random(0), eps=0.0,
                                   n_games=n_games, capture=True, **kw)


def test_expand_only_yields_the_learner_team():
    """`opp_mix=1.0`：重放产出的座位必须**只有学习那一队**。"""
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0):
        learner = {s for (_o, _a, _i, s, _h) in caps}
        pts, y = replay.expand(rec)
        seats = {s for (_o, _a, _i, s, _h) in pts}
        assert seats <= learner, \
            f"重放产出了非学习队的座位：{sorted(seats - learner)}"
        assert len(pts) == len(caps)
        assert y == _y


def test_expand_reproduces_the_learner_points_bit_for_bit():
    """过滤之后，剩下的每一点仍要与现场抓到的**逐点相同**（顺序、座位、下标）。

    过滤要是把局面推进也一起跳过了，这里立刻对不上 —— 见 Review Focus 1。
    """
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0):
        pts, _y2 = replay.expand(rec)
        assert [(s, i) for (_o, _a, i, s, _h) in pts] == \
               [(s, i) for (_o, _a, i, s, _h) in caps]


def test_expand_is_unchanged_for_pure_selfplay():
    """纯自对弈（`opp_mix=0`）时四家都学 —— 老行为不许变。"""
    for rec, _caps, _y in _play(opp_mix=0.0):
        pts, _y2 = replay.expand(rec)
        assert {s for (_o, _a, _i, s, _h) in pts} == {0, 1, 2, 3}


def test_learn_all_seats_restores_the_old_behaviour():
    """`learn_all_seats=True` = A/B 的**对照臂**：对手照旧换，但四家照旧都学。"""
    for rec, caps, _y in _play(opp_mix=1.0, greedy_share=1.0, learn_all_seats=True):
        pts, _y2 = replay.expand(rec)
        assert {s for (_o, _a, _i, s, _h) in pts} == {0, 1, 2, 3}
        assert {s for (_o, _a, _i, s, _h) in caps} == {0, 1, 2, 3}, \
            "对照臂连现场抓取也必须是四家 —— 否则两臂差的就不止一处"
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_expand_learn.py -q
```
Expected: FAIL —— 前两条红（`expand` 现在产出四家），后两条可能绿。

- [ ] **Step 3: 改 `train/replay.py`**

`GameRecord` 加一个字段（**放最后并给默认值**，老记录仍可构造）：

```python
@dataclass(frozen=True)
class GameRecord:
    """一局的紧凑记录 —— 足以重放出全部决策点。

    `actions` 是每一步在 `env.legal()` 候选里的**下标**（含「过」那条），
    不是牌张 —— 重放时会重新枚举，下标对得上就行。

    `learn`：**哪几个座位是学习的**。`None` = 四家都学（纯自对弈 / 老记录）。
    跟牌的固定对手那一队不该进训练目标（记进去等于拿它当老师），
    而重放侧**没有别的办法**知道这件事 —— 所以它必须跟记录一起走。
    """
    level: int
    first: int
    hands: tuple
    actions: tuple
    learn: tuple = None

    @staticmethod
    def of(e: "env.GuandanEnv", actions, hands0, learn=None) -> "GameRecord":
        return GameRecord(level=e.hand.level, first=e.hand.steps[0].seat,
                          hands=tuple(tuple(sorted(h)) for h in hands0),
                          actions=tuple(actions),
                          learn=tuple(learn) if learn is not None else None)
```

`expand` 加过滤（**只过滤产出，不动 `e.step(i)`**）：

```python
def expand(rec: GameRecord):
    """把记录重放成 `(决策点, 终局 reward)`。

    `rec.learn` 里的座位才产出决策点（`None` = 四家都产出，老行为）。
    ⚠️ **局面必须每一步都往前走** —— 过滤只发生在 `points.append` 那一行。
    提前 `continue` 会让重放错位（这是本任务最容易被写错的地方）。
    """
    e = env.GuandanEnv(seed=0)
    e.reset(level=rec.level, hands=[set(h) for h in rec.hands], first=rec.first)
    learn = set(rec.learn) if rec.learn else None
    points = []
    obs = e.observe()
    for i in rec.actions:
        acts = e.legal()
        seat = e.hand.turn
        if learn is None or seat in learn:
            points.append((obs, acts, i, seat, env.encode_history(e.hand, seat)))
        obs, _r, _done, _info = e.step(i)
    ranks = e.ranks
    y = [rules.reward(ranks, seat) for (_o, _a, _i, seat, _h) in points]
    return points, y
```

- [ ] **Step 4: 改 `train/selfplay.py`**

`generate_batch` 加参数并把 `learn[k]` 收成**一处**（现场抓取与记录共用它，
这样对照臂不可能只改一半）：

```python
def generate_batch(net, rng, eps, n_games, capture=True, opp_mix=0.0,
                   greedy_share=0.8, learn_all_seats=False):
    """...
    `learn_all_seats`：**A/B 的对照臂**（spec §3.0）。开了之后对手照旧换，
    但四家照旧都学 —— 也就是改动前的行为，用来量「修 `expand` 到底值多少」。
    """
    ...
        if rng.random() < opp_mix:
            opp = rng.randrange(2)
            kind = "greedy" if rng.random() < greedy_share else "random"
            fixed.append((kind, opp))
            learn.append(tuple(s for s in rules.SEATS if rules.TEAM[s] != opp))
        else:
            fixed.append(None)
            learn.append(tuple(rules.SEATS))
        if learn_all_seats:
            learn[-1] = tuple(rules.SEATS)   # 对照臂：一处改，现场抓取与记录一起跟
    ...
    for k in range(n_games):
        e = envs[k]
        rec = replay.GameRecord.of(e, log[k], hands0[k], learn=learn[k])
        ...
```

`train()` / `train_parallel()` 各加一个 `learn_all_seats: bool = False` 形参并传给
`generate_batch`；`main()` 加：

```python
    if "--learn-all-seats" in argv:
        kw["learn_all_seats"] = True
```

- [ ] **Step 5: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_expand_learn.py -q        # 4 passed
.venv/Scripts/python.exe -m pytest tests/ -q                            # 不许退步
.venv/Scripts/python.exe -m train.selfplay 20 --workers 2               # 两条路都走通
.venv/Scripts/python.exe -m train.selfplay 20 --workers 2 --learn-all-seats
```
Expected: 全绿（现有 355 passed + 1 个预存的 `test_tribute_records` 红）；
两条 20 秒短跑都能正常收尾并打印判据。

- [ ] **Step 6: 提交**

```bash
git add train/replay.py train/selfplay.py tests/test_expand_learn.py
git commit -m "fix(train): expand 只产出学习那一队的决策点（修 17.3% 的标签污染）"
```

---

## 第 1 期

### Task 2: `--init` 热启动

**Files:**
- Modify: `train/selfplay.py`（`_load_init`、`train`、`train_parallel`、`main`）
- Modify: `train/worker.py`（`worker_cfg` 带 `init`；`run_worker` 开局装上）
- Test: `tests/test_train_init.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `train(..., learn_all_seats=)`
- Produces:
  - `selfplay.load_init(net, path) -> None`（路径为 `None` 时什么都不做）
  - `train(..., init=None)` / `train_parallel(..., init=None)`
  - `worker.worker_cfg(seed, eps, opp_mix, greedy_share, batch_games, init=None)`

**为什么要它**：池子只有在模型足够强时才有意义。没有热启动，验证池子得先练 9 小时；
而且 A/B 两臂都从**同一个起点**出发才可比（spec §3.5）。

- [ ] **Step 1: 写失败的测试**

```python
"""`--init` 必须**真的**把权重装进去。

否则「热启动」是假的：两臂都从随机初始化开始，A/B 量的是别的东西。
这里用端到端断言 —— 1 秒训练后权重离 checkpoint 近、离随机初始化远。
"""
import os

import torch

from train import selfplay
from train.net import QNet


def test_init_warm_starts_from_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    src = QNet()
    with torch.no_grad():                      # 造一份「好认」的权重
        for p in src.parameters():
            p.mul_(3.0)
    ck = os.path.join(str(tmp_path), "init.pt")
    torch.save({"net": src.state_dict()}, ck)

    r = selfplay.train(seconds=1, init=ck, out_dir=str(tmp_path), log=lambda *a: None,
                       eval_games=1, eval_every=10 ** 9, batch_games=2)
    got = torch.load(os.path.join(r["out_dir"], "last.pt"), map_location="cpu")["net"]
    fresh = QNet().state_dict()
    near = max((got[k] - v).abs().max().item() for k, v in src.state_dict().items())
    far = max((got[k] - v).abs().max().item() for k, v in fresh.items())
    assert near < far * 0.5, f"热启动没生效（near={near:.3f} far={far:.3f}）"


def test_load_init_is_a_noop_without_a_path():
    """不传 `--init` 时不许改变任何东西（默认行为不变）。"""
    torch.manual_seed(0)
    net = QNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    selfplay.load_init(net, None)
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k])
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_train_init.py -q
```
Expected: FAIL —— `train()` 不接受 `init`；`selfplay.load_init` 不存在。

- [ ] **Step 3: 实现**

`train/selfplay.py`：

```python
def load_init(net, path: str = None) -> None:
    """把 checkpoint 装进 `net`。`path=None` 时什么都不做（默认行为不变）。

    只认 `{"net": state_dict}` 这一种（与 `best.pt` / `last.pt` 同格式）；
    装不上**必须炸** —— 静默从随机开始会让热启动变成假的，而日志上看不出来。
    """
    if not path:
        return
    ck = torch.load(path, map_location="cpu")
    net.load_state_dict(ck["net"] if isinstance(ck, dict) else ck)
```

`train()` 与 `train_parallel()` 各加 `init: str = None` 形参，在建完 `net` 之后：

```python
    net = QNet().to(DEVICE)
    load_init(net, init)
    if init:
        log(f"热启动：{init}")
```

`train_parallel` 里 worker 的构造要带上 `init`（worker 开局自己那一份也得是它，
否则广播到达之前的第一批是随机的）：

```python
                               worker_mod.worker_cfg(seed + 1 + k, EPS_START,
                                                     opp_mix, greedy_share,
                                                     batch_games, init=init)))
```

`train/worker.py`：

```python
def worker_cfg(seed, eps, opp_mix, greedy_share, batch_games, init=None) -> dict:
    return {"seed": seed, "eps": eps, "opp_mix": opp_mix,
            "greedy_share": greedy_share, "batch_games": batch_games,
            "init": init}


def run_worker(send_q, ctrl_q, cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    net = QNet().to(worker_device()).eval()
    selfplay.load_init(net, cfg.get("init"))
    ...
```

`main()` 加两个开关：

```python
    if "--init" in argv:
        kw["init"] = argv[argv.index("--init") + 1]
    if "--out-dir" in argv:
        kw["out_dir"] = argv[argv.index("--out-dir") + 1]
```

**`--out-dir` 是 A/B 的前提**：两臂必须落在不同目录，否则后跑的那一臂会覆盖
`best.pt` / `last.pt` —— 而 `net/advise.py::newest_weights` 按修改时间挑权重，
面板会**静默换源**。

- [ ] **Step 4: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_train_init.py -q     # 2 passed
.venv/Scripts/python.exe -m pytest tests/ -q                        # 不许退步
.venv/Scripts/python.exe -m train.selfplay 20 --workers 2 --init runs/rl/20260926-1407/best.pt
```
Expected: 全绿；第三条打印「热启动：…」且正常收尾。

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py train/worker.py tests/test_train_init.py
git commit -m "feat(train): --init 热启动（池子实验必须两臂同起点）"
```

---

### Task 3: `train/pool.py` —— 记账 + PFSP 采样 + `GameRecord.opp`

**Files:**
- Create: `train/pool.py`
- Modify: `train/replay.py`（`GameRecord.opp`；`of()` 加参数）
- Modify: `train/selfplay.py`（`generate_batch` 记录 `opp`）
- Test: `tests/test_pool.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `learn`
- Produces:
  - `pool.PoolMember(mid, path)`
  - `pool.WinRates(window=PFSP_WINDOW)`：`.record(mid, learner_won)` / `.games(mid)` / `.rate(mid)`
  - `pool.pfsp_weights(rates: dict, games: dict, uniform=PFSP_UNIFORM, min_games=PFSP_MIN_GAMES) -> dict`
  - `pool.pick_opponent(rng, member_ids, weights, greedy_share=GREEDY_SHARE) -> tuple`
  - `replay.GameRecord.opp`：`None` = 纯自对弈；`("greedy", 队号)` / `("random", 队号)` / `("member", mid)`
    ⚠️ **第二个元素的口径不统一**：对 `member` 是**成员 id**，对其余是**队号**。
    两者都是小整数，写错了不会报错 —— 所以 `opp[0]` 必须先判，别只取 `opp[1]`。

**为什么 `opp` 必须进记录**：PFSP 要知道「这一局打的是谁」才能归因胜负。
胜者不用加 —— `expand` 重放出来的 `y` 符号就告诉我们哪一队赢了（spec §3.2）。

- [ ] **Step 1: 写失败的测试**

```python
"""对手池的记账与采样 —— 纯函数，不碰进程、不碰 torch。"""
import random

import pytest

from train import pool


def test_pfsp_peaks_at_even_matchups():
    """专挑五五开的：p=0.5 权重最大，打穿/打不过的两头都小。"""
    w = pool.pfsp_weights({0: 0.5, 1: 0.05, 2: 0.95, 3: 0.5},
                          games={0: 999, 1: 999, 2: 999, 3: 999})
    assert w[0] == pytest.approx(w[3])
    assert w[0] > w[1] and w[0] > w[2]
    assert sum(w.values()) == pytest.approx(1.0)


def test_pfsp_keeps_a_uniform_floor_at_both_extremes():
    """两头都趋 0 但**不为 0** —— 防饿死；且两个极端对称。"""
    w = pool.pfsp_weights({0: 1.0, 1: 0.0}, games={0: 999, 1: 999})
    assert w[0] > 0 and w[1] > 0
    assert w[0] == pytest.approx(w[1], rel=1e-9)


def test_pfsp_is_uniform_during_warmup():
    """局数不够的成员按**均匀**算 —— 它的 p 还是噪声（spec §3.2 预热）。"""
    w = pool.pfsp_weights({0: 0.5, 1: 0.0}, games={0: 999, 1: 3}, min_games=20)
    assert w[0] == pytest.approx(w[1])


def test_winrate_window_forgets_the_past():
    """胜率必须滑窗 —— 学习者在变强，老胜率会过期（spec §3.2）。"""
    wr = pool.WinRates(window=10)
    for _ in range(10):
        wr.record(0, True)                     # 先连赢 10 局
    assert wr.rate(0) > 0.8
    for _ in range(10):
        wr.record(0, False)                    # 再连输 10 局，窗口里只剩这些
    assert wr.rate(0) < 0.2, "窗口没把老的胜绩冲掉"


def test_pick_opponent_reserves_a_share_for_greedy():
    """混合局里贪心拿**固定份额**，且不进 PFSP（它是判据的尺子 + 防池子退化）。"""
    rng = random.Random(0)
    ids = [0, 1, 2]
    w = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    picks = [pool.pick_opponent(rng, ids, w, greedy_share=0.2) for _ in range(2000)]
    n_greedy = sum(1 for p in picks if p == ("greedy",))
    assert 0.16 < n_greedy / 2000 < 0.24, f"贪心份额 {n_greedy / 2000:.2%}"
    assert any(p[0] == "member" for p in picks), "池成员一次都没被抽到"


def test_effective_member_count_flags_a_collapsed_pool():
    """池子塌成一个成员要**看得出来**（spec §1.4）。"""
    assert pool.effective_members({0: 1.0}) == pytest.approx(1.0)
    assert pool.effective_members({0: 0.5, 1: 0.5}) == pytest.approx(2.0)
    assert pool.effective_members({0: 0.9, 1: 0.1}) < 1.3


def test_record_carries_the_opponent():
    """记录里要能读出「这一局打的是谁」—— PFSP 靠它归因（spec §3.2）。"""
    import torch
    from train import selfplay
    from train.net import QNet
    torch.manual_seed(0)
    out = selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=8,
                                  capture=False, opp_mix=1.0, greedy_share=1.0)
    assert all(rec.opp and rec.opp[0] == "greedy" for rec, _p, _y in out)
    out = selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=4,
                                  capture=False, opp_mix=0.0)
    assert all(rec.opp is None for rec, _p, _y in out)
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool.py -q
```
Expected: FAIL —— `No module named 'train.pool'`。

- [ ] **Step 3: 写 `train/pool.py`**

```python
"""对手池的记账与采样（spec §3.1/§3.2）。

**只管两件事**：记住「每个成员打了多少局、学习者赢了多少」，按 PFSP 算出采样权重。
不碰进程、不碰 torch、不碰文件 —— 所以它能在任何地方单测。

为什么规格是 `p·(1−p)`：`p` 是学习者对该成员的胜率。`p → 1`（已经打穿）没信息，
`p → 0`（完全打不过）没梯度，**只有接近五五开才学得到东西**。两头都趋 0，
再加一个**均匀下限**防饿死、兜住预热期与陈旧的胜率估计。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: 局数少于这个数的成员，`p` 还是噪声（0/0）—— 按均匀算。
PFSP_MIN_GAMES = 20
#: 胜率滑窗。学习者在变强，老胜率会过期。
PFSP_WINDOW = 200
#: 均匀下限：占采样权重的这个比例。防饿死 + 兜底。
PFSP_UNIFORM = 0.2
#: 混合局里贪心的固定份额。它是判据的尺子，也是防池子退化的锚。
GREEDY_SHARE = 0.2
#: Beta 平滑的伪计数（`(wins+a)/(games+a+b)`）。
PRIOR = 2.0
#: 有效成员数低于这个值就报「池子塌了」。
COLLAPSE_BELOW = 2.0


@dataclass(frozen=True)
class PoolMember:
    mid: int
    path: str


class WinRates:
    """每个成员一个**滑窗**的胜负计数（**学习者视角**：True = 学习者赢）。"""

    def __init__(self, window: int = PFSP_WINDOW):
        self.window = window
        self._w = {}

    def record(self, mid: int, learner_won: bool) -> None:
        self._w.setdefault(mid, deque(maxlen=self.window)).append(bool(learner_won))

    def games(self, mid: int) -> int:
        return len(self._w.get(mid, ()))

    def rate(self, mid: int) -> float:
        d = self._w.get(mid)
        if not d:
            return 0.5                       # 没打过 -> 中性，配合预热期按均匀算
        wins = sum(d)
        return (wins + PRIOR) / (len(d) + 2 * PRIOR)


def pfsp_weights(rates: dict, games: dict = None, uniform: float = PFSP_UNIFORM,
                 min_games: int = PFSP_MIN_GAMES) -> dict:
    """`{mid: p}` -> `{mid: 采样权重}`（已归一化）。

    局数 < `min_games` 的成员按**均匀**算（`p` 还是噪声）。
    """
    ids = sorted(rates)
    k = len(ids)
    if k == 0:
        return {}
    games = games or {}
    raw = {}
    for i in ids:
        # 预热成员**把 p 当中性的 0.5** —— 不能给一个「特殊的大常数」：
        # p(1-p) 最大只有 0.25，给 1.0 就是让它比健康成员重 4 倍（测试抓到过这个）
        p = 0.5 if games.get(i, 0) < min_games else rates[i]
        raw[i] = p * (1.0 - p)
    tot = sum(raw.values())
    if tot <= 0:                             # 全塌成 0（理论上不会，防一手）
        return {i: 1.0 / k for i in ids}
    # 均匀下限按成员数摊，再与 PFSP 项按 (1-uniform) 混合
    return {i: (1.0 - uniform) * raw[i] / tot + uniform / k for i in ids}


def pick_opponent(rng, member_ids, weights: dict, greedy_share: float = GREEDY_SHARE):
    """这一局的固定对手是谁。

    先按 `greedy_share` 决定「是不是贪心」，否则按 `weights` 在池成员里抽一个。
    """
    if rng.random() < greedy_share or not member_ids:
        return ("greedy",)
    ids = sorted(member_ids)
    tot = sum(weights.get(i, 0.0) for i in ids)
    if tot <= 0:
        # 还没收到第一次 PFSP 广播 —— 按均匀抽一个，**别崩也别静默退回贪心**
        # （退回贪心会让「池子刚起步那一段」偷偷变成纯贪心局）
        return ("member", ids[rng.randrange(len(ids))])
    r = rng.random()
    acc = 0.0
    for i in ids:
        acc += weights.get(i, 0.0)
        if r <= acc:
            return ("member", i)
    return ("member", ids[-1])               # 浮点兜底


def effective_members(weights: dict) -> float:
    """有效成员数 `1 / Σw²` —— 池子塌成一个成员时会趋近 1（spec §1.4）。"""
    s = sum(w * w for w in weights.values())
    return 1.0 / s if s > 0 else 0.0
```

- [ ] **Step 4: 改 `train/replay.py` 与 `train/selfplay.py`**

`GameRecord` 再加一个字段（同样放最后给默认值）：

```python
    opp: tuple = None        # 这一局的固定对手：None = 纯自对弈
                             # ("greedy", 队号) / ("random", 队号) / ("member", mid)
                             # ⚠️ 第二个元素对 member 是**成员 id**、其余是**队号**
```

`of()` 加 `opp=None` 形参并透传。

`generate_batch` 建成记录时带上（`fixed[k]` 已经是这个形状，直接用）：

```python
        rec = replay.GameRecord.of(e, log[k], hands0[k], learn=learn[k],
                                   opp=tuple(fixed[k]) if fixed[k] else None)
```

- [ ] **Step 5: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool.py -q    # 8 passed
.venv/Scripts/python.exe -m pytest tests/ -q                 # 不许退步
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
git add train/pool.py train/replay.py train/selfplay.py tests/test_pool.py
git commit -m "feat(pool): 对手池的记账与 PFSP 采样 + 记录带上对手"
```

---

### Task 4: 定期快照 + 池子装载

**Files:**
- Modify: `train/pool.py`（`snapshot_path` / `prune_snapshots` / `load_members`）
- Modify: `train/selfplay.py`（`SNAP_EVERY_GAMES`；`_maybe_eval` 里存快照）
- Test: `tests/test_pool_snapshots.py`（新建）

**Interfaces:**
- Consumes: Task 3 的 `PoolMember`
- Produces:
  - `pool.SNAP_EVERY_GAMES = 20_000`、`pool.POOL_SIZE = 20`
  - `pool.snapshot_path(out_dir, games) -> str`（`<out_dir>/pool/snap_<games>.pt`）
  - `pool.prune_snapshots(out_dir, keep=POOL_SIZE) -> list[str]`（返回被删的）
  - `pool.load_members(paths) -> list[PoolMember]`
  - `selfplay.train(..., snap_every=pool.SNAP_EVERY_GAMES, pool_size=pool.POOL_SIZE)`
    与 `train_parallel(...)` 同签名；`main()` 加 `--snap-every N` / `--pool-size N`

**为什么快照要跟「有没有进步」解耦**：现在只在刷新 `best.pt` 时存，
1407 那 144 万局只落了几个点 —— 池子原料不够。

- [ ] **Step 1: 写失败的测试**

```python
"""快照必须**嵌套 + 不叫 best.pt** —— 否则面板会误加载（静默换源）。"""
import glob
import os

from train import pool


def test_snapshot_path_is_nested_and_not_named_best(tmp_path):
    p = pool.snapshot_path(str(tmp_path), 20000)
    assert os.path.basename(p).startswith("snap_")
    assert os.path.dirname(p).endswith("pool")


def test_snapshots_are_invisible_to_newest_weights(tmp_path):
    """`newest_weights()` 按修改时间挑 `runs/rl/*/best.pt` —— 快照不能被它命中。

    命中的后果是**静默换源**：用户面板上的建议会换成另一个模型，而日志上看不出来。
    """
    from net.advise import newest_weights
    os.makedirs(pool.snapshot_path(str(tmp_path), 20000).rsplit(os.sep, 1)[0],
                exist_ok=True)
    open(pool.snapshot_path(str(tmp_path), 20000), "wb").close()
    assert newest_weights(str(tmp_path)) is None


def test_prune_keeps_the_newest_and_reports_what_it_dropped(tmp_path):
    for g in range(10, 110, 10):
        p = pool.snapshot_path(str(tmp_path), g * 1000)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()
        os.utime(p, (g, g))                       # 让修改时间有先后
    dropped = pool.prune_snapshots(str(tmp_path), keep=3)
    left = {os.path.basename(p)
            for p in glob.glob(os.path.join(str(tmp_path), "pool", "snap_*.pt"))}
    # ⚠️ 别用 sorted(...)[-1] 判「留了最新的」—— basename 是**字典序**，
    # `snap_100000.pt` 排在 `snap_80000.pt` 前面。用集合比。
    assert left == {"snap_80000.pt", "snap_90000.pt", "snap_100000.pt"}
    assert len(dropped) == 7


def test_prune_is_a_noop_below_the_limit(tmp_path):
    p = pool.snapshot_path(str(tmp_path), 20000)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()
    assert pool.prune_snapshots(str(tmp_path), keep=20) == []
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool_snapshots.py -q
```
Expected: FAIL —— `pool.snapshot_path` 不存在。

- [ ] **Step 3: 实现**

`train/pool.py` 追加：

```python
SNAP_EVERY_GAMES = 20_000
POOL_SIZE = 20


def snapshot_path(out_dir: str, games: int) -> str:
    """快照路径：`<out_dir>/pool/snap_<games>.pt`。

    ⚠️ **故意嵌套一层、且不叫 `best.pt`** —— `net/advise.py::newest_weights()`
    glob 的是 `runs/rl/*/best.pt`，两层都命不中。命中就等于**静默换源**。
    """
    return os.path.join(out_dir, "pool", f"snap_{games}.pt")


def prune_snapshots(out_dir: str, keep: int = POOL_SIZE) -> list:
    """只留最新的 `keep` 个，返回被删掉的路径。"""
    found = sorted(glob.glob(os.path.join(out_dir, "pool", "snap_*.pt")),
                   key=os.path.getmtime)
    gone = found[:-keep] if keep > 0 else found
    for p in gone:
        os.remove(p)
    return gone


def load_members(paths) -> list:
    """磁盘上的快照 -> `PoolMember`（`mid` 按路径排序，稳定）。"""
    return [PoolMember(mid=i, path=p) for i, p in enumerate(sorted(paths))]
```

`train/selfplay.py`：`_maybe_eval` 里存（**与 `best.pt` 的判断无关**）：

```python
def _maybe_eval(net, games, curve, best_greedy, eval_games, eval_every,
                batch_games, out_dir, log, snap_every=pool.SNAP_EVERY_GAMES,
                pool_size=pool.POOL_SIZE):
    ...
    if snap_every and games and games % snap_every < batch_games:
        from train import pool as pool_mod
        p = pool_mod.snapshot_path(out_dir, games)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"net": net.state_dict(), "games": games}, p)
        gone = pool_mod.prune_snapshots(out_dir, pool_size)
        log(f"     >> 池子快照 {os.path.basename(p)}"
            + (f"（挤掉 {len(gone)} 个）" if gone else ""))
    ...
```

`train()` 与 `train_parallel()` 各加 `snap_every` / `pool_size` 形参并**透传给 `_maybe_eval`**
（默认值就是那两个常量，所以不传时行为不变）；`main()` 加：

```python
    if "--snap-every" in argv:
        kw["snap_every"] = int(argv[argv.index("--snap-every") + 1])
    if "--pool-size" in argv:
        kw["pool_size"] = int(argv[argv.index("--pool-size") + 1])
```

**为什么必须做成可调**：默认 2 万局一档，1 小时的 A/B（约 16 万局）只长出 8 个成员；
调成 `--snap-every 5000` 就有 30 多个可选，池子才谈得上「有强度谱」。

- [ ] **Step 4: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool_snapshots.py -q   # 4 passed
.venv/Scripts/python.exe -m pytest tests/ -q                          # 不许退步
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 60
```
Expected: 全绿；bench 跑完后 `runs/ab/…/pool/` 下出现 `snap_*.pt`，
且 `python -c "from net.advise import newest_weights; print(newest_weights())"`
仍然指向 `runs/rl/20260926-1407/best.pt`（**没被快照抢走**）。

- [ ] **Step 5: 提交**

```bash
git add train/pool.py train/selfplay.py tests/test_pool_snapshots.py
git commit -m "feat(pool): 定期权重快照（与 best.pt 解耦，嵌套目录免得面板误加载）"
```

---

### Task 5: worker 按权重分组做批量前向

**Files:**
- Modify: `train/selfplay.py`（`generate_batch` 的出手分组；`_fixed_pick` 保留）
- Test: `tests/test_grouped_forward.py`（新建）

**Interfaces:**
- Consumes: Task 3 的 `pick_opponent`
- Produces:
  - `selfplay.plan_step(learn_seats, turn, fixed) -> tuple`（纯函数，可单测）
  - `selfplay.generate_batch(..., members=None, pick_fixed=None)`

**这是本改动唯一的架构级改动**：现在二分（学习队问网络 / 固定对手走便宜的 Python），
要改成按**谁在出手**分组，每组各做一次批量前向。「当前权重」只是众多组里的一组。
**贪心/随机必须继续走便宜路径**（占混合局的 20%，白花前向会拖垮吞吐）。

- [ ] **Step 1: 写失败的测试**

```python
"""出手分组：谁出手 -> 用哪份权重。**纯函数单测 + 真跑一次**。"""
import random

import pytest
import torch

from train import selfplay
from train.net import QNet


def test_plan_step_dispatches_by_who_acts():
    """分组是纯函数，先把这张表钉死 —— 它错了会静默用错权重。"""
    learn = (0, 2)
    assert selfplay.plan_step(learn, 0, None) == ("learner", None)
    assert selfplay.plan_step(learn, 2, ("greedy", 1)) == ("learner", None)
    assert selfplay.plan_step(learn, 1, ("greedy", 1)) == ("fixed", "greedy")
    assert selfplay.plan_step(learn, 3, ("random", 1)) == ("fixed", "random")
    assert selfplay.plan_step(learn, 1, ("member", 7)) == ("member", 7)
    # 纯自对弈（四家都学）：谁都走 learner
    assert selfplay.plan_step((0, 1, 2, 3), 1, None) == ("learner", None)


def test_a_missing_member_raises_loudly():
    """抽到不在 `members` 里的成员**必须炸**。

    静默退回贪心会让池子悄悄少一个成员，而 A/B 的结果就没法解释
    —— 本仓库纪律：失败必须响。
    """
    torch.manual_seed(0)
    with pytest.raises(KeyError):
        selfplay.generate_batch(QNet().eval(), random.Random(0), eps=0.0, n_games=2,
                                capture=False, opp_mix=1.0, members={},
                                pick_fixed=lambda r: ("member", 7))


def test_members_actually_play_with_their_own_weights():
    """不同成员的出手必须**真的不同** —— 都一样说明权重没用对。

    做法：两个「极端」成员（一个只会出第一选、一个只会出最后一选），
    各打一批，比它们的着法分布。用真实的 `q_values` 太间接，这里用桩网络：
    `q_argmax_batch` 只要求 `net(state, action, hist)` 返回每行一个标量。
    """
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))   # ⚠️ 必须有：见下

    class First(_Stub):
        def forward(self, state, action, hist):
            return torch.zeros(state.shape[0])

    class Last(_Stub):
        def forward(self, state, action, hist):
            return torch.arange(state.shape[0], dtype=torch.float32)

    def run(member):
        torch.manual_seed(0)
        out = selfplay.generate_batch(QNet().eval(), random.Random(3), eps=0.0,
                                      n_games=4, capture=False, opp_mix=1.0,
                                      greedy_share=0.0, members={9: member},
                                      pick_fixed=lambda r: ("member", 9))
        return [rec.actions for rec in (o[0] for o in out)]

    assert run(First()) != run(Last()), "两个极端成员打出了一模一样的东西"
```

> ⚠️ **桩必须带一个哑参数**：`q_argmax_batch` 第一行就取
> `next(net.parameters()).device`，没有参数的 `nn.Module` 会抛 `StopIteration`。

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_grouped_forward.py -q
```
Expected: FAIL —— `selfplay.plan_step` 不存在；`generate_batch` 不接受 `members`。

- [ ] **Step 3: 实现**

`train/selfplay.py` 加纯函数 + 改内层循环：

```python
def plan_step(learn_seats, turn, fixed) -> tuple:
    """这一步由谁出手 -> `("learner", None)` / `("member", mid)` / `("fixed", kind)`。

    **纯函数**，故意抽出来 —— 分组错了会静默用错权重（池子里全变成一个模型），
    而那种错在日志上完全看不出来。
    """
    if turn in learn_seats:
        return ("learner", None)
    if fixed is None:
        # 「不属学习队、又没有固定对手」是不该出现的状态（learn 与 fixed 一起定的）
        # -> **炸掉**，别猜一个。猜的后果是静默退回贪心，日志上看不出来
        raise ValueError(
            f"座位{turn} 不属于学习队 {tuple(learn_seats)}、又没有固定对手 —— "
            f"分组的前提被破坏了")
    if fixed[0] == "member":
        return ("member", fixed[1])
    return ("fixed", fixed[0])
```

`generate_batch` 的内层循环（替换原来那段二分）：

```python
        # 按「这一步谁在出手」分组，**每组各做一次批量前向**
        # （spec §3.3）：「当前权重」只是众多组里的一组
        picks = [None] * len(pending)
        groups = {}
        for j, k in enumerate(alive):
            who = plan_step(learn[k], envs[k].hand.turn, fixed[k])
            groups.setdefault(who, []).append(j)
        for (who, mid), js in groups.items():
            if who == "fixed":
                for j in js:                 # 贪心/随机：便宜的 Python 路径，不占前向
                    picks[j] = _fixed_pick(fixed[alive[j]], pending[j], rng)
                continue
            if who == "member":
                if mid not in members:
                    raise KeyError(
                        f"对手池里没有成员 {mid}（有 {sorted(members)}）—— "
                        f"不许静默换个对手，那会让池子悄悄少一个成员")
            neti = net if who == "learner" else members[mid]
            if who == "learner" and eps >= 1.0:
                # 纯随机阶段不必前向（早期是 ε=1.0，省掉这一大截）
                for j in js:
                    picks[j] = rng.randrange(len(pending[j][1]))
                continue
            got = q_argmax_batch(neti, [pending[j] for j in js])
            for j, p in zip(js, got):
                # ⚠️ ε **只属于学习者**；对手一律 argmax（spec §4）
                picks[j] = (rng.randrange(len(pending[j][1]))
                            if who == "learner" and rng.random() < eps else p)
```

签名与对手来源：

```python
def generate_batch(net, rng, eps, n_games, capture=True, opp_mix=0.0,
                   greedy_share=0.8, learn_all_seats=False, members=None,
                   pick_fixed=None):
    """...
    `members`：`{mid: 网络}` —— 池子成员。`pick_fixed(rng) -> ("greedy",) |
               ("random",) | ("member", mid)` 决定这一局的固定对手是谁；
               不给就沿用老的 `greedy_share` 二分（便于与历史跑对照）。
    """
```

对手选择处：

```python
        if rng.random() < opp_mix:
            opp = rng.randrange(2)      # 哪一队当固定对手 —— **统一在这里抽**（池成员也一样）
            if pick_fixed is not None:
                kind = pick_fixed(rng)  # ("greedy",) | ("random",) | ("member", mid)
                # 形状统一成 (kind, x)：member 的 x 是**成员 id**，其余是**队号**
                fixed.append(("member", kind[1]) if kind[0] == "member"
                             else (kind[0], opp))
            else:
                kind = "greedy" if rng.random() < greedy_share else "random"
                fixed.append((kind, opp))
            learn.append(tuple(s for s in rules.SEATS if rules.TEAM[s] != opp))
```

> ⚠️ **队号必须对所有对手都抽**（包括池成员）。原计划里 member 那一路没抽队号，
> 于是 `learn` 会变成「四家都学」→ `plan_step` 永远返回 learner →
> `test_a_missing_member_raises_loudly` 根本不会触发（池子少一个成员会**静默**）。

> ⚠️ `fixed` 的形状要**统一**成 `(kind, …)`：`("greedy", opp)` / `("random", opp)` /
> `("member", mid)`。`plan_step` 只看 `fixed[0]`；`("member", mid)` 的第二个元素是
> **成员 id**，不是队号 —— 别混。

- [ ] **Step 4: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_grouped_forward.py -q   # 3 passed
.venv/Scripts/python.exe -m pytest tests/ -q                           # 不许退步
.venv/Scripts/python.exe -m train.selfplay 20 --workers 2              # 老路没坏
```
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add train/selfplay.py tests/test_grouped_forward.py
git commit -m "feat(train): 按「谁在出手」分组做批量前向（池成员用自己的权重）"
```

---

### Task 6: worker 持池子 + 增量广播 + 健康度

**Files:**
- Modify: `train/worker.py`（常驻池子、`_drain_ctrl` 收 `("member", …)`、按 PFSP 抽对手）
- Modify: `train/selfplay.py`（`train_parallel`：装种子池、增量广播、健康度上报）
- Test: `tests/test_pool_wiring.py`（新建）

**Interfaces:**
- Consumes: Task 3 的 `WinRates` / `pfsp_weights` / `pick_opponent`；Task 4 的 `load_members`；Task 5 的 `members=` / `pick_fixed=`
- Produces:
  - `worker.worker_cfg(..., init=None, members=None)`（`members` 是 `{mid: state_dict}` 的**初始**池）
  - `worker.run_worker` 内部维护 `members`；控制消息三类：
    `("member", (mid, state_dict))` / `("weights", (sd, eps, pfsp_weights))` / `("stop", None)`
  - `selfplay.pool_report(wr, weights, log)`（健康度表 + 塌陷报警）

**为什么必须增量发成员**：每次广播全池 = 20 × 7.8MB = 156MB，
按 1000 局（约 30 秒）一次算是 **5MB/s × worker 数**，直接把 learner 拖死。

- [ ] **Step 1: 写失败的测试**

```python
"""池子接进多进程：真起进程，看记录里的 `opp` 是不是跟着控制消息变。"""
import multiprocessing as mp
import random

import torch

from train import pool, worker
from train.net import QNet


def _cfg(**kw):
    d = dict(seed=3, eps=0.0, opp_mix=1.0, greedy_share=0.0, batch_games=2,
             init=None, members=None)
    d.update(kw)
    return d


def test_worker_takes_a_new_member_from_the_control_queue():
    """喂一个成员进去，worker 下一批就用它当对手 —— 证据是记录里的 `opp`。

    这条同时证明两件事：控制消息被接住了，以及**记录里能读出用了谁**
    （PFSP 的归因就靠它）。
    """
    ctx = mp.get_context("spawn")
    send_q, ctrl_q = ctx.Queue(), ctx.Queue()
    torch.manual_seed(0)
    sd = QNet().state_dict()
    ctrl_q.put(("member", (7, sd)))
    cfg = worker.worker_cfg(seed=3, eps=0.0, opp_mix=1.0, greedy_share=0.0,
                            batch_games=2, init=None,
                            members={},
                            pick_all="member", member_id=7)
    p = ctx.Process(target=worker.run_worker, args=(send_q, ctrl_q, cfg), daemon=True)
    p.start()
    try:
        recs = send_q.get(timeout=120)
    finally:
        ctrl_q.put(("stop", None))
        p.join(timeout=30)
    assert all(r.opp == ("member", 7) for r in recs), \
        f"记录里的对手不对：{[r.opp for r in recs]}"


def test_pool_report_warns_when_the_pool_collapses(caplog):
    """池子塌成一个成员必须**响**（spec §1.4）—— 不是只写一行日志。"""
    from train import selfplay
    wr = pool.WinRates()
    for _ in range(200):
        wr.record(0, True)
    lines = []
    collapsed = selfplay.pool_report(wr, {0: 1.0}, lines.append)
    assert collapsed is True
    assert any("塌" in s for s in lines)


def test_pool_report_is_quiet_for_a_healthy_pool():
    from train import selfplay
    wr = pool.WinRates()
    for i in range(3):
        for _ in range(50):
            wr.record(i, True)
    lines = []
    assert selfplay.pool_report(wr, {0: 0.34, 1: 0.33, 2: 0.33}, lines.append) is False
```

- [ ] **Step 2: 跑测试，确认失败**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool_wiring.py -q
```
Expected: FAIL —— `worker_cfg` 不认识 `members`；`selfplay.pool_report` 不存在。

- [ ] **Step 3: 实现 `train/worker.py`**

```python
def worker_cfg(seed, eps, opp_mix, greedy_share, batch_games, init=None,
               members=None, pick_all=None, member_id=None) -> dict:
    """`members`：`{mid: state_dict}` 的**初始**池。
    `pick_all` **只给测试用**：强制所有混合局都用 `member_id` 当对手
    （生产上对手由 learner 侧的 PFSP 权重决定，worker 自己按 `pfsp` 抽）。
    """
    return {"seed": seed, "eps": eps, "opp_mix": opp_mix,
            "greedy_share": greedy_share, "batch_games": batch_games,
            "init": init, "members": dict(members or {}),
            "pick_all": pick_all, "member_id": member_id}


def _drain_ctrl(ctrl_q, net, members, state):
    """非阻塞收控制消息。`state` 是可变字典（存最新的 ε 与 PFSP 权重）。"""
    while True:
        try:
            kind, payload = ctrl_q.get_nowait()
        except Exception:                       # noqa: BLE001 - 队列空就是没消息
            return
        if kind == "stop":
            raise SystemExit(0)
        if kind == "member":
            mid, sd = payload
            m = QNet().to(worker_device()).eval()
            m.load_state_dict(sd)               # 装不上会抛 —— 不许静默少一个成员
            members[mid] = m
        if kind == "weights":
            sd, eps, w = payload
            net.load_state_dict(sd)
            net.eval()
            state["eps"] = eps
            state["pfsp"] = w
```

`run_worker`：

```python
def run_worker(send_q, ctrl_q, cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    net = QNet().to(worker_device()).eval()
    selfplay.load_init(net, cfg.get("init"))
    members = {}
    for mid, sd in cfg["members"].items():
        m = QNet().to(worker_device()).eval()
        m.load_state_dict(sd)
        members[mid] = m
    rng = random.Random(cfg["seed"])
    state = {"eps": cfg["eps"], "pfsp": {}}

    def pick_fixed(r):
        if cfg["pick_all"] == "member":         # 只给测试用
            return ("member", cfg["member_id"])
        return pool.pick_opponent(r, list(members), state["pfsp"],
                                  cfg["greedy_share"])

    while True:
        # ⚠️ **先收控制消息，再打。** 反过来的话新装的成员要等下一批才生效，
        # 而「成员还没到就先抽到它」会直接 KeyError 把 worker 打死
        # （Task 6 那条测试就是这么喂成员的）。
        # 顺带把「广播到达之前第一批用的是开局旧权重」这个老窗口一起关掉。
        _drain_ctrl(ctrl_q, net, members, state)
        recs = worker_batch(net, rng, state["eps"], cfg["batch_games"],
                            opp_mix=cfg["opp_mix"], greedy_share=cfg["greedy_share"],
                            members=members, pick_fixed=pick_fixed)
        send_q.put(recs)                        # 队满则阻塞 = 天然背压
```

`worker_batch` 跟着扩两个参数（**必须带默认值** —— `tests/test_worker.py` 直接调它，
它是「worker 的打法与进程内逐局一致」那条硬保证的载体，别删掉另写一份）：

```python
def worker_batch(net, rng, eps, n_games, opp_mix=0.5, greedy_share=0.8,
                 members=None, pick_fixed=None):
    return [rec for rec, _pts, _y in selfplay.generate_batch(
        net, rng, eps, n_games, capture=False, opp_mix=opp_mix,
        greedy_share=greedy_share, members=members, pick_fixed=pick_fixed)]
```

`pick_fixed` 直接用 `train/pool.py` 的（**不再包一层**，池子为空/权重未到时它自己兜得住）：

其中 `pool_opponent` 是 `train/pool.py::pick_opponent` 的薄包装（`members` 可能为空
⇒ 退回贪心，池子还没长出成员时不能崩）。

- [ ] **Step 4: 实现 `train/selfplay.py` 的 learner 侧**

```python
def pool_report(wr, weights, log) -> bool:
    """打印池子健康度表，返回**是否塌陷**（spec §1.4）。

    塌了是「失败」，不是一行日志 —— 调用方据此报告（本仓库纪律：失败必须响）。
    """
    eff = pool.effective_members(weights)
    log(f"  池子（有效成员数 {eff:.2f}）:")
    for mid in sorted(weights):
        log(f"    #{mid:<3d} 打了 {wr.games(mid):5d} 局  学习者胜率 {wr.rate(mid):5.1%}  "
            f"采样权重 {weights[mid]:5.1%}")
    if eff < pool.COLLAPSE_BELOW:
        log(f"  ⚠️ **池子塌了**（有效成员数 {eff:.2f} < {pool.COLLAPSE_BELOW}）"
            f"—— 对手退化成同一个模型，训练会绕圈")
        return True
    return False
```

`train_parallel` 里新增：

```python
    # 种子池：把现有快照装进内存，开局广播一次（39MB，只发一次）
    seed_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*", "pool", "snap_*.pt")))
    seed_paths += sorted(glob.glob(os.path.join(RUNS_DIR, "*", "best.pt")))[:5]
    seed_sds = {}
    for i, p in enumerate(pool.load_members(seed_paths)):
        try:
            seed_sds[i] = torch.load(p.path, map_location="cpu")["net"]
        except Exception as exc:                # noqa: BLE001
            raise RuntimeError(f"种子池成员读不出来：{p.path}（{exc}）") from exc
    ...
    for q in ctrls:
        for mid, sd in seed_sds.items():
            q.put(("member", (mid, sd)))
    wr = pool.WinRates()
    ...
        # 每批记录：按 opp 归因胜负（y 的符号就是哪队赢）
        for rec in recs:
            buf.add(rec); games += 1
            if rec.opp and rec.opp[0] == "member":
                pts, y = replay.expand(rec)
                if pts:
                    wr.record(rec.opp[1], y[0] > 0)      # 学习者的第一个点
    ...
        if games - last_sync >= WEIGHT_SYNC_GAMES:
            w = pool.pfsp_weights({i: wr.rate(i) for i in seed_sds},
                                  {i: wr.games(i) for i in seed_sds})
            sd = {k: v.cpu() for k, v in net.state_dict().items()}
            for q in ctrls:
                q.put(("weights", (sd, eps_for(games, eps_games), w)))
            last_sync = games
    ...
            # 新快照也要进池（与 Task 4 的存盘同拍）
            ...
    # 收尾
    pool_report(wr, w, log)
```

> ⚠️ `y[0] > 0` 取的是**学习者的第一个决策点**的符号 ——
> `expand` 只产出学习那一队（Task 1），所以 `y` 里每个元素的符号都一致，
> 取哪个都行。这里取第一个只是最简单。

- [ ] **Step 5: 跑测试 + 回归**

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pool_wiring.py -q    # 3 passed
.venv/Scripts/python.exe -m pytest tests/ -q                        # 不许退步
.venv/Scripts/python.exe -m train.selfplay 60 --workers 2 --opp-mix 0.5
```
Expected: 全绿；60 秒短跑里能看到「池子（有效成员数 …）」那张表。

- [ ] **Step 6: 提交**

```bash
git add train/worker.py train/selfplay.py tests/test_pool_wiring.py
git commit -m "feat(pool): worker 常驻池子 + 增量广播 + 健康度上报"
```

---

### Task 7: A/B 短跑 + 台账 + 文档

**Files:**
- Create: `docs/superpowers/plans/2026-09-27-opponent-pool-delivery.md`（台账）
- Modify: `docs/superpowers/specs/2026-09-27-current-scheme.md`（§四.7 的对手配置、§十一 的下一步）
- Modify: `HANDOFF.md`（训练命令加新开关）

- [ ] **Step 1: 量基线（对照臂 = 老行为）**

```powershell
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -m train.selfplay 3600 --workers 2 --init runs/rl/20260926-1407/best.pt --learn-all-seats --eps-games 60000 --out-dir runs/ab/A_old
```
- [ ] **Step 2: 量处理臂（池子）**

```powershell
.venv/Scripts/python.exe -m train.selfplay 3600 --workers 2 --init runs/rl/20260926-1407/best.pt --eps-games 60000 --out-dir runs/ab/B_pool
```
⚠️ **两臂的 `--eps-games` 必须相同且调小**，否则都在近乎随机地探索，测不出差别
（这个坑本仓库踩过：默认 25 万局，跑 1 小时 ε 还有 0.86）。

- [ ] **Step 3: 同一把尺子上比**

```powershell
.venv/Scripts/python.exe -m tools.bench_train --workers 2 --seconds 180   # 顺带重测吞吐
```
然后对两臂的 `last.pt` 在**同一批牌**上量：

```python
# 对每个臂分别：
#   match(pol, greedy_policy, games=400, seed=1002)     —— vs 贪心（保可比）
#   bomb_waste(pol, games=60, seed=3001, opponent=<池成员>)  —— 真正的判据
```
**每个 `bomb_waste` 都对若干个池成员分别量**（它一次只吃一个对手），取中位。

- [ ] **Step 4: 写台账**（照 Plan 1/2/3 的体例）

一句话结论 · 怎么跑 · 两臂的数字（`vs 贪心` + **对池成员的 `bomb_waste`**）·
吞吐 · **池子健康度曲线**（有效成员数、各成员胜率）· 与杀停条件对照 ·
已知风险（不可复现的定义、队列、显存）· 没做完的。

- [ ] **Step 5: 更新文档**

- `current-scheme.md` §四.7 的对手配置改成实际值（`GREEDY_SHARE` 从 0.8 降到 0.2）；
  §四.9 的「三个薄弱点」按实测结论更新（哪条被证实、哪条没有）；§十一 的下一步重排
- `HANDOFF.md`：训练命令加 `--init` / `--learn-all-seats`；写明池子的存在与池目录位置
- ⚠️ 若杀停条件触发（`bomb_waste` 没降 1 个标准差），**照实写进台账**，
  并把 `current-scheme.md` §四.9 第 1 条（标签信噪比）标成「下一步该动这里」

- [ ] **Step 6: 提交**

```bash
git add docs/ HANDOFF.md
git commit -m "docs: 对手池交付台账（A/B 数字 + 池子健康度 + 杀停结论）"
```

---

## 收尾自检

**规格覆盖**：§1 判据 → 任务 3/6/7；§2 事实 → 各任务的测试注释；§3.0（0 期）→ 任务 1；
§3.1（池子）→ 任务 4；§3.2（PFSP）→ 任务 3；§3.3（分组前向）→ 任务 5；
§3.4（增量广播）→ 任务 6；§3.5（`--init`）→ 任务 2；§5 验收 → 各任务 Step 5 + 任务 7；
§6 风险 → Review Focus 六条逐条落到测试；§7 不在范围内 → 不建任务。

**没做的（明确在计划外）**：MCTS/rollout、reward shaping、改判据、
replay buffer 换共享内存、多机分布式、自适应 `opp_mix`。
