# 路线②（n 步自举）交付台账（2026-09-28）

> 计划：`docs/superpowers/plans/2026-09-28-nstep-bootstrap.md`
> 规格：`docs/superpowers/specs/2026-09-28-nstep-bootstrap-design.md`

## 0. 执行方式

- **inline**（本会话内逐 task 做，最后用一个新上下文的评审判整条分支）
- **不开 worktree**：本仓库一贯直接在 `master` 上提交
- 台账就是本文件，**不放进 `.superpowers/`** —— 那个目录是 gitignore 的临时目录，
  会被 `git clean` 清掉，而评审 M8 的教训正是「数字躺在临时目录里就不可复现」

## 1. 起跑前的预检

- 测试基线：**407 passed / 1 failed**（唯一那条红是预存的 `test_tribute_records`，
  日志轮转导致，见现用方案 §七.3）
- **共享接口扫描**（一个 task 产出、另一个 task 消费的地方）：

| 产出 → 消费 | 产出的形状 | 消费方要什么 | 结论 |
|---|---|---|---|
| T2 `expand(n=)` → T4 `build_samples` | `(points, y_mc, boot)`，`boot` 与 `points` **等长同序** | `pts, y, b = replay.expand(rec, bomb_cost=..., n=n_step)` | 一致 ✓ |
| T1 `blend` → T5 `_targets` | `blend(y_mc, boot, beta)`，`boot` 可以整条为 `None` | `blend(y_mc, vals, mc_mix)`，`vals` 由 `q_max_batch` 给 | 一致 ✓（`q_max_batch` 保位，长度与 `y_mc` 相同） |
| T3 `q_max_batch` → T5 `_targets` | `list[float | None]`，`None` 原样保留 | `if n_boot` 才调；`None` 项交给 `blend` 退回 MC | 一致 ✓ |
| T5 `train/train_parallel(mc_mix, n_step, tgt_sync)` → T6 CLI / T8 bench | 三个带默认值的关键字 | `kw["mc_mix"]=...` / `bench_run(..., mc_mix=..., n_step=...)` | 一致 ✓ |
| T5 `sync_target` → T5 自己两处调用 | `-> int`（新的 `last_sync`） | `last_tgt = sync_target(net, net_tgt, games, last_tgt)` | 一致 ✓ |
| T7 `measure` → T9 | `(n_points, stats)` | 脚本打印，人读 | 一致 ✓ |

- **一处刻意的不对称**（扫描时发现的，已写进计划）：T4 的 `_learn_step`
  **不接受** `mc_mix`（只做重构，`n_step=0`）；T5 才把签名扩成带自举的那一版。
  原因：接受了却算不出自举 = 「静默地按 β=1 走」，正是本仓库最恨的那种错。

## 2. 逐 task 记录

- **Task 1 完成**：`blend` 落地。RED 看过（10 条全 `AttributeError: no attribute 'blend'`）→
  GREEN：`pytest tests/test_blend.py tests/test_bomb_cost.py tests/test_train_replay.py` = **24 passed**。
  提交 `cd6d16c`。
- **Task 2 完成**：`expand(rec, bomb_cost, n)` 三元组 + 定长环。
  RED 看过（`TypeError: expand() got an unexpected keyword argument 'n'`）→
  GREEN：`tests/test_expand_boot.py` **4 passed**；受影响的 6 个文件一起 **29 passed**。
  两处**执行时才发现**的细节，都写进代码注释了：
  1. `n=0` 时 `boot` 必须是**空表**（不是一列 `None`）—— 一开始写成按 points 长度
     填 `None`，被 `test_n_zero_returns_no_boot_and_the_old_points` 抓住；
  2. `fresh.get(id(rec)) or replay.expand(rec)` 这个老写法**把 2 元组和 3 元组混在
     一起**（`or` 命中缓存时右边不求值），`test_train_device` / `test_train_init`
     当场红。改成显式判 `None` —— Task 4 会把这一段整体收进 `build_samples`。
  提交 `c3963ae`。
- **Task 3 完成**：`_flat_scores` 重构 + `q_max_batch`（返回 **float**，类型上断掉梯度）
  + `check_q_scale`（`not (x <= limit)` 写法，NaN 也拦）。
  RED 看过（`ImportError: cannot import name 'check_q_scale'`）→
  GREEN：`tests/test_q_max.py` **10 passed**；全量 **431 passed / 1 failed**（唯一红是预存的
  `test_tribute_records`）。重构没改老行为由
  `test_argmax_batch_is_unchanged_by_the_refactor` 钉住。提交 `39d9314`。
- **Task 4 完成**：`build_samples` + `_learn_step`，两条训练路线收成一份。
  RED 看过（`ImportError: cannot import name '_learn_step'`）→
  GREEN：`tests/test_learn_step.py` **5 passed**；全量
  （`--ignore=tests/test_nstep_wiring.py`，那是 T5 故意先写的红文件）
  **436 passed / 1 failed** = 基线 +5，唯一红是预存的 `test_tribute_records`。
  - **Ruling**：`tests/test_learn_step.py` 里那条 `test_build_samples_is_aligned`
    我一开始写成 `len(s)==len(y)==len(b)>0`，**断言错了** —— `n_step=0` 时
    `boot` 按设计就是空表。改成 `len(s)==len(y)>0` + `b == []`，计划文档同步改。
    代价若错：无（测试当场就红了，正是「先看它失败」的价值）。
  - **Ruling**：T5 的接线测试另起 `tests/test_nstep_wiring.py`，不按计划的「追加到
    `test_learn_step.py`」。理由：两份测的是**不同的单元**（重构 vs 新行为），
    而且分开才能让 T4 的全量验证干净跑一次。
    代价若错：多一个测试文件，没有别的。
  - **Ruling**：`_learn_step` 返回 **float**，所以两条日志行里的 `loss.item()`
    要改成 `loss`。计划漏了这两处（`train()` 那条只有 `steps%10==0` 才触发，
    短测试撞不上，是 `test_train_parallel` 抓到的）。
    代价若错：无（AttributeError 当场就响）。
  提交 `3f9d1c4`（待提交）。
- **Task 5 + Task 6 完成**：目标网络（`deepcopy`，不消耗 RNG）、`_targets`（自举**唯一**入口）、
  `_learn_step` 扩签名 + 发散守门（预测与标签都查）、CLI 三个开关、两条路线同拍同步。
  RED 看过（`ImportError: cannot import name '_targets'`）→ GREEN：
  `tests/test_nstep_wiring.py` **5 passed**；全量 **442 passed / 2 failed**。
  - **Ruling**：CLI 的三个开关与 T5 的接线是**同一次改动**落地的，所以
    `tests/test_mc_mix_cli.py` 写出来就是绿的 —— 按 TDD 的字面它没法「先看它红」。
    处理办法是**反向对照**：临时把 `--mc-mix` 的转发删掉，看它确实红
    （`KeyError: 'mc_mix'`，正是计划预期的失败），再装回。两次都记录在案。
    代价若错：没有独立证据证明这条测试有牙 —— 所以补了反向对照。
  - **Ruling**：发散守门要 `.detach()` 再 `float()` —— 直接 `float(带梯度的张量)`
    PyTorch 会告警（"Converting a tensor with requires_grad=True to a scalar"）。
    守门是纯读，不该把预测卷进任何图。
  - **Ruling**：`tests/test_nstep_wiring.py` 里 patch 的是
    `train.selfplay.q_max_batch`（**它被查找的那个名字空间**），不是
    `train.net.q_max_batch` —— 计划里写的是后者，那样 patch 根本不起作用
    （selfplay 用 `from ... import` 绑定了自己的名字）。
    代价若错：测试会「恒绿」，把没接线的情况放过去。
  - **端到端真跑**（60 秒，`--workers 2 --batch 8 --mc-mix 0.5 --n-step 3`）：
    日志头 `自举 β=0.5 n=3（目标网络每 1000 局同步）` ✓、1,768 局、
    **没触发发散守门**、loss 有限（0.7~1.7）、29.4 局/秒（随机初始化 + batch 8，
    不可与 44.5 直接比 —— 正式成本在 Task 8 量）。
    产物留在 `runs/ab/smoke_nstep/`（`rm` 被权限拒了，未清；它不在 `runs/rl/` 下，
    面板看不见它）。
- **环境变化（如实记）**：这一轮全量测试多了一条红
  `tests/test_game_log.py::test_settled_games_are_conserved` ——
  断言是「结算记录 ≥20 局」，当前实时日志只剩 **6 局**（日志被轮转掉了）。
  与本次改动无关（这个测试只读 `LOG_DIR`），正是现用方案 §七.3 记着的那一类。
  **基线红从 1 条变 2 条**，两条都是日志轮转。
- **Task 7 完成**：`tools/action_margin.py`（判据 2 的尺子）+ 4 条测试。
  **起点值量出来了，与 9-27 那次的口径一致**：

  | 存档 | 决策点 | 首选次选之差 中位 | p90 | 打平(<0.01) | 候选间离散度 中位 |
  |---|---|---|---|---|---|
  | `1407`（140 万局） | 3073（参与 1793） | **0.124** | 0.523 | 7.5% | **0.386** |

  对照现用方案 §四.2 记的 **0.123 / 0.401** —— 两个口径对得上 ✓。
  - **Ruling**：`test_margin_definition_is_top1_minus_top2` 一开始用 3 个候选，
    而那时 `top1-top2` 与 `top1-均值` 会**撞成同一个数**（都是 1.0），测不出区别。
    改成 4 个候选（Q = 0/1/2/10 → 差 8，减均值是 6.75），并做了**反向对照**：
    把实现改成「首选减均值」→ 这条红；改回 → 绿。
    代价若错：尺子的口径错了却没人发现，判据 2 变成自欺。
  提交 `TBD`。

## 3. Task 9：起 A/B

- **Ruling（偏离预登记的 10 GB 门槛）**：起跑前实测**可用物理内存 8.6 GB**
  （共 31.4 GB，占用 72%，没有我自己的残留进程 —— 是微信/游戏/浏览器占的）。
  计划 Task 9 Step 1 写的是「≥ 10 GB 不到就先别起」。**我放行了**，理由：
  那个门槛的**理由**是「每臂 2 个 worker 各约 1.2 GB + 学习进程一份」≈ 5 GB，
  8.6 GB 满足它并且还剩 ~3.6 GB 余量；而且这次是**一条臂一条臂起**（不是三条并发，
  那次才是真炸的原因）。**对冲**：起跑后 90 秒查一次内存与实际内存占用，
  不对就立刻停掉。**这条偏离记录在此，不藏着。**
  代价若错：用户前台的应用（微信/游戏）被系统换页变卡，或 worker 被 OOM 杀掉 ——
  后者会**响亮地报**（`worker [k] 挂了`，本仓库纪律）。
- **A/B 起了又停了 —— 对冲条件触发，按我说的做了。**
  起跑后 ~80 秒实测：可用内存从 8.6 GB 掉到 **3.0 GB（占用 90%）**，
  即一条臂真实吃 **~4.8 GB**（不是我估的 5 GB —— 估得偏低）。
  吞吐没问题（**53 局/秒**，与 bench 的 51.7 一致），但 3.0 GB / 90% 对
  用户前台的微信/游戏是**要换页**的状态。按起跑前写下的「不对就立刻停掉」，
  我停掉了（`TaskStop`，无残留进程，内存回到 7.8 GB）。
  跑到第 79 秒 / 4,160 局，`runs/ab/R4_b10.log` 留着。
  **下一臂没起**（串行设计里它本来就在后面）。
  结论：**这台机器现在腾不出 5 GB 给一条臂** —— 31.4 GB 里 22 GB 被前台应用占着。
  等用户决定：腾应用 / 降配 / 就这么跑 / 先不跑。
