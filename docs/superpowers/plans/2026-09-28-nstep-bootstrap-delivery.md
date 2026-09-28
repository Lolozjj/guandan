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
  提交 `b68ac7d`（待提交）。
