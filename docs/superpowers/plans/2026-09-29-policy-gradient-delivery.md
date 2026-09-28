# 策略式（REINFORCE）交付台账（2026-09-29）

> 计划：`docs/superpowers/plans/2026-09-29-policy-gradient.md`
> 规格：`docs/superpowers/specs/2026-09-29-policy-gradient-design.md`

## 0. 执行方式

- **inline**（本会话内逐个 task；最后一个新上下文的评审判整条分支）
- **不开 worktree**：本仓库一贯直接在 `master` 提交
- 台账就是本文件（不放 `.superpowers/` —— 那是 gitignore 的临时目录，会被 `git clean` 清掉）

## 1. 起跑前的预检

- 测试基线：**484 passed / 2 failed**（两条都是**日志轮转**的环境红：
  `test_tribute_records` / `test_game_log::test_settled_games_are_conserved`）
- **共享接口扫描**（一个 task 产出、另一个消费的地方）：

| 产出 → 消费 | 形状 | 结论 |
|---|---|---|
| T1 `log_prob_and_entropy` → T3 `_pg_step` | `(lp, ent, zmax)`，lp/ent 是 1-D tensor | 一致 ✓（**自检时把它从二元组改成三元组的**，理由见计划的自检记录） |
| T1 `check_entropy(h, log_k, frac)` → T3 | 纯函数，不达标 raise | 一致 ✓ |
| T1 `_flat_scores(grad=)` → T1/T2 | `grad=False` 默认不变 | 一致 ✓（**关键**：老调用方 `q_max_batch`/`q_argmax_batch` 必须不受影响） |
| T2 `policy_sample_batch(net, pending, rng)` → T4 `generate_batch(sample=True)` | `pending=[(obs,acts,hist)]`，返回下标 | 一致 ✓ |
| T3 `_pg_step(net, opt, samples, rewards, base, games, beta_ent)` → T4 两条训练路 | `samples` 是 `replay.expand` 的产出形状 | 一致 ✓（T4 里单进程用 `caps`、多进程用 `expand`，**形状相同**） |
| T4 `generate_batch(sample=)` → T4 worker 侧 | 布尔 | 一致 ✓ |
| T5 bench → T6 预算 | `--algo/--beta-ent/--weight-sync-games` 透传 | 一致 ✓ |

- **一处与计划的偏差（预先声明）**：计划 Task 1 的
  `test_one_optimizer_step_actually_changes_the_parameters` 里写了
  `torch.tensor([2.0, 1.0, -1.0, 3.0])` 当 reward，但 `_samples(3)` 返回的是
  **每局 4 个点 × 3 局 = 12 个** ⇒ 长度对不上、会 `TypeError`。
  **Ruling**：测试里改成 `[2.0] * len(samples)`（语义不变：只要 reward 非零、梯度就能动）。
  代价若错：无（测试当场会红）。
- **Task 1 完成**：`log_prob_and_entropy`（三元组 `(lp, ent, zmax)`，带梯度）+ `check_entropy` + `_flat_scores(grad=)`。
-  RED 看过（collection error：`log_prob_and_entropy` 不存在）→ GREEN：`tests/test_pg_logprob.py` **11 条**；连同 `test_q_max.py` 共 **20 passed**，输出干净。
-  **我自己写错的四处**（都当场被测试抓住，记下来）：
   1. 机制测试从输出层偏置取梯度 —— **错的**：输出层是 `Linear(d,1)`，偏置只有 1 维，根本没有"每候选一个梯度"。改成直接盯 logits（monkeypatch `_flat_scores`）。
   2. 符号断反了 —— `loss=−logπ(a)·adv` ⇒ **梯度**对选中项为**负**、其余为正；更新走 `−梯度` 才把选中项抬上去。顺便把"更新方向"也钉了一层。
   3. 计划里那条测试的 reward 列表长度写死 4，而样本是 12 个 ⇒ 会 TypeError（已在台账 §1 预先声明为 Ruling）。
   4. `float(带梯度的张量)` 告警两处 —— 与 `selfplay.py` 里同一个坑，加 `detach()`。
-  提交 `_TBD_`。
- **Task 1 全量验证**：`494 passed / 2 failed`（= 基线 484 + 新增 10；两条红仍是日志轮转的环境红）✓
- **Task 2 完成**：`policy_sample_batch`（从 π 采样，走调用方的 `rng` 逆累积分布）。
  RED 看过（collection error）→ GREEN：`tests/test_pg_sample.py` **4 条**，包含「60 次采样不能只出一个结果」
  （防它其实在 argmax）。
