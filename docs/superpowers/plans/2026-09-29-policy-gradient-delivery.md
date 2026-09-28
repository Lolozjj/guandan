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
