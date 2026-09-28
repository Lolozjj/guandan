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

## 4. 收口

**测试**：改动前基线 `407 passed / 1 failed` → 最终 **`446 passed / 2 failed`**
（净增 39 条）。两条红**都是日志轮转**（`test_tribute_records` 是预存的；
`test_game_log::test_settled_games_are_conserved` 是本轮中途日志转到只剩 6 局才出现的），
**都与本次改动无关**（它们只读日志目录 / 语料快照）。

**提交**：`bfb81d4`（计划）→ `cd6d16c` → `c3963ae` → `39d9314` → `8781663` →
`237e0e1` → `45dc509` → `5768b73` → `a4d02a8` → `e961249`。工作树干净。

**代码交付物**

| 文件 | 内容 |
|---|---|
| `train/replay.py` | `blend`（自举进入标签的唯一一处）；`expand(rec, bomb_cost, n)` 三元组 + 定长环产出 `s_{t+n}` |
| `train/net.py` | `_flat_scores`（批量前向收成一份）；`q_max_batch`（返回 **float**）；`check_q_scale`（NaN 也拦） |
| `train/selfplay.py` | `build_samples`/`_targets`/`_learn_step`（两条路线共用一份）；`sync_target`；目标网络；发散守门；`--mc-mix/--n-step/--tgt-sync` |
| `tools/action_margin.py` | 判据 2 的尺子（落进 `tools/`，不再躺临时目录） |
| `tools/bench_train.py` | `--mc-mix/--n-step/--init` 透传 |
| `tests/` 新增 7 个文件 | 40 条 |

**没做、明确记着的**（免得被当成遗漏）：
- **A/B 没跑**（机器内存，用户决定改天）。恢复命令见 `HANDOFF.md` 顶栏。
- 不做重要性采样（spec §4 的取舍）；γ 固定 1.0 没做成参数。
- 「只对一部分决策点自举」这个省钱法**没有实现** —— 它只在 Task 8 的第三档里作为
  「停下来讨论」的选项存在（YAGNI）。实测落在第二档，用不上。
- **Minor（deferred）**：单进程 `train()` 在 `β<1` 时仍会 `generate_batch(capture=True)`
  抓一遍决策点然后丢掉（因为 `build_samples` 里 `fresh` 让路）—— 每轮多分配约 38 MB，
  不是正确性问题；真跑用的是 `--workers 2`（worker 那边 `capture=False`），不受影响。

---

## 5. 整条分支评审（2026-09-28，新上下文，最贵的模型）

范围 `d1adcd1..HEAD`。**它挑出一条 Critical、两条 Important、七条 Minor。**
五条 Review Focus 里第 2/3/4/5 条它逐条查过**没找到问题**（并各自说明怎么查的，
包括手推 `n=0/1/3` 的对齐、另写一份针对混合局的重放核对、实测 `data_ptr`
确认目标网络是真复制不是别名），第 1 条判定为流程问题、代码层面无从判。

### Critical（已修，带回归测试）

**C1 自举项的「视角」接反了 —— 默认 n=3 时约 89% 的自举值符号是错的。**

`Q(s,a)` 学的是「**出手人那一队**」的收益（`encode_state` 一切以出手人为原点、
标签是 `rules.reward(ranks, seat)`），而 `V(s_{t+n}) = max_a Q(s_{t+n}, a)`
取的是 **`s_{t+n}` 处出手人**的视角。出手顺序是 `0→3→2→1`（`rules.NEXT`），
**奇数 n 的那个位置在对家**。后果是 `(1−β)V + βR` 两项互相抵消、标签被往 0 拉 ——
正是 spec §3.1 专门要避免的「塌成常数」，而且**没有任何测试、断言或 loss 异常能看见**
（loss 从 ~3.0 掉到 ~0.9，看着像学好了）。

**我自己独立复现了一遍**（12 局，`greedy_policy`，1407）：n=1/3 时
`corr(V, 标签) = −0.353 / −0.145`，n=2/4 是 `+0.271 / +0.273`；
自举源与标签同队的比例 5.4% / 16.0% vs 89.2% / 88.3%。**方向与量级都对得上评审。**

修法（`train/replay.py::_boot_source_ok`）：自举源必须**与标签同队**、
且**属于学习座位**（后者是老纪律：固定对手的状态不是我们的值），
否则该点**整项退回 MC**。**不取负** —— 取负等于把「对家按最大打」的价值当成我们的，
而行为策略并不是最大，那是另一种偏差，本轮不引入。
`N_STEP` 默认 **3 → 2**（必须偶数；奇数等于没自举）。
日志头对奇数 n 会打一个 ⚠️。

**时序是万幸**：A/B 还没跑，所以这个缺陷**没有代价** —— 否则处理臂测的是坏机制，
然后按预登记的「杀停」把路线②判死，而这条路从没被真正测过。

### Important（已修）

- **I1 `--tgt-sync` 在多进程那条路上被静默夹到 1000。** 同步写在
  `if games - last_sync >= WEIGHT_SYNC_GAMES`（=1000）**里面**，而日志头照写用户给的值
  —— 传 500 实际是 1000，**参数说谎**（违「换源必须可见」）。修法：把同步移出那个 `if`。
  另外 `every <= 0` 原来是静默空操作（`--tgt-sync 0` = **永不同步**，靶子不动），改成 raise。
- **I2 没有任何测试钉「自举用的是目标网络」。** 把 `q_max_batch(net_tgt, ...)` 写成
  `q_max_batch(net, ...)`（追自己的尾巴）**全部测试照样绿**。修法：把 `_targets` 那个
  没用到的 `net` 参数**删掉** —— 让这件事由**签名**排除，不靠注释（本仓库对「明牌泄漏」
  用的是同一招）。外加一条会红的测试：扰动目标网络 → 目标值必须变。

### Minor

**已修**：M1（`test_no_grad_leaks_into_the_network` 是**假牙** —— 只前向不 backward，
`grad` 本来就是 `None`；改成 spy `QNet.forward` 看 `torch.is_grad_enabled()`，
并**做了反向对照**：删掉 `no_grad` → 红）、M2（没用的参数，见 I2）、
M3（`β<1` 配 `n_step<=0` 是**自相矛盾**的配置，会「日志说自举、实际没自举」→ 现在 raise）、
M4（台账里两个提交号是死的）、M6（发散守门的报错文案写死「自举发散」，而 β=1 时它也在跑）、
M7（我留的临时文件）。
**deferred**：M5（`tools/action_margin.py` 每个决策点算两遍 `q_values` —— 白花一倍时间，
不是正确性问题）。

### 评审的独立复核（它自己跑过的，与台账对得上）

`−33%` 吞吐、动作边际 `0.124`、`446 passed / 2 failed`（两条都是日志轮转）——
它**独立复现**过，一致。它还核实了 `runs/ab/R4_b10/` 是空的、
`R4_b10.log` 停在 79 秒、机器上没有残留 worker 进程。

### 修复后的测试

全量 **`456 passed / 2 failed`**（两条红仍是日志轮转的环境红，与改动无关）。

## 6. 顺手做的两件（不在计划里，用户当场要的）

- **`tools/show_game.py`**：把一局自对弈打成人类可读战报（四家手牌 + 每一手 +
  模型当时的候选打分 + 自动标出「白炸」）。用户要「看模型到底怎么打的」。
- **`train/eval.py`** 里把「白炸」的判定抽成 `bomb_opportunity` / `is_wasted_bomb`，
  给统计（`_bomb_stats`）与战报**共用一份** —— 免得「什么算白炸」有两套口径。
  抽完既有 `test_bomb_rate.py` 全绿（行为没变），另加一条四种情形逐个钉的测试。
- **`docs/文件地图.md`**：每个目录每个文件干什么 + **面板加载哪个模型** + 速查表。
- **`tools/game_viewer.py`**（2026-09-28，用户当场要的）：自对弈回放的**图形版**，
  底部两个按钮「上一步 / 下一步」（键盘 ←/→ 也行，Home/End 跳首尾），
  候选**写出 Q 值**（用户选的；实机面板仍不写数字）。
  三件事保证了它不会变成第二份实现：
  1. **共用重放** —— `show_game.replay_game` 产出 `Frame` 列表，文字版和图形版都吃它
     （重构后文字战报与原文**逐字一致**，用 diff 验的）；
  2. **共用渲染器** —— 牌桌就是 `net/table.py::TableWindow`，跟实机面板同一个
     `draw()`，所以两边长得一模一样；
  3. **共用判定** —— 「白炸」用 `train/eval.py` 的 `bomb_opportunity`/`is_wasted_bomb`。
  `table.py` 只多了三个**默认不变**的可选参数（`hand_label` / `top_right` / `show_q`），
  实机面板一行行为都没改（`tests/test_table_panel.py` 3 passed、`smoke_panel.py` 7 帧照旧）。
  测试 5 条，其中一条**从渲染器自己身上抽字段清单**（AST 扫 `draw`/`_seat_area` 里的
  `st.X`）—— 渲染器将来多要一个字段它会**自动红**，不用等人从界面上看出半个牌桌。
  反向对照：从适配器里拿掉 `passes` → 立刻红。

---

## 7. 换固定对手：规则式基线（2026-09-28，用户当场要的）

**用户的要求**：「固定对手的策略太弱了，实际上人类不会这么打，去网上学优秀的掼蛋打法」。

### 怎么找到的（工具被封，绕道本机网络）

- `WebSearch` 报 API 错误、`WebFetch` 报域名校验连不上 —— **两个工具都用不了**
- 但 **本机 `curl` 通**：于是自己写了个抓取+剥标签的小工具，走机器自己的网络
- **抓不到的**：中文攻略站（知乎 / 百度百科 / 头条 / y7x7）**都有反爬**，
  而且这台机走代理、机房 IP 被拒；reader 代理（r.jina.ai / allorigins / corsproxy）、
  Wayback 也全不通
- **抓到的**：GitHub 上的掼蛋 AI 开源实现。核心来源是
  **`QinlinChen/guandan-ai`**（`client/ai_client.py` + `utils.py`，逐行读过，
  一份完整的**规则式** AI）；另有 598★ 的 `Calix-L/DanKS`（金山 AI 产品中心）确认技术框架

### 第一步的实测（**不训练**，各 400 局同种子 `seed=1002`）

| 变体 | vs 贪心 | 压队友 | 用炸率 | 白炸 |
|---|---|---|---|---|
| `greedy`（现状） | 基准 | **100.0%**（198/198） | 4.85 手/局 | 0.0%（构造如此） |
| A 照搬源规则 | **67.5%** | 62.3% | — | — |
| B 队友赢着就过（危险除外） | 68.0% | 37.9% | — | — |
| **C 队友赢着就让（能走完除外）—— 已采纳** | **67.0%** | **1.2%** | **1.70** | **0.0%** |

对照 `greedy vs greedy` = **50.7%**（尺子本身没偏）。400 局的二项 sd ≈ 2.5%，
所以 67% 是 **~7 个 sd**；三个「让队友」写法的胜率**全在噪声内** ⇒
**「不压队友」是免费的**，取最像人的那个（压队友 100% → 1.2%）。

### 偏离源规则的那一处（已记在代码里）

源规则**只在队友那手是火力牌或主点 ≥ K 时才让**，队友出小牌时照压 —— 而那正是
用户抱怨的「人类不会这么打」。**危险时也不压队友**：他本来就赢着这一手，
压他并不能解决「对手快走完」那个问题。→ C。

### 用户补的第 11 条规则（比源规则更准）

源规则只看「对手剩 1~2 张」。用户 2026-09-28 补：**3 张（三张）、5 张（三带二/顺子）、
6 张（三连对/钢板）都可能一手走完**，要**按剩的张数估「一手走完的风险」**再决定压不压。
实现：`rule_policy.finish_risk(n, pool)` —— 只看**未露面的牌池**（公开信息），
按 n 对应的牌型需求给一个**启发式刻度**（明确标注「不是概率」）。
领出侧则用更一般的写法：**领出的张数 ≠ 他剩的张数** ⇒ 他一手就走不完
（跟牌必须同牌型同张数）。

### 交付物

- `train/rule_policy.py`（新增）：`table_owner` / `unseen_pool` / `finish_risk` /
  `hand_partition` / `rule_choose` / `rule_policy` —— 全是**只吃 `obs`（公开信息）**的纯函数
- `tests/test_rule_policy.py`：**18 条**，每条钉一条规则，都不开窗口、不跑牌局
- 全量 **479 passed / 2 failed**（两条仍是日志轮转的环境红）

### 没做（等用户定）

- **A/B（第二步）没跑** —— 用户说「先只做第一步，回来给你看」
- `--opp-kind` 的 CLI 接线**故意没加**：A/B 没批准之前不加开关（YAGNI）

## 8. 路线② A/B 起跑（2026-09-28，用户批准）

- **起跑前内存 12.8 GB / 59%** —— 过了 10 GB 门槛（用户腾了机器）✓
- 命令见 `HANDOFF.md` 顶栏；两臂串行，**加了一道保险**：对照臂的日志里没有收尾行
  （`总共`）就**不起处理臂** —— 免得第一条 OOM 死了第二条又去撞同一堵墙
- 起跑确认：`自举关（β=1）`（对照臂）、`ε 起点 0.30`、`池子关`、48 局/秒
- 跑完怎么判：§5 那张预登记表（判据 1 `vs 贪心` 不掉 ≥1sd、判据 2 动作边际变宽，
  两者一起读；配对点 26 万局）
