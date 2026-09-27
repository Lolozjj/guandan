# 当前方案（现用全图）

> 2026-09-27 复盘时写。**只写「现在真正在用的东西」**，不写历史方案。
>
> 与 `HANDOFF.md` 的关系：HANDOFF 是流水账（里面大量是「已完成」的历史任务，
> 代码地图也停在截图那条线为主的时期）。**本文是现状快照；两者冲突时以本文为准。**
>
> 本文同时是本轮死代码清理的记录（见 §七、§八）。

---

## 一、用户要什么

PC 微信开着掼蛋，**旁边挂一个实时面板**，显示：

1. 当前级别（打几）
2. 我的手牌
3. 现在轮到谁
4. 每家出过什么牌（记牌）
5. 轮到我时给出**出牌推荐**（2026-09-24 定案：走自对弈 RL，**不写规则版**）

**明确排除**：替用户自动出牌。

---

## 二、链路全图

```
 微信小程序「大掼蛋」
   │
   ├──(wss，服务器→客户端是明文 protobuf)──► mitmproxy ──► net/events.jsonl
   │                                          net/addon.py    （事件流）
   │                                          net/rawdump.py  （全量载荷，给验收用）
   │
   └──(游戏自带日志)──► …/HappySDKLogFiles/*.log      ← 级别只在这里
                          net/levelwatch.py

                   net/state.py  GameState（手牌/出牌/轮次/座位/接风/一局清场）
                          │
        ┌─────────────────┼──────────────────┬──────────────────┐
        ▼                 ▼                  ▼                  ▼
   net/table.py     net/panel.py       net/shadow.py      net/advise.py
   （图形面板）      （终端版/回放）     （影子模式记录）     （出牌建议）
        │                 │                  │                  │
        └──── 屏幕/终端 ──┘                  ▼                  ▼
                                       net/shadow.jsonl   train/net.py  QNet
                                                                ▲
                                        train/selfplay.py 训练 ─┘
                                                │
                                     runs/rl/<日期-时分>/best.pt
```

**级别是唯一的例外**：网络报文里没有本局级别，只有游戏日志里有。
而且日志里 `UpgradeInfo.TrumpValue` 是**下一局**的级别，不是本局的 —— 这个坑踩过。

---

## 三、每一环的现状

| 环节 | 文件 | 说明 |
|---|---|---|
| 一键启停 | `net/launcher.py` | 装证书 → 起抓包（带看门狗）→ 开面板 → 退出**全部还原**。任何一条路径都不能留下断网 |
| 协议解码 | `net/protocol.py` | wss 帧 / msgid / 字段表。出牌 28/28 逐字段对过 |
| 牌 ID ↔ 牌面 | `net/cards.py` | 查表实现（`is_card` 412→91ns、`slot` 186→88ns） |
| 牌局状态机 | `net/state.py` | 手牌 / 出牌 / 轮次 / 座位识别 / 接风 / 一局清场；另有动作流水 `steps` |
| 级别 | `net/levelwatch.py` | 只从游戏日志来；取**结算行**的 `UpgradeInfo.TrumpValue` |
| 图形面板 | `net/table.py` | tkinter；右侧「模型建议」栏用**牌面图片**画（首选大图 + 两个备选小图） |
| 终端面板 | `net/panel.py` | 备用；`--replay` 可离线回放抓包 |
| 影子模式 | `net/shadow.py` | 每个决策点算建议、记实际着法、记局末结果 → `net/shadow.jsonl` |
| 推理链 | `net/advise.py` | `GameState → Observation → 候选 → QNet 打分`；**不加载模拟器**（但复用它的编码器与裁判，避免副本漂） |
| 规则引擎 | `net/sim/meld.py` `rules.py` | 牌型判定 / 牌局规则 |
| RL 环境 | `net/sim/env.py` | 700 维状态（按出牌人相对化）、143 个动作、15×147 历史 |
| 模型 | `train/net.py` | LSTM(128) + 6×512 MLP |
| 训练 | `train/selfplay.py` | DouZero 式 DMC；32 局同步推进；`--workers N` 多进程 |
| 回放缓冲 | `train/replay.py` | 存**紧凑记录**（几百字节/局），采样时重放（约 27ms/局） |
| worker | `train/worker.py` | 多进程里的打牌进程（只用 CPU） |
| 评测 | `train/eval.py` | `match`（座位对调、同步推进）+ `bomb_waste` |

---

## 四、怎么跑

### 打牌时（面板 + 影子模式）

```powershell
.venv/Scripts/python.exe -m net.launcher
```

| 参数 | 用途 |
|---|---|
| `--dry-run` | 只检查环境，不动系统代理 |
| `--no-panel` | 只起抓包，不开面板（调链路用） |
| `--no-advice` | 关掉影子模式（不加载权重） |
| `--no-show-advice` | 算建议但不显示（记录里如实写 `advice_shown:false`） |
| `--level N` | 手输级别（日志读不到时） |
| `--console` | 用终端版面板 |
| `--yes` | 跳过确认 |

### 训练

```powershell
$env:GUANDAN_DEVICE="cpu"      # 实测 CPU 反而快：这个循环以 Python 为主，GPU 只抢显存
.venv/Scripts/python.exe -m train.selfplay 32400 --opp-mix 0.5 --workers 2
```

- **判据（退出码）**：`vs 贪心 ≥ 55%`（spec §7 的分水岭：50% 只是五五开）
- 收尾会打印：`vs 随机`、`vs 贪心`（**两个种子，看方差**）、**炸弹浪费率**、曲线
- `--eps-games`：ε 退火到 0.1 所需的**局数**（默认 25 万）。
  ⚠️ **跑短实验一定要把它调小**，否则前大半段都在近乎随机地探索，两组实验测不出差别
- `--buffer`：默认 5 万局（约 92MB）；spec 原数是 50 万局（923MB）
- ⚠️ **`--workers` 的推荐值尚未钉准**，见 §六.4

### 验收

```powershell
.venv/Scripts/python.exe -m tools.accept_meld      # ① 牌型引擎
.venv/Scripts/python.exe -m tools.accept_sim       # ② 牌局引擎（轮转/接风/终局/记账）
.venv/Scripts/python.exe -m tools.accept_tribute   # ③ 进贡/还贡（证据分级）
.venv/Scripts/python.exe -m tools.accept_shadow    # ④ 影子模式（整条推理链，六项）
.venv/Scripts/python.exe -m pytest tests/ -q       # 356 条
```

`tools/snapshot_logs.py` 把当前日志语料冻结成 `data/game_corpus.json`。
**日志只留 2 天、会被轮转删除** —— 验收的真值靠这份快照活着。

---

## 五、实测数字（口径写在这里，引用时别丢）

### 三版权重，同一把尺子

| 存档 | 训练量 | vs 贪心（400 局，同种子） | 炸弹浪费率（面对贪心，60 局） |
|---|---|---|---|
| `runs/rl/20260926-0033/best.pt` | 20 万局 · **纯自对弈** | 87.8% | 5.1% |
| `runs/rl/20260926-0940/best.pt` | 18 万局 · 混对手 | 87.2% | 5.1% |
| **`runs/rl/20260926-1407/best.pt`** | **144 万局 · 混对手** | **95.2%** | **4.2%** |

- 1407 直接对打 0033：**62.3%**（400 局）
- 1407 **到结束前 16 分钟还在刷新 best** → **没到顶，继续练有收益**
- 吞吐 **44.5 局/秒**（1,441,760 局 / 9 小时 1 分）

### 评测是**确定性的**

`match` 的牌由固定种子抽出（每次评测都是**同一批牌**），策略全是确定性的
（网络 argmax、贪心取最小、随机用固定 rng）。实测：同一个网络连跑两次，
结果**一字不差**（90.0% / 90.0%，与存档自报一致）。

→ 所以日志里 `vs 贪心` 的涨跌是**真实的行为变化，不是采样噪声**。
→ 但反过来说：**它只是同一批 200 局上的表现**，不能当泛化能力的唯一依据。
（`selfplay.py` 里那句「200 局的噪声约 ±2%」说的是**换种子**时的不确定度，不是重复评测。）

### 炸弹浪费率有**两个**数

`train/eval.py::bomb_waste` 的 docstring 明写：**自对弈**和**面对贪心**是两个数，
**玩家看到的是后者**。两个都要看，别只量一个。

| 存档 | 自对弈 | 面对贪心 |
|---|---|---|
| 0033 | 5.2% | 5.1% |
| 0940 | 4.2% | 5.1% |
| 1407 | 6.4% | 4.2% |

### 真实对局（`net/shadow.jsonl`）

- 7 次开会话，权重依次 0033 → 0940 → 1407。**文件是追加累积的 —— 读的时候必须按 `session` 行切分**，否则会把老版本的问题算到新版本头上
- 12 局 / 223 个决策点；与建议第一选一致的比例 **62.6%**
- 最后一次会话（1407）：40 个决策点，**跟建议 90%**（之前几次是 40~66%）—— 明显更信它了
- 「桌面有牌」的决策点里**模型首选是炸弹**的比例：
  0033 会话 **56/140（40%）** → 1407 会话 **3/34（9%）**；
  其中「本来能用普通牌压」（= 真浪费）的：**22/140 → 1/34**

---

## 六、已知问题 / 没做完

### 1. 炸弹浪费**没治好**（用户最初的抱怨）

面对贪心 5.1% → 4.2%，**只有约 1.6 个标准差 —— 不足以称为改善**。
真实对局里症状明显轻了（40% → 9%），但残差还在。

**根因（2026-09-27 诊断）**：`opp_mix=0.5 / greedy_share=0.8` 意味着
**一半的训练局在跟一个已经被打穿的对手下棋**（95% 胜率）。
贪心现在教不了任何东西；更要命的是它**惩罚不了浪费** ——
它从不主动炸、也压不住你，你白炸一手它抓不住。
所以炸弹浪费在训练分布里**至今没有代价**。

方向：把固定对手从贪心换成**模型自己的历史版本**（league），而不是继续加算力。

### 2. `replay.expand` 会把固定对手的着法当训练目标（真 bug）

设计明写「固定对手的着法**不进训练目标**（记进去等于拿它当老师）」，
`generate_batch` 的现场抓取（`caps`）也确实只记学习那一队 ——
**但单进程那条路几乎不用现场抓取**：

```python
pts, y = fresh.get(id(rec)) or replay.expand(rec)
# buf.sample 从 5 万局里有放回抽 32 局，撞上刚打那 32 局的期望是 0.02 局/步
```

而 `GameRecord` 只有 `level/first/hands/actions`，**没有「哪一队是学习的」这个信息**，
`expand` 只能把四家全产出成训练点。实测 `opp_mix=0.5`：
6008 个重放训练点里 **1040 个（17.3%）是固定对手（贪心）的着法**，
标签却是那一局的胜负。`train_parallel` 更彻底 —— 它**永远**走 `expand`。

⚠️ 但要如实说：**1407 带着这个 bug 练到了 95.2%**，它不是拦路虎。

### 3. 验收的「换源」有缺口（那条红测试的真因）

`tests/test_tribute_records.py` 红，是因为**日志轮转**（最早的实时日志只剩 09-24，
进贡记录被转掉了）。而 `load_tributes()` **没有快照兜底**：
`load_corpus()` 有（`data/game_corpus.json` 在，报告里也会写「来源是快照还是实时日志」），
`load_tributes()` 直接读实时日志目录 —— 于是进贡那一环
**读不到证据，却没说数据是从哪来的**。这正是本项目「**换源必须可见**」纪律的缺口。

### 4. 多进程的收尾没做完

- 10 分钟短跑 + HANDOFF 更新 + learner 走 GPU 的那次测量（计划里的 Task 4）
- **吞吐曲线要重测**：`tools/bench_train.py` 每档只跑 45 秒，
  把 `spawn` + `import torch` 的启动成本全摊进去了 ——
  它报 workers=2 是 33.8 局/秒，而实际 9 小时跑出 **44.5**。
  **短跑测出来的吞吐不能拿来选并行度。**

### 5. 两份 replay 实现曾经并存

`net/panel.py --replay` 与 `net/replay.py` 是两套独立的「把抓包当实时流回放」，
同一个 `DEFAULT_CAPTURE`、同一件事。`net/replay.py` 已删，
但**这说明「副本会漂」在这条线上真实发生过一次**，值得记进纪律。

### 6. `net/rawdump.py` 是**隐式依赖**

它没有被任何代码 import（mitmproxy 用 `-s` 按路径加载），
所以**任何「谁引用谁」的自动检查都看不出它还在用**。删代码时这类东西最容易误伤。

---

## 七、本轮复盘：六个判断失误

1. **用 45 秒短跑测吞吐** → 把进程启动成本摊进去了，低估稳态（报 33.8，实际 44.5）。
   **短跑不能拿来选并行度。**
2. **建议在 22.7 万局停掉 0940**，依据是「vs 贪心」在固定 200 局上的下降趋势。
   1407 证明继续练有大收益 —— **固定小样本上的趋势不是可靠路标。**
3. **把 `replay.expand` 污染当成 0940 退化的主因** → 过度归因。
   1407 带着它练到了 95.2%。
4. **`bomb_waste` 的 docstring 是我自己写的**，里面明写「两个数都要看、
   玩家看到的是面对贪心那个」—— 我量的时候还是只量了自对弈那个，
   得出了「没改善」的错误中间结论。
5. **读 `shadow.jsonl` 时没先按会话切分** —— 那是追加累积的文件。
   混着读会把 0033 的问题算到 1407 头上（第一次数出「53 个首选炸弹」，
   实际 1407 会话只有 3 个）。
6. **划死代码时，把「留下的东西的依赖」划进了删除档**
   （`match_played.py` / `table_gt.py` 被保留的 `eval_table.py` import；
   `tools/snapshot_logs.py` 被 `game_log.py` + 一个测试引用）。
   **靠逐文件引用核查才拦住。**

---

## 八、本轮清理（2026-09-27）

删了 **22 个跟踪文件 + 2 份大日志**（提交 `ae39ffb`、`27d4e63`）：

`main.py`（内容是 PyCharm 默认模板）· `_patch_*.py` ×12（一次性改文件脚本）·
`probe_ws.py`（探协议用的，已被 `net/protocol.py` 取代）·
`test_pretrained.py` / `test_rotation.py`（HANDOFF 自己记着「已完成使命」）·
`net/verify.py` · `net/replay.py` · `tools/bench_sim.py` · `tools/census_ws.py` ·
`train/smoke.py` · `check_env.py` · `train_guandan7.log`（4MB）· `train_run.log`

同时修掉 3 处会指向「已不存在的文件」的注释
（`net/sim/meld.py`、`train/net.py`、`train/selfplay.py`）。

### 没删的，以及为什么

| 东西 | 为什么留着 |
|---|---|
| `match_played.py` / `table_gt.py` | 被保留的 `eval_table.py` import；`table_gt.py` 里是**人工逐帧放大核对的真值，不可再生** |
| `tools/snapshot_logs.py` | 是「把日志语料冻结成快照」的发生器，被 `game_log.py` + 测试引用（§六.3 那个缺口正是它要防的事） |
| `net/rawdump.py` | 没被 import，但 **mitmproxy 正在用**（`-s` 按路径加载） |
| `live/` `synth/` `predict_cards.py` `eval_real.py` `eval_table.py` + 配套分析脚本 | **YOLO 兜底线**，2026-09-27 决定整条保留（含 `tests/test_rules_adapter.py`，它是唯一拖着 `live/` 和 `synth/` 的测试） |

⚠️ **`runs/rl/` 绝不能删** —— 面板加载的就是 `runs/rl/*/best.pt`
（`net/advise.py::newest_weights` 按修改时间挑最新的那个）。

### 连带结论

只要 YOLO 兜底线还在，`cv2` / `mss` / `pywin32` / `ultralytics` 这四个重依赖就都得留着
（它们**只**被 `live/`、`synth/` 和 `tests/test_rules_adapter.py` 用到）。
想卸掉整个 CV 栈，只有一条路：整条兜底线一起删。

---

## 九、现用文件地图

```
net/            主链路
  launcher.py     一键启停（证书 / 抓包 / 面板 / 还原）
  addon.py        mitmproxy 插件：解事件 → events.jsonl
  rawdump.py      mitmproxy 插件：全量载荷 → raw.jsonl（给验收用）
  protocol.py     wss 协议解码
  cards.py        牌 ID ↔ 牌面 + 掼蛋大小排序
  state.py        牌局状态机
  levelwatch.py   级别（只在这里读游戏日志）
  table.py        图形面板（tkinter）
  panel.py        终端面板 / --replay
  shadow.py       影子模式记录器
  advise.py       推理链（唯一的「明牌 → 策略」窄口）
  sim/
    meld.py       牌型引擎
    rules.py      牌局规则
    env.py        RL 环境（700 维状态 / 143 动作 / 15×147 历史）

train/
  selfplay.py     训练主循环（DMC；--workers 多进程）
  worker.py       多进程的打牌进程
  replay.py       紧凑记录 + 重放
  net.py          QNet
  policies.py     贪心 / 随机 / 网络策略
  eval.py         match（胜率）+ bomb_waste（炸弹浪费率）

tools/            验收与运维
  accept_meld.py / accept_sim.py / accept_tribute.py / accept_shadow.py
  game_log.py     读游戏日志语料（快照优先）
  snapshot_logs.py 冻结语料快照
  decision_points.py / bench_train.py

tests/            356 条；fixtures/ 里是协议样本
data/game_corpus.json  日志语料快照（验收的真值靠它活着）

live/  synth/  predict_cards.py  eval_real.py  eval_table.py
                  └─ YOLO 兜底线（整条保留，非主路径）

docs/superpowers/plans/   各期计划与交付台账
docs/superpowers/specs/   设计与本文
HANDOFF.md                历史流水账（代码地图部分已过期）
```

---

## 十、下一步（按优先级）

1. **修 §六.3 的换源缺口** —— `load_tributes()` 加快照兜底，并让报告写明来源。
   修完那条红测试自然就绿了。（小、独立、有明确验收）
2. **修 §六.2 的 `expand` 污染** —— 给 `GameRecord` 加 `learn` 座位字段。
   小改动，有 RED→GREEN 测试。（收益未知，但它是真 bug）
3. **换对手：league 而不是贪心** —— 治 §六.1 炸弹浪费的根。这是唯一能惩罚浪费的方向。
4. **补 §六.4 的多进程收尾** —— 短跑 + 重测吞吐曲线（这次每档跑足 3 分钟）+ HANDOFF。
5. 继续训练 —— 1407 到结束还在涨，没到顶。
