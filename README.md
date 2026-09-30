# 掼蛋出牌建议

读微信小程序「大掼蛋」的**网络报文** → 还原牌局 → 桌边挂一个实时面板，显示
**当前级别 / 我的手牌 / 现在轮到谁 / 每家出过什么牌 / 出牌建议**。

**只建议，不代打** —— 不模拟点击、不往游戏发任何报文。这是产品红线。

> **2026-09-30 重排仓库**：截图识别（YOLO）那条兜底线、历史设计与台账、旧模型快照
> 全部移出工作区，代码按职责重新归到 `guandan/` 下。原文都在 **`master` 分支**的历史里。
> 这份 README 是工作区**唯一**的文档。

---

## 一、现在什么水平（先看结论）

**流程是通的；模型还不好用。**

| | 状态 |
|---|---|
| 抓包 → 牌局状态 → 面板 → 影子记录 | ✅ 真机跑通过；离线验收三层全绿（§五） |
| 出牌建议 | ⚠️ **还不够好** —— 最新权重 `models/best.pt` 对**规则式脚本对手** 71.2%（6 种子配对 × 400 局） |
| 缺的是什么 | **训练目标（信用分配）**，不是流程 |

⚠️ 上面那个 71.2% 是「打一个脚本对手」的数，**不能**读成「打人也能用」。
实机感受是：建议偏保守、关键局不灵。这正是下一步要改的东西（§七）。

---

## 二、目录

```
guandan/
  launcher.py        一键启停：装证书 → 起抓包（带看门狗）→ 开面板 → 退出全部还原
  paths.py           全仓库的路径常量（**想换目录只改这一份**）
  capture/           抓包与协议
    protocol.py        wss 帧 / msgid / 字段表解码
    cards.py           牌 ID ↔ 牌面 + 掼蛋牌序
    state.py           牌局状态机（手牌 / 出牌 / 轮次 / 接风 / 一局清场）
    levelwatch.py      级别（打几）—— 只从游戏自带日志读
    addon.py           mitmproxy 插件：事件流 → runtime/events.jsonl
    rawdump.py         mitmproxy 插件：全量原始帧 → runtime/raw.jsonl（验收素材）
  sim/               牌局引擎（训练与上线的**同一份**，不许有副本）
    meld.py            牌型引擎：合法着法枚举 + 大小比较
    rules.py           牌局规则：轮转 / 接风 / 进贡 / 名次
    env.py             RL 环境：700 维状态 / 143 动作 / 15×147 历史
  advice/            出牌建议（明牌 → 策略的唯一窄口）
    advise.py          GameState → Observation → 候选 → QNet 打分 → 建议
    shadow.py          影子模式：每个决策点记账 → runtime/shadow.jsonl
  ui/                面板
    table.py           图形面板（tkinter）
    panel.py           终端面板 / --replay 离线回放
  rl/                自对弈训练（DouZero 式 DMC）
    net.py             QNet（LSTM(128) + 6×512 MLP）
    selfplay.py        训练主循环（32 局同步推进；--workers 多进程）
    worker.py          多进程里的打牌进程（只用 CPU）
    replay.py          紧凑记录 + 采样时重放（拿 CPU 换内存）
    policies.py        贪心 / 随机 / 网络 策略
    rule_policy.py     规则式「像人」对手（也当固定对手用）
    eval.py            评测：match 胜率 + bomb_waste 炸弹浪费率
    pool.py            对手池（**已证否**，默认关）
tools/               验收与测量脚本（§五）
tests/               472 条
data/game_corpus.json  日志语料快照（验收的真值，**入库、别删**）
models/best.pt       面板加载的权重（**不入库**：7.8 MB）
runtime/             运行时产物（不入库）：events.jsonl / raw.jsonl / shadow.jsonl / mitm.log
runs/                训练产出（不入库）：<臂名>/best.pt、pool/snap_*.pt
```

---

## 三、怎么跑

### 1. 打牌时：面板 + 影子记录

```powershell
.venv/Scripts/python.exe -m guandan.launcher
```

一条命令做完四件事，**任何一条失败路径都不许留下断网**：
装 mitmproxy 证书 → 起抓包（带看门狗）→ 等代理真的连通 → 开面板；退出时全部还原。

| 参数 | 用途 |
|---|---|
| `--dry-run` | 只检查环境，不动系统代理 |
| `--no-panel` | 只起抓包，不开面板（调链路用） |
| `--no-advice` | 关掉影子模式（不加载权重） |
| `--no-show-advice` | 算建议但不显示（记录里如实写 `advice_shown:false`） |
| `--level N` | 手输级别（日志读不到时兜底） |
| `--console` | 用终端版面板 |
| `--selftest N` | 走 N 局把流程走顺再退出（演示前排练还原路径） |
| `--yes` | 跳过确认 |

启动时会自己报加载了哪份权重 —— 这一行是判断「换没换源」的唯一依据：

```
影子模式就绪 —— 模型：guandan/models/best.pt（210,016 局）
```

### 2. 看对局（不用真机）

```powershell
# 图形回放：牌桌与实机面板同一个渲染器（黄底 = 实际出的牌，绿底 = 模型推荐的）
.venv/Scripts/python.exe -m tools.game_viewer --seed 61 --opp-kind rule

# 文字战报：四家手牌 + 每一手 + 当时候选打分 + 自动标出「白炸」
.venv/Scripts/python.exe -m tools.show_game --seed 7 --opp-kind rule
```

### 3. 训练

```powershell
$env:GUANDAN_DEVICE="cpu"      # ⚠️ 实测最优，不是将就：这个循环以 Python 为主，上 GPU 只抢显存
.venv/Scripts/python.exe -u -m guandan.rl.selfplay 32400 `
  --init models/best.pt `
  --opp-kind rule --opp-mix 0.5 --workers 2 `
  --out-dir runs/R12 *> runs/R12.log
```

第一个位置参数是**秒数**（32400 = 9 小时）。常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--init <ckpt>` | 无 | 热启动。**ε 起点自动压到 0.3**（`--eps-start` 可覆盖） |
| `--opp-kind {greedy,rule}` | `greedy` | 固定对手：贪心（弱、已饱和）或规则式（像人，**推荐**） |
| `--opp-mix F` | `0.5` | 每局以 F 的概率把一个队换成固定对手 |
| `--workers N` | `1` | 多进程自对弈 |
| `--out-dir <dir>` | `runs/<时间戳>` | 两臂必须分开，否则后跑的覆盖 `best.pt` |
| `--eps-games G` | `250000` | ε 退火到 0.1 所需的**局数**。**跑短实验一定调小** |
| `--buffer N` | `50000` | replay 容量（局） |
| `--algo {dmc,pg}` | `dmc` | ⚠️ `pg`（REINFORCE）**会塌**，别拿它跑长训练 |

**三条训练纪律（都是踩出来的）**

1. **绝不并发两条臂** —— 一条约 4.8 GB，这台机器只够一条（并发过、打爆过）。
2. **热启动短于 ~50 万局基本白跑** —— ε 从 0.3 退到 0.1 要 25 万局，大半预算花在探索上。
   实测：54 万局的续训只值 +0.7pp（测不出来），78 万局才值 +3.8pp。
3. **`Ctrl-C` 是安全的**（走收尾逻辑）；快照每 2 万局一份，**最多丢 2 万局**。
   恢复只需 `--init` 指到最新快照，但**局数计数器会重启**。

### 4. 量水平（四把尺子）

```powershell
# 主尺子：多臂 × 多种子**配对**，比谁强
.venv/Scripts/python.exe -m tools.ruler models/best.pt runs/R12/best.pt --base models/best.pt

# 单臂体检：vs 贪心 / vs 规则式 / 白炸 / 用炸率
.venv/Scripts/python.exe -m tools.ab_compare runs/R12

# 规则式自己有多强 + 三条行为护栏
.venv/Scripts/python.exe -m tools.rule_bench

# 动作边际（候选间 top1−top2）
.venv/Scripts/python.exe -m tools.action_margin models/best.pt
```

**测量纪律**：单种子 400 局的 sd 约 2.1pp，比大多数要判的效应还大 ⇒ **至少 3 个种子**，
而且**必须配对**（同种子 = 同一副牌、同样谁坐哪，逐局对消）。两次独立测量之差**不能**当两臂之差。

### 5. 验收

```powershell
.venv/Scripts/python.exe -m tools.accept_meld      # ① 牌型引擎（59 局真实对局）
.venv/Scripts/python.exe -m tools.accept_sim       # ② 牌局引擎：轮转/接风/终局/记账
.venv/Scripts/python.exe -m tools.accept_tribute   # ③ 进贡/还贡（证据分级）
.venv/Scripts/python.exe -m tools.accept_shadow    # ④ 影子模式整条推理链（需 runtime/raw.jsonl）
.venv/Scripts/python.exe -m pytest tests/ -q       # 全量：472 过 / 16 跳
```

①②③ 现在全绿。**④ 需要素材**：先用 `$env:GUANDAN_RAW=1; python -m guandan.launcher`
抓一场全量包（落 `runtime/raw.jsonl`），否则它会明确报「一项都没检查到」——
那是**假绿防御**，不是故障。

`tools/snapshot_logs.py` 把当前日志语料冻结成 `data/game_corpus.json`：
**游戏日志只留两天、会被轮转删除**，验收的真值靠这份快照活着。

---

## 四、权重怎么换（只有一条路，而且看得见）

优先级：**显式传参 → `GUANDAN_WEIGHTS` 环境变量 → `models/best.pt`**（`advise.resolve_weights`）。

这里**故意没有**「扫目录挑最新的」那套 —— 老版本按修改时间扫 `runs/rl/*/best.pt`，
换过两次源、屏幕上都没人知道。要换权重：

```powershell
Copy-Item runs/R12/best.pt models/best.pt          # ① 换文件
$env:GUANDAN_WEIGHTS="runs/R12/best.pt"            # ② 或临时指一份（不动 models/）
```

训练快照落在 `runs/<臂>/pool/snap_*.pt`，**结构上不可能被面板加载**（嵌套目录 + 不叫 best.pt）。

---

## 五、运行时产物

| 文件 | 是什么 |
|---|---|
| `runtime/events.jsonl` | 抓包解出的**语义事件流**（谁出了什么牌），面板跟读它 |
| `runtime/raw.jsonl` | **全量原始帧**（`GUANDAN_RAW=1` 才写），验收④的素材 |
| `runtime/shadow.jsonl` | **影子模式**：每个决策点「模型建议了什么 / 我实际出了什么 / 这手最后赢没赢」 |
| `runtime/mitm.log` | mitmproxy 自己的输出 |

`shadow.jsonl` 是**追加累积**的 —— 读的时候必须**按 `session` 行切分**，
否则会把老权重的问题算到新权重头上（这个坑踩过）。

---

## 六、环境

| 项 | 值 |
|---|---|
| Python | **3.12.10**，虚拟环境在仓库根 `.venv/` |
| 主链路依赖 | `torch 2.11+cu128` / `numpy` / `matplotlib` / `pillow` / `pywin32` |
| 抓包 | **mitmproxy 装在另一个独立 venv**：`C:\Users\17837\mitmtool\Scripts\mitmdump.exe` |
| 证书目录 | `C:\Users\17837\.mitmproxy`（launcher 把 CA 装进当前用户证书库） |

⚠️ **没有 `requirements.txt`** —— 重建环境照上表装。
⚠️ **本机现在没装 mitmproxy**：`C:\Users\17837\mitmtool\Scripts\mitmdump.exe` 与
`C:\Users\17837\.mitmproxy` 都不存在，所以 `guandan.launcher --dry-run` 会报
「mitmdump : !! 没找到」，抓包那条路现在起不来。要真机跑面板，先照上表装一遍 mitmproxy，
并把 `VENV_MITM` 指到实际的 `mitmdump.exe`。
⚠️ **RTX 50 系是 Blackwell（sm_120）**，必须 cu128 及以上构建，否则报
`no kernel image is available for execution on the device`。
⚠️ **`mitmdump` 的路径是写死的绝对路径**（`guandan/launcher.py` 的 `VENV_MITM`）——
换机器 / 换用户名要改那一行。

---

## 七、已知的坑与下一步

### 坑

| 坑 | 说明 |
|---|---|
| 级别有两个来源 | 游戏日志（准，最坏滞后 20 秒）与网络报文（快）。**日志里 `UpgradeInfo.TrumpValue` 是下一局的级别**，不是本局 —— 拿它当本局级别会算出全错的建议 |
| `shadow.jsonl` 是累积的 | 读之前先按 `session` 切分 |
| `best.pt` 的挑选口径 | 训练内评测只有 200 局（噪声 ±2pp），而一代的真实进步约 3pp ⇒ **噪声和信号同量级**，挑出来的常常是「运气最好的那一次」。要选权重请用 `tools.ruler` |
| 后台长跑会被回收 | 内存紧张时后台任务可能被杀（不是训练失败）。快照每 2 万局一份，够到配对点就无害 |

### 下一步：换训练目标（信用分配）

根因不在流程：现在的标签是**整手牌的终局结果**，而 95% 的局都赢
⇒ 标签里 95% 是「+3」，「炸 vs 不炸」的区别全挤在那 5% 里 ——
**不是没被惩罚，是惩罚淹在噪声里**。首选次选之差中位只有 0.123 也印证这点。

**这些路已经走过并证否，别再重复跑**（每条都留下了数字）：

| 路线 | 结果 |
|---|---|
| 对手池（league + PFSP） | 每局不比对照好，每小时少跑 39% 的局 |
| n 步自举 | 判据掉 2.0 sd，**动作边际反而窄了 28%**（方向是反的） |
| 候选排序（listwise） | 推演即否 |
| 模仿学习 / 换更强对手 | 反事实探针：分歧点上听规则式的**平均 −0.06 点**，正效应上界 +0.025 —— 不值得 |
| 策略式（REINFORCE，`--algo pg`） | **会塌**：熵守门约 30 分钟逮到策略退化成 argmax。根因是信任域缺失（熵奖励是软约束，β 从 0.01 扫到 0.1 都不治） |

⇒ 框架级候选只剩**信用分配**这一条（设计一份新 spec，别在时长上加码：
同配方边际已经落到 **1~2pp/百万局**，烧的是电）。

> 代码注释里出现的 `spec §N`、`HANDOFF`、`plans/…` 都是 2026-09 那批设计与台账，
> 已移出工作区 —— 原文在 `master` 分支的历史里。
