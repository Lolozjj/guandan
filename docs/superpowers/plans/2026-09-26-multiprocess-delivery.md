# 多进程自对弈交付台账

> 2026-09-26。设计：`docs/superpowers/specs/2026-09-26-multiprocess-selfplay-design.md`
> 计划：`docs/superpowers/plans/2026-09-26-multiprocess-selfplay.md`（**Task 4 还没做**，见文末）
> 用户当时的决定：先用多进程把 20 个核用起来，**Rust 内核留到以后再议**。

## 一、一句话结论

**训练循环拆成了「worker 进程打牌 + 主进程学习」，实测 33.8 局/秒 —— 是原来单进程
13.2 局/秒的 2.6×**（推荐 `--workers 2`）。代码路径 `--workers 1` 仍是老行为。

## 二、怎么跑

```powershell
cd C:\Users\17837\PycharmProjects\yolo
$env:GUANDAN_DEVICE="cpu"
.venv/Scripts/python.exe -m train.selfplay 32400 --opp-mix 0.5 --workers 2

# 量吞吐（worker 数各跑一段，报局/秒与队列积压峰值）
.venv/Scripts/python.exe -m tools.bench_train --workers 1 2 3 4 6 --seconds 45 --quiet
```

## 三、实测吞吐曲线（2026-09-26，每档 45 秒，量的是训练阶段）

| worker | 局/秒 | 队列积压峰值 |
|---|---|---|
| 1 | 30.5 | 2 |
| **2** | **33.8** | 4 |
| 3 | 31.1 | 6 |
| 4 | 27.3 | 8 |
| 6 | 21.9 | 12 |

**两个与设计预测不符的地方（都是好消息 + 一个要记住的教训）**：

1. **单 worker 就 30.5，不是我预测的 13.2** —— 因为拆分之后「打牌」与「重放+训练」
   在**两条进程里并行了**；老单进程是把这两件事串起来做的。所以这一步的收益
   一半来自并行、一半来自多核。
2. **设计里写的天花板 23 局/秒偏保守**（实测学习进程能吃下 ~30）。但方向没错：
   **再加 worker 只会变慢**（3/4/6 掉到 31/27/22），队列积压一路涨到 12 ——
   说明学习进程饱和后，多出来的 worker 只是把记录堆在队列里、外加 IPC 开销。
   曲线与积压列互相印证，这是「别凭直觉加并行度」的现场证据。

⚠️ 45 秒的短跑只能定性；**要更稳的数就加 `--seconds`**（Task 4 的 10 分钟短跑本来要做这个）。

## 四、这次改动做成了什么

| 文件 | 是什么 |
|---|---|
| `train/worker.py` | worker 进程：`worker_batch`（只取紧凑 `GameRecord`，不传张量）+ `run_worker`（收权重/stop 的控制循环） |
| `train/selfplay.py` | 新增 `train_parallel` + `--workers`；抽出 `_maybe_eval` / `_finish` 两个公共件（两条路共用评测/存档/报告） |
| `tools/bench_train.py` | 吞吐曲线工具（+ `elapsed`/`qmax` 两个测量字段） |
| `tests/test_worker.py` | **跨进程逐局逐位一致** + 真起进程收发 + worker 用 CPU |
| `tests/test_train_parallel.py` | worker 死了**必须炸** + 短跑走通并存档 |

**三条纪律都钉在测试里**：worker 死了必须报错（不许静默变慢）；队列有界（背压，
不堆内存）；ε 由主进程按**全局局数**算完广播（各 worker 自己算会「每个都以为自己是全部」）。

## 五、还没做的

1. **Task 4 的 10 分钟短跑**（2 worker）：确认 loss 正常下降、结尾 `vs 贪心` 与
   `炸弹浪费率` 都正常。**这是唯一还没验的**（前面全是 ≤45 秒的短跑）。
2. **HANDOFF 更新**（训练命令加 `--workers 2`、瓶颈已从「Python 引擎」移到
   「学习进程的重放」、曲线表）。
3. **学习进程走 GPU 的那次测量**（设计 §4 说好本轮只量）：拆分后 torch 占比上升，
   GPU 可能从「更慢」变成「更快」——值得量一次再决定默认值。
4. **Rust 内核**：用户已同意留到以后。按剖面算只值 2.4×（换语言省不掉 torch 那 16%），
   而多进程这一步已经拿到 2.6×。
