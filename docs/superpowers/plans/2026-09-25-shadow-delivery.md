# 影子模式交付台账（spec §8.2）

> 2026-09-25/26。**本文档是「影子模式做到哪儿了」的唯一台账**，配 `HANDOFF.md` 一起读。
> 计划：`docs/superpowers/plans/2026-09-25-shadow-mode.md`
> 规格：`docs/superpowers/specs/2026-09-24-guandan-rl-advisor-design.md` §8
> 前作：`docs/superpowers/plans/2026-09-25-plan3-training-delivery.md`（模型练成）
>
> **下一个会话看这里**：用户要的是「先投入使用，我看看效果」——
> 影子模式已经做成，现在等**实机积累几十局**，然后看第四节的读法。

---

## 一、一句话结论

**轮到我出牌时，面板会悄悄算一次模型建议、写进 `net/shadow.jsonl`，并在我真的出手之后回填「我实际出了什么」、在那局结束时追一行「这局赢没赢」。** 全程不动游戏、不发任何报文、不上屏建议（用户 2026-09-25 定：上了屏人会被影响，分歧数据就废了）。

离线验收在**真机素材**上全绿：两局 48 个决策点，网络那条路与日志真值两条独立重建
**700 维状态向量与 15×147 历史矩阵逐位相等**，建议相同，输赢与日志 `Rank` 一致。

## 二、怎么跑

```powershell
cd C:\Users\17837\PycharmProjects\yolo

# 验收（不用联网、不用装证书）
.venv/Scripts/python.exe -m pytest tests/ -q              # 323 passed
.venv/Scripts/python.exe -m tools.accept_shadow           # 六项 [OK] -> 退出码 0

# 离线冒烟：不开游戏也能产出影子日志（14 秒自动关窗口）
.venv/Scripts/python.exe -m net.table --replay --level 9 --seconds 14

# 实机（要装证书；退出自动卸）
python -m net.launcher
```

权重默认取 `runs/rl/*/best.pt` 里**最新的那一个**（按修改时间）；要指定就设
`GUANDAN_WEIGHTS=<路径>`。**加载失败不会崩面板** —— 影子模式整体降级成「不记录」，
面板底部会明说原因（比如「找不到权重（runs/rl/*/best.pt）」）。

`--no-advice` 可以整个关掉影子模式（只显示牌局）。

## 三、这一增量做成了什么

| 文件 | 是什么 |
|---|---|
| `net/advise.py` | 推理链：`GameState -> Observation -> 候选 -> 打分`。**七种「算不了」的原因**都返回 reason 不抛异常 |
| `net/shadow.py` | 影子记录器：决策点去重、落盘、回填实际着法、局末结果、退出收尾 |
| `net/state.py` | 加了**动作流水** `steps`（历史就建在它上）、局号、座位确认位、出完顺序；**修了三个真 bug**（见第四节） |
| `net/panel.py` / `net/table.py` | 两个面板都挂上记录器：`--level` / `--no-advice`，回放也能离线产出影子日志 |
| `tools/accept_shadow.py` | 第三层验收：抓包 + 日志双源逐位对齐（六项 + 地板） |
| `tools/accept_sim.py` | `replay()` 加了 `record` 回调（真值侧复用同一个回放循环，不另写一份） |
| `tests/test_state_steps.py` 等 5 个 | 新增 16 个单测 |

**推理链只加载裁判与权重，不加载模拟器**（spec §8.1）：`net/advise.py` 不建
`GuandanEnv`、不跑对局循环，编码/裁判/打分一律复用
（`env.encode_*`、`meld.legal_moves/as_meld`、`train.net.q_values`）—— 本仓库为
「副本会漂」吃过亏，推理侧另写一份编码器，训练与上线就会悄悄不一致。

## 四、实测与更正

### 三条此前记错的事实（都是这次实测推翻的）

**① 「面板从 msgid 3008 拿级别」不成立。** 3008 的 `3.9.22.12` 是
**两队各自的级别**（2026-09-25 那帧是 `[10, 9, 10, 9]`，座位 0/2 一队打 10、1/3 一队打 9），
而那一局的 `Trump` 是 **9**（= 我那一队的级别，因为上一局我们输了）。
**级别仍然只从日志来**（`LevelWatcher`），3008 没接进面板 —— 接进去反而会错。

**② 我自己「要不起」，报文里是发的。** Plan 3 台账与 HANDOFF 原来写的是「服务器只通知别人、
不通知自己」，据此在状态机里做了一套「从轮转推断我过了」的逻辑。全量抓包里实测：
`seat` 与我的座位一致的那些 3006 一共 36 条（两局 21 + 15），**那就是我自己的要不起**。
原来那个结论来自**被截断的老抓包**（每帧只存 256 字节，大消息的正文根本没存下来）。
推断逻辑**保留**（它仍然能兜住「服务器某次没发」的情况，而且它对已出完的座位不补步），
但它不再是被依赖的主路径。

**③ 座位每局都会变，而且是同一段抓包里就变。** 用「日志发牌给我的 27 张」×
报文里的 `LeftCardList` 子集关系证实：11:11 那局我是 **seat1**（9/9 命中）、
11:15 那局我是 **seat2**（7/7 命中）。**换局后到我第一次出牌之间，状态机的座位是陈旧的** ——
影子模式这段一个决策点都不算（计入跳过原因「座位未确认」）。

### 验收抓出来的三个真 bug（都在 `net/state.py`）

| # | bug | 后果 | 证据 |
|---|---|---|---|
| 1 | `on_play` 清空 `passes` 的条件写成「只有领出的人再次出牌才清」 | 有人压过一手之后，前面那些「要不起」还挂在名单上 → **动作流水少记** | 验收①：抓包 77 步 vs 日志真值 101 步，当场红 |
| 2 | **桌面牌从不清空** | 一轮结束后「待压」还挂着上一手 → 面板显示错，**建议也错**（以为必须压牌，其实是我领出） | 验收②：4 个决策点的桌面与真值不符，连带那几手的建议全错（验收④） |
| 3 | 清桌时没顺手清 `passes` | 同一处，`passed` 那 4 维与真值不符 | 验收②：差异维恰好落在 `_OFF_PASSED` 这 4 格上 |

第 1、3 条的判据与 `rules.Hand.play` / `_advance` **完全一致**（那套在 55 局真实对局上验过）：
「一手打出来 = 重新开一轮」「一轮结束就清桌清要不起」。第 2 条新加了
`_maybe_clear_table`，判据与 `rules.Hand._advance` 同源，注释里写明了必须保持一致。

**这三条都影响过用户看到的面板**（尤其第 2 条：建议会以为我在跟牌，其实我在领出）。

### 两条量出来的数（不是猜的）

- **级别滞后**：日志要 ~20 秒才落盘，所以换局后有一小段窗口级别还是上一局的。
  按真实帧时间戳建模后实测：**0/48 个决策点**会带着上一局的级别 ——
  因为「座位未确认」那段窗口（换局到我第一次出牌）本来就更宽，把风险盖住了。
  验收⑥ 会把这两个数都打出来。
- **`actual_rank == -1` 必须按「形状」比**：`melds_from` 的契约是「每个形状一条代表」
  （spec §13.2），候选里的牌常常是换了花色的同形状牌 —— 拿牌张比会把合法着法
  误报成「不在候选里」（实测 2 处误报，其中一手是单张 ♣9）。

## 五、影子日志怎么读

`net/shadow.jsonl`，一行一个 JSON、追加写（与 `net/events.jsonl` 同风格）。

**一行决策点**（离线回放实测产物，真数据）：

```json
{"type": "decision", "t": 1790352859.107, "deal": 0, "pos": 4, "seat": 1, "level": 9,
 "level_age_s": null,
 "hand": [17, 19, 27, 28, 37, 39, 41, 42, 43, 51, 56, 57, 67, 70, 73, 77, 281, 297, 299, 301, 307, 333],
 "table": [33, 44, 45, 58, 59], "table_kind": 4, "left": [22, 22, 27, 27], "passed": [2, 3],
 "n_cand": 14,
 "top": [{"cards": [19, 51, 67, 307], "names": ["3♠", "3♣", "3♣(二副)", "3♦"], "kind": 8, "q": 0.1831},
         {"cards": [], "names": [], "kind": 0, "q": 0.153},
         {"cards": [19, 41, 51, 67, 297, 307], "names": ["9♥", "9♥(二副)", "3♠", "3♣", "3♣(二副)", "3♦"], "kind": 8, "q": 0.1121}],
 "actual": [19, 51, 67, 307], "actual_names": ["3♠", "3♣", "3♣(二副)", "3♦"], "actual_is_me": true,
 "actual_shape": [8, 4, 3], "actual_rank": 0, "resolved": true}
```

逐字段：

| 字段 | 含义 |
|---|---|
| `deal` / `pos` | 第几局 / 局内第几步（`pos` = 动作流水里的下标，也是决策点的唯一键） |
| `seat` / `level` | 我在报文里的座位 / 本局级别（打几） |
| `level_age_s` | 级别是多久以前更新的（`null` = 面板启动到现在没更新过级别） |
| `hand` / `table` / `table_kind` | 我的手牌 / 桌面待压的牌 / 桌面牌型（`kind` 见 `net/sim/meld.py`：4=顺子、8=炸弹、0=我领出） |
| `left` / `passed` | 按**绝对座位**给的：各家剩几张 / 本轮谁要不起 |
| `n_cand` / `top` | 候选个数 / 模型的前三名（`q` 是 Q 值，`cards` 为空 = 「过」） |
| `actual` / `actual_names` / `actual_rank` | 我实际出的牌 / 牌面 / **它在候选里的 Q 名次（0 = 就是头名，`-1` = 不在候选里=告警）** |
| `actual_shape` | 实际着法的 `(牌型, 张数, 主点数)` —— 离线统计分歧时用，不用再判一次牌型 |
| `actual_is_me` | 那一步是不是我出的（不是的话说明状态机把人归错了，要查） |
| `resolved` | 是否已回填；`false` 的带 `unresolved_reason`（换局/退出时未回填） |

**一行局末**：

```json
{"type": "deal_end", "deal": 0, "seat": 1, "level": 9, "finish": [], "me_team_won": null,
 "n_decisions": 17, "n_unresolved": 1, "skips": {}}
```

`finish` = 出完的先后（空 = 回放窗口没覆盖到那一局的结尾）；
`me_team_won` = 我这队赢没赢（`null` = 没看到结束）；`skips` = 各种「算不了」的次数。

**第一行是会话行**：`{"type":"session","schema":1,"weights":"runs/rl/.../best.pt","games":40000,"winrate_greedy":0.795,...}`
—— 权重路径与训练元信息，用来溯源「这批记录是哪版权重产出的」。

**三条现成命令**：

```powershell
# 1) 分歧率 + 告警
.venv/Scripts/python.exe -c "
import json
d=[json.loads(l) for l in open('net/shadow.jsonl',encoding='utf-8') if l.strip()]
d=[r for r in d if r['type']=='decision' and r.get('resolved')]
n=len(d)
print('决策点', n)
print('模型与人一致（actual_rank==0）:', sum(1 for r in d if r.get('actual_rank')==0), '/', n)
print('模型前三里含人的选择:', sum(1 for r in d if r.get('actual_rank') in (0,1,2)), '/', n)
print('实际着法不在候选里（告警，应当为 0）:', sum(1 for r in d if r.get('actual_rank')==-1))
print('那一步不是我出的（告警）:', sum(1 for r in d if r.get('actual_is_me') is not True))
"

# 2) 跳过原因分布
.venv/Scripts/python.exe -c "
import json, collections
rs=[json.loads(l) for l in open('net/shadow.jsonl',encoding='utf-8') if l.strip()]
c=collections.Counter()
for r in rs:
    if r['type']=='deal_end': c.update(r['skips'])
print('跳过原因合计:', dict(c) or '（没有跳过）')
print('各局决策点:', [(r['deal'], r['n_decisions'], r['n_unresolved']) for r in rs if r['type']=='deal_end'])
print('各局赢没赢:', [(r['deal'], r['me_team_won']) for r in rs if r['type']=='deal_end'])
"

# 3) 分歧点逐条看（人打了什么、模型想打什么）
.venv/Scripts/python.exe -c "
import json
d=[json.loads(l) for l in open('net/shadow.jsonl',encoding='utf-8') if l.strip()]
d=[r for r in d if r['type']=='decision' and r.get('resolved') and r.get('actual_rank')!=0]
for r in d:
    top=r['top'][0] if r['top'] else {'names':[],'q':0}
    print(f\"局{r['deal']} #{r['pos']} 桌面 {' '.join(map(str,r['table'])) or '清'} | 模型: {' '.join(top['names']) or '过'}(q={top['q']}) | 我: {' '.join(r['actual_names']) or '过'}(第{r['actual_rank']+1}名)\")
"
```

**判据**（下一个会话据此下结论）：
- `actual_rank == -1` 与 `actual_is_me != true` 都应当**接近 0** —— 不接近说明状态重建有错，先查那个。
- 「跳过原因」必须**可解释**，不能一片全是「座位未确认」（那说明座位认定坏了）。
- `actual_rank == 0` 的比例 = 「模型与人不谋而合」；**这个数不是模型好坏的判据**
  （人不采纳不等于模型错，反之亦然 —— 所以这份数据用于分析，不直接当训练标签，spec §8.3）。

## 六、已知限制（照实写）

1. **进贡/还贡**：那两个字段网络里还没定位，状态机那几手不准。验收里没把进贡窗口计入分母；
   实机日志里如果看到某局开头几步的桌面不对劲，先怀疑这个。
2. **级别**：换局后日志要 ~20 秒才写出新 `Trump`，这段窗口级别可能是上一局的。
   记录里带 `level_age_s`；本次两局实测量到 **0 个决策点**受影响（窗口被「座位未确认」盖住）。
3. **面板启动在一局中间**时，「本局」编号是 0（没有换局事件），局末汇总的 `finish` 可能是空的
   —— 那一局的 `me_team_won` 会是 `null`。
4. **`actual_rank` 按「形状」比**（见第四节）—— 同形状换花色的候选算同一个。
5. 模拟器训练时**进贡是关着的**（`EnvConfig.tribute=False`），级别是**均匀采样**的；
   真人是从打 2 一路爬上去的。影子模式不受影响（它不模拟、只读实时状态），
   而且正好能量出这个偏差有多大（拿实机日志的级别分布比训练的均匀分布）。

## 七、没做的（明确在计划外）

| 东西 | 为什么 |
|---|---|
| 反馈回收（spec §8.3） | 影子模式之后：要先有几十局基线数据 |
| vs 人类着法那条统计（spec §7 第三行） | 下一个会话拿影子日志顺带算，不用额外素材 |
| msgid 3008 接进面板 | 这次已证明它不是**本局**级别（是两队各自的），接进去反而错 |
| 进贡/还贡字段定位 | 不影响影子模式（只影响那几手的桌面） |
| 图形面板上的「复盘视图」 | 现在的复盘靠上面那三条命令；上了屏反而可能被建议影响 |

## 八、本次的提交

数它们用这条命令（**不写死数字** —— 文档自己也在提交，写死了必错）：

```powershell
git log --oneline 115b842..HEAD
```

内容：`feat(state)` 动作流水 + 换局重认座位 · `feat(advise)` 推理链 ·
`feat(shadow)` 影子记录器 · `feat(panel)` 两个面板挂上记录器 ·
`feat(accept)` 影子模式离线验收（修掉三个真 bug）· 本台账。
