"""全仓库的路径常量 —— **想换目录只改这一份**。

三条规矩（2026-09-30 重排仓库时立的）：

1. `models/best.pt` 是**面板唯一会加载**的权重。这里没有「按修改时间挑最新」
   那套魔法 —— 它当年换过两次源、屏幕上一点提示都没有。训练产出留在 `runs/`，
   要上屏就**手动复制**成 `models/best.pt`：一步，而且看得见。
2. 运行时产物（抓包事件流、影子日志、mitmproxy 日志）一律落 `runtime/`，与代码分开。
   `runtime/` 不入库。
3. 验收的真值（日志语料快照）在 `data/game_corpus.json` —— **入库、别删**：
   游戏日志只留两天，被轮转掉之后验收就没有素材了。
"""
from __future__ import annotations

from pathlib import Path

#: 仓库根（本文件在 `<根>/guandan/paths.py`）
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RUNTIME = ROOT / "runtime"          # 运行时产物，不入库
MODELS = ROOT / "models"            # 权重，不入库
RUNS = ROOT / "runs"                # 训练产出，不入库

EVENTS = RUNTIME / "events.jsonl"   # 抓包事件流（面板跟读它）
RAW = RUNTIME / "raw.jsonl"         # 全量载荷转储（验收素材）
SHADOW = RUNTIME / "shadow.jsonl"   # 影子模式记录
MITM_LOG = RUNTIME / "mitm.log"     # mitmproxy 自己的输出

BEST = MODELS / "best.pt"           # 面板加载的权重
CORPUS = DATA / "game_corpus.json"  # 日志语料快照（验收真值）

# 让第一次跑不必先手动建目录
RUNTIME.mkdir(parents=True, exist_ok=True)

#: mitmproxy 的两个插件脚本（launcher 用 `-s <路径>` 挂它们）
ADDON = ROOT / "guandan" / "capture" / "addon.py"
RAWDUMP = ROOT / "guandan" / "capture" / "rawdump.py"
