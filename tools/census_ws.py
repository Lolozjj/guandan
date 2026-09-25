"""把原始帧全量转储（net/raw.jsonl）做一次普查。

要回答的问题：**网络里到底有没有「级别（打几）」**。此前结论「确实没有」不成立 ——
那份抓包只存了每帧前 256 字节，发牌(310B)/结算/进贡这些大消息的正文从未被看过。

跑法：
    .venv/Scripts/python.exe -m tools.census_ws

它做四件事：
  ① 完整性自检：帧真实长度 vs 实际存的字节数 —— **必须全等**，否则结论要打折
  ② msgid 普查：哪些 msgid 出现过、各自多少帧（不受长度限制）
  ③ 字段普查：**每个** msgid 出现过的字段路径与取值样例（这是找新字段的正路）
  ④ 级别候选：找「取值落在 1..13、且一局之内不变」的整数字段
"""
from __future__ import annotations

import collections
import json
import os
import sys

from net import protocol

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "net", "raw.jsonl")


def load(path=RAW):
    if not os.path.exists(path):
        sys.exit(f"没有原始转储：{path}\n"
                 f"先抓一局：set GUANDAN_RAW=1 && python -m net.launcher")
    frames, truncated = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") != "frame":
                continue
            body = bytes.fromhex(r["hex"])
            if len(body) != r["n"]:
                truncated += 1
            frames.append({"t": r["t"], "dir": r["dir"], "n": r["n"], "body": body})
    return frames, truncated


def main() -> int:
    frames, truncated = load()
    s2c = [f for f in frames if f["dir"] == "S→C"]
    print(f"帧 {len(frames)} 条（S→C {len(s2c)}，C→S {len(frames) - len(s2c)}）")

    print(f"\n[①] 完整性：真实长度 != 实存字节的帧 = {truncated}"
          + ("  ✓ 无截断" if truncated == 0 else "  ✗ 有截断，下面结论要打折"))
    lens = collections.Counter(f["n"] for f in s2c)
    big = [n for n in lens if n > 256]
    print(f"    长度分布：{len(lens)} 种，最长 {max(lens)}B；"
          f"**超过 256B 的有 {len(big)} 种**（这些正是以前看不到的）")

    ids = collections.Counter()
    parsed = []
    for f in s2c:
        m = protocol.parse(f["body"])
        if m:
            ids[m["msgid"]] += 1
            parsed.append((f, m))
    print(f"\n[②] S→C 能解出协议结构的 {len(parsed)}/{len(s2c)}")
    for k, v in ids.most_common():
        print(f"    {k}  {protocol.MSG_NAMES.get(k, '?'):<16} {v} 帧")

    print(f"\n[③] 字段普查（每个 msgid 见过的字段路径；★ = 只在大帧里出现）")
    for mid in sorted(ids):
        paths, samples, bigonly = collections.Counter(), {}, set()
        for f, m in parsed:
            if m["msgid"] != mid:
                continue
            for kind, path, val in m["fields"]:
                key = f"{kind} {path}"
                paths[key] += 1
                samples.setdefault(key, val)
                if f["n"] > 256:
                    bigonly.add(key)
        print(f"    --- msgid {mid}（{protocol.MSG_NAMES.get(mid, '?')}）---")
        for key, cnt in paths.most_common(40):
            mark = "★" if key in bigonly else " "
            print(f"      {mark} {key:<22} x{cnt:<5} 例={str(samples[key])[:60]}")

    print(f"\n[④] 级别候选（整数字段，取值 1..13，且在同一连接内长时间不变）")
    seen = collections.defaultdict(collections.Counter)
    for f, m in parsed:
        for kind, path, val in m["fields"]:
            if kind == "int" and isinstance(val, int) and 1 <= val <= 13:
                seen[path][val] += 1
    hits = [(p, c) for p, c in seen.items() if len(c) <= 3]
    if not hits:
        print("    没找到 —— 级别可能不在 S→C，或编码方式不同")
    for p, c in sorted(hits)[:20]:
        print(f"    {p}  取值分布={dict(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
