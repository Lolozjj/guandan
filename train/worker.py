"""自对弈的 worker 进程：打牌、把**紧凑记录**发回给学习进程。

为什么不发张量：一局编码后约 1.6 MB，100 局/秒就是 160 MB/s 的 IPC（spec §4 已否）。
`replay.GameRecord` 只有几百字节，而且它自带 `expand` 能把奖励重算出来
（所以 worker **不需要**回传名次 —— 「不复制」原则的又一次适用）。

控制通道三类消息（spec §3.4）：

| 消息 | 何时发 | 载荷 |
|---|---|---|
| `("member", (mid, state_dict))` | **只在池子新增成员时** | ~7.8 MB |
| `("weights", (sd, eps, pfsp))` | 每 1000 局 | ~7.8 MB + K 个 float |
| `("stop", None)` | 退出 | — |

**为什么成员要增量发**：每次广播全池是 20 × 7.8MB = 156MB，
按 1000 局（约 30 秒）一次算是 5MB/s × worker 数 —— 直接把 learner 拖死。
"""
from __future__ import annotations

import random

import torch

from train import pool, selfplay
from train.net import QNet


def worker_device() -> str:
    """worker 一律用 CPU：它只做推理，显存留给 learner。

    实测（spec §2）：这个循环的瓶颈在 Python 侧（牌型引擎+编码占 60%+），
    worker 上 GPU 只会抢显存，不换速度。
    """
    return "cpu"


def worker_batch(net, rng, eps, n_games, opp_mix=0.5, greedy_share=0.8,
                 learn_all_seats=False, members=None, pick_fixed=None):
    """打一批局，只取紧凑记录（张量与奖励都由 learner 侧重放出来）。

    ⚠️ 新参数**必须带默认值** —— `tests/test_worker.py` 直接调这个函数，
    它是「worker 的打法与进程内逐局一致」那条硬保证的载体，别删掉另写一份。
    """
    return [rec for rec, _pts, _y in selfplay.generate_batch(
        net, rng, eps, n_games, capture=False,
        opp_mix=opp_mix, greedy_share=greedy_share,
        learn_all_seats=learn_all_seats, members=members, pick_fixed=pick_fixed)]


def _drain_ctrl(ctrl_q, net, members: dict, state: dict) -> None:
    """非阻塞收控制消息，**就地**改 `net` / `members` / `state`。

    `state` 里存最新的 ε 与 PFSP 权重 —— 不用返回值传，因为一次可能收到好几条。
    """
    while True:
        try:
            kind, payload = ctrl_q.get_nowait()
        except Exception:                       # noqa: BLE001 - 队列空就是没消息
            return
        if kind == "stop":
            raise SystemExit(0)
        if kind == "member":
            mid, sd = payload
            m = QNet().to(worker_device()).eval()
            m.load_state_dict(sd)               # 装不上会抛 —— 不许静默少一个成员
            members[mid] = m
        elif kind == "weights":
            sd, eps, w = payload
            net.load_state_dict(sd)
            net.eval()
            state["eps"] = eps
            state["pfsp"] = w
            if w:
                # learner 的池子是有界的（只留最近 POOL_SIZE 个）。它挤掉谁，
                # 采样权重里就没有谁 —— 据此同步删，**不另开一种消息类型**。
                # ⚠️ `w` 空 = 还没算过采样权重，这时**不能删**（会把开局种子池清空）。
                keep = set(w)
                for mid in [m for m in members if m not in keep]:
                    del members[mid]


def run_worker(send_q, ctrl_q, cfg: dict) -> None:
    """进程体。`cfg` 由 `worker_cfg` 造。

    **入口是模块级函数**（Windows 用 spawn，闭包不能跨进程）。
    """
    # 与 learner 同一个种子 -> 初始权重一致；之后靠广播保持同步
    torch.manual_seed(cfg["seed"])
    net = QNet().to(worker_device()).eval()
    # worker 开局自己那一份也得是热启动的那份 —— 否则 learner 的第一次广播到达之前，
    # 第一批是用**随机**权重打的（那个窗口老代码里就有，热启动让它更刺眼）
    selfplay.load_init(net, cfg.get("init"))
    members = {}
    for mid, sd in (cfg.get("members") or {}).items():
        m = QNet().to(worker_device()).eval()
        m.load_state_dict(sd)
        members[mid] = m
    rng = random.Random(cfg["seed"])
    state = {"eps": cfg["eps"], "pfsp": {}}

    def pick_fixed(r):
        if cfg.get("pick_all") == "member":     # 只给测试用
            return ("member", cfg["member_id"])
        return pool.pick_opponent(r, list(members), state["pfsp"],
                                  cfg["greedy_share"])

    # ⚠️ 池子关的时候要传 **`None` 这个参数**（让 generate_batch 走老的 greedy_share 二分），
    # 而不是让 `pick_fixed` 返回 None —— 函数照旧会被调用，`kind[0]` 会炸。
    picker = pick_fixed if (cfg.get("use_pool") or cfg.get("pick_all")) else None

    while True:
        # ⚠️ **先收控制消息，再打牌。** 反过来的话新装的成员要等下一批才生效，
        # 而「成员还没到就先抽到它」会直接 KeyError 把 worker 打死。
        # 顺带把「广播到达之前第一批用的是开局旧权重」这个老窗口一起关掉。
        _drain_ctrl(ctrl_q, net, members, state)
        recs = worker_batch(net, rng, state["eps"], cfg["batch_games"],
                            opp_mix=cfg["opp_mix"], greedy_share=cfg["greedy_share"],
                            learn_all_seats=cfg.get("learn_all_seats", False),
                            members=members, pick_fixed=picker)
        send_q.put(recs)                        # 队满则阻塞 = 天然背压（spec §6）


def worker_cfg(seed, eps, opp_mix, greedy_share, batch_games,
               learn_all_seats=False, init=None, members=None,
               pick_all=None, member_id=None, use_pool=False) -> dict:
    """`members`：`{mid: state_dict}` 的**初始**池。

    `use_pool=False` 时**不抽池成员**，走老的 `greedy_share` 二分 ——
    A/B 的对照臂就靠它（两臂只能差「有没有池子」这一个变量）。

    `pick_all` / `member_id` **只给测试用**：强制所有混合局都用 `member_id` 当对手
    （生产上对手由 learner 侧的 PFSP 权重决定，worker 按 `pfsp` 抽）。
    """
    return {"seed": seed, "eps": eps, "opp_mix": opp_mix,
            "greedy_share": greedy_share, "batch_games": batch_games,
            "learn_all_seats": learn_all_seats, "init": init,
            "members": dict(members or {}),
            "pick_all": pick_all, "member_id": member_id, "use_pool": use_pool}
