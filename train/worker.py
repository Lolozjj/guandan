"""自对弈的 worker 进程：打牌、把**紧凑记录**发回给学习进程。

为什么不发张量：一局编码后约 1.6 MB，100 局/秒就是 160 MB/s 的 IPC（spec §4 已否）。
`replay.GameRecord` 只有几百字节，而且它自带 `expand` 能把奖励重算出来
（所以 worker **不需要**回传名次 —— 「不复制」原则的又一次适用）。

控制通道：learner 定期广播 `("weights", (state_dict, eps))` ——
不广播的话 worker 永远用开局那版策略，「学的」和「打的」就是两回事。
"""
from __future__ import annotations

import random

import torch

from train import selfplay
from train.net import QNet


def worker_device() -> str:
    """worker 一律用 CPU：它只做推理，显存留给 learner。

    实测（spec §2）：这个循环的瓶颈在 Python 侧（牌型引擎+编码占 60%+），
    worker 上 GPU 只会抢显存，不换速度。
    """
    return "cpu"


def worker_batch(net, rng, eps, n_games, opp_mix=0.5, greedy_share=0.8,
                 learn_all_seats=False):
    """打一批局，只取紧凑记录（张量与奖励都由 learner 侧重放出来）。

    ⚠️ 新参数**必须带默认值** —— `tests/test_worker.py` 直接调这个函数，
    它是「worker 的打法与进程内逐局一致」那条硬保证的载体，别删掉另写一份。
    """
    return [rec for rec, _pts, _y in selfplay.generate_batch(
        net, rng, eps, n_games, capture=False,
        opp_mix=opp_mix, greedy_share=greedy_share,
        learn_all_seats=learn_all_seats)]


def _drain_ctrl(ctrl_q, net):
    """非阻塞收控制消息；返回最新收到的 ε（没收到就返回 None）。"""
    eps = None
    while True:
        try:
            kind, payload = ctrl_q.get_nowait()
        except Exception:                       # noqa: BLE001 - 队列空就是没消息
            return eps
        if kind == "stop":
            raise SystemExit(0)
        if kind == "weights":
            sd, eps = payload
            net.load_state_dict(sd)
            net.eval()


def run_worker(send_q, ctrl_q, cfg: dict) -> None:
    """进程体。`cfg` 由 `worker_cfg` 造。

    **入口是模块级函数**（Windows 用 spawn，闭包不能跨进程）。
    """
    # 与 learner 同一个种子 -> 初始权重一致；之后靠广播保持同步
    torch.manual_seed(cfg["seed"])
    net = QNet().to(worker_device()).eval()
    # worker 开局自己那一份也得是热启动的那份 —— 否则 learner 的第一次广播到达之前，
    # 第一批是用**随机**权重打的（那个窗口在老代码里就有，热启动让它更刺眼）
    selfplay.load_init(net, cfg.get("init"))
    rng = random.Random(cfg["seed"])
    eps = cfg["eps"]
    while True:
        recs = worker_batch(net, rng, eps, cfg["batch_games"],
                            opp_mix=cfg["opp_mix"], greedy_share=cfg["greedy_share"],
                            learn_all_seats=cfg.get("learn_all_seats", False))
        send_q.put(recs)                        # 队满则阻塞 = 天然背压（spec §6）
        got = _drain_ctrl(ctrl_q, net)
        if got is not None:
            eps = got


def worker_cfg(seed, eps, opp_mix, greedy_share, batch_games,
               learn_all_seats=False, init=None) -> dict:
    return {"seed": seed, "eps": eps, "opp_mix": opp_mix,
            "greedy_share": greedy_share, "batch_games": batch_games,
            "learn_all_seats": learn_all_seats, "init": init}
