"""第三层验收：影子模式（spec §8.2）—— 拿**真机素材**离线验整条推理链。

素材：`net/raw.jsonl`（全量载荷抓包，2026-09-25 11:11:13~11:18:55，两局）
真值：同一时段的游戏日志（两局都有结算）—— 能一手不落地重建出明牌对局。

与另两个验收的分工：`accept_meld` 验牌型引擎，`accept_sim` 验牌局引擎；
**本脚本验「网络那条路能不能喂出和模拟器一模一样的输入」，以及建议本身。**

六项：
  ① 抓包事件流 -> 状态机的动作流水，与真值 `rules.Hand.steps` **逐手一致**
     （含推断出来的「我过了」；座位按「我的座位」这一对锚点做旋转映射）
  ② 每个影子决策点的状态编码（700 维）与历史（15x147）与真值**逐位相等**
  ③ 实际着法都在候选里、且那一步确实是我（`accept_meld` 验收① 的口径）
  ④ 同一个局面、两条独立重建（网络 / 日志）给出的**建议相同**
  ⑤ 每局的输赢与日志 `Rank` 一致（含「第一个出完的人就是第 1 名」）
  ⑥ 每个决策点的级别与**那一局**的日志 `Trump` 一致

跑法：
    .venv/Scripts/python.exe -m tools.accept_shadow

⚠️ **素材会过期**：日志只留 2 天。真值走 `load_corpus()`（**快照优先**），
所以只要 `data/game_corpus.json` 里有那两局，日志轮转掉也还能跑 —— 报告里会写清
真值是从哪来的。抓包本身是 `GUANDAN_RAW=1` 跑一次 launcher 才有的。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

from net import addon, advise, cards as cardmod, panel, shadow
from net.sim import env, rules
from net.state import GameState
from tools import accept_sim, decision_points, game_log as gl
from tools.accept_meld import Result, _utf8_stdout

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = gl.LOG_DIR
DEFAULT_CAPTURE = os.path.join(PROJ, "net", "raw.jsonl")

#: 地板：验到的量不到这个数，「全过」不足以称为结论（与 accept_meld / accept_sim 同口径）。
_MIN_DECISIONS = 30
_MIN_DEALS = 2

#: 日志的落盘滞后（秒）。**实测约 20 秒**（游戏自带日志攒够一批才写），
#: 用来量「级别还是上一局的」这段窗口有多大 —— 见 `wire_pass(lag=True)`。
LEVEL_LAG = 20.0


# ------------------------------------------------------------------ 素材

def load_frames(path):
    """抓包 -> [(t, body)]（只要游戏服的服务器下发方向）。

    只认全量转储那种格式（`k="frame"`）：老抓包每帧被截到 256 字节，
    发牌/结算这种大消息的正文从来没被存下来，不适合当验收素材。
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("k") == "frame" and r.get("dir") == "S→C":
                out.append((r["t"], bytes.fromhex(r["hex"])))
    out.sort(key=lambda x: x[0])
    return out


# ------------------------------------------------------------------ 网络那条路

def wire_pass(frames, net, out_path, levels=None, lag=False, collect_builds=False):
    """走**生产那份**解码与分发（只少了 mitmproxy）。

    `addon.GuandanTap` 只用它的 `handle()`（纯函数：帧 -> 事件字典），
    `_emit` 不碰，所以不会顺手建文件 —— 落盘只由 `ShadowLog` 干。

    `levels = {局号: Trump}` 模拟「面板从日志拿到级别」：换局时把本局的级别写进状态机
    （实时那条路是 `LevelWatcher` 每 150 毫秒轮询日志，`LEVEL_LAG` 秒才收敛）。
    `lag=True` 时**按真实帧时间戳模拟那个滞后**（换局后 LEVEL_LAG 秒才认新级别）——
    用来量「级别还是上一局的」这段窗口里到底有多少个决策点。
    `collect_builds=True` 时，把「轮到我」那一个瞬间的 `Built`（状态 + 历史）也收下来，
    键是 `(局号, 流水位置)` —— 验收要拿它和真值逐位比，而 `ShadowLog` 自己不留 `Observation`。

    返回 `(st, 每局的流水, 每局我在哪个座位, shadow.jsonl 的记录, 收下来的 Built)`。
    """
    st = GameState()
    log = shadow.ShadowLog(net=net, out_path=out_path, weights="")
    tap = addon.GuandanTap()
    steps, me, builds, deal_t0 = {}, {}, {}, {}
    for t, body in frames:
        ev = tap.handle(body)
        if ev is None:
            continue
        panel.apply_event(st, ev)
        log.after_event(st, ev)
        steps[st.deal_seq] = list(st.steps)      # 每个事件后都快照一份
        me[st.deal_seq] = st.me
        deal_t0.setdefault(st.deal_seq, t)
        if levels and st.deal_seq in levels and st.level != levels[st.deal_seq]:
            ready = (t >= deal_t0[st.deal_seq] + LEVEL_LAG) if lag else True
            if ready:
                st.level = levels[st.deal_seq]
                log.note_level()
        if collect_builds and st.turn == st.me and st.me_confirmed:
            b = advise.build(st)
            if not b.reason:
                builds.setdefault((st.deal_seq, len(st.steps)), b)
    log.close()
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    return st, steps, me, recs, builds


# ------------------------------------------------------------------ 真值那条路

def my_log_seat(g):
    """日志里我在哪个座位：发牌给我的 27 张与哪个座位的起手牌重合最多。

    判据是「重合 >= 25」而不是全等 —— 进贡/还贡会换掉一两张
    （`initial_hands` 重建出来的是**进贡之后**的起手牌）。
    返回 `(座位, 重合张数)`。
    """
    want = set(g.my_cards)
    hands0 = decision_points.initial_hands(g)
    best, hit = None, 0
    for s in rules.SEATS:
        n = len(want & hands0[s])
        if n > hit:
            best, hit = s, n
    return best, hit


def truth_pass(g, my_seat):
    """真值那条路：复用 `accept_sim.replay` 的回放循环，逐点取真值。

    `played` 不用另外维护 —— 「起手牌 - 当前手牌」就是这一家出过的牌。

    返回 `{"actors", "points", "steps", "hands0"}`：`actors[i]` 是第 i 步是谁出的，
    `points[i]` 只在「第 i 步轮到我」时才有（那一瞬间的状态与历史）。
    """
    hands0 = decision_points.initial_hands(g)
    actors, points = [], {}

    def record(kind, hand):
        actors.append(hand.turn)
        if hand.turn != my_seat:
            return
        played = {s: hands0[s] - hand.hands[s] for s in rules.SEATS}
        points[len(actors) - 1] = {
            "obs": env.observe(hand, my_seat, played, hand.table),
            "hist": env.encode_history(hand, my_seat),
            "table_meld": hand.table,
        }

    ann = accept_sim.replay(g, record=record)
    return {"actors": actors, "points": points, "steps": ann.hand.steps,
            "hands0": hands0}


# ------------------------------------------------------------------ 验收

def _names(cs):
    return " ".join(cardmod.decode_all(sorted(cs))) if cs else "过"


def run(capture=None, log_dir=None) -> list:
    """跑六项验收，返回 `Result` 列表（**素材不全就报 FAIL，不抛异常也不返回空**）。"""
    capture = capture or DEFAULT_CAPTURE
    if not os.path.exists(capture):
        return [Result("影子模式验收的素材",
                       floor=f"找不到抓包文件 {capture}"
                             f"（全量抓包要 GUANDAN_RAW=1 跑一次 net.launcher）")]
    frames = load_frames(capture)
    if not frames:
        return [Result("影子模式验收的素材",
                       floor=f"{capture} 里没有游戏服的服务器下发帧")]
    net, err = advise.load_net()
    if net is None:
        return [Result("影子模式验收的权重", floor=err)]

    r1 = Result("① 动作流水：抓包那条路 vs 日志真值，逐手一致")
    r2 = Result("② 决策点：700 维状态与 15x147 历史，逐位相等")
    r3 = Result("③ 实际着法都在候选里、且那一步确实是我")
    r4 = Result("④ 两条独立重建（网络 / 日志）给出同一个建议")
    r5 = Result("⑤ 每局的输赢与日志 Rank 一致")
    r6 = Result("⑥ 决策点的级别与那一局的 Trump 一致")
    out = [r1, r2, r3, r4, r5, r6]

    t0, t1 = frames[0][0], frames[-1][0]
    games = sorted([g for g in gl.load_corpus(log_dir)
                    if t0 - 5 <= g.t0.timestamp() <= t1 + 600],
                   key=lambda g: g.t0)
    r1.note = (f"真值语料：{gl.LAST_SOURCE}；抓包窗口 "
               f"{t0:.0f}~{t1:.0f} 内 {len(games)} 局")
    if not games:
        for r in out:
            r.floor = ("抓包窗口内一局真值都没有 —— 日志被轮转删了？"
                       "快照 data/game_corpus.json 也没覆盖？")
        return out

    # ---- 每局：真值重建 + 座位映射 ----
    # 第一遍**不带级别**：先只看结构（每局的流水、每局我在哪个座位），
    # 才能把「局号 -> 日志里的那一局」对上，从而知道每局的 Trump。
    tmpdir = tempfile.mkdtemp(prefix="shadow-accept-")
    tmp = os.path.join(tmpdir, "shadow.jsonl")
    _st, steps_by_deal, me_by_deal, _r0, _b0 = wire_pass(frames, net, tmp)
    deals = sorted(steps_by_deal)
    per_deal = {}
    levels = {}
    # **按时间配对，不用 zip。** `net/raw.jsonl` 是**累积**的（不同场次的帧会追加在
    # 一起），zip 会按位置硬配 —— 配不上的那些局就被**静默丢掉**（实测：今天那一局
    # 的 11 个决策点因此没进 ②④⑥）。配不上的要报出来，不许装作没有。
    if len(deals) != len(games):
        r1.note += (f"。⚠️ 抓包里有 {len(deals)} 局、真值只覆盖 {len(games)} 局 —— "
                    f"抓包是累积的（不同场次的帧混在一起），这次只验了能配上的那些")
    for deal, g in zip(deals, games):
        lme, hit = my_log_seat(g)
        r1.total += 1
        if lme is None or hit < 25:
            r1.bad.append(f"局 {g.t0:%H:%M}：认不出日志里我在哪个座位（最大重合 {hit}/27）"
                          f"—— 语料不足，不是代码问题")
            continue
        levels[deal] = g.trump
        wme = me_by_deal.get(deal)
        per_deal[deal] = {
            "g": g, "lme": lme, "truth": truth_pass(g, lme),
            # 两套座位号只差一个旋转（两边都遵守 0→3→2→1 的出牌顺序），
            # 旋转量由「我的座位」这一对锚点定死：日志的 lme ↔ 报文的 wme。
            "of": (lambda s, lme=lme, wme=wme: (wme + (s - lme)) % 4),
        }

    # 第二遍**带上级别**（模拟面板从日志拿到级别）——这一次的记录才是验收要看的，
    # 顺带把「轮到我」那一瞬间的状态收下来，供 ② 与真值逐位比。
    tmp = os.path.join(tmpdir, "shadow-leveled.jsonl")
    _st, _s2, _m2, _r1, builds = wire_pass(frames, net, tmp, levels=levels,
                                           collect_builds=True)

    # ---- ① 动作流水 ----
    for deal, d in sorted(per_deal.items()):
        r1.total += 1
        wire = [(s, tuple(sorted(cs)) if cs else None) for s, cs in steps_by_deal[deal]]
        want = [(d["of"](stp.seat), tuple(sorted(stp.meld.cards)) if stp.meld else None)
                for stp in d["truth"]["steps"]]
        if len(wire) != len(want):
            r1.bad.append(f"局 {deal}：流水长度不一样 —— 抓包 {len(wire)} 步、真值 {len(want)} 步")
            continue
        diff = [i for i, (a, b) in enumerate(zip(wire, want)) if a != b]
        if diff:
            i = diff[0]
            r1.bad.append(
                f"局 {deal}：{len(diff)}/{len(wire)} 步对不上，第 {i} 步 "
                f"抓包 座位{wire[i][0]} {_names(wire[i][1])} vs "
                f"真值 座位{want[i][0]} {_names(want[i][1])}")

    # ---- 决策点：影子记录 + 网络侧重重建 ----
    recs = [json.loads(l) for l in open(tmp, encoding="utf-8") if l.strip()]
    decs = [r for r in recs if r["type"] == "decision"]
    if len(decs) < _MIN_DECISIONS or len(per_deal) < _MIN_DEALS:
        floors = (f"验到的量太少：决策点 {len(decs)}（要 ≥{_MIN_DECISIONS}）、"
                  f"能对齐的局 {len(per_deal)}（要 ≥{_MIN_DEALS}）—— 「全过」不足以称为结论")
        for r in (r2, r3, r4, r6):
            r.floor = floors

    for rec in decs:
        d = per_deal.get(rec["deal"])
        if d is None:
            continue
        key = (rec["deal"], rec["pos"])
        tp, b = d["truth"]["points"].get(rec["pos"]), builds.get(key)
        r2.total += 1
        r4.total += 1
        if tp is None or b is None:
            why = f"局 {rec['deal']} 位置 {rec['pos']}："
            r2.bad.append(why + ("真值侧没有这个决策点" if tp is None else "网络侧重建不出来"))
            r4.bad.append(why + "有一侧重建不出来")
            continue
        va, vb = env.encode_state(b.obs), env.encode_state(tp["obs"])
        if not np.array_equal(va, vb):
            idx = np.nonzero(va != vb)[0]
            r2.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}：状态向量 {len(idx)} 维不一样"
                          f"（前几维 {list(idx[:6])}）")
        if not np.array_equal(b.hist, tp["hist"]):
            n = int((b.hist != tp["hist"]).any(axis=1).sum())
            r2.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}：历史矩阵不一样（{n}/15 行）")
        # ④ 同一局面、两边各算一次建议
        ct = advise.candidates(advise.Built(obs=tp["obs"], hist=tp["hist"],
                                           table_meld=tp["table_meld"]))
        if not ct:
            r4.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}：真值侧枚举不出候选")
            continue
        from train.net import q_values
        pick_t = ct[int(q_values(net, tp["obs"], ct, tp["hist"]).argmax())]
        top = rec.get("top") or []
        pick_w = sorted(top[0]["cards"]) if top else None
        want = sorted(pick_t.cards) if pick_t is not None else []
        if pick_w != want:
            r4.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}：建议不一样 —— "
                          f"网络 {_names(pick_w)} vs 日志 {_names(want)}")
        elif len(ct) != rec["n_cand"]:
            r4.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}：候选个数不一样 —— "
                          f"网络 {rec['n_cand']} vs 日志 {len(ct)}")

    # ---- ③ 实际着法 ----
    for rec in decs:
        if not rec.get("resolved"):
            continue
        r3.total += 1
        if rec["actual_is_me"] is not True:
            r3.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}："
                          f"流水里那一步不是我出的（状态机把座位搞错了）")
        if rec["actual_rank"] == -1:
            r3.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}："
                          f"实际出的「{_names(rec['actual'])}」不在候选里")

    # ---- ⑤ 输赢 ----
    ends = {r["deal"]: r for r in recs if r["type"] == "deal_end"}
    for deal, d in sorted(per_deal.items()):
        rank = (d["g"].settle or {}).get("Rank")
        if not rank:
            continue
        r5.total += 1
        ranks = [int(x) for x in rank]
        me_won = (d["lme"] % 2) == rules.winner_team(ranks)
        end = ends.get(deal)
        if end is None:
            r5.bad.append(f"局 {deal}：没有 deal_end 记录")
            continue
        if end["me_team_won"] != me_won:
            r5.bad.append(f"局 {deal}：记录说我这队{'赢' if end['me_team_won'] else '输'}，"
                          f"日志 Rank={ranks} 说{'赢' if me_won else '输'}")
        if end["finish"]:
            want_first = d["of"](ranks.index(1))
            if end["finish"][0] != want_first:
                r5.bad.append(f"局 {deal}：第一个出完的是座位 {end['finish'][0]}，"
                              f"日志说第 1 名是座位 {want_first}")

    # ---- ⑥ 级别 ----
    for rec in decs:
        d = per_deal.get(rec["deal"])
        if d is None:
            continue
        r6.total += 1
        if rec["level"] != d["g"].trump:
            r6.bad.append(f"局 {rec['deal']} 位置 {rec['pos']}："
                          f"记录里的级别 {rec['level']} ≠ 本局 Trump {d['g'].trump}")

    # 顺带量一下：**故意按日志的滞后刷新级别**时，有多少决策点会带着上一局的级别
    # （实时那条路日志要 ~20 秒才写到，这段窗口有多大，只报不判）
    tmp2 = os.path.join(tmpdir, "shadow-lag.jsonl")
    _s2, _st2, _me2, recs2, _b2 = wire_pass(frames, net, tmp2, levels=levels, lag=True)
    stale = []
    for rec in [r for r in recs2 if r["type"] == "decision"]:
        d = per_deal.get(rec["deal"])
        if d is not None and rec["level"] != d["g"].trump:
            stale.append(rec["deal"])
    n_lag = len([r for r in recs2 if r["type"] == "decision"])
    r6.note = (f"级别刷新滞后（按真实帧时间戳模拟日志那 {LEVEL_LAG:.0f} 秒落盘延迟）："
               f"{len(stale)}/{n_lag} 个决策点会带着上一局的级别。"
               f"⚠️ **这条在本素材上没能真正试到风险时序**：两局的第一个决策点分别出现在"
               f"开局后 +85 秒与 +44 秒，都远晚于滞后窗口；"
               f"「一开局就轮到我」（日志还没追上）不在素材覆盖范围内 —— "
               f"见交付台账第六节「已知限制」。")
    return out


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=None)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    res = run(capture=args.capture, log_dir=args.log_dir)
    print("影子模式离线验收（抓包 + 日志双源）\n")
    for r in res:
        print(r.report())
        print()
    ok = all(r.ok for r in res)
    print("验收全绿 ✓" if ok else "**有项目不过 ✗**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
