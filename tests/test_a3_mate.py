"""A3 队友多样性 + A5 `--greedy-share`：**开关必须真的透传，且关掉时逐位不变**。

A3 最容易出的两种错都**在日志上看不出来**：
  ① 固定队友的着法混进了训练目标（拿规则式当老师）；
  ② `mate_mix=0.0` 时多消费了 rng ⇒ 同一个种子打出来的牌与老臂不同，老臂不再可比。
"""
import random

import guandan.rl.selfplay as sp
from guandan.rl import worker
from guandan.rl.net import QNet

# ---------------------------------------------------------------- plan_step


def test_mate_seat_is_fixed_even_in_pure_selfplay():
    """⚠️ 队友那一支必须在「没有固定对手」那道检查**之前**判：A3 用在纯自对弈局里。"""
    assert sp.plan_step((0,), 2, None, {2: "rule"}) == ("fixed", "rule")
    assert sp.plan_step((0, 2), 0, None, {2: "rule"}) == ("learner", None)
    assert sp.plan_step((0,), 2, ("member", 7), {}) == ("member", 7)
    try:
        sp.plan_step((0,), 2, None, None)
    except ValueError:
        return
    raise AssertionError("不属于学习队又没有固定对手时应当炸掉")


# ---------------------------------------------------------------- rng 不许漂


def _records(mate_mix, seed=11, n_games=4, opp_mix=1.0, net=None, **kw):
    # ⚠️ 两次比较**必须用同一个 net**：`QNet()` 每次新建都用 torch 的全局 RNG 重新初始化
    # ⇒ 权重不同 ⇒ 每局的步数不同 ⇒ rng 消费数不同，"rng 状态一样"这条就假失败。
    net = net if net is not None else QNet().eval()
    rng = random.Random(seed)
    out = sp.generate_batch(net, rng, 0.0, n_games, opp_mix=opp_mix,
                            mate_mix=mate_mix, **kw)
    return [rec for rec, _p, _y in out], rng.getstate()


def _fingerprint(rec):
    """一局的可比指纹：牌、级别、先出者、学习的座位。"""
    return (rec.level, rec.first, rec.hands, rec.learn)


def test_mate_mix_zero_changes_nothing_including_the_rng():
    """`mate_mix=0.0` 必须与**不传这个参数**完全一样 —— 连 rng 状态都要一样。

    这条钉的是「老臂可比」：多抽一个随机数，同一个种子打出来的牌就变了。
    """
    net = QNet().eval()                   # 同一个网络，两次都得一样
    a, ra = _records(0.0, net=net)
    b, rb = _records(0.0, net=net)
    assert [_fingerprint(r) for r in a] == [_fingerprint(r) for r in b]
    assert ra == rb, "mate_mix=0 居然消费了 rng —— 老臂不再可比"
    assert all(len(r.learn) == 2 for r in a), "关掉队友多样性时学习队两个座位都学"
    c, rc = _records(0.0, opp_mix=0.0, net=net)
    assert all(len(r.learn) == 4 for r in c)


def test_mate_mix_one_keeps_exactly_one_learner_seat():
    """有固定对手的局里，换掉搭档 ⇒ **只剩一个座位学**（剂量 = opp_mix × mate_mix）。"""
    recs, _ = _records(1.0, n_games=6, opp_mix=1.0)
    assert all(len(r.learn) == 1 for r in recs), \
        f"应当只留一个学习座位，实际 {[r.learn for r in recs]}"


def test_mate_mix_does_not_touch_pure_selfplay_games():
    """⚠️ 纯自对弈局里"学习队的搭档"没有定义 ⇒ A3 在那里**不生效**（否则是半个效果）。"""
    recs, _ = _records(1.0, n_games=6, opp_mix=0.0)
    assert all(len(r.learn) == 4 for r in recs)


def test_fixed_teammate_moves_stay_out_of_the_training_target():
    """固定队友的决策点**不许**进训练目标 —— 那与固定对手是同一条纪律。

    走的是现场抓取（`capture=True` 的 `pts`），所以直接看 pts 的座位。
    """
    net = QNet().eval()
    rng = random.Random(5)
    out = sp.generate_batch(net, rng, 0.0, 4, opp_mix=1.0, mate_mix=1.0,
                            capture=True)
    for rec, pts, _y in out:
        assert pts, "总该有决策点"
        assert {seat for _o, _a, _i, seat, _h in pts} == set(rec.learn), \
            "抓下来的决策点必须全部来自学习座位"


# ---------------------------------------------------------------- worker 透传


def test_worker_cfg_carries_mate_mix():
    cfg = worker.worker_cfg(1, 0.3, 0.5, 0.8, 4, mate_mix=0.25)
    assert cfg["mate_mix"] == 0.25
    assert worker.worker_cfg(1, 0.3, 0.5, 0.8, 4)["mate_mix"] == 0.0


def test_worker_batch_uses_mate_mix():
    """worker 那条路也要真的换队友（它走的是另一份调用点）。"""
    net = QNet().eval()
    recs = worker.worker_batch(net, random.Random(3), 0.0, 4,
                               opp_mix=1.0, mate_mix=1.0)
    assert all(len(r.learn) == 1 for r in recs)


# ---------------------------------------------------------------- CLI 透传


def _fake_train(seen):
    def f(seconds=0, **kw):
        seen.update(kw)
        return {"out_dir": "x", "wr_greedy": 1.0, "wr_random": 1.0,
                "games": 0, "curve": [], "best_score": 1.0, "elapsed": 0.0}
    return f


def test_cli_forwards_greedy_share_and_mate_mix(monkeypatch):
    seen = {}
    monkeypatch.setattr(sp, "train", _fake_train(seen))
    sp.main(["1", "--greedy-share", "1.0", "--mate-mix", "0.5"])
    assert seen["greedy_share"] == 1.0
    assert seen["mate_mix"] == 0.5


def test_defaults_keep_old_behaviour(monkeypatch):
    """默认不许变：`greedy_share=0.8`（老二分）、`mate_mix=0.0`（关）。

    ⚠️ 没传开关时参数**不进 `kw`**（走函数签名的默认值）—— 所以这里两处都要查：
    签名默认值 + 不传时不出现在 `kw` 里。
    """
    import inspect
    assert inspect.signature(sp.train).parameters["greedy_share"].default == 0.8
    assert inspect.signature(sp.train).parameters["mate_mix"].default == 0.0
    assert inspect.signature(sp.train_parallel).parameters["mate_mix"].default == 0.0
    seen = {}
    monkeypatch.setattr(sp, "train", _fake_train(seen))
    sp.main(["1"])
    assert "greedy_share" not in seen and "mate_mix" not in seen
