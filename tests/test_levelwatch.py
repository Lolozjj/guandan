"""级别来源（游戏 3.2.2 之后的现实）。

老格式里级别来自发牌行 `SendCardsService set roundID:… k : {"Trump":N,…}`，
**3.2.2 把那条整个删了**（2026-09-26 实测：`SendCardsService` 命中 0 次），
于是面板一个决策点都算不出来（全以「级别未知」跳过）。

新来源是**结算行**（还在）里的 `UpgradeInfo`。实测语义（拿 34 局老格式样本对，
那时本局 Trump 是确定的）：

    UpgradeInfo.TrumpValue = **赢家升级后、下一局要打的级别**   ← 34/34 对上
    UpgradeInfo.trump[]    = 升级后每座位的级别（两队各一个）
    UpgradeInfo.TrumpOwner = 拥有这个 trump 的座位

⚠️ 它**不是本局级别**（34 局里 0 局相等）—— 这个坑踩过：拿它当本局级别算出来的
建议全是错的（连牌型都判不出来）。
"""
import os

from net.levelwatch import LevelWatcher

#: 今天那一局真实的结算行（截掉无关字段，保留 UpgradeInfo）
SETTLE = ('2026-09-26|06:51:35:064|INFO|G|GameLogger|520|520|3222027089|'
          'EVA1B001结算协议 = {"Result":0,"Rank":[1,2,3,4],"DonerChange":[2000,-2000,2000,-2000],'
          '"UpgradeInfo":{"trump":[11,9,11,9],"Upgrade":2,"TrumpOwner":0,"TrumpValue":11,'
          '"RoundCount":1,"UsedSeriesTime":208}}')

#: 老格式的发牌行（万一游戏回退，这条路要还在）
DEAL = ('2026-09-25|11:11:14:116|INFO|G|GameLogger|520|520|3222027089|'
        'SendCardsService set roundID:9,9,9,9 k : '
        '{"CardLen":27,"Cards":[17,19,281,27,28],"nWhoIsFirstOut":1,"Trump":9}')


def _log(tmp_path, *lines):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = os.path.join(str(tmp_path), "2026-09-26-06.log")
    with open(p, "w", encoding="utf-8") as fh:
        for l in lines:
            fh.write(l + "\n")
    return str(tmp_path)


def test_settle_line_gives_the_next_deals_level(tmp_path):
    """结算行 -> 下一局的级别（TrumpValue），并且要说清来源。"""
    d = _log(tmp_path, SETTLE)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 11, "TrumpValue = 赢家升级后下一局要打的级别"
    assert w.level_src.startswith("结算行"), f"来源要说清：{w.level_src}"
    assert w.last_levels == [11, 9, 11, 9], "顺便记住每座位的级别（两队各一个）"


def test_the_old_deal_line_still_works(tmp_path):
    """老发牌行那条路不许丢 —— 游戏回退时还得能用。"""
    d = _log(tmp_path, DEAL)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 9
    assert w.level_src.startswith("发牌行")


def test_settle_wins_over_a_stale_deal_line(tmp_path):
    """同一个日志里两条都有时，**结算行更新**（它是后写的那条）。"""
    d = _log(tmp_path, DEAL, SETTLE)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 11, "结算行在后 -> 它说的算"


def test_nothing_readable_means_no_level(tmp_path):
    """读不到就是读不到 —— 不许瞎给一个（面板要能据此明说「还没拿到级别」）。"""
    d = _log(tmp_path, "2026-09-26|06:47:21:241|INFO|G|t|520|520|1|ShopDetailsMod init")
    w = LevelWatcher(log_dir=d)
    assert w.poll() is None
    assert w.level is None


def test_it_looks_far_enough_back_on_the_first_read(tmp_path):
    """**首次**读要往前看够远：结算行可能落在 300KB 之外。

    实测踩过：今天那局日志 7 分钟长了 2.4 MB，06:51 的结算行根本不在
    最后 300 KB 里 —— 级别读成 None，于是又是「一局全跳过」。
    """
    d = _log(tmp_path)                       # 先建目录
    p = os.path.join(d, "2026-09-26-06.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(SETTLE + "\n")              # 关键的结算行在最前面
        for i in range(4000):                # 后面灌 4000 行噪声（约 0.5 MB）
            fh.write(f"2026-09-26|06:52:{i % 60:02d}:000|INFO|G|t|520|520|1|"
                     f"噪声噪声噪声噪声噪声噪声噪声噪声 {i}\n")
    assert os.path.getsize(p) > LevelWatcher.TAIL
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 11, "首次读要往前看够远，别只看尾部"


def test_the_deal_line_far_back_still_overrides_a_manual_level(tmp_path):
    """**手输的级别必须被日志里的发牌行覆盖** —— 用户 2026-09-26 就栽在这儿。

    他 `--level 9` 开面板，而那一局日志里的发牌行写着 8；面板的轮询只看日志
    最后 300 KB，发牌行在更前面 → 一直没覆盖 → 那一场 71 个决策点的建议
    全是在错级别下算出来的（级别错 → 逢人配/级牌/候选全错）。
    """
    d = _log(tmp_path)
    p = os.path.join(d, "2026-09-26-07.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(DEAL + "\n")                     # 发牌行（打 9）在最前面
        for i in range(4000):
            fh.write(f"2026-09-26|07:41:{i % 60:02d}:000|INFO|G|t|520|520|1|"
                     f"噪声噪声噪声噪声噪声噪声噪声噪声 {i}\n")
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 9, "发牌行在 300KB 之外，也必须读到（否则手输的级别永远不被覆盖）"
    assert w.level_src.startswith("发牌行")
