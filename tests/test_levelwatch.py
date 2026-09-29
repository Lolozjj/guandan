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


def test_the_settlement_no_longer_overrides_a_this_deal_source(tmp_path):
    """⚠️ **这条原来是反的**：以前断言「结算行后出现就算数」。

    2026-09-29 用现场数据推翻：结算行的 `TrumpValue` **不再等于「下一局打几」**——
    15:03 那条说 16，紧接着 15:42 的发牌行却是 `Trump=11`。
    而且「后出现的算」这条规则本身让级别每局一结束就被顶错（用户报的那次：
    面板拿 13 去打了真实级别 6 的一局）。

    ⇒ 期望反过来：**本局来源（发牌行 / 界面自报）优先，结算行只做兜底。**
    """
    d = _log(tmp_path, DEAL, SETTLE)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 9, "发牌行是本局级别，不该被后面那条结算行顶掉"


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


# ------------------------------------------------ 2026-09-29：又换了一次来源

#: 游戏界面自己打印的（`GDTableView`）——**本局级别**，每局开头连着 3 条。
#: 17:11:07 的真实行：面板当时显示「打K」，而游戏里是 6。
UI = ('2026-09-29|17:11:07:051|INFO|G|GameLogger|520|520|3222027089|'
      'GDTableView selfTrump = 6 otherTrump = 6')

#: 今天那条结算行（17:14:54，**那一局结束时**的）。它的 TrumpValue=10 ——
#: 而同一段日志里 15:03 的结算说 TrumpValue=16、紧接着 15:42 的发牌行却是 Trump=11
#: ⇒ **`TrumpValue` 不再等于「下一局打几」**（老结论是拿已删掉的发牌行验的）。
SETTLE_TODAY = ('2026-09-29|17:14:54:269|INFO|G|GameLogger|520|520|3222027089|'
                'EVA1B001结算协议 = {"Result":0,"Rank":[1,3,2,4],'
                '"UpgradeInfo":{"trump":[10,6,10,6],"Upgrade":0,"TrumpOwner":0,'
                '"TrumpValue":10,"RoundCount":1,"UsedSeriesTime":300}}')


def test_the_ui_self_report_gives_this_deals_level(tmp_path):
    """**用户 2026-09-29 报的**：面板显示「打K」，游戏里其实是打 6。

    `GDTableView selfTrump = N` 是游戏界面**自己打印**的当前级别 ——
    每局开头连着 3 条（比发牌行还全：今天 17 点那个文件里发牌行 0 条、它 3 条）。
    两处独立交叉验证：17:11 这条 =6 与用户截图「我方 6」一致；
    15:42 那条 =11 与同局的发牌行 `Trump=11` 一致。
    """
    d = _log(tmp_path, UI)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 6
    assert w.level_src == "界面自报（本局）"


def test_a_this_deal_source_beats_a_later_settlement_line(tmp_path):
    """⚠️ **这条是今天那个 bug 的回归测试。**

    原来是「两条都在时**后出现的那条**算」，而结算行**总在**发牌行之后 ——
    于是每局一结束，级别就被那条不可靠的结算行顶掉，一直错到下一局开头。
    今天的现场：面板拿 15:45 那条结算的 13 去打了 17:11 那局（真实级别 6）。

    ⇒ 改成本局来源（界面自报 / 发牌行）**永远优先**，结算行只做最后兜底。
    """
    d = _log(tmp_path, UI, SETTLE_TODAY)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 6, "被后面那条结算行顶掉了 —— 就是今天那个 bug"


def test_the_old_deal_line_also_beats_a_later_settlement(tmp_path):
    """发牌行同样是**本局**来源（15:42 实测与界面自报互相印证），所以它也优先。"""
    d = _log(tmp_path, DEAL, SETTLE)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 9, "发牌行的 Trump 是本局级别，不该被结算行顶掉"


def test_the_settlement_still_fills_in_when_no_this_deal_source_exists(tmp_path):
    """兜底不能丢：一个本局来源都没见过时（万一游戏又改回去），结算行还得能用。"""
    d = _log(tmp_path, SETTLE)
    w = LevelWatcher(log_dir=d)
    assert w.poll() == 11
    assert w.level_src == "结算行（下一局）"
