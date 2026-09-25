"""live/rules.py 适配层的测试（Task 8）。

分四块：

1. **生产词表** —— 输入直接取自真实生产者（`synth/layout.py` 的 `CLASSES`
   与 `live/level.py` 的 `'T'`）。**这是本文件最重要的一块**：适配层就是
   「唯一的名字边界」，边界两侧对不上，整套东西等于没验过。第一版适配层
   只认 `"S10"` / `"JOKER_SMALL"`（自己编的词表），实测 1740 手真实着法里
   29% 抛错 —— 而 `live/main.py` 的 `render_lines` 那时不在 `tick` 的 try 里，
   抛错会**永久打断面板刷新链**（该机制现已由 `live/main.py` 的
   `safe_render_lines` 修掉，见本文件第 5 块）。
2. **6 个已证实缺陷**（brief 的那 6 条，用长写法 `"S10"` 写；两种写法都要收）。
3. **旧实现做对、不能被弄丢的行为**（同花顺不能降级成顺子）。
4. **编码与报错契约**（逐张对表、认不出必须抛 ValueError）。
"""
import pytest

from live import rules
from net import cards
from net.sim import meld


# =========================================================== 1. 生产词表
# 生产链路：live/main.py:36 `from layout import CLASSES`（= ROOT/synth/layout.py，
# 由 main.py 的 sys.path.insert(str(ROOT / "synth")) 接上），
# 级别来自 live/level.py（_normalize 把 "10"/"1" 都规成 "T"）。


def test_production_vocabulary_is_understood():
    """**输入取自真实生产者**，不是我们手写的字符串。

    `synth.layout.CLASSES` 就是模型输出的 54 个类名，`live/main.py:229` 把它们
    原样喂给 `classify_play`。所以：十是 `ST`（不是 `S10`）、王是 `JOKER_S` /
    `JOKER_B`（不是长写法）、打十的级别是 `'T'`（不是 `'10'`）。

    这条测试在**第一版适配层**（只认长写法）下会红 —— 54 类里 34 类直接抛错。
    没有它，适配层的全部意义（唯一的名字边界）就等于没验过。
    """
    from synth.layout import CLASSES

    # 54 类逐类都必须能转成一张合法牌，一条都不许抛错
    bad = [c for c in CLASSES if not cards.is_card(meld.cid_from_name(c))]
    assert bad == []
    assert len(CLASSES) == 54

    # 十（T）、王、级别 T：面板真正会遇到的三种输入
    assert rules.classify(["ST", "HT", "DT"], "2") == "三张"
    assert rules.classify(["JOKER_S", "JOKER_S"], "2") == "对子"
    assert rules.classify(["JOKER_B", "JOKER_S", "JOKER_B", "JOKER_S"], "2") == "四大天王"
    assert rules.classify(["C9", "CT", "CJ", "CQ", "CK"], "8") == "同花顺"
    assert rules.classify(["H2", "H2", "S2", "HT"], "T") == "4 张炸（含逢人配）"


def test_all_54_classes_parse_to_distinct_ids():
    """54 个类名要映射到 54 张不同的牌（不能有两类撞到同一个 ID）。"""
    from synth.layout import CLASSES

    ids = [meld.cid_from_name(c) for c in CLASSES]
    assert len(set(ids)) == 54


def test_no_deck_suffix_in_production_vocabulary():
    """生产词表**不带副数**：模型分不出两副牌，所以适配层也不认 `(二副)`
    （`net/cards.decode` 的显示后缀是另一套格式，见 live/rules.py 的 docstring）。

    喂进来要**明着抛错**，不许悄悄当成第一副 —— 否则「格式不对」会被洗成
    一个看着正常的牌型结论。
    """
    with pytest.raises(ValueError):
        rules.classify(["ST(二副)", "HT", "DT"], "2")


# ======================================================= 2. 6 个已证实缺陷
# 用 brief 的长写法（"S10" / "JOKER_SMALL"）写；生产写法在上面的测试里。
# 说明：旧实现下真会红的只有 4 条（ace-low / joker-pair / 三带二带王 / 二连对），
# `test_ten_is_handled`、`test_illegal_returns_none`、`test_describe_falls_back`
# 旧实现本来就过 —— 它们是回归网，**不是缺陷证据**（见 task-8-report §2）。


def test_ten_is_handled():
    """bug #1（`"10"` vs `"T"`）。"""
    assert rules.classify(["S10", "H10", "D10"], "2") == "三张"
    assert rules.classify(["ST", "HT", "DT"], "2") == "三张"


def test_ace_low_straight():
    """旧实现判非法（bug #2）。"""
    assert rules.classify(["SA", "H2", "D3", "C4", "S5"], "9") == "顺子"


def test_two_small_jokers_are_a_pair():
    """旧实现说「两个王不是对子」（bug #3）。"""
    got = rules.classify(["JOKER_SMALL", "JOKER_SMALL"], "9")
    assert got is not None and "对" in got


def test_three_twos_with_two_jokers_is_triple_pair():
    """bug #3 的真实着法那一半：三个 2 + 两张小王是**三带二**（旧实现判 None）。

    王不能凑三张，但**能当三带二里的对子**（net/sim/meld.py 的 `pairs` 不排除王）。
    """
    assert rules.classify(
        ["S2", "D2", "C2", "JOKER_SMALL", "JOKER_SMALL"], "9") == "三带二"


def test_two_pair_run_is_illegal():
    """旧实现把 4 张二连对判合法（bug #4），游戏里 4 张只有炸弹。"""
    assert rules.classify(["S4", "H4", "S5", "H5"], "2") is None


def test_illegal_returns_none():
    assert rules.classify(["S5", "H7", "D9"], "2") is None


def test_describe_falls_back():
    assert rules.describe(["S5", "H7", "D9"], "2") == "不合法"


# ================================================ 3. 旧实现做对、别弄丢的
# 「同花顺优先于顺子」这一条旧实现是对的（`return "同花顺" if len(set(suits)) == 1`），
# 换适配层不能把它降级 —— `melds_from` 对同一组牌会给出顺子与同花顺两条解释，
# 而顺子在枚举顺序里靠前，取第一条就会降级。


def test_wild_makes_four_bomb():
    """真实着法：2♥2♥2♠10♥，打 10，是 4 张炸（不是三张、不是连对）。"""
    assert rules.classify(["S2", "H2", "D2", "H10"], "10") == "4 张炸（含逢人配）"


def test_wild_straight_flush_beats_plain_straight():
    """真实着法：9♣10♣J♣Q♣+♥2（打 2）是同花顺 —— 同一组牌也能解释成顺子，
    必须取更强的那条（同花顺是炸弹，顺子不是，面板和比大小都靠它）。"""
    got = rules.classify(["C9", "C10", "CJ", "CQ", "H2"], "2")
    assert got == "同花顺（含逢人配）"


def test_natural_straight_flush_not_downgraded():
    assert rules.classify(["S5", "S6", "S7", "S8", "S9"], "2") == "同花顺"


def test_pair_run_keeps_old_display_name():
    """3 个连续对子必须还是叫「三连对」——live/main.py:230 把牌型名原样打在面板上，
    改成「连对」会让面板文字在用户眼皮底下变（老实现返回的就是「三连对」）。"""
    assert rules.classify(["S5", "H5", "S6", "H6", "S7", "H7"], "2") == "三连对"
    # 老实现那个 4 张的「连对」真的不存在（bug #4），只剩炸弹
    assert rules.classify(["S4", "H4", "S5", "H5"], "2") is None


# ==================================== 4. 编码 / 报错契约
# 硬约束：不许静默返回 None（那会把「认不出」冒充成「不合法」），
# 也不许把认不出的名字当某一张牌。


@pytest.mark.parametrize("name", ["SX", "X10", "S", "", "JOKER_MIDDLE", "3S", "10S",
                                  "S1", "S11", "S14", "S37", "JOKER_SMALLL"])
def test_unknown_card_name_raises(name):
    with pytest.raises(ValueError):
        meld.cid_from_name(name)


def test_unknown_level_raises():
    # 'T' 是**合法**级别（打十），所以这里用一个真认不出的
    with pytest.raises(ValueError):
        rules.classify(["S5"], "X")
    with pytest.raises(ValueError):
        rules.classify(["S5"], "TEN")


def test_level_idx_accepts_only_real_levels():
    """正例：1~13 的数值形式与 'A'/'T'/'J'/'Q'/'K'/'10' 都能转。"""
    for i in range(1, 14):
        assert meld.level_idx(str(i)) == i          # 字符串形式（日志口径）
        assert meld.level_idx(i) == i               # 整数形式
    assert meld.level_idx("T") == meld.level_idx("10") == 10
    assert meld.level_idx("A") == 1
    assert meld.level_idx(None) is None             # None 只表示「没有级牌」


@pytest.mark.parametrize("bad", ["0", "00", "99", "14", "15", 0, 14, 99, -1])
def test_level_out_of_range_raises(bad):
    """数值不是 1~13 的级别必须**抛** —— 不许 `isdigit()` 放过去。

    旧实现 `if text.isdigit(): return int(text)` 会让 `'0'` / `'99'` 原样通过，
    而它的 docstring 写着「认不出的级别抛 ValueError —— 不返回 None 蒙混过去」：
    实现与自己的承诺相反。

    这不是理论问题：**live/level.py 的字形表里有 '0'**（打十时面板上是「1」+「0」
    两个字形，`_normalize` 只把 '1'/'10' 折成 'T'）—— 打十被切坏、只剩「0」
    就会读出 `'0'`，然后 `0` 一路传进引擎，把**合法着法判成「不合法」**，
    正是 spec §6⑥ 要防的那种「错结论端到用户脸上」。
    实测可达路径：live/main.py:229 `classify_play(cards, d.get("level") or "2")`。
    """
    with pytest.raises(ValueError):
        meld.level_idx(bad)


def test_t_and_10_are_the_same_ten():
    assert meld.level_idx("T") == meld.level_idx("10") == 10
    assert meld.cid_from_name("ST") == meld.cid_from_name("S10")
    assert meld.cid_from_name("HT") == meld.cid_from_name("H10")


def test_unknown_deck_raises():
    with pytest.raises(ValueError):
        meld.cid_from_name("S3", 3)


def test_bad_name_does_not_silently_return_none():
    """认不出的牌名要抛异常，**不能**被 classify 吞成 None。"""
    with pytest.raises(ValueError):
        rules.classify(["S5", "H7", "ZZ9"], "2")


@pytest.mark.parametrize("letter,suit", [("S", "♠"), ("H", "♥"), ("C", "♣"), ("D", "♦")])
@pytest.mark.parametrize("rank,idx", [("A", 1)] + [(str(i), i) for i in range(2, 11)]
                          + [("T", 10), ("J", 11), ("Q", 12), ("K", 13)])
def test_every_card_name_maps_to_its_id(letter, suit, rank, idx):
    """四个花色 x 十三个点数（10 的两种写法都测）逐张对 net/cards.py 的编码。"""
    cid = meld.cid_from_name(letter + rank)
    assert cards.parts(cid) == (idx, suit, 1)
    assert meld.cid_from_name(letter + rank, 2) == cid + 256
    assert cards.parts(meld.cid_from_name(letter + rank, 2)) == (idx, suit, 2)


def test_jokers():
    for short, long_, idx in (("JOKER_S", "JOKER_SMALL", 14),
                              ("JOKER_B", "JOKER_BIG", 15)):
        assert cards.parts(meld.cid_from_name(short)) == (idx, "", 1)
        assert meld.cid_from_name(short) == meld.cid_from_name(long_)
        assert meld.cid_from_name(short, 2) == meld.cid_from_name(long_, 2) == idx + 256


# =============== 5. 真·端到端：面板的渲染入口（live/main.py）
# 「live/main.py 还能用」不能只靠 `import live.main` 证明 —— 出错的地方在
# `render_lines`，它由 tkinter 的 after 回调调用、且**不在** tick 的 try 内，
# 抛错会永久断掉刷新链。所以这里直接调它，输入就是模型类名。


def _panel_input(table, level="T"):
    """live/main.py:render_lines 要的那份字典（字段照 Worker 的产物）。"""
    return {"phase": "normal", "status": "ok", "level": level, "mine": False,
            "turn_side": "left", "hand": [("ST", .9), ("JOKER_S", .9)],
            "table": table, "elapsed": .05, "level_conf": .9, "orange": .0}


def test_panel_render_path_handles_ten_jokers_and_level_T():
    """打十那一局、桌上出现 10 与王 —— 面板必须照常渲染出结论。

    第一版适配层在这里抛 ValueError（词表用的是 `"S10"`），而这条路径当时一抛错
    面板就再也不刷新了（桌上那张牌还是用户自己打出去的）。这条测试就是那个
    Critical 的**触发条件**回归网：它跑的是**真实渲染入口**，不是 `classify` 单点。
    （**机制**那一半——抛错也不许断链——在下面
    `test_panel_render_chain_survives_an_exception` 里。）
    """
    import live.main as panel        # 依赖 cv2/mss/win32，本机有

    def render(table):
        return "".join(t for t, _ in panel.render_lines(_panel_input(table)))

    empty = {"机器人1": [], "机器人3": [], "队友": [], "我": []}
    # 三张 10（打十）：十在词表里是 ST，级别是 T —— 第一版这里就炸了
    assert "三张" in render({**empty, "机器人1": [("ST", .9), ("HT", .9), ("DT", .9)]})
    # 单张王
    assert "单张" in render({**empty, "机器人1": [("JOKER_S", .9)]})
    # 四大天王
    assert "四大天王" in render({**empty, "机器人1": [("JOKER_S", .9), ("JOKER_S", .9),
                                                   ("JOKER_B", .9), ("JOKER_B", .9)]})
    # 三连对：面板文字必须是「三连对」（见 _KIND_NAMES 的说明）
    assert "三连对" in render({**empty, "机器人1": [("S5", .9), ("H5", .9), ("S6", .9),
                                                ("H6", .9), ("S7", .9), ("H7", .9)]})
    # 认不出/不合法的组合走「警告」分支，**不是**抛错（这是 live/main.py 自己的措辞）
    assert "不是合法牌型" in render({**empty, "机器人1": [("S5", .9), ("H7", .9), ("D9", .9)]})
    # 一张小王 + 一张大王不是对子（各有各的那张，配不成对）
    assert "不是合法牌型" in render({**empty, "机器人1": [("JOKER_S", .9), ("JOKER_B", .9)]})


def test_panel_render_chain_survives_an_exception(monkeypatch, capsys):
    """渲染抛错时，刷新链**不能死**，错误还必须**显示出来**（不许吞）。

    `render_lines` 由 tkinter 的 `after` 回调调用，而它当时**不在** `tick` 的 try
    里 —— 抛一次错，`root.after` 就再也不会被排上，面板**永久、静默地**定格。
    上面那条测试修的是**触发条件**（词表对齐），这条守的是**机制**：
    今后任何一次渲染异常都只是显示一行错误，链继续跑。

    断言三件事：safe_render_lines 不把异常放出去、返回的文本里有错误标记、
    stderr 上有完整栈（不吞）。
    """
    import live.main as panel

    def boom(_d):
        raise ValueError("模拟：认不出的牌名/级别")

    monkeypatch.setattr(panel, "render_lines", boom)
    lines = panel.safe_render_lines({"level": "2"})       # 不抛
    text = "".join(t for t, _ in lines)
    assert "面板渲染出错" in text
    assert "模拟：认不出的牌名/级别" in text               # 错误本身要可见
    assert capsys.readouterr().err.count("ValueError") >= 1   # 栈打到 stderr
