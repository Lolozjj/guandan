"""live/rules.py 适配层的测试（Task 8）。

分四块：

1. **生产词表** —— 输入直接取自真实生产者（`synth/layout.py` 的 `CLASSES`
   与 `live/level.py` 的 `'T'`）。**这是本文件最重要的一块**：适配层就是
   「唯一的名字边界」，边界两侧对不上，整套东西等于没验过。第一版适配层
   只认 `"S10"` / `"JOKER_SMALL"`（自己编的词表），实测 1740 手真实着法里
   29% 抛错 —— 而 `live/main.py` 的 `render_lines` 不在 `tick` 的 try 里，
   抛错会**永久打断面板刷新链**。
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
