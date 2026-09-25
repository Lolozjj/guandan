"""进贡记录的解析测试。**用真日志跑**，但日志没了要明着说，不能静默返回空列表。"""
import os

import pytest

from tools import game_log


@pytest.mark.skipif(not os.path.isdir(game_log.LOG_DIR),
                    reason="实时日志已被轮转删除 —— 这不是失败，是跳过的理由")
def test_tribute_records_parse_and_every_return_is_at_most_10():
    """**25/25 条硬证据（spec §13.6）**：还贡的牌点数全部落在 2..10。

    A=1 不在里面 —— `idx <= 10` 会把 A 也算成「≤10」，那正是最难还出去的牌。
    """
    from net import cards
    recs = game_log.load_tributes()
    assert len(recs) >= 10, f"只解析出 {len(recs)} 条进贡记录，太少了（格式变了？）"
    bad = [(r.giver, cards.decode(r.card)) for r in recs
           if r.kind == "return" and not 2 <= cards.parts(r.card)[0] <= 10]
    assert not bad, f"这些还贡牌点数不在 2..10：{bad}"


@pytest.mark.skipif(not os.path.isdir(game_log.LOG_DIR), reason="日志不在")
def test_load_tributes_raises_when_the_directory_is_gone():
    """目录不在必须**炸**，不许返回空列表 —— 空列表会让验收「0 项全过」。"""
    with pytest.raises(FileNotFoundError):
        game_log.load_tributes(log_dir=os.path.join(game_log.LOG_DIR, "不存在"))
