"""验收脚本自己也要有地板 —— `total == 0` 必须算失败（假绿）。"""
from tools import accept_shadow


def test_missing_capture_reports_failure_not_silence(tmp_path):
    res = accept_shadow.run(capture=str(tmp_path / "nope.jsonl"))
    assert res, "找不到素材也必须给出结果，不能返回空"
    assert not any(r.ok for r in res), "缺素材时不许有任何一项报 OK"
