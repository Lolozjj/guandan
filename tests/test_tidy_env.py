"""`GUANDAN_TIDY` 的完整串解析（`margin:0.25,leads=1,coop=1`）—— 面板就靠它。"""
import os


def test_full_switch_string():
    from guandan.advice.advise import tidy_flags, tidy_margin, tidy_mode
    old = os.environ.get("GUANDAN_TIDY")
    try:
        os.environ["GUANDAN_TIDY"] = "margin:0.25,leads=1,coop=1"
        assert tidy_mode() == "margin:0.25,leads=1,coop=1"
        assert tidy_margin() == 0.25                    # 逗号前那一段
        assert tidy_flags() == {"leads": True, "coop": True}
        os.environ["GUANDAN_TIDY"] = "margin:0.15"
        assert tidy_margin() == 0.15 and tidy_flags() == {"leads": False, "coop": False}
        os.environ["GUANDAN_TIDY"] = "1"
        assert tidy_margin() == 0.0 and tidy_flags() == {"leads": False, "coop": False}
    finally:
        if old is None:
            os.environ.pop("GUANDAN_TIDY", None)
        else:
            os.environ["GUANDAN_TIDY"] = old
