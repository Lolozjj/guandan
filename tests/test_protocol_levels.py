"""级别（打几）在网络里的位置 —— 用真实抓包的一帧当 fixture。

为什么单独立一个文件：这条以前被判定为「网络里确实没有」，是因为旧抓包把每帧截到
256 字节，而这条报文 2906 字节。fixture 是那一帧的**完整**十六进制，
所以这条测试同时守着「别再被截断」这件事。
"""
from __future__ import annotations

import os

from net import protocol

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fixtures", "msg3008_levels.hex")


def _frame():
    with open(FIX, encoding="utf-8") as fh:
        return bytes.fromhex(fh.read().strip())


def test_fixture_is_a_whole_frame():
    """fixture 必须是完整帧 —— 截断了这条测试就没意义了。

    判据用协议自己的帧长字段（前 2 字节大端 == 实际长度），不靠人记。
    """
    body = _frame()
    assert protocol.declared_length(body) == len(body)


def test_decodes_as_msgid_3008():
    m = protocol.parse(_frame())
    assert m is not None and m["msgid"] == 3008


def test_levels_are_four_seat_values():
    """4 个座位的级别，取值都是合法级别（A=1 … K=13）。"""
    m = protocol.parse(_frame())
    lv = protocol.decode_levels(m["fields"])
    assert lv is not None and len(lv) == 4
    assert all(1 <= v <= 13 for v in lv), lv
    # 实测那局：两队各一个级别，座位 0/2 与 1/3 分别相同
    assert lv[0] == lv[2] and lv[1] == lv[3], lv
    assert set(lv) == {9, 10}, lv      # 用户当时打的两把就是 9 和 10


def test_other_messages_yield_none():
    """不是 3008 的消息必须返回 None，不能瞎给一个级别。"""
    assert protocol.decode_levels([]) is None
    assert protocol.decode_levels([("int", (3, 9, 22, 12), 9)]) is None   # 只有 1 个
