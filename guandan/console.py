"""控制台输出的小事 —— 全仓库**只此一份**，工具与面板都从这里拿。

这个模块存在的理由：`♠♥♣♦`、`✓`（U+2713）这些字符不在 GBK（cp936）里，
而 Windows 下 **stdout 一旦被重定向或接管道**（`> out.txt`、`| tee`、CI、后台任务），
Python 就从「Windows 控制台 API（PEP 528，本身 UTF-8）」退回本地编码 cp936，
于是 `print` 直接抛 `UnicodeEncodeError: 'gbk' codec can't encode character`。

这个坑在本仓库踩过两次，两次都很隐蔽：

1. 验收脚本**四项全绿却 exit 1** —— 最后那行「验收全绿 ✓」的 `✓` 编不出来；
   验收脚本报假红比报假绿好不到哪去，而且恰恰在最需要留证据（存日志）的时候犯。
2. 一个调试用的 `print(decode(77))`（打印 `K♦`）让 `import guandan.capture.cards`
   **直接崩** —— 面板连 import 都过不去，而 PyCharm 的控制台是 UTF-8，看不出来。
"""
from __future__ import annotations

import sys


def utf8_stdout() -> None:
    """把 stdout / stderr 切到 UTF-8。**入口的第一件事就该调它。**

    幂等，可以重复调；流不支持 `reconfigure`（pytest 的捕获流、被替换过的流）时跳过。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
