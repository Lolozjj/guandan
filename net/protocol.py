"""掼蛋 wss 协议解码 —— 只解「服务器 → 客户端」这个方向。

为什么只解这个方向：实测（字节熵判据，见项目记忆 `guandan-websocket-verified`）
**服务器下发是明文 protobuf，客户端上行是密文**。而我们需要的牌局状态
（我的手牌、谁出了什么、轮到谁）全都是服务器下发的那一半。

报文外形：

    [2B 大端帧长][2B 保留][6B ff…fd][4B 魔数 c0 0c 3b 51][…帧头…][protobuf 载荷]

帧头长度**随消息类型变化**（KeepAlive 34B、游戏消息 ~131B），所以载荷起点不写死，
靠「从这里能不能解出一条合法的游戏消息」来定。

载荷起点怎么定得稳：字段 1 是游戏自己的 msgid（3000~3010 这一段），
日志里会把每个 msgid 打印出来（`DispatchGameSoMsg Type = 3005`），
所以消息目录不用逆 —— 见 MSG_NAMES。
"""

import struct

from . import cards

MAGIC = bytes.fromhex("c00c3b51")
MSGID_MIN, MSGID_MAX = 3000, 3099
MAX_HEADER_SCAN = 240        # 载荷起点最多往后找这么多字节

# 与游戏日志里 `DispatchGameSoMsg Type = N` / `QueueCmdManager push cmd=N` 对照得到。
# 只列已经确认或高度疑似的；其余等离线核对后补。
MSG_NAMES = {
    3003: "未知3003",
    3004: "同步/回放信息",
    3005: "出牌",
    3006: "要不起",              # 3.7.2=座位(缺省即0) 3.7.3=下一手
    3009: "发牌/换牌",
    3019: "手牌/状态同步",       # 周期性重发，内含我的手牌（27 张）与各家牌数组
}


# ------------------------------------------------------------------ 变长整数

def varint(buf, i):
    """读一个变长整数，返回 (值, 新位置)；读不出来返回 (None, i)。"""
    val, shift = 0, 0
    while i < len(buf) and shift < 64:
        b = buf[i]
        val |= (b & 0x7F) << shift
        i += 1
        if not b & 0x80:
            return val, i
        shift += 7
    return None, i


def declared_length(data):
    """帧头前 2 字节是大端帧长，用来自检帧有没有截断。"""
    return int.from_bytes(data[0:2], "big") if len(data) >= 2 else None


# ------------------------------------------------------------------ 结构遍历

def _as_packed(sub):
    """这段字节是不是「整组都是牌的数组」。

    只有整组都通过 is_card 才认 —— 这样既能把牌数组挑出来，
    又不会把嵌套消息（08 1b 12 26 …）误判成数组。
    """
    i, vals = 0, []
    while i < len(sub):
        v, j = varint(sub, i)
        if v is None or j <= i:
            return None
        vals.append(v)
        i = j
    if vals and all(cards.is_card(v) for v in vals):
        return tuple(vals)
    return None


def _as_text(sub):
    """整段基本都是可打印字符 -> 当字符串。"""
    if not sub:
        return None
    ok = sum(1 for b in sub if 32 <= b < 127 or b in (9, 10, 13))
    return sub.decode("utf-8", "replace") if ok / len(sub) >= 0.9 else None


def walk(data, start, end=None, path=(), depth=0, out=None):
    """把 data[start:end] 当 protobuf 走一遍。

    返回 [(类型, 字段路径, 值)]，类型是 int / arr / str / dbl / f32。
    碰到读不通的地方就停下并返回已有结果 —— 帧被截断时也能用。
    """
    if out is None:
        out = []
    if end is None:
        end = len(data)
    if depth > 8:
        return out
    i = start
    while i < end:
        key, i2 = varint(data, i)
        if key is None:
            return out
        field, wt = key >> 3, key & 7
        if field == 0:
            return out
        p = path + (field,)
        if wt == 0:
            v, i3 = varint(data, i2)
            if v is None:
                return out
            out.append(("int", p, v))
            i = i3
        elif wt == 1:
            if i2 + 8 > end:
                return out
            out.append(("dbl", p, struct.unpack("<d", data[i2:i2 + 8])[0]))
            i = i2 + 8
        elif wt == 2:
            ln, i3 = varint(data, i2)
            if ln is None or i3 + ln > end:
                return out
            sub = data[i3:i3 + ln]
            packed = _as_packed(sub)
            if packed is not None:
                out.append(("arr", p, packed))
            else:
                txt = _as_text(sub)
                if txt is not None:
                    out.append(("str", p, txt))
                else:
                    walk(data, i3, i3 + ln, p, depth + 1, out)
            i = i3 + ln
        elif wt == 5:
            if i2 + 4 > end:
                return out
            out.append(("f32", p, data[i2:i2 + 4]))
            i = i2 + 4
        else:
            return out
    return out


# -------------------------------------------------------------- 载荷起点 / msgid

def candidates(data):
    """扫描出所有「能解出一条合法游戏消息」的载荷起点。

    判据：字段 1 的取值落在 msgid 区间，且从这里能解出至少 2 个字段。
    """
    out = []
    top = min(len(data) - 2, MAX_HEADER_SCAN)
    for s in range(0, max(0, top)):
        key, i = varint(data, s)
        if key != 0x08:                     # 字段 1 + 变长整数
            continue
        v, _ = varint(data, i)
        if v is None or not (MSGID_MIN <= v <= MSGID_MAX):
            continue
        fields = walk(data, s)
        if len(fields) >= 2:
            out.append((s, v, fields))
    return out


def _score(fields):
    """候选起点打分。

    帧头长度会变，错误的起点也能"解析成功"，只是解出来的字段路径漂了、
    拿不到正确的同级字段。所以按**能不能解出一条有意义的牌局消息**来选，
    这比"解出最多字段"可靠得多。
    """
    if decode_play(fields) is not None:
        return (3, len(fields))
    if decode_hand(fields) is not None:
        return (2, len(fields))
    has_arr = any(k == "arr" for k, _, _ in fields)
    return (1 if has_arr else 0, len(fields))


def parse(data):
    """解一条服务器下发的报文。

    返回 dict：msgid / name / offset / fields / card_arrays / texts，
    解不出来返回 None。
    """
    cands = candidates(data)
    if not cands:
        return None
    s, msgid, fields = max(cands, key=lambda c: (_score(c[2]), -c[0]))
    return {
        "msgid": msgid,
        "name": MSG_NAMES.get(msgid, f"未知{msgid}"),
        "offset": s,
        "fields": fields,
        "card_arrays": [(p, v) for k, p, v in fields if k == "arr"],
        "texts": [v for k, _, v in fields if k == "str"],
    }


# ------------------------------------------------------------------ 具体消息

def _siblings(fields, path, kinds=("int",)):
    """取与 path 同层的字段，返回 {字段号: 值}。path 是元组。"""
    prefix = path[:-1]
    out = {}
    for k, p, v in fields:
        if k in kinds and len(p) == len(prefix) + 1 and p[:-1] == prefix:
            out[p[-1]] = v
    return out


def _field(fields, path):
    """按字段路径取值。"""
    for k, p, v in fields:
        if tuple(p) == path:
            return v
    return None


def decode_play(fields):
    """出牌（msgid 3005）。字段号对照游戏日志的 NotifyGiveCards JSON 得到：

        3.6.2 座位号   3.6.3 下一手轮到谁   3.6.4 牌型   3.6.5 该家剩余张数
        3.6.6 这手几张 3.6.7 出的牌         3.6.8 炸弹倍数 3.6.9 出牌时限
        3.6.10 LeftCardList —— **只有自己出牌时才有值**，见 decode_my_seat

    已用 10 次出牌与日志逐字段核对一致。
    """
    for kind, path, val in fields:
        if kind != "arr" or not (1 <= len(val) <= 12):
            continue
        sib = _siblings(fields, path)
        if sib.get(6) != len(val):
            continue
        # protobuf 会省略值为 0 的字段 —— 座位 0 的出牌不会写 seat 字段，
        # 所以「字段不存在」必须当成 0，不能当成「解不出来」。
        seat = sib.get(2, 0)
        if not (0 <= seat <= 3):
            continue
        return {
            "seat": seat,
            "next": sib.get(3, 0),
            "card_type": sib.get(4, 0),
            "left": sib.get(5, 0),
            "count": len(val),
            "cards": list(val),
            "bomb": sib.get(8, 0),
            "seconds": sib.get(9, 0),
            "left_cards": _field(fields, (3, 6, 10)),
        }
    return None


def decode_my_seat(fields):
    """从出牌报文里认出「我」是哪个座位。

    **`3.6.10`（LeftCardList）只有自己出牌时才有值** —— 服务器只给自己的
    出牌附上「你还剩哪些牌」。实测：14:05 那局只有 seat1 的报文带这个字段，
    而且那 76 张 100% 落在我的起手牌里。

    这一点很关键：**报文里的座位号每局都不一样**（同一台机器，16:01 那局我是
    seat3、16:33 那局我是 seat1），写死任何值都会错。所以座位必须运行时认。

    返回 (座位, 剩余手牌) 或 (None, None)。
    """
    for kind, path, val in fields:
        if kind != "arr" or not (1 <= len(val) <= 27):
            continue
        sib = _siblings(fields, path)
        if sib.get(6) != len(val):          # CardLen 要对得上，确认是 CardList
            continue
        left = _field(fields, (3, 6, 10))
        if left:
            return sib.get(2, 0), list(left)
    return None, None


def decode_pass(fields):
    """要不起（msgid 3006）。

    结构对照游戏日志的 `OrUpService 其他玩家要不起, 其SeatID = N` 得到：

        3.7.2 = 要不起的座位（**protobuf 省略 0**，所以缺省就是座位 0）
        3.7.3 = 下一个该行动的人  <-- 服务器直接给了，不用自己推
        3.7.4 / 3.7.5 = 倒计时之类

    第一版把 3006 当成「牌局状态」、把 3003 当成要不起，结果要不起的消息
    根本没进状态机，轮次一直推不动（预测正确率只有 54%）。
    """
    seat = nxt = None
    seen = False
    for kind, path, val in fields:
        if kind != "int" or len(path) != 3 or path[:2] != (3, 7):
            continue
        seen = True
        if path[2] == 2:
            seat = val
        elif path[2] == 3:
            nxt = val
    if not seen:
        return None
    # 两个字段都可能因为「值是 0」而被省略 —— 缺省就是 0，不是"没有"。
    # 实测：14:06:44 要不起 seat3 且 3.7.3 缺省，紧接着 14:06:45 就是 seat0 出牌。
    return {"seat": seat if seat is not None else 0,
            "next": nxt if nxt is not None else 0}


def decode_hand(fields, low=18, high=27):
    """手牌。

    来自 msgid 3019（周期性重发，约每 40 秒一次），结构是
    `3.26.3.2.1 = 张数` + `3.26.3.2.2 = 牌数组`。
    这里不写死字段路径 —— 帧头长度会变、路径会跟着漂，
    只按「整组都是牌 + 张数在 18~27」来认，已经和日志的真值核对一致。

    周期性重发这点很重要：**任何时候接进来都能拿到当前手牌**，
    不需要等在发牌那一刻。
    """
    for kind, path, val in fields:
        if kind != "arr" or not (low <= len(val) <= high):
            continue
        sib = _siblings(fields, path)
        if len(val) in sib.values() or not sib:
            return {"count": len(val), "cards": list(val)}
    return None
