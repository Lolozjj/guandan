"""给 phase.py 补一个「亮手牌」状态。

用户实测报的 bug：一局打完，游戏会在中间亮出「队友手牌」（或「对手手牌」），
面板把这些牌当成了「我出的牌」。337 帧真实素材里 **73 帧（21.7%）** 是这个状态 ——
比另外三种换局状态加起来还多，原来完全没覆盖。

判据：画面中部 y≈760~900 处有「手牌」两个字的金色标题。
模板是「队友手牌」四个字里**右边两个字**（`assets/reveal_text.png` 的右半），
这样「队友手牌 / 对手手牌」都能匹配上。

分离度（实测）：正例 0.885~1.000，正常帧 0.315~0.351。
"""
from pathlib import Path

p = Path('live/phase.py')
s = p.read_text(encoding='utf-8')

old = """BADGE_PATH = ROOT / "assets" / "gong_badge.png\""""
new = """BADGE_PATH = ROOT / "assets" / "gong_badge.png"
REVEAL_PATH = ROOT / "assets" / "reveal_text.png\""""
assert old in s
s = s.replace(old, new)

old = """_badge: np.ndarray | None = None"""
new = """# 亮手牌：「队友手牌 / 对手手牌」的标题
REVEAL_BAND = (760, 900)        # y 范围（画面宽 1698 时）
REVEAL_THR = 0.75               # 实测正例 0.885~1.0 / 正常帧 0.315~0.351

_badge: np.ndarray | None = None
_reveal: np.ndarray | None = None"""
assert old in s
s = s.replace(old, new)

old = """def detect_phase(img: np.ndarray) -> tuple[str, str]:"""
new = """def _get_reveal() -> np.ndarray | None:
    \"\"\"亮手牌的标题模板 —— 只取右半边的「手牌」两个字，兼容队友/对手。\"\"\"
    global _reveal
    if _reveal is None:
        img = cv2.imread(str(REVEAL_PATH))
        _reveal = img[:, img.shape[1] // 2:] if img is not None \\
            else np.zeros((1, 1, 3), np.uint8)
    return _reveal if _reveal.size > 3 else None


def detect_phase(img: np.ndarray) -> tuple[str, str]:"""
assert old in s
s = s.replace(old, new)

old = """    # 3) 进贡：红色「贡」小标记"""
new = """    # 3) 亮手牌：队友/对手剩下的牌被亮出来（一局结束后）
    tpl = _get_reveal()
    if tpl is not None:
        sc = img.shape[1] / 1698.0
        y1, y2 = int(REVEAL_BAND[0] * sc), int(REVEAL_BAND[1] * sc)
        band = img[y1:y2, :]
        if band.shape[0] >= tpl.shape[0] and band.shape[1] >= tpl.shape[1]:
            r = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
            if float(r.max()) > REVEAL_THR:
                return "reveal", "亮手牌（队友/对手剩下的牌）"

    # 4) 进贡：红色「贡」小标记"""
assert old in s
s = s.replace(old, new)
Path('live/phase.py').write_text(s, encoding='utf-8')
print('phase.py 已加 reveal 状态')
