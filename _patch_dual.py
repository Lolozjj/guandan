"""给面板加「双分辨率两遍推理」。

背景：用户报「队友打了 5 个 8，只认出 4 个」，实测那第 5 张在 imgsz=960 下
模型完全不输出（降到 0.03 都没有），但 **imgsz=1600 下能认出来（0.31）**。

所以：960 跑一遍（主力，指标最好）+ 1600 跑一遍（补 960 漏掉的难牌），
两遍结果 dedup 合并。实测：
    用户那个 case   队友 4 张 -> 5 张 ✓
    验收真值        召回 100% 不变，误检 1.3% -> 2.6%
    而多出来那 1 个「误报」正好落在 f00433 —— 那是一帧「队友手牌」画面，
    很可能是真牌、只是我的真值不全。所以这个合并基本是白捡的。

代价：单帧 43ms -> 约 150ms，实时面板完全够用。
"""
from pathlib import Path

# --- live/main.py ---
p = Path('live/main.py')
s = p.read_text(encoding='utf-8')

old = """    def _model_run(self, m, names, img):
        # 用「低门槛」跑模型：轮到我时，UI 按钮会盖住左右两家牌的花色符号，
        # 那些牌的置信度会掉到 0.1 左右，用常态门槛根本不会输出。跑完再按
        # tolerate_ui_occlusion 过滤回正常门槛（只对按钮底下那几块放宽）。
        res = m.predict(img, imgsz=self.args.imgsz,
                        conf=min(self.args.conf, CONF_UNDER_UI),
                        verbose=False)[0]
        raw = dedup([(names[int(b.cls)],
                      float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                      float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                      float(b.conf)) for b in res.boxes])"""
new = """    @staticmethod
    def _predict(m, names, img, imgsz, conf):
        res = m.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        return [(names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf)) for b in res.boxes]

    def _model_run(self, m, names, img):
        # 第一遍用「低门槛」跑主力分辨率：轮到我时 UI 按钮会盖住左右两家牌的
        # 花色符号，那些牌置信度掉到 0.1 左右，用常态门槛根本不会输出。
        # 跑完再按 tolerate_occlusion 过滤回正常门槛。
        raw = dedup(self._predict(m, names, img, self.args.imgsz,
                                  min(self.args.conf, CONF_UNDER_UI)))
        # 第二遍高分辨率：补第一遍完全认不出的难牌（实测 5 张 8 只能认出 4 张、
        # 但 1600 下第 5 张能出来）。两遍合并后再统一过滤。
        if self.args.imgsz2 and self.args.imgsz2 != self.args.imgsz:
            raw = dedup(raw + self._predict(m, names, img, self.args.imgsz2,
                                            self.args.imgsz2_conf))"""
assert old in s
s = s.replace(old, new)

old = """    ap.add_argument("--imgsz", type=int, default=960)"""
new = """    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--imgsz2", type=int, default=1600,
                    help="第二遍（高分辨率）补漏用的尺寸；0 = 关掉")
    ap.add_argument("--imgsz2-conf", type=float, default=0.30,
                    help="第二遍的置信度门槛")"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('live/main.py 已加双分辨率')

# --- eval_table.py ---
p = Path('eval_table.py')
s = p.read_text(encoding='utf-8')
old = """    def f(img):
        # 和面板保持一致：低门槛跑，再按 UI 遮挡规则过滤
        res = model.predict(img, imgsz=args.imgsz,
                            conf=min(args.conf, CONF_UNDER_UI), verbose=False)[0]
        raw = [(model.names[int(b.cls)],
                float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                float(b.conf)) for b in res.boxes]"""
new = """    def _predict(img, imgsz, conf):
        res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        return [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf)) for b in res.boxes]

    def f(img):
        # 和面板保持一致：低门槛跑 + 高分辨率补一遍，再按遮挡规则过滤
        raw = dedup(_predict(img, args.imgsz, min(args.conf, CONF_UNDER_UI)))
        if args.imgsz2 and args.imgsz2 != args.imgsz:
            raw = dedup(raw + _predict(img, args.imgsz2, args.imgsz2_conf))"""
assert old in s
s = s.replace(old, new)

old = """    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="输出标注图")"""
new = """    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz2", type=int, default=1600,
                    help="第二遍（高分辨率）补漏用的尺寸；0 = 关掉")
    ap.add_argument("--imgsz2-conf", type=float, default=0.30)
    ap.add_argument("--save", action="store_true", help="输出标注图")"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('eval_table.py 已加双分辨率')
