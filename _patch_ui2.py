"""把 UI 遮挡容忍接进面板和验收脚本。

- live/main.py 的 _model_run：按 min(conf, CONF_UNDER_UI) 跑模型，再过滤。
  必须用低门槛跑，否则 0.108 那个框根本不会输出。
- eval_table.py：用同一套逻辑，否则验收测的不是面板实际的行为。
"""
from pathlib import Path

# --- main.py ---
p = Path('live/main.py')
s = p.read_text(encoding='utf-8')
old = """from predict_cards import (card_value, dedup, split_hand_table,  # noqa: E402
                           table_by_player)"""
new = """from predict_cards import (card_value, dedup, split_hand_table,  # noqa: E402
                           table_by_player, tolerate_ui_occlusion,
                           CONF_UNDER_UI)"""
assert old in s
s = s.replace(old, new)

old = """    def _model_run(self, m, names, img):
        res = m.predict(img, imgsz=self.args.imgsz, conf=self.args.conf,
                        verbose=False)[0]
        dets = dedup([(names[int(b.cls)],
                       float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                       float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                       float(b.conf)) for b in res.boxes])"""
new = """    def _model_run(self, m, names, img):
        # 用「低门槛」跑模型：轮到我时，UI 按钮会盖住左右两家牌的花色符号，
        # 那些牌的置信度会掉到 0.1 左右，用常态门槛根本不会输出。跑完再按
        # tolerate_ui_occlusion 过滤回正常门槛（只对按钮底下那几块放宽）。
        res = m.predict(img, imgsz=self.args.imgsz,
                        conf=min(self.args.conf, CONF_UNDER_UI),
                        verbose=False)[0]
        raw = dedup([(names[int(b.cls)],
                      float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                      float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                      float(b.conf)) for b in res.boxes])
        mine, _orange = is_my_turn(img)
        dets = tolerate_ui_occlusion(raw, img.shape[1], mine, self.args.conf)"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('live/main.py 已接入')

# --- eval_table.py ---
p = Path('eval_table.py')
s = p.read_text(encoding='utf-8')
old = """from predict_cards import dedup, split_hand_table, table_by_player  # noqa: E402"""
new = """from predict_cards import (dedup, split_hand_table, table_by_player,  # noqa: E402
                           tolerate_ui_occlusion, CONF_UNDER_UI)"""
assert old in s
s = s.replace(old, new)

old = """    def f(img):
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        return [(model.names[int(b.cls)],
                 float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                 float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.conf)) for b in res.boxes]

    return f"""
new = """    def f(img):
        # 和面板保持一致：低门槛跑，再按 UI 遮挡规则过滤
        res = model.predict(img, imgsz=args.imgsz,
                            conf=min(args.conf, CONF_UNDER_UI), verbose=False)[0]
        raw = [(model.names[int(b.cls)],
                float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                float(b.conf)) for b in res.boxes]
        mine, _ = is_my_turn(img)
        return tolerate_ui_occlusion(raw, img.shape[1], mine, args.conf)

    return f"""
assert old in s
s = s.replace(old, new)

old = """from table_gt import GROUND_TRUTH, LOW_CONFIDENCE             # noqa: E402"""
new = """from table_gt import GROUND_TRUTH, LOW_CONFIDENCE             # noqa: E402
from turn import is_my_turn                                   # noqa: E402"""
assert old in s
s = s.replace(old, new)
Path('eval_table.py').write_text(s, encoding='utf-8')
print('eval_table.py 已接入')
