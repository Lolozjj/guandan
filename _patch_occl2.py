"""把 panel / eval_table 接到新的通用遮挡容忍上。"""
from pathlib import Path

# --- live/main.py ---
p = Path('live/main.py')
s = p.read_text(encoding='utf-8')
old = """from predict_cards import (card_value, dedup, split_hand_table,  # noqa: E402
                           table_by_player, tolerate_ui_occlusion,
                           CONF_UNDER_UI)"""
new = """from predict_cards import (card_value, dedup, split_hand_table,  # noqa: E402
                           table_by_player, tolerate_occlusion,
                           occluders, CONF_UNDER_UI)"""
assert old in s
s = s.replace(old, new)

old = """        mine, _orange = is_my_turn(img)
        dets = tolerate_ui_occlusion(raw, img.shape[1], mine, self.args.conf)"""
new = """        # 先用常态门槛找出「确定是手牌」的那些，它们的位置就是遮挡物之一
        # （自己的手牌会盖住自己刚出的牌）。再据此过滤出最终检出。
        mine, _orange = is_my_turn(img)
        hand_hi, _ = split_hand_table([d for d in raw if d[3] >= self.args.conf])
        dets = tolerate_occlusion(raw, self.args.conf,
                                  occluders(img.shape[1], mine, hand_hi))"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('live/main.py 已更新')

# --- eval_table.py ---
p = Path('eval_table.py')
s = p.read_text(encoding='utf-8')
old = """from predict_cards import (dedup, split_hand_table, table_by_player,  # noqa: E402
                           tolerate_ui_occlusion, CONF_UNDER_UI)"""
new = """from predict_cards import (dedup, split_hand_table, table_by_player,  # noqa: E402
                           tolerate_occlusion, occluders, CONF_UNDER_UI)"""
assert old in s
s = s.replace(old, new)

old = """        mine, _ = is_my_turn(img)
        return tolerate_ui_occlusion(raw, img.shape[1], mine, args.conf)"""
new = """        mine, _ = is_my_turn(img)
        hand_hi, _ = split_hand_table([d for d in raw if d[3] >= args.conf])
        return tolerate_occlusion(raw, args.conf,
                                  occluders(img.shape[1], mine, hand_hi))"""
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
print('eval_table.py 已更新')
