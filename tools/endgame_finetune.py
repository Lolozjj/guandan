"""在**残局专家标签**上小步微调现役权重（监督学习，不动自对弈主循环）。

为什么用监督而不是改自对弈目标：① 标签是现成的（`tools/build_endgame_set.py`）；
② 不动主循环 ⇒ 风险低、可回滚；③ 只练"残局那一段"，正是干预实验指出的短板
（22% 的残局决策上差 0.36 点）。

⚠️ 两条纪律：
- **小学习率 + 少轮次**（默认 3e-5 / 2 轮）：这是微调，不是重训；大了会灾难性遗忘
  （把整局的能力换掉换不来残局那点分）。
- 练完**必须**用 `tools/ruler` 复核（≥3 种子配对）：监督微调最容易"在训练集上变好、
  在真局里变差"。

    .venv/Scripts/python.exe -m tools.endgame_finetune --data runs/ab/endgame_set.npz \
        --init models/best.pt --out runs/ab/finetune_endgame.pt --epochs 2 --lr 3e-5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from guandan.console import utf8_stdout
from guandan.rl.net import QNet, load_state


def main(argv=None) -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="残局专家标签上的监督微调")
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default="models/best.pt")
    ap.add_argument("--out", default="runs/ab/finetune_endgame.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=("mse", "rank"), default="mse",
                    help="mse = 回归绝对回报（默认为老行为）；rank = 成对排序损失（只学谁更好）")
    a = ap.parse_args(argv)

    d = np.load(a.data)
    st, ac, hi, y = d["states"], d["actions"], d["hists"], d["values"]
    grp = d["groups"] if "groups" in d.files else np.arange(len(y))
    n = len(y)
    rng = np.random.default_rng(a.seed)
    # ⚠️ **按组切分**：同一个残局局面的几个候选必须落在同一边 ——
    # 按样本随机切会把组切散，于是验证集里几乎没有"可配对的样本"，
    # 排序损失报出来的准确率就成了 4/7 这种毫无意义的分数（2026-10-01 踩过）。
    uniq = np.unique(grp)
    rng.shuffle(uniq)
    n_val_g = max(1, int(len(uniq) * a.val_frac))
    val_groups = set(uniq[:n_val_g].tolist())
    val = np.where(np.isin(grp, list(val_groups)))[0]
    train = np.where(~np.isin(grp, list(val_groups)))[0]
    print(f"样本 {n}（训练 {len(train)} / 验证 {len(val)}）；标签均值 {y.mean():+.3f}")

    net = QNet()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    load_state(net, ck["net"] if isinstance(ck, dict) else ck)
    dev = next(net.parameters()).device
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def rank_loss(ix):
        """**成对排序损失**：只惩罚「把更差的候选排在更好的前面」。

        为什么需要它（2026-10-01 夜的教训）：残局标签的 sd = 1.58 点，而候选之间真正的差
        只有 ~0.36 点 ⇒ MSE 回归是在学「每个局面的绝对水平」，**不是在学排序**；
        两次绝对回归微调的独立复核都没复现（+1.22→+0.10、+0.79→+0.23）。
        排序损失对每个局面的整体平移不变 ⇒ 只学我们真正要的那部分。
        """
        pred, t = fwd(ix)
        # 组内两两配对（值不同才有信号）
        g = grp[ix]
        vals = t.detach()
        pi, pj = [], []
        for gi in np.unique(g):
            m = np.where(g == gi)[0]
            for a_ in m:
                for b_ in m:
                    if vals[a_] > vals[b_]:
                        pi.append(int(a_)); pj.append(int(b_))
        if not pi:
            return pred.sum() * 0.0, 0.0
        d_ = pred[pi] - pred[pj]
        loss = torch.nn.functional.softplus(-d_).mean()
        acc = float((d_ > 0).float().mean())
        return loss, acc

    def fwd(ix):
        s_, c_, h_, t_ = (torch.from_numpy(st[ix]).to(dev), torch.from_numpy(ac[ix]).to(dev),
                          torch.from_numpy(hi[ix]).to(dev), torch.from_numpy(y[ix]).to(dev))
        return net(s_, c_, h_), t_

    def evaluate(ix):
        net.eval()
        with torch.no_grad():
            if a.loss == "rank":
                _l, acc = rank_loss(ix)
                return acc
            pred, t = fwd(ix)
            return float(torch.nn.functional.mse_loss(pred, t))
        net.train()

    before = evaluate(val)
    for ep in range(a.epochs):
        perm = rng.permutation(train)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            ix = perm[i:i + a.batch]
            if a.loss == "rank":
                loss, _acc = rank_loss(ix)
            else:
                pred, t = fwd(ix)
                loss = torch.nn.functional.mse_loss(pred, t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(ix)
        name = "排序损失" if a.loss == "rank" else "MSE"
        print(f"  epoch {ep + 1}/{a.epochs}  训练 {name} {tot / len(perm):.4f}  "
              f"验证 {{'成对准确率' if a.loss == 'rank' else 'MSE'}} {evaluate(val):.4f}")
    after = evaluate(val)
    net.eval()
    torch.save({"net": net.state_dict(), "games": None,
                "note": f"endgame finetune on {a.data} from {a.init}"}, a.out)
    metric = "成对准确率" if a.loss == "rank" else "验证 MSE"
    print(f"\n{metric}：微调前 {before:.4f} → 微调后 {after:.4f}")
    print(f"权重 -> {a.out}")
    print("⚠️ 下一步必须复核：`tools/ruler <新件> models/best_r11_backup.pt --base models/best.pt "
          "--seeds ...`（监督微调最容易训练集变好、真局变差）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
