"""训练必须把网络放到 `train.net.DEVICE` 上 —— 这个坑真踩过。

实测（2026-09-26）：Plan 3 那 5 万局与这一次的 9 小时跑，日志第一行都是
`device=cpu`，而 `torch.cuda.is_available()` 从始至终是 True —— 也就是
**GPU 从来没被训练用上**（`net = QNet()` 后面漏了 `.to(DEVICE)`）。
训练步的瓶颈本来就是网络前向（spec §14.3），落在 CPU 上等于白等。
"""
from train import net as netmod
from train import selfplay


def test_training_puts_the_net_on_the_configured_device(tmp_path):
    """用 1 秒预算真跑一次训练，看它把网络放在了哪个设备上。

    `eval_games=1` 是为了让收尾那两次评测快（那两次跟本测试无关）。
    """
    lines = []
    selfplay.train(seconds=1, eval_games=1, out_dir=str(tmp_path),
                   log=lines.append)
    assert lines, "训练一句日志都没打出来"
    assert f"device={netmod.DEVICE}" in lines[0], \
        f"网络没放到 {netmod.DEVICE} 上：{lines[0]}"
