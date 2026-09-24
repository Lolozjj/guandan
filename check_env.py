"""环境自检：确认 PyTorch 能真正驱动 RTX 5060 (Blackwell / sm_120)。

用法：python check_env.py
"""
import sys

print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")

# --- PyTorch ---
try:
    import torch
except ImportError:
    print("\n[FAIL] 未安装 torch。")
    sys.exit(1)

print(f"PyTorch     : {torch.__version__}")
print(f"torch.version.cuda : {torch.version.cuda}")
print(f"CUDA 可用   : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n[FAIL] CUDA 不可用。八成是装成了 CPU 版，或装的 CUDA 版本低于 12.8")
    print("       重装：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)  # (major, minor)，Blackwell 应为 (12, 0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"GPU         : {name}")
print(f"算力        : sm_{cap[0]}{cap[1]}")
print(f"显存        : {vram:.1f} GB")

# --- 真正跑一次运算，验证 kernel 存在（这一步才是 Blackwell 的照妖镜）---
try:
    a = torch.randn(1024, 1024, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"矩阵运算    : OK  (结果均值 {b.mean().item():.4f})")
except RuntimeError as e:
    print(f"\n[FAIL] GPU 运算失败：{e}")
    print("       这通常是 sm_120 kernel 缺失 → torch 版本太老，需要 cu128+ 的构建")
    sys.exit(1)

# --- Ultralytics ---
try:
    import ultralytics
except ImportError:
    print("\n[FAIL] 未安装 ultralytics。")
    sys.exit(1)

print(f"Ultralytics : {ultralytics.__version__}")

# --- 组装一句结论 ---
if cap[0] == 12:
    print("\n[DONE] 环境就绪：Blackwell 架构 + CUDA 版 torch，可以开始训练。")
else:
    print(f"\n[DONE] 环境可用，但检测到 sm_{cap[0]}{cap[1]}（预期 Blackwell 为 sm_120）。")
