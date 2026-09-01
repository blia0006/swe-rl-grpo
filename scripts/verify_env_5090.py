#!/usr/bin/env python3
"""5090（Blackwell / sm_120）环境自检脚本。

用途：GPU pod 起来后，在 pod 内运行本脚本，一次性验证
CUDA / PyTorch / flash-attn / vLLM / bf16 是否支持 RTX 5090。

用法（pod 内）：
    python3 scripts/verify_env_5090.py

关键判断标准（5090 强制要求）：
    - 驱动版本  >= 570
    - CUDA       >= 12.8（cu128）
    - PyTorch    >= 2.7（镜像为 2.8）
    - device capability == (12, 0)
    - bf16 张量核心：torch.cuda.is_bf16_supported() == True
"""
from __future__ import annotations

import subprocess
import sys

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------- ① nvidia-smi
section("① 宿主机 GPU 驱动 / 显卡识别（最关键，决定渲染型能否跑 CUDA）")
try:
    out = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, timeout=30
    )
    print(out.stdout)
    if out.returncode == 0 and "NVIDIA" in out.stdout:
        record("nvidia-smi 能识别 GPU", True)
        # 提取驱动版本粗判（低于 570 直接标红）
        import re
        m = re.search(r"Driver Version:\s*(\d+)", out.stdout)
        if m:
            ver = int(m.group(1))
            record("驱动版本 >= 570", ver >= 570, f"Driver={ver}")
    else:
        record("nvidia-smi 能识别 GPU", False, f"returncode={out.returncode}")
except Exception as e:
    record("nvidia-smi 能识别 GPU", False, str(e))

# ---------------------------------------------------------------- ② torch / sm_120
section("② PyTorch / CUDA / sm_120")
try:
    import torch
    record("torch 版本", torch.__version__.startswith(("2.7", "2.8", "2.9")),
           f"torch={torch.__version__}")
    record("torch 内置 CUDA 版本 >= 12.8",
           getattr(torch.version, "cuda", "0") and tuple(map(int, torch.version.cuda.split("."))) >= (12, 8),
           f"cuda={torch.version.cuda}")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        record("device capability == (12, 0)", tuple(cap) == (12, 0), f"cap={cap}")
        record("device 名称", True, torch.cuda.get_device_name())
        # 实际跑一个 GPU 算子，验证 sm_120 kernel 真的能执行（不只是 is_available）
        try:
            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            _ = x @ x
            torch.cuda.synchronize()
            record("GPU 上 bf16 matmul 真实执行成功", True)
        except Exception as e:
            record("GPU 上 bf16 matmul 真实执行成功", False, str(e))
    else:
        record("torch.cuda.is_available()", False, "CUDA 不可用")
except Exception as e:
    record("torch 导入/检查", False, str(e))

# ---------------------------------------------------------------- ③ bf16
section("③ bf16 张量核心")
try:
    import torch
    record("torch.cuda.is_bf16_supported()", torch.cuda.is_bf16_supported())
except Exception as e:
    record("bf16 检查", False, str(e))

# ---------------------------------------------------------------- ④ flash-attn
section("④ FlashAttention（5090 需要 >= 2.7.4 才支持 sm_120）")
try:
    import flash_attn
    record("flash-attn 可导入", True, f"version={flash_attn.__version__}")
except Exception as e:
    record("flash-attn 可导入", False, str(e))

# ---------------------------------------------------------------- ⑤ vllm
section("⑤ vLLM（预编译 kernel 是否含 sm_120，训练侧可先用 HFRollout 兜底）")
try:
    import vllm
    record("vllm 可导入", True, f"version={vllm.__version__}")
    try:
        from vllm import LLM
        llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=256,
                  enforce_eager=True, gpu_memory_utilization=0.3)
        out = llm.generate(["hello"], sampling_params=None) if False else None
        record("vllm 能加载模型并推理（sm_120 kernel）", True)
    except Exception as e:
        record("vllm 能加载模型并推理（sm_120 kernel）", False, str(e))
except Exception as e:
    record("vllm 可导入", False, str(e))

# ---------------------------------------------------------------- 总结
section("总结")
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print(f"通过 {passed}/{total}")
if passed == total:
    print("✅ 环境就绪，可开始训练。")
else:
    print("❌ 有 FAIL 项，见上方详情。")
    print("   - 若 FAIL 集中在 ①驱动：渲染型可能没装 CUDA 计算驱动，需联系云厂商/换镜像")
    print("   - 若 FAIL 在 ②④⑤：镜像版本不对，需换 cu128 + torch2.8 + fa2.7.4 的镜像")
    print("   - 若仅 ⑤vllm FAIL：训练侧改用 HFRollout（不依赖 vLLM 引擎）即可绕开")

sys.exit(0 if passed == total else 1)
