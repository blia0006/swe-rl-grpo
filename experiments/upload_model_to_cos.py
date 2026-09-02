"""Phase 1.5：把经 ModelScope 预取到本地的模型权重上传到 COS，供 TKE GPU 节点
在训练前直接从 COS 拉取（避免 GPU 节点直连 HuggingFace/ModelScope 浪费计费时间，
见 plan.md 2.4 节 / Phase 1.5 checklist）。

用法：
    cd <repo_root>
    source .venv/bin/activate
    python3 experiments/upload_model_to_cos.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_LOCAL_DIR = ROOT / ".model_cache" / "Qwen" / "Qwen2___5-Coder-1___5B-Instruct"
COS_PREFIX = "models/Qwen2.5-Coder-1.5B-Instruct/"
# 只上传训练/推理真正需要的文件，跳过 modelscope 内部元数据（.mdl/.msc/.mv）
UPLOAD_FILES = [
    "config.json", "configuration.json", "generation_config.json",
    "merges.txt", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "LICENSE", "README.md",
]


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def main() -> int:
    load_env()
    from clients.cos import upload_file, list_objects

    bucket = os.environ.get("COS_BUCKET", "YOUR_COS_BUCKET")

    if not MODEL_LOCAL_DIR.exists():
        print(f"[FATAL] 找不到本地模型目录 {MODEL_LOCAL_DIR}，请先用 ModelScope 下载")
        return 1

    print(f"上传到 bucket={bucket}，前缀={COS_PREFIX}")
    total = 0
    for fname in UPLOAD_FILES:
        local_path = MODEL_LOCAL_DIR / fname
        if not local_path.exists():
            print(f"  [skip] {fname} 不存在")
            continue
        size_mb = local_path.stat().st_size / 1024 / 1024
        key = COS_PREFIX + fname
        cos_path = upload_file(bucket, key, str(local_path))
        total += 1
        print(f"  [ok] {fname} ({size_mb:.1f}MB) -> {cos_path}")

    print(f"\n共上传 {total} 个文件。核对 COS 上的对象列表：")
    for k in list_objects(bucket, prefix=COS_PREFIX):
        print(f"  · {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
