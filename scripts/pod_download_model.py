"""在 GPU Pod 内运行：把 COS 上预取好的模型权重下到本地磁盘。
用法（pod 内）：
    cd /workspace/repo && python3 scripts/pod_download_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.cos import list_objects, download_file  # noqa: E402

BUCKET = os.environ.get("COS_BUCKET", "YOUR_COS_BUCKET")
PREFIX = "models/Qwen2.5-Coder-1.5B-Instruct/"
LOCAL_DIR = "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"


def main() -> int:
    keys = list_objects(BUCKET, prefix=PREFIX)
    if not keys:
        print(f"[FATAL] COS 上没找到 {PREFIX} 下的文件")
        return 1
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for key in keys:
        fname = key[len(PREFIX):]
        if not fname:
            continue
        local_path = os.path.join(LOCAL_DIR, fname)
        print(f"downloading {key} -> {local_path}")
        download_file(BUCKET, key, local_path)
    print(f"完成，共 {len(keys)} 个文件，模型目录：{LOCAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
