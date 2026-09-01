"""诊断 6：验证"共享工具 + image_override"模式是否是问题根源。
对比：
  A. 用共享工具 + image_override（现有 verl_reward_fn.py 的方式）→ 已确认稳定返回错误内容
  B. 新建一个专属工具，把镜像直接写在 CreateSandboxTool（不走 override）→ 看内容是否正确
如果 B 正确、A 错误，则证明是"共享工具 + 实例级 override"这个模式在平台侧有 bug/缓存问题。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

from clients.ags import AGSClient, AGSError  # noqa: E402

REPO_DIR = "/workspace/repo"


def run(sbx, cmd):
    try:
        r = sbx.commands.run(cmd, user="root", timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<EXC {type(e).__name__}: {e}>"


def main() -> int:
    from e2b_code_interpreter import Sandbox

    ags = AGSClient()

    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        task0 = json.loads(f.readline())

    image = task0["image"]
    dedicated_tool_name = "diag-dedicated-0001"

    # 清理可能存在的同名旧工具
    existing = ags.find_tool(dedicated_tool_name)
    if existing:
        print(f"发现已存在的诊断工具 {dedicated_tool_name}，先删除…")
        ags.delete_tool(existing["tool_id"])
        time.sleep(5)

    print(f"新建专属工具，直接绑定镜像 {image} …")
    tool_id = ags.create_tool(dedicated_tool_name, image, description="diag-dedicated")
    ags.wait_tool_active(dedicated_tool_name, timeout=180)
    print("工具已 ACTIVE，tool_id:", tool_id)

    instance_id, effective_image = ags.start_instance(tool_id, timeout="10m")  # 不传 image_override
    print("instance_id:", instance_id, "effective_image:", effective_image)
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        print(f"[专属工具-无override] pyproject.name = {pkgname}  (期望包含 CacheControl)")
    finally:
        ags.stop_instance(instance_id)

    print("\n清理诊断工具…")
    ags.delete_tool(tool_id)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
