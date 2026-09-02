"""修复尝试：删除现有的（可能被污染缓存的）共享沙箱工具，重新创建一个干净的，
再对 6 道已知坏题重新验证，看是否恢复正常。
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

from clients.ags import AGSClient  # noqa: E402

TOOL_NAME = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")
REPO_DIR = "/workspace/repo"
BAD_TASK_IDS = [
    "swe-synth-0001", "swe-synth-0002", "swe-synth-0003",
    "swe-synth-0004", "swe-synth-0005", "swe-synth-0006",
]


def run(sbx, cmd):
    try:
        r = sbx.commands.run(cmd, user="root", timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<EXC {type(e).__name__}: {e}>"


def check_task(ags, tool_id, task):
    from e2b_code_interpreter import Sandbox

    instance_id, _ = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        expect = task["repo"].split("/")[-1].lower()
        ok = expect in pkgname.lower()
        return ok, pkgname
    finally:
        ags.stop_instance(instance_id)


def main() -> int:
    ags = AGSClient()

    print("=== 第一步：删除所有现存的 swe-synth-shared-runner 工具 ===")
    tools = ags.list_tools()
    dup_tool_ids = [t["tool_id"] for t in tools if t["name"] == TOOL_NAME]
    print("待删除:", dup_tool_ids)
    for tid in dup_tool_ids:
        try:
            ags.delete_tool(tid)
            print(f"  已删除 {tid}")
        except Exception as e:  # noqa: BLE001
            print(f"  删除 {tid} 失败：{e}")
    time.sleep(8)

    print("\n=== 第二步：重新创建干净的共享工具 ===")
    # 占位镜像随便用一个已知正确的（从环境变量读取，避免硬编码 TCR 命名空间）
    placeholder_image = os.environ.get(
        "PLACEHOLDER_IMAGE",
        "ccr.ccs.tencentyun.com/<tcr-namespace>/swe-synth-0016:v1",
    )
    tool_id = ags.create_tool(TOOL_NAME, placeholder_image, description="swe-rl reward runner (rebuilt)")
    print("新 tool_id:", tool_id)
    ags.wait_tool_active(TOOL_NAME, timeout=180)
    print("工具已 ACTIVE")

    print("\n=== 第三步：重新验证 6 道已知坏题 ===")
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        all_tasks = {json.loads(line)["task_id"]: json.loads(line) for line in f}

    n_ok = 0
    for tid in BAD_TASK_IDS:
        task = all_tasks[tid]
        ok, pkgname = check_task(ags, tool_id, task)
        print(f"  {tid} 期望repo={task['repo']:25s} 实际={pkgname:35s} [{'FIXED-OK' if ok else 'STILL-BAD'}]")
        if ok:
            n_ok += 1

    print(f"\n修复结果：{n_ok}/{len(BAD_TASK_IDS)} 恢复正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
