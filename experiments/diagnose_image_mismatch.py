"""诊断 3（简化版）：逐条执行、互不依赖，避免 && 链在某一步失败时整体中断。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

from clients.ags import AGSClient  # noqa: E402

TOOL_NAME = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")
REPO_DIR = "/workspace/repo"


def run(sbx, cmd):
    try:
        r = sbx.commands.run(cmd, user="root", timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<EXC {type(e).__name__}: {e}>"


def check_task(ags, tool_id, task):
    from e2b_code_interpreter import Sandbox

    print(f"\n{'='*60}\ntask_id={task['task_id']} 记录的repo={task['repo']} 记录的modified_files={task.get('modified_files')}")
    instance_id, _ = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    try:
        sbx = Sandbox.connect(instance_id)
        print("--- ls repo ---")
        print(run(sbx, f"ls -la {REPO_DIR}"))
        print("--- pyproject name ---")
        print(run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml"))
        print("--- modified_file exists? ---")
        mf = task.get("modified_files", ["NA"])[0]
        print(run(sbx, f"ls -la {REPO_DIR}/{mf}"))
        print("--- modified_file content head ---")
        print(run(sbx, f"head -40 {REPO_DIR}/{mf}"))
    finally:
        ags.stop_instance(instance_id)


def main() -> int:
    ags = AGSClient()
    tool = ags.find_tool(TOOL_NAME)
    tool_id = tool["tool_id"]

    tasks = []
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    for idx in [0, 14]:
        check_task(ags, tool_id, tasks[idx])

    return 0


if __name__ == "__main__":
    sys.exit(main())
