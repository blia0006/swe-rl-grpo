"""诊断 4：确认「共享沙箱工具」的 image_override 是否真的生效，
还是所有实例都被缓存卡在同一份镜像内容上。
"""
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

    instance_id, effective_image = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        print(f"task_id={task['task_id']:20s} 期望repo={task['repo']:25s} "
              f"请求镜像={task['image'][-30:]:32s} effective_image末尾={str(effective_image)[-30:]:32s} "
              f"实际pyproject.name={pkgname}")
    finally:
        ags.stop_instance(instance_id)


def main() -> int:
    ags = AGSClient()
    tool = ags.find_tool(TOOL_NAME)
    print("工具当前默认(创建时)绑定镜像:", tool.get("image"))
    print("工具 tool_id:", tool["tool_id"], "status:", tool.get("status"))
    tool_id = tool["tool_id"]

    tasks = []
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    for idx in [0, 1, 2, 3, 6, 14]:
        check_task(ags, tool_id, tasks[idx])

    return 0


if __name__ == "__main__":
    sys.exit(main())
