"""诊断 7：确认根因——是不是因为撞上了重名的另一个 swe-synth-shared-runner 工具。
分别对两个同名工具的 tool_id 显式发起 start_instance(image_override=...)，
对比返回内容，确认哪一个才是"真正干净、按 override 生效"的工具。
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

REPO_DIR = "/workspace/repo"


def run(sbx, cmd):
    try:
        r = sbx.commands.run(cmd, user="root", timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<EXC {type(e).__name__}: {e}>"


def check(ags, tool_id, task, label):
    from e2b_code_interpreter import Sandbox

    instance_id, effective_image = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        ok = task["repo"].split("/")[-1].lower() in pkgname.lower()
        print(f"[{label}] tool_id={tool_id} task={task['task_id']} pyproject={pkgname} match={'OK' if ok else '!!MISMATCH!!'}")
    finally:
        ags.stop_instance(instance_id)


def main() -> int:
    ags = AGSClient()
    tools = ags.list_tools()
    dup_tool_ids = [t["tool_id"] for t in tools if t["name"] == "swe-synth-shared-runner"]
    print("重名工具 tool_id 列表:", dup_tool_ids)

    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        task0 = json.loads(f.readline())

    for tid in dup_tool_ids:
        check(ags, tid, task0, f"tool={tid}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
