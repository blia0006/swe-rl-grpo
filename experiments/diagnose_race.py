"""诊断 5：验证"镜像切换竞态"假设——
对同一个 task 反复 start_instance 多次（每次都 stop 干净），
看 pyproject.name 是否稳定正确，还是随机命中别的镜像内容。
同时测试：如果 start_instance 后先 sleep 一段时间再读文件，是否能避免错误。
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


def run(sbx, cmd):
    try:
        r = sbx.commands.run(cmd, user="root", timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<EXC {type(e).__name__}: {e}>"


def check_once(ags, tool_id, task, extra_sleep=0.0):
    from e2b_code_interpreter import Sandbox

    instance_id, effective_image = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    if extra_sleep > 0:
        time.sleep(extra_sleep)
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        # 也读一下容器内 image id / 文件 mtime，看看是不是真的换了容器
        image_id = run(sbx, "cat /etc/hostname 2>&1; readlink -f /proc/1/root 2>&1")
        ok = task["repo"].split("/")[-1].lower() in pkgname.lower()
        print(f"  sleep={extra_sleep:>4.0f}s instance={instance_id[:20]:22s} "
              f"pyproject.name={pkgname:35s} match={'OK' if ok else '!!MISMATCH!!'} hostname={image_id.splitlines()[0] if image_id else ''}")
        return ok
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

    task0 = tasks[0]  # swe-synth-0001, cachecontrol
    print(f"反复请求同一题 {task0['task_id']} (期望repo={task0['repo']}) 共 5 次，不加额外等待：")
    for i in range(5):
        check_once(ags, tool_id, task0, extra_sleep=0.0)

    print(f"\n反复请求同一题 {task0['task_id']} 共 3 次，start后额外 sleep 5s 再读：")
    for i in range(3):
        check_once(ags, tool_id, task0, extra_sleep=5.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
