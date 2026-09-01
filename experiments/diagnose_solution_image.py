"""诊断 2：直接进入 solution_image 沙箱，检查被测试的源文件内容，
判断修复代码是否真的存在于镜像里、以及 run_tests.sh/verify.sh 到底在跑什么。
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


def main() -> int:
    from e2b_code_interpreter import Sandbox

    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        task = json.loads(f.readline())

    sol_image = task["solution_image"]
    print("task_id:", task["task_id"])
    print("solution_image:", sol_image)
    print("modified_files:", task.get("modified_files"))

    ags = AGSClient()
    tool = ags.find_tool(TOOL_NAME)
    tool_id = tool["tool_id"]
    instance_id, effective_image = ags.start_instance(tool_id, image_override=sol_image, timeout="15m")
    print("instance_id:", instance_id, "effective_image:", effective_image)

    try:
        sbx = Sandbox.connect(instance_id)

        # 看 modified_files 的实际内容
        for mf in task.get("modified_files", []):
            r = sbx.commands.run(f"cat {REPO_DIR}/{mf}", user="root", timeout=30)
            print(f"\n--- {mf} (head 2000 chars) ---")
            print((r.stdout or "")[:2000])
            if r.stderr:
                print("STDERR:", r.stderr[:300])

        # 看 run_tests.sh / verify.sh 内容
        r = sbx.commands.run("cat /task/run_tests.sh", user="root", timeout=30)
        print("\n--- /task/run_tests.sh ---")
        print(r.stdout)

        r = sbx.commands.run("cat /task/verify.sh 2>&1 || echo NO_VERIFY_SH", user="root", timeout=30)
        print("\n--- /task/verify.sh ---")
        print(r.stdout)

        # 直接手动跑 pytest 看真实报错
        r = sbx.commands.run(
            f"cd {REPO_DIR} && python3 -m pytest tests/test_retry_history_synth.py -x 2>&1 | tail -60",
            user="root", timeout=120,
        )
        print("\n--- 手动 pytest 输出 (tail 60 lines) ---")
        print(r.stdout)
        print("exit_code:", r.exit_code)

        # 看 git log / git diff 确认代码是否真的跟原始镜像不同
        r = sbx.commands.run(f"cd {REPO_DIR} && git log --oneline -5", user="root", timeout=30)
        print("\n--- git log ---")
        print(r.stdout)

    finally:
        try:
            ags.stop_instance(instance_id)
            print(f"\n已释放 instance_id={instance_id}")
        except Exception as e:  # noqa: BLE001
            print(f"释放失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
