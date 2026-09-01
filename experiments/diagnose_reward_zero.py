"""诊断脚本：排查训练里 reward 恒为 0 的原因。

策略：分两步隔离问题
  1. 用「已经修复好」的 solution_image 直接起沙箱、不打任何 patch，直接跑
     verify.sh，看 result.json 判分是不是应该给满分的用例真的给了满分。
     —— 如果这一步就是 0，说明问题在 verify.sh / result.json / compute_reward
        判分链路本身，跟模型生成的 patch 质量无关。
  2. 用原始 image（未修复），跑 verify.sh，应该判 0（fail），作为对照组，
     确认判分链路在"应该失败"的场景下确实输出 0（不是永远输出固定值的假象）。

不影响正在运行的训练（新建独立沙箱实例，用完立即释放，不进 _InstancePool）。
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
from pipeline.reward import compute_reward  # noqa: E402

TOOL_NAME = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")
REPO_DIR = "/workspace/repo"


def run_verify(image: str, label: str) -> None:
    from e2b_code_interpreter import Sandbox

    print(f"\n=== [{label}] image={image} ===")
    ags = AGSClient()
    tool = ags.find_tool(TOOL_NAME)
    if not tool:
        print(f"!! 工具 {TOOL_NAME} 不存在")
        return
    tool_id = tool["tool_id"]

    instance_id, effective_image = ags.start_instance(
        tool_id, image_override=image, timeout="15m",
    )
    print(f"instance_id={instance_id} effective_image={effective_image}")
    try:
        sbx = Sandbox.connect(instance_id)

        ls_res = sbx.commands.run(f"ls -la {REPO_DIR} | head -5", user="root", timeout=30)
        print("repo ls (head):", ls_res.stdout[:200])

        ls_task = sbx.commands.run("ls -la /task/ 2>&1", user="root", timeout=30)
        print("/task ls:", ls_task.stdout[:300], ls_task.stderr[:200])

        try:
            verify_res = sbx.commands.run(
                f"cd {REPO_DIR} && bash /task/verify.sh",
                user="root", timeout=300,
                envs={"PYTEST_ADDOPTS": "--color=no"},
            )
            print("verify exit_code:", verify_res.exit_code)
            print("verify stdout (tail 1000):", (verify_res.stdout or "")[-1000:])
            print("verify stderr (tail 500):", (verify_res.stderr or "")[-500:])
        except Exception as e:  # noqa: BLE001
            print(f"verify.sh 抛异常（可能是非0退出码触发SDK异常）: {type(e).__name__}: {e}")

        try:
            result_raw = sbx.files.read("/task/result.json", user="root")
            print("result.json raw:", result_raw[:1000])
            result_json = json.loads(result_raw)
            rr = compute_reward(result_json)
            print(f">>> reward={rr.reward} reason={rr.reason}")
        except Exception as e:  # noqa: BLE001
            print(f"读取/解析 result.json 失败：{type(e).__name__}: {e}")
    finally:
        try:
            ags.stop_instance(instance_id)
            print(f"已释放 instance_id={instance_id}")
        except Exception as e:  # noqa: BLE001
            print(f"释放实例失败（可忽略）：{e}")


def main() -> int:
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        task = json.loads(f.readline())

    print("task_id:", task["task_id"])
    print("image (未修复):", task["image"])
    print("solution_image (已修复):", task.get("solution_image"))

    # 对照组：未修复镜像，预期 verify.sh 判 fail，reward 应该是 0
    run_verify(task["image"], "对照组-未修复镜像")

    # 实验组：已修复镜像，预期 verify.sh 判 pass，reward 应该 > 0
    sol_image = task.get("solution_image")
    if sol_image:
        run_verify(sol_image, "实验组-已修复镜像(golden)")
    else:
        print("!! 该题没有 solution_image 字段，跳过实验组")

    return 0


if __name__ == "__main__":
    sys.exit(main())
