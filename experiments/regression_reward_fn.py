"""Phase 1.5 回归测试：直接调用生产入口 `pipeline.verl_reward_fn.compute_score`
（不是重新实现的测试逻辑），验证 tar 快照 bug 修正后，训练时真正会跑的这条代码路径
在多道不同题目上都正确。

覆盖两个场景 × N 道题目：
  - solution_str 携带该题的 golden.patch → 期望 compute_score 返回 1.0
  - solution_str 为空 → 期望 compute_score 返回 0.0
  - 且验证同一 task_id 复用实例（第二次调用应该更快，日志里能看出）

golden.patch 内容**从沙箱内部读取**（`/opt/solution/golden.patch`，镜像自带的真实数据源），
不使用本地 `课题三-数据合成/data/proofs/*/golden.patch` 镜像文件——2026-08-23 实测发现该本地
目录部分题目的 golden.patch 与当前镜像内容不一致（task_id 编号与镜像内容错位，属于历史数据
过期问题），用沙箱内部文件才能保证测的是当前镜像的真实标准答案。

用法：
    cd <repo_root>
    source .venv/bin/activate
    AGS_REWARD_TOOL_NAME=swe-synth-shared-runner python3 experiments/regression_reward_fn.py
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
os.environ.setdefault("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")

N_TASKS = 5


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def load_tasks(n: int) -> list[dict]:
    tasks = []
    with open(ROOT / "data" / "tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("image"):
                tasks.append(t)
            if len(tasks) >= n:
                break
    return tasks


def fetch_golden_patch_from_sandbox(ags, tool_id: str, image: str) -> str:
    """临时起一个实例，读取镜像内 /opt/solution/golden.patch 后立即释放。"""
    from e2b_code_interpreter import Sandbox

    instance_id, _ = ags.start_instance(tool_id, image_override=image, timeout="15m")
    try:
        sbx = Sandbox.connect(instance_id)
        return sbx.files.read("/opt/solution/golden.patch", user="root")
    finally:
        ags.stop_instance(instance_id)


def main() -> int:
    load_env()
    from pipeline.verl_reward_fn import compute_score, _POOL
    from clients.ags import AGSClient

    ags = AGSClient()
    tool = ags.find_tool(os.environ["AGS_REWARD_TOOL_NAME"])
    tool_id = tool["tool_id"]

    tasks = load_tasks(N_TASKS)
    print(f"回归测试题目：{[t['task_id'] for t in tasks]}")

    n_pass = 0
    n_total = 0
    for t in tasks:
        task_id = t["task_id"]
        image = t["image"]
        ground_truth = json.dumps({"task_id": task_id, "image": image})

        try:
            golden_patch = fetch_golden_patch_from_sandbox(ags, tool_id, image)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {task_id}: 读取沙箱内 golden.patch 失败：{e}")
            continue
        solution_golden = f"```diff\n{golden_patch}\n```"

        t0 = time.time()
        r_golden = compute_score("swe_rl", solution_golden, ground_truth)
        dt_golden = round(time.time() - t0, 2)
        n_total += 1
        ok_golden = r_golden == 1.0
        n_pass += ok_golden
        print(f"  [{task_id}] golden patch -> reward={r_golden} ({dt_golden}s) {'OK' if ok_golden else 'FAIL!'}")

        t1 = time.time()
        r_empty = compute_score("swe_rl", "我无法解决这个问题。", ground_truth)
        dt_empty = round(time.time() - t1, 2)
        n_total += 1
        ok_empty = r_empty == 0.0
        n_pass += ok_empty
        print(f"  [{task_id}] empty solution -> reward={r_empty} ({dt_empty}s, 复用实例应更快) {'OK' if ok_empty else 'FAIL!'}")

    print(f"\n========== 结果：{n_pass}/{n_total} 通过 ==========")
    _POOL.shutdown_all()
    print("已清空实例池，避免持续计费。")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
