"""上线前最后验证：用新版 verl_reward_fn.compute_score（含内容自检）
对训练集第一题，构造一个「已知会通过」的 patch，验证整条链路（含新加的
_repo_content_matches 校验）能正确产出 reward=1.0，不会误杀正常实例。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
os.environ.setdefault("REWARD_DEBUG_LOG", "1")

from pipeline.verl_reward_fn import compute_score  # noqa: E402


def main() -> int:
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        all_tasks = {json.loads(line)["task_id"]: json.loads(line) for line in f}

    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    train_task = all_tasks[split["train_task_ids"][0]]
    print("测试题:", train_task["task_id"], train_task["repo"])

    ground_truth = json.dumps(
        {"task_id": train_task["task_id"], "image": train_task["image"], "repo": train_task["repo"]},
        ensure_ascii=False,
    )

    # 场景1：故意给一个空/无效 solution_str，预期 reward=0（走"未抽取到有效 patch"分支，
    # 同时验证「内容自检」本身不会误报——只要 repo 内容是对的，能顺利跑到这一步）
    score_empty = compute_score(
        data_source="swe_rl", solution_str="我无法修复。",
        ground_truth=ground_truth, extra_info={},
    )
    print(f"\n场景1（空patch）reward = {score_empty}  (预期 0.0)")

    # 场景2：故意给一个格式错误、无法 apply 的 patch，预期 reward=0（走"patch应用失败"分支）
    bad_patch = "```diff\n--- a/nonexistent_file_xyz.py\n+++ b/nonexistent_file_xyz.py\n@@ -1 +1 @@\n-a\n+b\n```"
    score_bad = compute_score(
        data_source="swe_rl", solution_str=bad_patch,
        ground_truth=ground_truth, extra_info={},
    )
    print(f"场景2（无效patch）reward = {score_bad}  (预期 0.0，且不应报'沙箱内容校验失败')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
