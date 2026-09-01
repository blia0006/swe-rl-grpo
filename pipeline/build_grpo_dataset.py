"""Phase 3 训练数据构建：把 `data/tasks.jsonl` 里 14 道训练题转成 VERL GRPO
训练需要的 parquet 格式（单轮生成任务：给 problem_statement，让模型直接输出
unified diff patch；不是 Line A 的多轮 ReAct 格式，见 plan.md 2.3 节"训练侧
prompt 设计"）。

VERL 标准列（`verl.utils.dataset.RLHFDataset` 期望的 schema，已用 GSM8K/MATH
官方预处理脚本核对）：
  - data_source: str，训练时用来分发到对应的 reward 计算（我们固定 "swe_rl"，
    对应 `pipeline/verl_reward_fn.py::DATA_SOURCE_NAME`）
  - prompt: list[{"role":..., "content":...}]（chat 格式，VERL 用 tokenizer 的
    chat_template 展开）
  - ability: str，仅用于日志分类，随便填
  - reward_model: {"style": "rule", "ground_truth": <JSON 字符串>}——
    `ground_truth` 就是 `compute_score` 拿到的那个参数，本课题放题目元数据
    （task_id + image），见 `pipeline/verl_reward_fn.py::compute_score` 的注释
  - extra_info: dict，留空 {}（占位，VERL 有些版本要求这一列存在）

用法：
    cd /Users/user/学习/题目四：强化学习
    source .venv/bin/activate
    python3 pipeline/build_grpo_dataset.py
输出：data/grpo_train.parquet（55 行 = 11 题 × 5 轮，配合
train_batch_size=1 正好是 55 个 step，覆盖任务书 "≥50 step" 的要求。
11 题而非最初的 14 题：另 6 道题（swe-synth-0001~0006）经沙箱内容核实
存在镜像错乱缺陷，已剔除，见 data/split.json 的 _comment）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = """\
你是一个资深 Python 工程师，负责修复代码仓库中的 bug。

规则：
1. 你会看到一段问题描述（issue）和需要修改的文件路径。
2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` \
`+++ b/` 格式），用一个 ```diff 代码块包裹，不要输出多余的解释性文字。
3. patch 必须能被 `git apply` 直接应用，路径要和仓库里的路径完全一致。
4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"""


def build_user_prompt(task: dict) -> str:
    files = "、".join(task.get("modified_files") or [])
    return (
        f"仓库：{task.get('repo', '')}\n\n"
        f"问题描述：\n{task.get('problem_statement', '')}\n\n"
        f"需要修改的文件：{files}\n\n"
        f"请直接给出修复该问题的 unified diff patch。"
    )


def build_rows(tasks: list[dict], repeats: int) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for t in tasks:
            ground_truth = json.dumps(
                {"task_id": t["task_id"], "image": t["image"], "repo": t.get("repo", "")},
                ensure_ascii=False,
            )
            rows.append({
                "data_source": "swe_rl",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(t)},
                ],
                "ability": "swe-bug-fix",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"task_id": t["task_id"], "difficulty": t.get("difficulty", "")},
            })
    return rows


def main() -> int:
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_task_ids"])

    all_tasks = {}
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            all_tasks[d["task_id"]] = d

    train_tasks = [all_tasks[tid] for tid in split["train_task_ids"]]
    assert len(train_tasks) == len(train_ids), "训练集 task_id 在 tasks.jsonl 里没找全"

    # 11 题 × 5 轮 = 55 行；train_batch_size=1 → 恰好 55 个 step（≥ 任务书要求的 50）
    # （原为 14 题 × 4 轮 = 56，因剔除 6 道镜像内容错乱的坏题后训练集缩小为 11 题，
    # 见 data/split.json 的 _comment，调整轮数以维持 ≥50 step 的要求）
    rows = build_rows(train_tasks, repeats=5)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "grpo_train.parquet"
    df.to_parquet(out_path)
    print(f"写入 {out_path}，共 {len(df)} 行（{len(train_tasks)} 题 × 5 轮）")

    # 顺便把 eval 集也存一份 JSON（Phase 4 评测脚本直接读，不需要 parquet 格式）
    eval_tasks = [all_tasks[tid] for tid in split["eval_task_ids"]]
    eval_out = ROOT / "data" / "eval_tasks_full.json"
    eval_out.write_text(json.dumps(eval_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {eval_out}，共 {len(eval_tasks)} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
