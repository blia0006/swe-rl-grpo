"""Phase 3 训练数据构建：把 `data/tasks.jsonl` 里 11 道训练题转成 VERL GRPO
训练需要的 parquet 格式（单轮生成任务：给 problem_statement + 文件实际内容，让模型
直接输出 unified diff patch；不是 Line A 的多轮 ReAct 格式）。

VERL 标准列（`verl.utils.dataset.RLHFDataset` 期望的 schema）：
  - data_source: str，训练时用来分发到对应的 reward 计算（固定 "swe_rl"）
  - prompt: list[{"role":..., "content":...}]（chat 格式）
  - ability: str，仅用于日志分类
  - reward_model: {"style": "rule", "ground_truth": <JSON 字符串>}
    `ground_truth` 就是 `compute_score` 拿到的参数，本课题放题目元数据
  - extra_info: dict，留空 {}

【关键改动】每次迭代模型 reward 都是 0 的根因是任务设计缺陷：模型只看到
issue 描述 + 文件路径，看不到文件实际内容 → 盲写 patch → git apply 失败。
本脚本在 prompt 里嵌入文件的实际内容（前 300 行），让模型能基于真实代码
生成精确 diff。

用法：
    cd <repo_root>
    python3 pipeline/build_grpo_dataset.py
输出：data/grpo_train.parquet（55 行 = 11 题 × 5 轮）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 文件内容最多展示前 N 行（避免 prompt 过长；超过则截断并标注）
MAX_FILE_LINES = 300

SYSTEM_PROMPT = """\
你是一个资深 Python 工程师，负责修复代码仓库中的 bug。

规则：
1. 你会看到一段问题描述（issue）、需要修改的文件路径，以及【该文件当前的完整内容】。
2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` \
`+++ b/` 格式），用一个 ```diff 代码块包裹，不要输出多余的解释性文字。
3. patch 必须能被 `git apply` 直接应用：
   · 路径必须和仓库里的路径完全一致
   · hunk header（`@@ -X,Y +A,B @@`）的行号和上下文必须和【文件当前内容】严格匹配
   · 只修改与问题直接相关的行
4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"""


def _truncate_content(content: str, max_lines: int) -> tuple[str, bool]:
    """截断过长文件内容，返回 (content, truncated)。"""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content, False
    shown = "\n".join(lines[:max_lines])
    return f"{shown}\n\n# ... 以下省略（共 {len(lines) - max_lines} 行）", True


def build_user_prompt(
    task: dict, file_contents: dict[str, str] | None = None
) -> str:
    tid = task["task_id"]
    files = task.get("modified_files") or []

    parts: list[str] = [
        f"仓库：{task.get('repo', '')}",
        "",
        f"问题描述：\n{task.get('problem_statement', '')}",
        "",
        f"需要修改的文件：{'、'.join(files) if files else '(无)'}",
    ]

    # 🔑 关键：嵌入文件实际内容，让模型"看着文件写 patch"
    if file_contents and files:
        parts.append("")
        parts.append("=" * 60)
        parts.append("【文件当前内容】请仔细阅读后再写 patch")
        parts.append("=" * 60)
        for fpath in files:
            key = f"{tid}:{fpath}"
            content = file_contents.get(key)
            if content is None:
                parts.append(f"\n### {fpath}\n（暂无该文件内容，仅凭问题描述生成 patch）")
                continue
            truncated, was_truncated = _truncate_content(content, MAX_FILE_LINES)
            parts.append(
                f"\n### {fpath}（共 {len(content.splitlines())} 行"
                + (f"，展示前 {MAX_FILE_LINES} 行" if was_truncated else "，完整展示")
                + "）"
            )
            parts.append("```python")
            parts.append(truncated)
            parts.append("```")

    parts.append("")
    parts.append("=" * 60)
    parts.append("请直接给出修复该问题的 unified diff patch。")
    parts.append("提示：")
    parts.append("  · 行号和上下文必须与上面【文件当前内容】严格匹配")
    parts.append("  · 修改要最小化，只改真正需要改的地方")
    parts.append("  · 输出格式示例：")
    parts.append("    ```diff")
    parts.append("    --- a/<path>")
    parts.append("    +++ b/<path>")
    parts.append("    @@ -X,Y +A,B @@")
    parts.append("     context_line")
    parts.append("    -old_line")
    parts.append("    +new_line")
    parts.append("    ```")

    return "\n".join(parts)


def build_rows(
    tasks: list[dict],
    repeats: int,
    file_contents: dict[str, str] | None = None,
) -> list[dict]:
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
                    {
                        "role": "user",
                        "content": build_user_prompt(t, file_contents),
                    },
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

    # 读文件内容（如果有）。缺文件不影响构建（prompt 会标注"暂无内容"），
    # 但训练会回到"盲写 patch"模式，reward 会持续 0。
    fc_path = ROOT / "data" / "file_contents.json"
    file_contents: dict[str, str] = {}
    if fc_path.exists():
        file_contents = json.loads(fc_path.read_text(encoding="utf-8"))
        print(f"已加载 {len(file_contents)} 个文件内容（来自 {fc_path.name}）")
    else:
        print(f"⚠️ {fc_path.name} 不存在，prompt 将不含文件内容")
        print(f"   先跑：python3 pipeline/extract_file_contents.py")

    # 11 题 × 5 轮 = 55 行
    rows = build_rows(train_tasks, repeats=5, file_contents=file_contents)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "grpo_train.parquet"
    df.to_parquet(out_path)
    print(f"写入 {out_path}，共 {len(df)} 行（{len(train_tasks)} 题 × 5 轮）")

    # 顺便把 eval 集也存一份 JSON
    eval_tasks = [all_tasks[tid] for tid in split["eval_task_ids"]]
    eval_out = ROOT / "data" / "eval_tasks_full.json"
    eval_out.write_text(json.dumps(eval_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {eval_out}，共 {len(eval_tasks)} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())