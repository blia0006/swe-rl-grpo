"""读取训练集 11 道题的 modified_files 实际内容，写入 data/file_contents.json。

【为什么需要这个】
当前训练 prompt 只有 issue 描述 + 文件路径，没有文件内容，导致 1.5B 模型盲写
unified diff（行号和上下文瞎编），git apply 报错，reward 全 0（任务设计缺陷）。
本脚本从 AGS 沙箱实例里读出文件实际内容，存到 file_contents.json，供 build_grpo_dataset.py
拼进 prompt，让模型"看着文件写 patch"。

【用法】在 Pod 里，已 source .env：
  cd /home/dpsk_a2a/repo
  python3 pipeline/extract_file_contents.py
预计耗时 ~2 分钟（11 道题 × ~11 秒冷启动）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import e2b 之前关闭 Key 校验（沿用课题三的实测结论）
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # noqa: E402

from clients.ags import AGSClient  # noqa: E402
from e2b_code_interpreter import Sandbox  # noqa: E402

DEFAULT_TOOL = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")
REPO_DIR_IN_SANDBOX = "/workspace/repo"


def main() -> int:
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    train_ids = split["train_task_ids"]

    tasks = {}
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_id"]] = d

    ags = AGSClient()

    # 找到 swe-synth-shared-runner 工具（之前 PROGRESS.md 验证过可用）
    tool = ags.find_tool(DEFAULT_TOOL)
    if tool is None:
        print(f"❌ 找不到工具 {DEFAULT_TOOL}（请检查 .env 凭证和 AGS 工具名）")
        return 1
    tool_id = tool["tool_id"]
    print(f"使用工具: {DEFAULT_TOOL} ({tool_id})\n")

    results: dict[str, str] = {}
    failed: list[str] = []

    for tid in train_ids:
        task = tasks[tid]
        image = task["image"]
        modified_files = task.get("modified_files", [])

        if not modified_files:
            print(f"[{tid}] 无 modified_files，跳过")
            continue

        print(f"[{tid}] 起沙箱（镜像 {image}）...")
        instance_id = None
        try:
            # 起沙箱（用题目 image_override）
            instance_id, _ = ags.start_instance(
                tool_id=tool_id,
                image_override=image,
            )

            sbx = Sandbox.connect(instance_id)

            for fpath in modified_files:
                full_path = f"{REPO_DIR_IN_SANDBOX}/{fpath}"
                content = None

                # 主路径：files.read（必须 user="root"，镜像里没有 "user" 用户）
                try:
                    content = sbx.files.read(full_path, user="root")
                except Exception as e1:
                    # 兜底路径：用 cat 读（更底层，绕开 SDK 的 user 处理）
                    try:
                        r = sbx.commands.run(f"cat {full_path}", user="root", timeout=60)
                        content = r.stdout
                    except Exception as e2:
                        print(f"  ✗ {fpath} - files.read: {e1} | cat: {e2}")
                        failed.append(f"{tid}:{fpath}")
                        continue

                if content:
                    results[f"{tid}:{fpath}"] = content
                    print(f"  ✓ {fpath} ({len(content)} bytes)")
                else:
                    print(f"  ✗ {fpath} - 读到空内容")
                    failed.append(f"{tid}:{fpath}")
        except Exception as e:
            print(f"  ✗ 起沙箱失败: {e}")
            failed.append(tid)
        finally:
            # 停掉沙箱（不是 pool 模式，用完就停，避免占资源）
            if instance_id:
                try:
                    ags.stop_instance(instance_id)
                except Exception:
                    pass

    # 保存结果
    out = ROOT / "data" / "file_contents.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ 写入 {out}")
    print(f"   成功: {len(results)} 个文件")
    if failed:
        print(f"   失败: {len(failed)} 个 → {failed}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())