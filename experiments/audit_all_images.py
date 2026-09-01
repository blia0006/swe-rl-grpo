"""诊断 8（全量普查）：对 tasks.jsonl 里全部任务的 `image` 逐一验证
pyproject.toml 的包名是否与期望 repo 匹配，统计出"损坏镶像"的完整清单。
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

    instance_id, _ = ags.start_instance(tool_id, image_override=task["image"], timeout="10m")
    try:
        sbx = Sandbox.connect(instance_id)
        pkgname = run(sbx, f"grep -m1 '^name' {REPO_DIR}/pyproject.toml").strip()
        expect = task["repo"].split("/")[-1].lower()
        ok = expect in pkgname.lower()
        return ok, pkgname
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

    print(f"共 {len(tasks)} 道题，逐一验证 image 内容是否匹配期望 repo：\n")
    n_ok, n_bad = 0, 0
    bad_list = []
    for t in tasks:
        ok, pkgname = check_task(ags, tool_id, t)
        status = "OK" if ok else "MISMATCH"
        print(f"  {t['task_id']:20s} 期望={t['repo']:25s} 实际={pkgname:40s} [{status}]")
        if ok:
            n_ok += 1
        else:
            n_bad += 1
            bad_list.append(t["task_id"])

    print(f"\n汇总：{n_ok} 正常 / {n_bad} 异常（共 {len(tasks)}）")
    print("异常 task_id 列表:", bad_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
