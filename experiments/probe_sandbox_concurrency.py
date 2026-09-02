"""Phase 1.5 门禁项：沙箱关键数字实测（并发度 / 复用耗时），结论用于定
`group_size` 与训练时 reward function 的并发数（见 plan.md 2.2 节）。

复用已注册的共享沙箱工具 `swe-synth-shared-runner`（不新建工具，账号沙箱工具
配额紧张），用 `data/tasks.jsonl` 里的真实题目镜像做压测，实测三件事：

  1. 冷启动耗时：`start_instance()` 到实例可连接、可执行命令，要多久
  2. 热复用耗时：同一个实例上用 tar 快照还原（镜像内 `/workspace/repo` 不含
     `.git`，`git checkout`/`git clean` 不可行，见下方"还原方案"说明）
     + `git apply`（打 golden patch）+ `bash verify.sh`（判分）一次完整循环要多久
     —— 这是训练时每个 rollout 样本打分的稳态耗时，直接决定单机吞吐
  3. 并发稳定性：同时起 N 个实例（分别对应不同题目），看是否都能成功、
     耗时是否比串行明显变差（判断账号地域资源是否够撑并发）

还原方案（2026-08-23 单实例真实沙箱实测确认）：
  镜像内 `/workspace/repo` 是纯文件拷贝，不含 `.git` 目录，因此不能用
  `git checkout -- . && git clean -fd`。改用 tar 快照：
    - 首次使用：`tar czf /tmp/pristine.tar.gz -C / workspace/repo` 建快照（实测 ~0.3s）
    - 复用还原：`rm -rf /workspace/repo && tar xzf /tmp/pristine.tar.gz -C /`（实测 ~0.05s）
  `git apply` 本身不依赖 `.git` 历史，只按文件路径打 patch，因此仍可正常使用。
  另外 `verify.sh` 判不通过时进程退出码非 0，`commands.run()` 默认会抛
  `CommandExitException`，需要 try/except 后改读 `/task/result.json` 判断，
  不能依赖调用是否抛异常。

用法：
    cd <repo_root>
    source .venv/bin/activate
    python3 experiments/probe_sandbox_concurrency.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

SHARED_TOOL_NAME = "swe-synth-shared-runner"
N_CONCURRENT = 3  # 与 N_CONCURRENT 道不同题目分别起实例
REPO_DIR = "/workspace/repo"
PRISTINE_SNAPSHOT = "/tmp/pristine.tar.gz"


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        print(f"[FATAL] 找不到 {env_path}，请先 cp .env.example .env 并填入真实值")
        sys.exit(1)
    from dotenv import load_dotenv
    load_dotenv(env_path)


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


def probe_one(ags, tool_id: str, task: dict) -> dict:
    task_id = task["task_id"]
    image = task["image"]
    result: dict = {"task_id": task_id}

    t0 = time.time()
    try:
        instance_id, effective_image = ags.start_instance(
            tool_id, image_override=image, timeout="15m",
        )
    except Exception as e:  # noqa: BLE001
        result["error"] = f"start_instance 失败：{e}"
        result["cold_start_sec"] = time.time() - t0
        return result
    result["cold_start_sec"] = round(time.time() - t0, 2)
    result["instance_id"] = instance_id

    from e2b_code_interpreter import Sandbox

    try:
        sbx = Sandbox.connect(instance_id)

        # 建立 pristine 快照（新实例首次使用时做一次，本次探测每题都是新实例）
        snap_res = sbx.commands.run(
            f"tar czf {PRISTINE_SNAPSHOT} -C / workspace/repo",
            user="root", timeout=30,
        )
        result["snapshot_exit"] = snap_res.exit_code

        # 场景 A：应用镜像自带的 golden.patch（/opt/solution/golden.patch）→ verify
        t1 = time.time()
        golden_check = sbx.commands.run(
            "test -f /opt/solution/golden.patch && echo yes || echo no",
            user="root", timeout=10,
        )
        has_golden = golden_check.stdout.strip() == "yes"
        if has_golden:
            apply_res = sbx.commands.run(
                f"cp /opt/solution/golden.patch /tmp/g.patch && cd {REPO_DIR} && "
                f"git apply --whitespace=nowarn /tmp/g.patch",
                user="root", timeout=30,
            )
            result["apply_exit"] = apply_res.exit_code
        else:
            result["apply_exit"] = None

        verify_exit = _run_verify(sbx)
        result["verify_exit"] = verify_exit
        result["warm_cycle_sec"] = round(time.time() - t1, 2)
        result["passed_after_golden"] = _read_passed(sbx)

        # 场景 B：tar 还原到干净态 → 不打任何 patch → 再次 verify（验证还原有效 + 空解应 fail）
        t2 = time.time()
        restore_res = sbx.commands.run(
            f"rm -rf {REPO_DIR} && tar xzf {PRISTINE_SNAPSHOT} -C /",
            user="root", timeout=30,
        )
        result["restore_exit"] = restore_res.exit_code
        verify_exit2 = _run_verify(sbx)
        result["restore_cycle_sec"] = round(time.time() - t2, 2)
        result["passed_after_restore_empty"] = _read_passed(sbx)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"沙箱内操作异常：{type(e).__name__}: {e}"
    finally:
        try:
            ags.stop_instance(instance_id)
        except Exception:  # noqa: BLE001
            pass
    return result


def _run_verify(sbx) -> int | str:
    """跑 verify.sh。判不通过时退出码非 0，SDK 默认对非 0 退出码抛异常（实测确认），
    这里统一吞掉异常，返回码只用于日志展示，真正判分靠读 result.json。"""
    try:
        r = sbx.commands.run(f"cd {REPO_DIR} && bash /task/verify.sh", user="root", timeout=300)
        return r.exit_code
    except Exception as e:  # noqa: BLE001
        return getattr(e, "exit_code", f"EXC:{e}")


def _read_passed(sbx) -> bool | None:
    try:
        raw = sbx.files.read("/task/result.json", user="root")
        return json.loads(raw).get("passed")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    load_env()
    from clients.ags import AGSClient

    ags = AGSClient()
    tool = ags.find_tool(SHARED_TOOL_NAME)
    if not tool:
        print(f"[FATAL] 共享工具 {SHARED_TOOL_NAME} 不存在，改用其他已注册工具或先创建")
        return 1
    tool_id = tool["tool_id"]
    print(f"复用工具：{SHARED_TOOL_NAME} (tool_id={tool_id})")

    tasks = load_tasks(N_CONCURRENT)
    if not tasks:
        print("[FATAL] data/tasks.jsonl 没有可用题目")
        return 1
    print(f"取 {len(tasks)} 道题目并发起实例：{[t['task_id'] for t in tasks]}")

    t_start = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futs = {pool.submit(probe_one, ags, tool_id, t): t["task_id"] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  [done] {r}")
    total_wall = round(time.time() - t_start, 2)

    print("\n========== 汇总 ==========")
    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"并发数：{len(tasks)}  成功：{n_ok}/{len(tasks)}  总耗时：{total_wall}s")
    cold_times = [r["cold_start_sec"] for r in results if "cold_start_sec" in r]
    warm_times = [r["warm_cycle_sec"] for r in results if "warm_cycle_sec" in r]
    restore_times = [r["restore_cycle_sec"] for r in results if "restore_cycle_sec" in r]
    if cold_times:
        print(f"冷启动耗时：min={min(cold_times)}s max={max(cold_times)}s avg={round(sum(cold_times)/len(cold_times),2)}s")
    if warm_times:
        print(f"golden patch 判分耗时：min={min(warm_times)}s max={max(warm_times)}s avg={round(sum(warm_times)/len(warm_times),2)}s")
    if restore_times:
        print(f"tar 还原+空解判分耗时：min={min(restore_times)}s max={max(restore_times)}s avg={round(sum(restore_times)/len(restore_times),2)}s")
    n_golden_pass = sum(1 for r in results if r.get("passed_after_golden") is True)
    n_empty_fail = sum(1 for r in results if r.get("passed_after_restore_empty") is False)
    print(f"golden patch 判分正确（True）：{n_golden_pass}/{len(tasks)}")
    print(f"还原后空解判分正确（False）：{n_empty_fail}/{len(tasks)}")

    out_path = ROOT / "data" / "sandbox_concurrency_probe_result.json"
    out_path.write_text(json.dumps({
        "n_concurrent": len(tasks), "n_ok": n_ok, "total_wall_sec": total_wall,
        "cold_start_sec": cold_times, "warm_cycle_sec": warm_times,
        "restore_cycle_sec": restore_times, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")
    return 0 if n_ok == len(tasks) else 2


if __name__ == "__main__":
    sys.exit(main())
