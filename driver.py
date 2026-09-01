#!/usr/bin/env python3
"""driver.py —— 本地唯一允许运行的东西：编排脚本（起沙箱 → 注入 agent.py →
运行 → 收 tracing → 传 COS），本身不读题、不生成代码、不跑测试、不做任何解题
判断。真正"解题"的逻辑（ReAct 循环、读文件、打 patch、跑测试）全部在
`sandbox_agent/agent.py` 里，且只会被注入到远程 AGS 沙箱内执行。

依赖极简（`requirements.txt`）：tencentcloud-sdk-python + e2b-code-interpreter +
cos-python-sdk-v5 + python-dotenv，都是调云端 API 的 SDK，没有 torch/vllm 这类
重量级依赖 —— 这是刻意设计：客户 clone 本仓库后，本地只需要
`pip install -r requirements.txt`（几秒钟量级）+ 配置 .env，就能立刻执行
`python driver.py rollout ...` 远程跑起整条链路，不需要在本机装任何 AI 相关
的运行时。

用法：
    python driver.py setup-tool   --image <占位镜像>                  # 建 1 个共享沙箱工具
    python driver.py rollout      --tasks data/tasks.jsonl --limit 3   # 真实跑（默认路径）
    python driver.py rollout      --mock  --mock-dir /tmp/xxx          # 本地 mock（仅开发自测，见下）
    python driver.py cleanup-tool                                      # 用完删除工具，释放配额

⚠️ `--mock` 仅用于我们开发期自测 driver.py 与 agent.py 的注入/收集协议是否
   对得上（不连真实 AGS，走本地临时目录模拟 sbx.files.write/commands.run/
   files.read），不是交付给客户使用的路径，也不产生任何"解题"结果。客户
   实际使用时默认（不加 --mock）就是全程远程。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 腾讯云 AGS 的 API Key 格式（"ark_xxx"）与 E2B 官方不同，e2b-code-interpreter
# SDK 默认会校验 Key 前缀格式，需要关掉这个校验才能连上 AGS（沿用课题三
# `sandbox_runner.py` 的实测结论）。必须在 import e2b_code_interpreter 之前设置。
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

AGENT_PY_PATH = ROOT / "sandbox_agent" / "agent.py"
DEFAULT_TASKS_PATH = ROOT / "data" / "tasks.jsonl"
DEFAULT_TOOL_NAME = os.environ.get("AGS_TOOL_NAME", "swe-rl-runner")


def load_env() -> None:
    """加载 .env（本地唯一读取的配置来源，凭证不落代码）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# ============================================================== 沙箱抽象
# 真实模式用 e2b_code_interpreter.Sandbox；mock 模式用下面这个本地实现，
# 两者暴露相同的 files.write / files.read / commands.run 接口，driver 主流程
# 因此不需要 if/else 分叉，只在"怎么拿到一个 sandbox 对象"这一步分叉。

class _MockFiles:
    def __init__(self, base: Path):
        self.base = base

    def _resolve(self, path: str) -> Path:
        """把一个路径映射到 mock 根目录下的真实文件系统路径。

        兼容两种输入：
          1. 已经是本机真实绝对路径且落在 `base` 之下（如 driver.py 直接把
             `f"{mock_dir}/task"` 当 task_dir 传给 agent.py 子进程用）——原样返回，
             不重复拼接；
          2. "沙箱风格"的绝对路径（如真实沙箱里的 "/task/x"）——按 sandbox 语义
             映射到 `base/x`。
        """
        p = Path(path)
        if p.is_absolute():
            try:
                p.relative_to(self.base)
                return p
            except ValueError:
                pass
        rel = str(path).lstrip("/")
        return self.base / rel

    def write(self, path: str, content: str, user: str | None = None) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read(self, path: str, user: str | None = None) -> str:
        p = self._resolve(path)
        return p.read_text(encoding="utf-8")


class _MockCommandResult:
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _MockCommands:
    def __init__(self, base: Path):
        self.base = base

    def run(self, cmd: str, user: str | None = None, timeout: int | None = None,
             envs: dict[str, str] | None = None) -> _MockCommandResult:
        full_env = os.environ.copy()
        full_env.update(envs or {})
        # 真实沙箱里文件系统是独立的容器 rootfs，"/root/agent.py" 天然存在；
        # mock 模式共用本机文件系统，没有独立 root，这里把 run_one_task 写死的
        # "/root/agent.py" 换算成本次 mock 用的临时目录下的实际路径
        # （仅 mock shim 内部处理，不影响真实路径下发的调用方代码）。
        resolved_cmd = cmd.replace("/root/agent.py", str(self.base / "root" / "agent.py"))
        proc = subprocess.run(
            resolved_cmd, shell=False if isinstance(resolved_cmd, list) else True,
            cwd=str(self.base), env=full_env,
            capture_output=True, text=True, timeout=timeout,
        )
        return _MockCommandResult(proc.stdout, proc.stderr, proc.returncode)


class MockSandbox:
    """本地目录模拟一个沙箱实例，供 --mock 自测用（见文件头说明）。"""

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.files = _MockFiles(self.base)
        self.commands = _MockCommands(self.base)


# ============================================================== 单题 rollout

def run_one_task(
    task: dict,
    *,
    sandbox_factory,
    llm_endpoint: str,
    llm_api_key: str,
    model_name: str,
    max_steps: int,
    repo_dir: str,
    task_dir: str,
    stop_instance_fn=None,
) -> dict:
    """在一个（已启动好的）沙箱里注入 agent.py 并跑一次 episode，返回该 episode dict。

    `sandbox_factory()` 返回一个具备 files/commands 接口的对象（真实 Sandbox 或
    MockSandbox），把"怎么拿到沙箱"这一步交给调用方，本函数只管注入 + 运行 + 收集，
    这是"driver 只做编排、不做解题判断"的具体体现——本函数甚至不解析 tracing 内容，
    原样透传给上层写盘/传 COS。
    """
    sbx = sandbox_factory()
    episode_id = f"ep-{task.get('task_id', 'unknown')}-{uuid.uuid4().hex[:8]}"

    agent_code = AGENT_PY_PATH.read_text(encoding="utf-8")
    sbx.files.write("/root/agent.py", agent_code, user="root")

    env = {
        "LLM_ENDPOINT": llm_endpoint,
        "LLM_API_KEY": llm_api_key,
        "MODEL_NAME": model_name,
        "MAX_STEPS": str(max_steps),
        "EPISODE_ID": episode_id,
        "REPO_DIR": repo_dir,
        "TASK_DIR": task_dir,
    }

    # 题目镜像里裸 "python3" 不在 PATH 里（登录 shell 也一样，实测确认），
    # 真正的解释器在题目仓库自带的 venv `/opt/venv311/bin/python3`
    # （镜像统一约定，`verify.sh` 内部也是引用这个路径，见课题三 Dockerfile 契约）。
    # agent.py 本身只用标准库，用这个 venv 的 python3 跑没有兼容性问题。
    result = sbx.commands.run(
        "/opt/venv311/bin/python3 /root/agent.py", user="root", timeout=900, envs=env,
    )

    tracing_path = f"{task_dir.rstrip('/')}/tracing.jsonl"
    try:
        tracing_raw = sbx.files.read(tracing_path, user="root")
    except Exception as e:  # noqa: BLE001
        return {
            "episode_id": episode_id, "task_id": task.get("task_id", "unknown"),
            "model_version": model_name, "steps": [], "final_reward": 0.0,
            "fail_to_pass_rate": "0/0", "pass_to_pass_rate": "0/0", "num_steps": 0,
            "started_at": time.time(), "finished_at": time.time(),
            "error": f"读取 tracing 失败：{e}；agent stdout(截取)：{(result.stdout or '')[:500]}",
        }
    finally:
        if stop_instance_fn:
            try:
                stop_instance_fn()
            except Exception:  # noqa: BLE001
                pass

    # tracing.jsonl 可能因为重试/多次运行有多行，取最后一行（本次运行产出的那条）
    lines = [l for l in tracing_raw.splitlines() if l.strip()]
    if not lines:
        return {
            "episode_id": episode_id, "task_id": task.get("task_id", "unknown"),
            "model_version": model_name, "steps": [], "final_reward": 0.0,
            "fail_to_pass_rate": "0/0", "pass_to_pass_rate": "0/0", "num_steps": 0,
            "started_at": time.time(), "finished_at": time.time(),
            "error": f"tracing.jsonl 为空；agent exit_code={result.exit_code}",
        }
    return json.loads(lines[-1])


# ============================================================== 子命令：setup-tool / cleanup-tool

def cmd_setup_tool(args: argparse.Namespace) -> int:
    from clients.ags import AGSClient, AGSError

    ags = AGSClient()
    existing = ags.find_tool(args.tool_name)
    if existing:
        print(f"工具 {args.tool_name} 已存在（tool_id={existing['tool_id']}，status={existing['status']}），不重复创建")
        tool_id = existing["tool_id"]
    else:
        print(f"创建沙箱工具 {args.tool_name}（占位镜像={args.image}，题目镜像会在 rollout 时用 image_override 逐题覆盖）...")
        tool_id = ags.create_tool(args.tool_name, args.image)
        print(f"已创建 tool_id={tool_id}，等待变为 ACTIVE...")
    try:
        ags.wait_tool_active(args.tool_name, timeout=args.wait_timeout)
        print("工具已就绪（ACTIVE）")
    except AGSError as e:
        print(f"❌ {e}")
        return 1
    return 0


def cmd_cleanup_tool(args: argparse.Namespace) -> int:
    from clients.ags import AGSClient

    ags = AGSClient()
    existing = ags.find_tool(args.tool_name)
    if not existing:
        print(f"工具 {args.tool_name} 不存在，无需清理")
        return 0
    ags.delete_tool(existing["tool_id"])
    print(f"已删除工具 {args.tool_name}（tool_id={existing['tool_id']}）")
    return 0


# ============================================================== 子命令：rollout

def cmd_rollout(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if args.task_id:
        tasks = [t for t in tasks if t.get("task_id") == args.task_id]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("没有匹配到任何题目，检查 --tasks / --task-id / --limit")
        return 1

    episodes: list[dict] = []

    if args.mock:
        print(f"[mock] 使用本地目录模拟沙箱：{args.mock_dir}")
        # 注意：repo_dir/task_dir 这里必须是本机真实存在的绝对路径——它们会被
        # 原样透传给 agent.py 子进程当 REPO_DIR/TASK_DIR 环境变量用
        # （agent.py 内部是直接 open()/subprocess cwd=，不经过任何 mock 路径转换）。
        # MockFiles._resolve 对"已经落在 base 之下的绝对路径"会原样直通，
        # 不会重复拼接，所以这里可以放心用真实路径。
        mock_repo_dir = args.mock_repo_dir or f"{args.mock_dir}/repo"
        mock_task_dir = args.mock_task_dir or f"{args.mock_dir}/task"
        for task in tasks:
            print(f"[mock] rollout task={task.get('task_id')}")
            ep = run_one_task(
                task,
                sandbox_factory=lambda: MockSandbox(args.mock_dir),
                llm_endpoint=args.llm_endpoint,
                llm_api_key=args.llm_api_key,
                model_name=args.model_name,
                max_steps=args.max_steps,
                repo_dir=mock_repo_dir,
                task_dir=mock_task_dir,
            )
            episodes.append(ep)
            print(f"  -> reward={ep.get('final_reward')} steps={ep.get('num_steps')} error={ep.get('error')}")
    else:
        from clients.ags import AGSClient, AGSError

        ags = AGSClient()
        tool = ags.find_tool(args.tool_name)
        if not tool:
            print(f"❌ 沙箱工具 {args.tool_name} 不存在，先执行 `python driver.py setup-tool --image <占位镜像>`")
            return 1
        tool_id = tool["tool_id"]

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            image = task.get("image")
            if not image:
                print(f"[skip] {task_id} 缺少 image 字段")
                continue
            print(f"rollout task={task_id} image={image} ...")
            try:
                instance_id, effective_image = ags.start_instance(
                    tool_id, image_override=image, timeout=args.instance_timeout,
                )
            except AGSError as e:
                print(f"  ❌ 起沙箱失败：{e}")
                episodes.append({
                    "episode_id": f"ep-{task_id}-failed", "task_id": task_id,
                    "model_version": args.model_name, "steps": [], "final_reward": 0.0,
                    "fail_to_pass_rate": "0/0", "pass_to_pass_rate": "0/0", "num_steps": 0,
                    "started_at": time.time(), "finished_at": time.time(),
                    "error": f"起沙箱失败：{e}",
                })
                continue

            from e2b_code_interpreter import Sandbox  # noqa: PLC0415

            def _factory(instance_id=instance_id):
                return Sandbox.connect(instance_id)

            def _stop(instance_id=instance_id):
                ags.stop_instance(instance_id)

            ep = run_one_task(
                task,
                sandbox_factory=_factory,
                llm_endpoint=args.llm_endpoint,
                llm_api_key=args.llm_api_key,
                model_name=args.model_name,
                max_steps=args.max_steps,
                repo_dir="/workspace/repo",
                task_dir="/task",
                stop_instance_fn=_stop,
            )
            episodes.append(ep)
            print(f"  -> reward={ep.get('final_reward')} steps={ep.get('num_steps')} error={ep.get('error')}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"\n共 {len(episodes)} 条 episode 写入 {out_path}")

    n_ok = sum(1 for ep in episodes if not ep.get("error") and ep.get("num_steps", 0) >= 1)
    print(f"其中成功产出 tracing（无 error）：{n_ok}/{len(episodes)}")

    if args.upload_cos and not args.mock:
        from clients import cos as cos_client
        key = args.cos_key or f"tracing/{int(time.time())}_{out_path.name}"
        url = cos_client.upload_file(args.cos_bucket, key, str(out_path))
        print(f"已上传 COS：{url}")

    return 0 if n_ok == len(episodes) else 2


# ============================================================== CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="题目四：强化学习 —— 本地编排入口（不做任何解题逻辑）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup-tool", help="创建/等待共享沙箱工具就绪")
    p_setup.add_argument("--tool-name", default=DEFAULT_TOOL_NAME)
    p_setup.add_argument("--image", required=True, help="占位镜像（真正题目镜像在 rollout 时逐题 image_override）")
    p_setup.add_argument("--wait-timeout", type=float, default=180)
    p_setup.set_defaults(func=cmd_setup_tool)

    p_cleanup = sub.add_parser("cleanup-tool", help="删除共享沙箱工具，释放配额")
    p_cleanup.add_argument("--tool-name", default=DEFAULT_TOOL_NAME)
    p_cleanup.set_defaults(func=cmd_cleanup_tool)

    p_roll = sub.add_parser("rollout", help="对题目池跑多轮 ReAct，产出 tracing")
    p_roll.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH))
    p_roll.add_argument("--task-id", default=None, help="只跑某一道题（调试用）")
    p_roll.add_argument("--limit", type=int, default=0, help="只跑前 N 道题，0=不限制")
    p_roll.add_argument("--tool-name", default=DEFAULT_TOOL_NAME)
    p_roll.add_argument("--llm-endpoint", default=os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    p_roll.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    p_roll.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "swe-rl-model"))
    p_roll.add_argument("--max-steps", type=int, default=15)
    p_roll.add_argument("--instance-timeout", default="20m")
    p_roll.add_argument("--output", default=str(ROOT / "data" / "tracing.jsonl"))
    p_roll.add_argument("--upload-cos", action="store_true")
    p_roll.add_argument("--cos-bucket", default=os.environ.get("COS_BUCKET", "COS_BUCKET"))
    p_roll.add_argument("--cos-key", default=None)
    p_roll.add_argument("--mock", action="store_true", help="仅开发自测：本地目录模拟沙箱，不连真实 AGS")
    p_roll.add_argument("--mock-dir", default="/tmp/swe_rl_mock_test")
    p_roll.add_argument("--mock-repo-dir", default=None)
    p_roll.add_argument("--mock-task-dir", default=None)
    p_roll.set_defaults(func=cmd_rollout)

    return ap


def main() -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
