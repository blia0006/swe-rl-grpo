"""沙箱内运行的 SWE Agent 主程序（ReAct 多轮循环）。

⚠️ 运行环境约束（务必遵守，否则会在真沙箱里跑不起来）：
  - 只能用 **Python 标准库**。沙箱镜像里题目仓库的 venv（/opt/venv311）不保证装了
    requests/openai 等第三方包，而且题目之间镜像互不相同，谁都不能假设。
    HTTP 调用一律用 urllib.request。
  - 不依赖本文件之外的任何模块（pipeline/ 目录在沙箱里不存在），tracing 相关的
    最小结构在本文件内直接实现，与 pipeline/schema.py 保持字段一致但不 import 它
    ——避免"沙箱里的 agent.py"和"本机/TKE 侧的 pipeline"产生隐式耦合部署问题。
  - 本文件由 driver.py 通过 sbx.files.write 整份注入沙箱，然后
    `python3 agent.py` 直接执行，不需要 pip install 任何东西。

职责：给定一道 SWE 题目（problem_statement + repo 已在 /workspace/repo 里 checkout
好 base_commit），在沙箱内自主跑多轮 ReAct（读文件 / 跑命令 / 打 patch / 跑测试 /
提交），把每一步和最终 reward 写进 /task/tracing.jsonl，供 driver.py 收走。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# 配置（全部走环境变量，由 driver.py 通过 CreateSandboxInstance 的 Env 注入，
# 不写死任何本机/云端地址，保证同一份 agent.py 在 mock / 真实沙箱下都能跑）
# --------------------------------------------------------------------------
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("MODEL_NAME", "swe-rl-model")
# TASK_DIR 对应镜像契约里固定的 /task（COPY task/ /task/，见课题三 Dockerfile），
# 仅在本地 mock 自测时通过环境变量覆盖为临时目录，真沙箱里永远是 /task。
TASK_DIR = os.environ.get("TASK_DIR", "/task")
# 实测真沙箱镜像契约：题目信息拆成两个文件（而非单一 task.json）：
#   /task/metadata.json        task_id/repo/test_cmd/FAIL_TO_PASS/PASS_TO_PASS 等结构化字段
#   /task/problem_statement.md 纯文本问题描述（Markdown）
METADATA_JSON_PATH = os.environ.get("METADATA_JSON_PATH", os.path.join(TASK_DIR, "metadata.json"))
PROBLEM_STATEMENT_PATH = os.environ.get(
    "PROBLEM_STATEMENT_PATH", os.path.join(TASK_DIR, "problem_statement.md")
)
REPO_DIR = os.environ.get("REPO_DIR", "/workspace/repo")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", os.path.join(TASK_DIR, "tracing.jsonl"))
VERIFY_SCRIPT_PATH = os.environ.get("VERIFY_SCRIPT_PATH", os.path.join(TASK_DIR, "verify.sh"))
RESULT_JSON_PATH = os.environ.get("RESULT_JSON_PATH", os.path.join(TASK_DIR, "result.json"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "15"))
LLM_TIMEOUT_SEC = int(os.environ.get("LLM_TIMEOUT_SEC", "120"))
CMD_TIMEOUT_SEC = int(os.environ.get("CMD_TIMEOUT_SEC", "300"))
MAX_OBS_CHARS = 4000
EPISODE_ID = os.environ.get("EPISODE_ID") or f"ep-{int(time.time() * 1000)}"

TOOLS_DOC = """\
你是一个自主修复代码仓库 bug 的 SWE Agent。你只能通过下面这些工具与环境交互，
每次回复必须是且只能是一个 JSON 对象（不要输出任何多余文字/代码块围栏），格式：

  {"tool": "<工具名>", "args": {...}}

可用工具：
  - read_file: {"path": "相对或绝对路径"}  查看文件内容
  - bash: {"cmd": "shell 命令"}  在仓库根目录执行任意 shell 命令（如 grep/find/ls）
  - apply_patch: {"patch": "unified diff 文本"}  用 git apply 应用一个 patch
  - run_tests: {}  运行判据脚本，返回通过/失败的用例统计（不会告诉你具体断言细节，
    避免你直接对着答案抄）
  - submit: {}  确认修复完成，结束本次任务（会自动跑一次最终判据）

策略建议：先用 read_file / bash 理解问题定位代码，再用 apply_patch 提交修复，
用 run_tests 检查效果，反复迭代直到测试通过或你判断已尽力后 submit。
"""


def _truncate(text: str, limit: int = MAX_OBS_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(truncated, total {len(text)} chars)"


def call_llm(messages: list[dict]) -> str:
    """调用 OpenAI 兼容的 chat/completions 接口（GPU Pod 内 `scripts/pod_hf_serve.py`
    提供，纯 transformers 实现，非 vLLM——实测 GPU 机型是 P4，vLLM 官方不支持其算力），
    纯 stdlib 实现。"""
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SEC) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"调用 LLM 失败（已重试 3 次）：{last_err}")


def parse_action(raw: str) -> dict:
    """从模型输出里提取 JSON 动作。容错：模型偶尔会包 ```json 代码块或加多余文字。"""
    raw = raw.strip()
    # 优先尝试整体就是 JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试提取第一个 {...} 块
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"tool": "invalid", "args": {"raw": raw}}


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------

def tool_read_file(args: dict) -> str:
    path = args.get("path", "")
    if not path:
        return "错误：缺少 path 参数"
    full = path if os.path.isabs(path) else os.path.join(REPO_DIR, path)
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except OSError as e:
        return f"读取失败：{e}"


def tool_bash(args: dict) -> str:
    cmd = args.get("cmd", "")
    if not cmd:
        return "错误：缺少 cmd 参数"
    try:
        # bash 是 Agent 修复代码的核心能力（在隔离沙箱容器内执行），非任意外部
        # 输入拼接注入——cmd 就是模型这一步"想执行的动作"本身。
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT_SEC,
        )
        out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"[exit={proc.returncode}]\n{out}" if out.strip() else f"[exit={proc.returncode}] (无输出)"
    except subprocess.TimeoutExpired:
        return f"命令超时（>{CMD_TIMEOUT_SEC}s）"
    except OSError as e:
        return f"执行失败：{e}"


def tool_apply_patch(args: dict) -> str:
    patch = args.get("patch", "")
    if not patch:
        return "错误：缺少 patch 参数"
    if not patch.endswith("\n"):
        patch += "\n"
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=REPO_DIR,
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return "patch 应用成功"
        # 失败时尝试 --check 给出更明确的错误原因
        return f"patch 应用失败（exit={proc.returncode}）：\n{proc.stderr}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"patch 应用异常：{e}"


def _read_result_json() -> dict | None:
    if not os.path.exists(RESULT_JSON_PATH):
        return None
    try:
        with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def tool_run_tests(args: dict) -> tuple[str, dict | None]:
    try:
        proc = subprocess.run(
            ["bash", VERIFY_SCRIPT_PATH],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return f"判据脚本超时（>{CMD_TIMEOUT_SEC}s）", None
    except OSError as e:
        return f"判据脚本执行异常：{e}", None

    result = _read_result_json()
    if result is None:
        return f"判据脚本运行完毕（exit={proc.returncode}），但未生成 result.json，可能是环境错误", None

    f2p = result.get("fail_to_pass", {})
    p2p = result.get("pass_to_pass", {})
    summary = (
        f"FAIL_TO_PASS: {f2p.get('passed', 0)}/{f2p.get('total', 0)} 通过；"
        f"PASS_TO_PASS: {p2p.get('passed', 0)}/{p2p.get('total', 0)} 通过（不能有回归）"
    )
    return summary, result


# --------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------

def load_task() -> dict:
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        task = json.load(f)
    try:
        with open(PROBLEM_STATEMENT_PATH, "r", encoding="utf-8") as f:
            task["problem_statement"] = f.read()
    except OSError:
        task.setdefault("problem_statement", "")
    return task


def compute_reward(result: dict | None) -> tuple[float, str, str]:
    """与 pipeline/reward.py 逻辑保持一致（此处独立实现，见文件头说明）。"""
    if not result:
        return 0.0, "0/0", "0/0"
    f2p = result.get("fail_to_pass", {})
    p2p = result.get("pass_to_pass", {})
    f2p_total = int(f2p.get("total", 0))
    f2p_passed = int(f2p.get("passed", 0))
    p2p_total = int(p2p.get("total", 0))
    p2p_passed = int(p2p.get("passed", 0))
    f2p_rate = f"{f2p_passed}/{f2p_total}"
    p2p_rate = f"{p2p_passed}/{p2p_total}"
    if result.get("collect_error") or f2p_total == 0:
        return 0.0, f2p_rate, p2p_rate
    if p2p_total > 0 and p2p_passed < p2p_total:
        return 0.0, f2p_rate, p2p_rate
    return f2p_passed / f2p_total, f2p_rate, p2p_rate


def run_episode() -> dict:
    task = load_task()
    messages = [
        {"role": "system", "content": TOOLS_DOC},
        {"role": "user", "content": (
            f"任务：{task.get('problem_statement', '')}\n\n"
            f"仓库已 checkout 到 {REPO_DIR}，请开始定位并修复问题。"
        )},
    ]

    steps: list[dict] = []
    final_result: dict | None = None
    error: str | None = None
    started_at = time.time()

    for step_idx in range(MAX_STEPS):
        try:
            raw = call_llm(messages)
        except RuntimeError as e:
            error = str(e)
            break

        action = parse_action(raw)
        tool = action.get("tool")
        args = action.get("args", {})
        done = False
        reward = 0.0

        if tool == "read_file":
            observation = tool_read_file(args)
        elif tool == "bash":
            observation = tool_bash(args)
        elif tool == "apply_patch":
            observation = tool_apply_patch(args)
        elif tool == "run_tests":
            observation, final_result = tool_run_tests(args)
        elif tool == "submit":
            # submit 时强制再跑一次判据兜底计分（防止模型没调用 run_tests 就 submit）
            observation, final_result = tool_run_tests(args)
            done = True
        elif tool == "invalid":
            observation = f"无法解析你的输出为合法动作 JSON，请严格按格式重试。原始输出：{_truncate(args.get('raw', ''), 300)}"
        else:
            observation = f"未知工具：{tool}，可用工具见系统提示"

        is_last = (step_idx == MAX_STEPS - 1)
        if is_last:
            done = True
            if final_result is None:
                # 超步数强制结束，兜底跑一次判据算分
                observation2, final_result = tool_run_tests({})
                observation = observation + f"\n（已达最大步数 {MAX_STEPS}，强制结束并跑最终判据：{observation2}）"

        steps.append({
            "step": step_idx,
            "action": {"tool": tool, "args": args},
            "observation": _truncate(observation),
            "reward": 0.0,  # 中间步 reward 先占位，最终统一在下面算 final_reward
            "done": done,
        })

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {_truncate(observation)}"})

        if done:
            break

    final_reward, f2p_rate, p2p_rate = compute_reward(final_result)
    if steps:
        steps[-1]["reward"] = final_reward

    episode = {
        "episode_id": EPISODE_ID,
        "task_id": task.get("task_id", "unknown"),
        "model_version": MODEL_NAME,
        "steps": steps,
        "final_reward": final_reward,
        "fail_to_pass_rate": f2p_rate,
        "pass_to_pass_rate": p2p_rate,
        "num_steps": len(steps),
        "started_at": started_at,
        "finished_at": time.time(),
        "error": error,
    }
    return episode


def main() -> int:
    try:
        episode = run_episode()
    except Exception as e:  # noqa: BLE001 - 顶层兜底，确保总能写出 tracing 记录错误
        episode = {
            "episode_id": EPISODE_ID,
            "task_id": "unknown",
            "model_version": MODEL_NAME,
            "steps": [],
            "final_reward": 0.0,
            "fail_to_pass_rate": "0/0",
            "pass_to_pass_rate": "0/0",
            "num_steps": 0,
            "started_at": time.time(),
            "finished_at": time.time(),
            "error": f"{type(e).__name__}: {e}",
        }

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    print(f"episode done: task={episode['task_id']} reward={episode['final_reward']} "
          f"steps={episode['num_steps']} error={episode['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
