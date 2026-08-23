"""VERL 自定义 reward function（线 B 训练侧唯一需要实现的钩子）。

===========================================================================
接口依据（已用官方文档 + 社区实测记录核实，结论写入 PROGRESS.md）
===========================================================================
VERL 的 `custom_reward_function` 扩展点要求实现一个函数：

    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

并通过训练启动命令行传入：
    custom_reward_function.path=<本文件绝对路径>
    custom_reward_function.name=compute_score   # 若函数名就是 compute_score 可省略

VERL 的 `RewardManager.__call__` 会对 batch 里每条样本调用一次本函数：
- `solution_str`：模型生成的响应字符串（已 detokenize），本课题里就是模型输出的
  一个动作序列 / 最终 unified diff patch（取决于我们喂给它的 prompt 设计）
- `ground_truth`：预处理阶段准备好的标准答案字符串，本课题里放的是
  **该样本对应的题目元数据 JSON**（task_id / image / base_commit 等），
  不是字符串意义上的"标准答案"——因为 SWE 任务的"对不对"要跑测试才知道，
  不能用字符串匹配，这是与 GSM8K/MATH 这类题目的本质区别，因此我们复用
  `ground_truth` 这个通道来传"怎么判这道题"的信息，符合该字段"逐样本可变、
  预处理阶段准备好"的设计本意
- `data_source`：数据集名，本课题固定为 "swe_rl"
- `extra_info`：留空未使用

===========================================================================
本课题的实现策略：调用 AGS 沙箱执行 verify.sh 打分（"SandBox 只管执行与
采集，TKE GPU 只管训练"的分工，见 plan.md 2.3 节）
===========================================================================
核心动作：
  1. 从 `ground_truth`（JSON 字符串）里取出题目信息（task_id、镜像地址）
  2. 从 `solution_str` 里抽取模型生成的 unified diff patch
  3. 找一个该题目的沙箱实例（实例池，见 `_InstancePool`）：
     - 有空闲实例 → 复用：`git checkout -- . && git clean -fd` 还原到干净态
     - 没有 → 起一个新实例（`AGSClient.start_instance`，image_override=题目镜像）
  4. `git apply` 应用 patch → 跑 `verify.sh` → 读 `result.json`
  5. 用 `pipeline/reward.py::compute_reward` 统一judge口径算分（与线 A 同一套逻辑，
     避免两条线判分不一致）
  6. 用完的实例放回池子（不 kill），供下一个 sample 复用——这是 plan.md 2.2 节
     "三层降本"里第①层（实例复用），②并发③缓存留给 Phase 1.5 实测后按需接入

失败兜底：沙箱调用异常 / patch 应用失败 / 判据脚本出错，统一返回 0.0（不让
基础设施错误被误判为"模型能力信号"，与 `pipeline/reward.py` 的口径一致）。

⚠️ 本文件运行在 TKE 训练侧（GPU 节点上的 VERL 进程内），不运行在沙箱内，
可以自由 import 第三方包（tencentcloud-sdk-python 等训练侧已装好的依赖）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 腾讯云 AGS 的 API Key 格式与 E2B 官方不同，需要关掉 e2b-code-interpreter SDK
# 的 Key 格式校验才能连上 AGS（沿用课题三 `sandbox_runner.py` 的实测结论）。
# 必须在本文件任何地方 import e2b_code_interpreter 之前设置。
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

from pipeline.reward import compute_reward  # noqa: E402

DATA_SOURCE_NAME = "swe_rl"
DEFAULT_TOOL_NAME = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-rl-runner")
VERIFY_TIMEOUT_SEC = int(os.environ.get("REWARD_VERIFY_TIMEOUT_SEC", "300"))
INSTANCE_START_TIMEOUT = os.environ.get("REWARD_INSTANCE_TIMEOUT", "15m")


# ---------------------------------------------------------------- patch 抽取

_DIFF_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def extract_patch(solution_str: str) -> str:
    """从模型输出里抽取 unified diff。优先取代码块，否则找 `--- a/` 起始的部分。"""
    if not solution_str:
        return ""
    m = _DIFF_BLOCK_RE.search(solution_str)
    if m:
        return m.group(1).strip()
    idx = solution_str.find("--- a/")
    if idx == -1:
        idx = solution_str.find("diff --git")
    if idx != -1:
        return solution_str[idx:].strip()
    return solution_str.strip()


# ---------------------------------------------------------------- 实例池（三层降本第①层）

class _InstancePool:
    """按 task_id 维护一小池「常驻」沙箱实例，跑完打分不 kill，留给下一个 sample 复用。

    Phase 1.5 实测后可调整 `max_per_task`；当前先给出正确、线程安全的最小实现，
    调用侧串行/并行都能工作（用 Lock 保护实例的借出/归还，不保护跑分过程本身，
    因此外部可以并发调用 `score_one`，天然支持 plan.md 2.2 节的第②层"并发"）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle: dict[str, list[str]] = {}  # task_id -> [instance_id, ...]
        self._tool_id: str | None = None

    def _ensure_tool(self):
        from clients.ags import AGSClient

        ags = AGSClient()
        if self._tool_id is None:
            tool = ags.find_tool(DEFAULT_TOOL_NAME)
            if not tool:
                raise RuntimeError(
                    f"沙箱工具 {DEFAULT_TOOL_NAME} 不存在，训练前需先执行 "
                    f"`python driver.py setup-tool --image <占位镜像>`"
                )
            self._tool_id = tool["tool_id"]
        return ags, self._tool_id

    def acquire(self, task_id: str, image: str):
        """借一个该题目的沙箱实例：优先复用空闲的，没有则新起一个。"""
        with self._lock:
            idle_list = self._idle.get(task_id, [])
            if idle_list:
                instance_id = idle_list.pop()
                return instance_id, True  # (instance_id, reused)

        ags, tool_id = self._ensure_tool()
        instance_id, _ = ags.start_instance(
            tool_id, image_override=image, timeout=INSTANCE_START_TIMEOUT,
        )
        return instance_id, False

    def release(self, task_id: str, instance_id: str) -> None:
        """打分完毕，把实例放回空闲池（不 kill，供下一个 sample 复用）。"""
        with self._lock:
            self._idle.setdefault(task_id, []).append(instance_id)

    def drop(self, task_id: str, instance_id: str) -> None:
        """实例已损坏（如复用还原失败），直接销毁，不放回池子。"""
        try:
            from clients.ags import AGSClient
            AGSClient().stop_instance(instance_id)
        except Exception:  # noqa: BLE001
            pass

    def shutdown_all(self) -> None:
        """训练结束调用：清空池子里所有常驻实例，避免持续计费。"""
        with self._lock:
            all_ids = [iid for ids in self._idle.values() for iid in ids]
            self._idle.clear()
        if not all_ids:
            return
        from clients.ags import AGSClient
        ags = AGSClient()
        for iid in all_ids:
            try:
                ags.stop_instance(iid)
            except Exception:  # noqa: BLE001
                pass


_POOL = _InstancePool()


# ---------------------------------------------------------------- 单次打分

def _score_via_sandbox(task_id: str, image: str, patch: str) -> tuple[float, str]:
    """借实例 → 还原干净态 → apply patch → verify.sh → 判分 → 还实例。返回 (reward, reason)。"""
    from e2b_code_interpreter import Sandbox

    instance_id, reused = _POOL.acquire(task_id, image)
    broken = False
    try:
        sbx = Sandbox.connect(instance_id)

        if reused:
            reset = sbx.commands.run(
                "cd /workspace/repo && git checkout -- . && git clean -fd",
                user="root", timeout=30,
            )
            if reset.exit_code != 0:
                broken = True
                return 0.0, f"实例还原失败（exit={reset.exit_code}），已丢弃该实例"

        if patch.strip():
            patch_content = patch if patch.endswith("\n") else patch + "\n"
            sbx.files.write("/tmp/reward.patch", patch_content, user="root")
            apply_res = sbx.commands.run(
                "cd /workspace/repo && git apply --whitespace=nowarn /tmp/reward.patch",
                user="root", timeout=30,
            )
            if apply_res.exit_code != 0:
                # patch 应用失败视为模型这次生成无效，判 0，不算基础设施错误，
                # 实例本身仍是干净可复用的（apply 失败不会改动仓库）
                return 0.0, f"patch 应用失败（模型输出的 diff 不合法）：{(apply_res.stderr or '')[:300]}"
        else:
            return 0.0, "solution_str 中未抽取到有效 patch"

        verify_res = sbx.commands.run(
            "cd /workspace/repo && bash /task/verify.sh",
            user="root", timeout=VERIFY_TIMEOUT_SEC,
            envs={"PYTEST_ADDOPTS": "--color=no"},
        )
        try:
            result_raw = sbx.files.read("/task/result.json", user="root")
            result_json = json.loads(result_raw)
        except Exception as e:  # noqa: BLE001
            return 0.0, f"读取/解析 result.json 失败：{e}（verify exit={verify_res.exit_code}）"

        rr = compute_reward(result_json)
        return rr.reward, rr.reason
    except Exception as e:  # noqa: BLE001
        broken = True
        return 0.0, f"沙箱调用异常：{type(e).__name__}: {e}"
    finally:
        if broken:
            _POOL.drop(task_id, instance_id)
        else:
            _POOL.release(task_id, instance_id)


# ---------------------------------------------------------------- 缓存（三层降本第③层）

_cache_lock = threading.Lock()
_reward_cache: dict[str, tuple[float, str]] = {}


def _cache_key(task_id: str, patch: str) -> str:
    import hashlib
    h = hashlib.sha256(patch.encode("utf-8")).hexdigest()[:16]
    return f"{task_id}:{h}"


# ---------------------------------------------------------------- VERL 入口函数

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """VERL `custom_reward_function` 的标准入口签名，见文件头说明。"""
    try:
        meta = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        return 0.0

    task_id = meta.get("task_id", "unknown")
    image = meta.get("image")
    if not image:
        return 0.0

    patch = extract_patch(solution_str)
    key = _cache_key(task_id, patch)

    with _cache_lock:
        cached = _reward_cache.get(key)
    if cached is not None:
        return cached[0]

    reward, reason = _score_via_sandbox(task_id, image, patch)

    with _cache_lock:
        _reward_cache[key] = (reward, reason)
    return reward


# ---------------------------------------------------------------- 本地自测（不连云，纯逻辑）

if __name__ == "__main__":
    # 只测 extract_patch 与 cache key 的纯函数逻辑，不连真实 AGS（那部分需要
    # Phase 1.5 云上预热之后、在 TKE 侧用真实沙箱做集成测试）。
    sample_1 = """好的，我来修复这个 bug：

```diff
--- a/mock.py
+++ b/mock.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
```
修复完成。"""
    p1 = extract_patch(sample_1)
    assert "return a + b" in p1, f"抽取失败：{p1!r}"
    print("[OK] extract_patch 代码块场景通过")

    sample_2 = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    p2 = extract_patch(sample_2)
    assert p2 == sample_2.strip(), f"抽取失败：{p2!r}"
    print("[OK] extract_patch 裸 diff 场景通过")

    sample_3 = "我无法修复这个问题。"
    p3 = extract_patch(sample_3)
    assert p3 == sample_3.strip()
    print("[OK] extract_patch 无 diff 场景通过（原样返回，后续会在 _score_via_sandbox 判 0）")

    k1 = _cache_key("t1", "patch-A")
    k2 = _cache_key("t1", "patch-A")
    k3 = _cache_key("t1", "patch-B")
    assert k1 == k2 and k1 != k3
    print("[OK] cache key 一致性通过")

    print("\n全部本地纯函数自测通过。云端集成测试（沙箱实例池/verify.sh 打分）见 Phase 1.5。")
