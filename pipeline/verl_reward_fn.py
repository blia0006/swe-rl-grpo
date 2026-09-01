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
     - 有空闲实例 → 复用：用 tar 快照还原到干净题目态（镜像内 `/workspace/repo`
       **不含 `.git`**，经 Phase 1.5 真实沙箱实测确认，`git checkout`/`git clean`
       不可行，改用 `tar czf` 首次建快照 + `tar xzf` 还原，实测仅 ~0.3s/~0.05s）
     - 没有 → 起一个新实例（`AGSClient.start_instance`，image_override=题目镜像），
       首次使用时先建一次 pristine 快照
  4. `git apply` 应用 patch（镜像内仓库虽无 `.git` 历史，但 `git apply` 本身只按
     文件路径打 patch，不依赖 `.git`，经实测可用）→ 跑 `verify.sh` → 读 `result.json`
  5. 用 `pipeline/reward.py::compute_reward` 统一judge口径算分（与线 A 同一套逻辑，
     避免两条线判分不一致）
  6. 用完的实例放回池子（不 kill），供下一个 sample 复用——这是 plan.md 2.2 节
     "三层降本"里第①层（实例复用），②并发③缓存留给 Phase 1.5 实测后按需接入

失败兜底：沙箱调用异常 / patch 应用失败 / 判据脚本出错，统一返回 0.0（不让
基础设施错误被误判为"模型能力信号"，与 `pipeline/reward.py` 的口径一致）。

**reward 分段设计（reward shaping，见下方 APPLY_SUCCESS_BONUS 注释）**：
  0.0                    → 没抽到 patch / patch 无法 git apply
  0.2                    → patch 合法（能 apply）但测试没修好
  0.2 + 0.8×(F2P 通过率)  → 部分修对
  1.0                    → F2P 全绿且无 P2P 回归

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
DEFAULT_TOOL_NAME = os.environ.get("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")
VERIFY_TIMEOUT_SEC = int(os.environ.get("REWARD_VERIFY_TIMEOUT_SEC", "300"))
INSTANCE_START_TIMEOUT = os.environ.get("REWARD_INSTANCE_TIMEOUT", "15m")
REPO_DIR = "/workspace/repo"
PRISTINE_SNAPSHOT = "/tmp/pristine.tar.gz"

# ---------------------------------------------------------------- 分段 reward
# 【为什么需要 reward shaping】
# 纯 outcome reward（只看 fail→pass 通过率）在本课题下会陷入死循环：
# Qwen2.5-Coder-1.5B 生成的 unified diff 有 ~75% 是 `corrupt patch`（算不准
# `@@ -X,Y +A,B @@` 的行号），git apply 全失败 → 一组 8 个采样 reward 全 0
# → GRPO 组内 advantage 全 0 → pg_loss=0 → grad_norm=0 → 模型参数完全不更新
# → 下一步还是全 0（实测 71 步 / 8 步两轮训练都复现了这个死循环）。
#
# 【做法】给"patch 能被 git apply 成功"一个小的基础分（0.2），
# 让"格式合法"这个中间能力可被 GRPO 感知，组内比较产生非零 advantage，
# 梯度信号出现，模型先学会写合法 diff，再逐步学会修对 bug。
#
# 【是否偏离课题要求】不偏离：课题定义的 `fail→pass 数 / 相关测试数`
# 仍是主体（TEST_WEIGHT=0.8 权重），P2P 回归判 0 的防 reward-hacking 规则
# 也完整保留（在 pipeline/reward.py 里）。0.2 只是引导项，会在 README 说明。
#
# 最终 reward 取值区间：
#   0.0                      → 没抽到 patch / patch 无法 apply
#   0.2                      → patch 合法但一个 F2P 测试都没修好
#   0.2 + 0.8×(F2P 通过率)   → 部分修对
#   1.0                      → F2P 全绿且无 P2P 回归
APPLY_SUCCESS_BONUS = float(os.environ.get("REWARD_APPLY_BONUS", "0.2"))
TEST_WEIGHT = float(os.environ.get("REWARD_TEST_WEIGHT", "0.8"))


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

def _repo_content_matches(sbx, expected_repo: str) -> bool:
    """校验沙箱内 `/workspace/repo` 的实际内容是否与期望的 repo 匹配。

    背景（重要，勿删）：2026-08-30 训练实测发现，AGS 共享沙箱工具在反复用
    `image_override` 切换镜像时，曾出现「实际拉起的容器内容与请求的镜像不符」
    的情况（`swe-synth-0001~0006` 六道题镶像被稳定复现地判成了别的题目的仓库
    内容；删除重建共享工具后仍有 5/6 未恢复，判定为镜像构建阶段的数据缺陷，
    而非单纯的实例缓存问题——但两种成因都可能在未来再次出现，因此在这里加
    一层运行时兜底校验，防止 reward 被静默算错而不自知）。
    校验方式：读 `pyproject.toml` 的 `name = "..."` 字段，看是否包含期望 repo
    的项目名（大小写不敏感）。读不到就退化为无法判断，返回 True（不误杀）。
    """
    expect = expected_repo.split("/")[-1].lower()
    try:
        _code, out, _err = _sbx_run(
            sbx,
            f"grep -m1 '^name' {REPO_DIR}/pyproject.toml 2>/dev/null || true",
            timeout=15,
        )
        pkgname = out.strip().lower()
    except Exception:  # noqa: BLE001
        return True
    if not pkgname:
        return True
    return expect in pkgname


_CONTENT_MISMATCH_MAX_RETRY = int(os.environ.get("REWARD_CONTENT_MISMATCH_RETRY", "2"))


def _score_via_sandbox(task_id: str, image: str, patch: str, expected_repo: str = "") -> tuple[float, str]:
    """`_score_via_sandbox_once` 的重试包装：若命中"沙箱内容与期望 repo 不符"
    （见 `_repo_content_matches`），说明借到的是一个坏实例，不是模型能力问题，
    丢弃后换一个新实例重试，最多 `_CONTENT_MISMATCH_MAX_RETRY` 次，避免这类
    基础设施问题被误判为模型输出差、静默污染训练信号。
    """
    last_reason = ""
    for attempt in range(_CONTENT_MISMATCH_MAX_RETRY + 1):
        reward, reason = _score_via_sandbox_once(task_id, image, patch, expected_repo)
        if "沙箱内容校验失败" not in reason:
            return reward, reason
        last_reason = reason
        if os.environ.get("REWARD_DEBUG_LOG"):
            print(f"[reward_debug] 内容校验失败，第 {attempt + 1} 次重试：{reason}", flush=True)
    return 0.0, f"{last_reason}（已重试 {_CONTENT_MISMATCH_MAX_RETRY} 次仍失败）"


def _sbx_run(sbx, cmd: str, timeout: int = 30, envs: dict | None = None) -> tuple[int, str, str]:
    """执行沙箱命令，返回 (exit_code, stdout, stderr)，**不抛异常**。

    e2b SDK 的 `commands.run` 在命令退出码非 0 时会直接 raise
    `CommandExitException`（而不是返回带 exit_code 的结果对象），导致调用方
    `if res.exit_code != 0` 这类判断永远走不到，所有命令失败都被外层
    `except` 兜成"沙箱调用异常"——把**模型输出不合法**误分类成**基础设施故障**
    （实测 78/79 次 `git apply` 失败全被误报为沙箱异常，失败归因完全失真）。

    本函数把退出码异常收敛成正常返回值，只有真正的连接/超时类异常才向上抛。
    """
    try:
        res = sbx.commands.run(cmd, user="root", timeout=timeout, envs=envs or {})
        return res.exit_code, (res.stdout or ""), (res.stderr or "")
    except Exception as e:  # noqa: BLE001
        # CommandExitException 带 exit_code/stdout/stderr 属性，说明命令确实执行了
        # 只是退出码非 0，属于"业务失败"，收敛为返回值
        code = getattr(e, "exit_code", None)
        if code is None:
            raise  # 连接失败/超时等真实基础设施异常，交给外层处理
        return int(code), str(getattr(e, "stdout", "") or ""), str(getattr(e, "stderr", "") or str(e))


# patch 应用策略级联：从最严格到最宽松，任一成功即视为 apply 成功。
# 【为什么需要级联】实测 1.5B 模型失败原因分布（78 次采样）：
#   corrupt patch at line N        → hunk header 里的行数 `@@ -X,Y +A,B @@` 算错
#   patch fragment without header  → hunk 头缺失或错乱
#   patch does not apply           → 上下文行对不上
# 前两类**纯粹是算术/格式问题，与修复逻辑对不对无关**：模型可能已经改对了代码，
# 只因数不清行数就被判 0。`git apply --recount` 正是为"手写 patch 行数不准"设计的，
# 会忽略 header 里的计数、按 hunk body 实际内容重新推算；`-C1` 放宽上下文匹配到
# 1 行；`patch -p1 --fuzz=3` 允许上下文模糊匹配。
# 【是否算作弊】不算：放宽的只是 diff 的**格式容错**，代码改动内容本身分毫未变，
# 最终仍由 verify.sh 跑真实测试判定对错，防 reward-hacking 的 P2P 规则也完整保留。
# 可用 REWARD_STRICT_APPLY=1 关掉级联，退回单一严格模式做对照实验。
_APPLY_STRATEGIES: list[tuple[str, str]] = [
    ("strict", "git apply --whitespace=nowarn {p}"),
    ("recount", "git apply --recount --whitespace=nowarn {p}"),
    ("recount+C1", "git apply --recount -C1 --unidiff-zero --whitespace=nowarn {p}"),
    ("patch-fuzz", "patch -p1 --fuzz=3 --no-backup-if-mismatch -i {p}"),
]


def _apply_patch(sbx, patch_path: str) -> tuple[bool, str, str]:
    """按 `_APPLY_STRATEGIES` 顺序尝试应用 patch。

    返回 (是否成功, 生效的策略名, 最后一次失败的 stderr)。
    """
    strategies = _APPLY_STRATEGIES
    if os.environ.get("REWARD_STRICT_APPLY"):
        strategies = _APPLY_STRATEGIES[:1]

    last_err = ""
    for name, tmpl in strategies:
        code, _out, err = _sbx_run(sbx, f"cd {REPO_DIR} && " + tmpl.format(p=patch_path))
        if code == 0:
            return True, name, ""
        last_err = err.strip()
    return False, "", last_err


def _score_via_sandbox_once(task_id: str, image: str, patch: str, expected_repo: str = "") -> tuple[float, str]:
    """借实例 → 还原干净态（tar 快照）→ apply patch → verify.sh → 判分 → 还实例。

    返回 (reward, reason)。

    还原方案说明：镜像内 `/workspace/repo` 不含 `.git`（Phase 1.5 真实沙箱实测确认），
    因此不能用 `git checkout -- . && git clean -fd`。改用 tar 快照：
      - 全新实例首次使用：`tar czf {PRISTINE_SNAPSHOT} -C / workspace/repo` 建一份快照
      - 复用实例：`rm -rf {REPO_DIR} && tar xzf {PRISTINE_SNAPSHOT} -C /` 秒级还原
    """
    from e2b_code_interpreter import Sandbox

    instance_id, reused = _POOL.acquire(task_id, image)
    broken = False
    try:
        sbx = Sandbox.connect(instance_id)

        if expected_repo and not _repo_content_matches(sbx, expected_repo):
            broken = True
            return 0.0, (
                f"沙箱内容校验失败：task_id={task_id} 期望 repo={expected_repo}，"
                f"但实例内容不匹配（疑似镶像内容错乱，已丢弃该实例，不计入模型能力信号）"
            )

        if reused:
            r_code, _o, _e = _sbx_run(sbx, f"rm -rf {REPO_DIR} && tar xzf {PRISTINE_SNAPSHOT} -C /")
            if r_code != 0:
                broken = True
                return 0.0, f"实例还原失败（exit={r_code}），已丢弃该实例"
        else:
            s_code, _o, _e = _sbx_run(sbx, f"tar czf {PRISTINE_SNAPSHOT} -C / workspace/repo")
            if s_code != 0:
                broken = True
                return 0.0, f"建立 pristine 快照失败（exit={s_code}），已丢弃该实例"

        if patch.strip():
            patch_content = patch if patch.endswith("\n") else patch + "\n"
            sbx.files.write("/tmp/reward.patch", patch_content, user="root")
            ok, strategy, apply_err = _apply_patch(sbx, "/tmp/reward.patch")
            if not ok:
                # patch 应用失败视为模型这次生成无效，判 0，不算基础设施错误，
                # 实例本身仍是干净可复用的（apply 失败不会改动仓库）
                return 0.0, f"patch 应用失败（模型输出的 diff 不合法）：{apply_err[:300]}"
        else:
            return 0.0, "solution_str 中未抽取到有效 patch"

        # verify.sh 判不通过时退出码非 0，不代表基础设施故障，
        # 统一用 result.json 的内容判分，退出码只用于日志。
        _sbx_run(sbx, f"cd {REPO_DIR} && bash /task/verify.sh",
                 timeout=VERIFY_TIMEOUT_SEC, envs={"PYTEST_ADDOPTS": "--color=no"})

        try:
            result_raw = sbx.files.read("/task/result.json", user="root")
            result_json = json.loads(result_raw)
        except Exception as e:  # noqa: BLE001
            # patch 已成功应用（走到这里说明 git apply 通过了），只是判据脚本
            # 读取失败，仍然给格式分，避免基础设施抖动把有效样本判成 0
            return APPLY_SUCCESS_BONUS, f"patch 应用成功但读取 result.json 失败：{e}"

        rr = compute_reward(result_json)
        # 分段 reward（reward shaping）：patch 能被 git apply 成功本身就是
        # 一个有价值的中间能力（1.5B 小模型最缺的正是"写出结构合法的
        # unified diff"），给一个小的基础分让 GRPO 的组内比较能产生非零
        # advantage，否则 8 个采样全 0 → advantage 全 0 → grad_norm=0 →
        # 模型永远不更新（实测 71 步 + 8 步都是这个死循环）。
        # 课题要求的 `fail→pass / 总数` 仍占主体权重（TEST_WEIGHT=0.8）。
        final_reward = APPLY_SUCCESS_BONUS + TEST_WEIGHT * rr.reward
        return final_reward, (
            f"[apply成功({strategy})+{APPLY_SUCCESS_BONUS}] {rr.reason}"
            f"（测试分 {rr.reward:.3f}×{TEST_WEIGHT} → 合计 {final_reward:.3f}）"
        )
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
    expected_repo = meta.get("repo", "")
    if not image:
        return 0.0

    patch = extract_patch(solution_str)
    key = _cache_key(task_id, patch)

    with _cache_lock:
        cached = _reward_cache.get(key)
    if cached is not None:
        return cached[0]

    reward, reason = _score_via_sandbox(task_id, image, patch, expected_repo)

    with _cache_lock:
        _reward_cache[key] = (reward, reason)

    if os.environ.get("REWARD_DEBUG_LOG"):
        print(
            f"[reward_debug] task_id={task_id} reward={reward} reason={reason} "
            f"solution_len={len(solution_str)} patch_len={len(patch)} "
            f"patch_head={patch[:200]!r}",
            flush=True,
        )

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
