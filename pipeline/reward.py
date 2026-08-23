"""奖励函数：解析 `/task/result.json`（`verify.sh` 的判据输出）→ reward。

对应 TASK-SPEC.md 第 3 节："以 fail→pass 测试数 / 总相关测试数作为 reward"，
以及 plan.md 3.2 节的 P2P 回归防 reward-hacking 规则。

`result.json` 的真实结构（`verify_gen.py` 生成，已用
`pipeline/testdata/*.json` 里的真实产出核实，见本文件底部单元测试）：

    {
      "task_id": "...",
      "passed": bool,
      "fail_to_pass": {"total": int, "passed": int, "failing": [str, ...]},
      "pass_to_pass": {"total": int, "passed": int, "failing": [str, ...]},
      "collect_error": bool,
      "pytest_returncode": int,
      "n_collected": int,
      "raw_log_path": str,
    }

纯 stdlib 实现，线 A（沙箱内 agent.py 收尾算分）和线 B（TKE 侧
`verl_reward_fn.py`）共用同一份逻辑，避免两处各写一套判分口径不一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardResult:
    reward: float
    fail_to_pass_rate: str  # "3/3" 形式，人类可读
    pass_to_pass_rate: str
    passed: bool
    reason: str  # 便于 tracing/日志排查为什么给了这个分


def compute_reward(result_json: dict[str, Any]) -> RewardResult:
    """核心判分逻辑：

        reward = FAIL_TO_PASS 中变为 pass 的用例数 / FAIL_TO_PASS 总用例数
        若 PASS_TO_PASS 中有回归 fail，则整体 reward = 0（防 reward hacking，
        例如模型删掉无关测试文件让 F2P "看似"通过）。

    额外的环境错误兜底（collect_error / pytest 根本没跑起来）同样判 0，
    避免把"环境坏了"误算成"模型答对了一部分"。
    """
    f2p = result_json.get("fail_to_pass") or {}
    p2p = result_json.get("pass_to_pass") or {}

    f2p_total = int(f2p.get("total", 0))
    f2p_passed = int(f2p.get("passed", 0))
    p2p_total = int(p2p.get("total", 0))
    p2p_passed = int(p2p.get("passed", 0))

    f2p_rate = f"{f2p_passed}/{f2p_total}"
    p2p_rate = f"{p2p_passed}/{p2p_total}"

    # 环境/流程错误：pytest 根本没能收集到用例，或收集阶段报错。
    # 这不代表"模型的修复是错的"，但按判分口径我们无法区分，统一给 0
    # （沙箱侧应对此类情况重试，而不是把 0 当作模型能力信号，具体在
    #  driver.py / verl_reward_fn.py 里做区分处理）。
    if result_json.get("collect_error"):
        return RewardResult(
            reward=0.0, fail_to_pass_rate=f2p_rate, pass_to_pass_rate=p2p_rate,
            passed=False, reason="collect_error=true（pytest 收集阶段失败，判 0）",
        )

    if f2p_total == 0:
        # 题目契约本身要求 FAIL_TO_PASS 非空（课题三 Agent2 双向验证已保证），
        # 出现 0 说明 metadata 或运行环境有问题，同样判 0 并给出明确原因。
        return RewardResult(
            reward=0.0, fail_to_pass_rate=f2p_rate, pass_to_pass_rate=p2p_rate,
            passed=False, reason="fail_to_pass.total=0，题目契约异常，判 0",
        )

    if p2p_total > 0 and p2p_passed < p2p_total:
        # P2P 回归：改动破坏了原本就通过的测试（reward hacking 的常见手法之一，
        # 比如直接删测试文件、改断言让判据失效）。
        return RewardResult(
            reward=0.0, fail_to_pass_rate=f2p_rate, pass_to_pass_rate=p2p_rate,
            passed=False,
            reason=f"PASS_TO_PASS 回归（{p2p_passed}/{p2p_total}），整体判 0 防 reward hacking",
        )

    reward = f2p_passed / f2p_total
    passed = bool(result_json.get("passed", reward == 1.0))
    reason = "F2P 全绿 + 无 P2P 回归" if passed else f"F2P 部分通过（{f2p_rate}），无 P2P 回归"
    return RewardResult(
        reward=reward, fail_to_pass_rate=f2p_rate, pass_to_pass_rate=p2p_rate,
        passed=passed, reason=reason,
    )


# ---------------------------------------------------------------- 自测
# 用真实产出的 verification.json 样本（vendored 到 pipeline/testdata/）做单元测试，
# 不依赖云端、也不依赖任何外部目录，纯本地文件 IO。
# 注：verification.json 是「双向验证」的外层结构，真正的判据输出（result.json
# 的等价物）在其 empty_run / golden_run 字段里，一并测两种场景：
#   - empty_run：应改判 reward=0（空解，F2P 应全 fail）
#   - golden_run：应改判 reward=1.0（标准答案，F2P 应全 pass）
if __name__ == "__main__":
    import glob
    import json
    import os
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_PATTERN = os.path.join(_HERE, "testdata", "*.json")

    pattern = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATTERN
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"未找到匹配文件：{pattern}")
        raise SystemExit(1)

    n_ok, n_fail = 0, 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            v = json.load(f)
        task_id = v.get("task_id", fp)

        empty = v.get("empty_run")
        golden = v.get("golden_run")
        if not empty or not golden:
            print(f"[SKIP] {task_id}：缺 empty_run/golden_run")
            continue

        r_empty = compute_reward(empty)
        r_golden = compute_reward(golden)

        ok = True
        if r_empty.reward != 0.0:
            ok = False
            print(f"[FAIL] {task_id} empty_run reward 应为 0，实际 {r_empty.reward}（{r_empty.reason}）")
        if r_golden.reward != 1.0:
            ok = False
            print(f"[FAIL] {task_id} golden_run reward 应为 1.0，实际 {r_golden.reward}（{r_golden.reason}）")

        if ok:
            n_ok += 1
            print(f"[OK]   {task_id}  empty=0.0  golden=1.0  (F2P golden={r_golden.fail_to_pass_rate})")
        else:
            n_fail += 1

    print(f"\n共 {len(files)} 题，通过 {n_ok}，失败 {n_fail}")
    raise SystemExit(1 if n_fail else 0)
