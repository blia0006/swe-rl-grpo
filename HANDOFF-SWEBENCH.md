# 题目四 · 迁移到 SWE-bench Verified 官方题库 —— 新项目交接文档

> **本文档用途**：作为新项目的启动依据。旧仓库（合成题版本）保留为备份，已完整交付、
> 可独立答辩，**不要删除**。新项目是"用官方题库重做一遍"，不是修 bug。
>
> **旧仓库**：`https://github.com/blia0006/swe-rl-grpo`（最新 `75249b3`，7 项验收 5 项达成）

---

## 0. 先读这一段：迁移的收益与代价

### 收益（真实存在）

| 项 | 说明 |
|---|---|
| **更贴合任务书** | `TASK-SPEC.md` 原文要求"SWE-bench Verified 子集或 swesmith"，官方题库是首选项；旧方案用合成题属于"关键决策差异"，需要额外解释 |
| **消除镜像质量疑虑** | 旧方案 21 题中 6 题因上游镜像内容错乱被剔除，答辩时需解释；官方镜像无此问题 |
| **结果可对标** | SWE-bench 有公开 leaderboard，pass@1 数字有参照系 |
| **题量充足** | Verified 有 500 题，可把评测集扩到 20+ 题，解决旧方案"4 题评测无统计功效"的硬伤（见旧 README §7.4） |

### 代价（必须提前知道）

| 项 | 实测/估算 |
|---|---|
| **镜像体积** | SWE-bench 采用三层镜像（base → env → instance）。官方文档给出：全量缓存约 **2TB**。取 15 题的话，实际落盘约 **30~60GB**（共享 base/env 层） |
| **镜像搬运** | 官方镜像在 Docker Hub，AGS 沙箱只认 TCR → 必须 `pull → tag → push` 到你的 TCR 命名空间。这是新增的主要工作量 |
| **判据契约重写** | 旧方案用 `/task/verify.sh` + `result.json`；SWE-bench 用 `FAIL_TO_PASS` / `PASS_TO_PASS`（测试 node id 列表）+ 官方 harness。需要写适配层 |
| **网络风险** | 已实测：GPU Pod **无法访问 github.com**（`curl` 返回 `000`），腾讯云 COS 正常。Docker Hub 是否可达**必须先验证**，这是最大的不确定性 |
| **时间** | 乐观 1 天，保守 2~3 天。主要耗在镜像搬运与判据适配 |

### ⚠️ 关键提醒：不要从零构建镜像

你原话是"重新建立镜像"，但**官方已提供全部 500 道 Verified 题的预构建镜像**
（`epoch-research/SWE-bench` 镜像仓库明确写着 Verified 集 500/500 全部可用）。

**从零构建 = 每题装一遍依赖，慢且容易失败；直接拉取预构建镜像可省掉 80% 工作量。**
第一件事就是验证能否拉到官方镜像。

---

## 1. 可直接复用的资产（不要重写）

旧仓库这些模块与题目来源**无关**，直接拷过来就能用：

| 文件 | 作用 | 迁移改动 |
|---|---|---|
| `pipeline/verl_reward_fn.py` | VERL custom_reward_function：沙箱实例池 + **apply 策略级联** + reward shaping + `_sbx_run` 异常收敛 | 只需改判据调用部分（§4） |
| `pipeline/reward.py` | F2P/P2P 判分 + 防 reward hacking | 改输入格式适配 |
| `scripts/run_grpo_training.sh` | GRPO 训练入口（5090 全部适配项已调通） | 几乎不用改 |
| `scripts/plot_reward_curve.py` | 从日志画 reward/grad_norm 曲线 | **零改动** |
| `scripts/eval_pass_at_1.py` | 闭环 pass@1 评测（含 LoRA 加载、防数据泄漏校验） | 改评测集加载 |
| `scripts/run_final_deliverables.sh` | 一键出交付物 | **零改动** |
| `clients/ags.py` / `clients/cos.py` | 沙箱 / COS 客户端 | **零改动** |
| `sandbox_agent/agent.py` | 沙箱内多轮 ReAct Agent（线 A） | 小改 |
| `deploy/gpu-pod.yaml` | GPU Pod（含 `/dev/shm` 4Gi 挂载） | **零改动** |

**特别强调 `verl_reward_fn.py` 里的 apply 策略级联**（`strict → recount → recount+C1 → recount+C0`）
和 `_sbx_run()` 异常收敛 —— 这两处是踩了大坑才做对的，**照搬，别重写**（原因见 §5）。

---

## 2. 必须改的部分

```
data/tasks.jsonl（合成题）        →  SWE-bench Verified 子集
/task/verify.sh + result.json     →  FAIL_TO_PASS / PASS_TO_PASS + 官方 harness
TCR 自建镜像                       →  官方镜像 pull → push TCR
prompt 里嵌入文件内容（前300行）    →  保留此设计（旧方案实测：不嵌内容会导致盲写 patch、reward 恒 0）
```

---

## 3. 分阶段计划（每阶段都有明确门禁，不通过不往下走）

### Phase 0：网络与镜像可达性验证（**最高优先级，2 小时内必须有结论**）

这一阶段决定整个方案是否可行，**不通过就要换路线，别往下做**。

**门禁 1：能否拉到官方镜像**

```bash
# 本地 Mac（已确认能访问外网）
docker pull swebench/sweb.eval.x86_64.<instance_id>:latest
```

> 镜像名的具体拼写规则需实测确认：instance_id 中的 `__` 在镜像名中会被编码替换
> （如 `astropy__astropy-12907` → `astropy_1776_astropy-12907`）。
> **不要凭记忆写死，先 `docker search` 或查 `swebench/harness/test_spec.py` 的生成逻辑确认。**

**门禁 2：能否推到 TCR**

```bash
docker tag <官方镜像> ccr.ccs.tencentyun.com/tcb-100008634787-zbaf/sweb-<id>:v1
docker push ccr.ccs.tencentyun.com/tcb-100008634787-zbaf/sweb-<id>:v1
```

**门禁 3：AGS 沙箱能否用该镜像起实例**

用旧仓库 `clients/ags.py` 的 `start_instance(image_override=...)` 直接试。

**门禁 4：镜像内能否跑通测试**

进沙箱执行 SWE-bench 的测试命令，确认 pytest 能正常收集与执行。

> **只要有一个门禁不通过，立即停下来评估**。特别是门禁 1 —— 如果本地也拉不到
> Docker Hub，整个方案要改成"用官方 Dockerfile 自行构建"，工作量翻数倍，
> 那时应该考虑是否值得继续。

### Phase 1：选题与数据准备

- 从 SWE-bench Verified 选 **15~20 题**（训练 12 + 评测 8，比旧方案的 11+4 更有统计功效）
- 选题建议：优先挑 `patch` 较短（改动 1~2 个文件）的题，1.5B 模型才有机会
- 生成 `data/tasks.jsonl`（沿用旧格式：`task_id` / `repo` / `image` / `problem_statement` / `modified_files`）
- 保留 `data/split.json` 的训练/评测划分机制与**防泄漏校验**

### Phase 2：判据适配层（核心工作量）

SWE-bench 的判据是数据集里的两个字段：

```
FAIL_TO_PASS: ["tests/test_x.py::test_a", ...]   # 修复后应从 fail 变 pass
PASS_TO_PASS: ["tests/test_y.py::test_b", ...]   # 修复前后都应 pass（防回归）
```

需要写一个 `verify.sh` 等价物，在沙箱内：
1. 应用模型 patch
2. `python -m pytest <F2P + P2P 的测试列表> -rA --tb=no`
3. 解析结果，输出与 `pipeline/reward.py` 兼容的 JSON：

```json
{
  "fail_to_pass": {"total": N, "passed": M, "failing": [...]},
  "pass_to_pass": {"total": N, "passed": M, "failing": [...]},
  "collect_error": false
}
```

**这样 `pipeline/reward.py` 与 `verl_reward_fn.py` 的判分逻辑就能零改动复用。**

### Phase 3：训练 + 交付物

直接复用旧脚本。**但要先应用下面这个改进**（见 §5.1）。

---

## 4. 必须带过去的改进（旧项目实测得出，不要重蹈覆辙）

### 4.1 reward 分档细化（旧项目最大的设计疏漏）

旧方案的分档：

```
0.0   无法 apply
0.2   apply 成功
0.2 + 0.8×F2P通过率
```

**问题**：实测 `collect_error` 稳定占 apply 成功的 **51%**（两个时间点 50% / 51%，
是结构性而非波动）—— 这些是 `-C0` 靠行号硬插、把文件语法插坏的样本，
但它们和"位置正确、测试可正常收集"的样本**同为 0.2 分**。

等于告诉模型"写歪的 diff 和写对的一样值钱"，格式能力自然学不动。
这被判定为旧方案 reward 曲线未上升的**主因**。

**新项目从第一天就用这个分档**：

```
0.00   未抽到 patch / 路径不存在 / 无法 apply
0.05   apply 成功但 collect_error（代码被插坏）    ← 新增档位
0.20   apply 成功且测试可正常收集
0.20 + 0.8×F2P通过率
```

### 4.2 缓解文件路径幻觉

旧项目实测发现模型会写错路径：

```
itsdangerous/encoding.py: No such file     ← 丢了 src/ 前缀
jd/tenacity/retry_after.py: No such file   ← 凭空多出 jd/ 前缀
```

**做法**：prompt 里把 `--- a/<完整路径>` / `+++ b/<完整路径>` 作为**固定模板行**直接给出，
让模型只需填 hunk 内容。

### 4.3 评测规模

旧方案 4 题 × k=1，实测**期望成功次数 0.07**，全 0 是统计必然、结论无效。
新项目**评测集至少 8 题 × k=8**，并在报告里附显著性检验。

### 4.4 考虑改用 search/replace 格式

旧项目 440 次采样中 **227 次 `corrupt patch`**（算不准 hunk header 行号）。
如果新项目仍用 1.5B 模型，**强烈建议把任务表示从 unified diff 改成 search/replace 块**：

```
<<<<<<< SEARCH
原始代码片段
=======
替换后代码
>>>>>>> REPLACE
```

**不需要算行号**，直接绕开小模型最大的短板，可能比任何 reward 调优都有效。

---

## 5. 环境侧的坑（全部实测踩过，直接抄配置即可）

### 5.1 GPU / 训练环境

| 坑 | 解法 |
|---|---|
| 5090（Blackwell sm_120）驱动 | 装 `nvidia-driver-570-open`（**必须 open 版**），**装完必须 reboot**，否则 `nvidia-smi` 报 `No devices were found` |
| 镜像 CUDA 版本 | 必须 cu128：`verlai/verl:vllm011.latest`。CUDA 12.1 镜像会报 `no kernel image is available` |
| `/dev/shm` 仅 64MB | Pod YAML 必须挂 `medium: Memory` emptyDir（4Gi）。否则 NCCL 初始化失败、训练零步崩溃，且**运行期无法补救**（非 privileged 时 remount 报 write-protected） |
| 兜底 | 训练脚本里 `export NCCL_SHM_DISABLE=1` / `NCCL_P2P_DISABLE=1`（单卡无跨卡通信，零性能损失） |
| verl 0.6.1 配置 | `actor.strategy=fsdp`、`model_dtype=bfloat16`（须写全名）、`use_orig_params=False`、`use_torch_compile=False` |

### 5.2 e2b SDK 的致命陷阱（**最容易导致结论错误**）

`sbx.commands.run()` 在**退出码非 0 时直接抛异常**，不返回 `exit_code`。
因此 `if res.exit_code != 0` 这类判断**永远走不到**，所有命令失败都被外层
`except` 兜成"沙箱调用异常"。

旧项目实测后果：**78 次 apply 失败被误报成"基础设施故障"**，失败归因完全失真，
差点据此写错结果分析。

**解法**：照抄旧仓库 `verl_reward_fn.py` 的 `_sbx_run()`，把退出码异常收敛为返回值。
顺带收益：修复后单步耗时从 **110s → 24s（4.6×）**。

### 5.3 网络

- GPU Pod **访问 github.com 会超时**（实测 `curl` 返回 `000`），但**腾讯云 COS 正常**（返回 `403` 即网络通）
- pip 用腾讯云镜像源可通：`-i https://mirrors.cloud.tencent.com/pypi/simple`
- **交付物回传**：Pod 打包 → 上传 COS → 本地拉回 → 本地 git push
  （旧仓库 `scripts/ship_deliverables_via_cos.py` 直接可用）

### 5.4 停训练的正确方式

```bash
ray stop; sleep 8; pkill -f "ray::"; sleep 3; rm -f /dev/shm/nccl-*
```

**不要用 `pkill -9`** —— 会让 Ray worker 来不及清理 `/dev/shm/nccl-*`，
导致下次训练零步崩溃，且产生大量 zombie 进程。

---

## 6. 风险清单与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| Docker Hub 不可达 | 中 | **致命** | Phase 0 门禁 1 先验证；不通则用腾讯云镜像加速服务，或退回旧方案交付 |
| 镜像太大传不动 | 中 | 高 | 只取 15~20 题；用 TCR 同地域加速；分批推送 |
| 判据适配层出 bug | 中 | 高 | 用 golden patch 验证：应得 reward=1.0；空 patch 应得 0.0 |
| 1.5B 在真实 SWE 题上完全做不出 | **高** | 中 | SWE-bench 比合成题难得多，**pass@1 很可能是 0**。应对：改 search/replace 格式（§4.4）、选最简单的题、如实报告 |
| 时间不够 | 中 | 中 | 旧仓库已完整交付，**随时可以停下来用旧的答辩** |

### ⚠️ 关于最后一项，必须说清楚

**SWE-bench Verified 的真实难度远高于合成题。** 公开 leaderboard 上，
即便是顶尖闭源模型配上完整 Agent 框架，pass@1 也就在 50%~70% 量级；
**1.5B 小模型单轮生成 patch，pass@1 极可能是 0。**

所以要有心理准备：**换成官方题库后，"训练后 pass@1 提升"这条验收标准可能更难达成**，
而不是更容易。它的价值在于"题目来源更权威、无镜像质量疑虑"，不在于结果更好看。

**决策建议**：把新项目定位为"锦上添花的第二版"，旧仓库作为保底交付物完整保留。
如果新项目在 Phase 0 或 Phase 2 卡住，果断用旧的答辩。

---

## 7. 新项目第一天的行动清单

```
□ 1. 从旧仓库拷贝可复用模块（§1 那张表）
□ 2. Phase 0 门禁 1：本地 docker pull 一个官方 SWE-bench 镜像 —— 成功才继续
□ 3. Phase 0 门禁 2：push 到 TCR
□ 4. Phase 0 门禁 3：AGS 沙箱用该镜像起实例
□ 5. Phase 0 门禁 4：沙箱内跑通 pytest，确认 F2P/P2P 测试可执行
□ 6. 四个门禁全过 → 写 PROGRESS.md 记录，进入 Phase 1
□ 7. 任一门禁失败 → 记录失败原因，评估是否继续（旧仓库可保底）
```

---

## 8. 旧项目的最终状态（备份基线）

| # | 验收标准 | 状态 |
|---|---|---|
| 1 | ≥10 题 tracing | ✅ 14 题 |
| 2 | 单条 ≥3 步 | ✅ 4~10 步 |
| 3 | ≥50 step 训练 | ✅ 55/55，0 报错 |
| 4 | reward 曲线上升 | ⚠️ 未达成（U 型），根因已定量 |
| 5 | 完整闭环 | ✅ |
| 6 | pass@1 对比 | ✅ 有数据；差异经 Welch t 检验不显著 |
| 7 | README | ✅ |

**关键数据（新项目可作为对照基线）**：

- 训练 55 step，`grad_norm` 全程 > 1e-8（此前两轮恒为 0，死循环已打破）
- 440 次采样中 apply 成功 152 次（34.5%）；其中 `strict` 仅 **8 次（1.8%）**
- 失败原因：`corrupt patch` **227** / `without header` 44 / `does not apply` 10
- `collect_error` 占 apply 成功的 **51%**（结构性瓶颈）
- 最高单步 reward **1.0**（有采样真正修对了题）

**这些数字在新项目里可以直接用作"合成题 vs SWE-bench 官方题"的对比论据，
本身就是有价值的实验结论 —— 所以旧仓库务必保留。**
