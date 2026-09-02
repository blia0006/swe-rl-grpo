# 题目四：强化学习 —— 基于 Agent SandBox + TKE GPU 的代码修复 RL 全流程

> 本文件整理原始需求，作为跨模型/跨会话继续本课题时的唯一事实来源。
> 若与 chat 记录冲突，以本文件为准；若本文件后续有修订，请同步更新此文件而非仅在对话中说明。

## 1. 背景

强化学习训练 Coding Agent 的核心瓶颈是执行反馈数据的获取：模型生成的代码修复必须在隔离环境中真实运行，
才能得到可验证的奖励信号。

- **Agent SandBox**（腾讯云）：毫秒级启动、CPU 沙箱，负责 SWE 题目的隔离执行与 tracing 采集。
- **TKE GPU 集群**：负责模型训练（VERL 做策略优化）。

架构分工：SandBox 只管执行与采集（纯 CPU），TKE GPU 只管训练。

## 2. 目标产出（组件）

| 组件 | 说明 |
|---|---|
| SWE 执行环境（SandBox 侧） | 为每个 SWE 题目构建独立沙箱（含指定 repo 分支 + 测试环境），Agent 在沙箱中执行修复操作，输出完整 tracing 日志 |
| Tracing 数据采集 | 每次解题过程的结构化 tracing：Agent 每步操作（读文件/编辑/执行命令/跑测试）+ 观察结果 + 最终奖励（测试通过率），格式对齐 VERL rollout 数据 |
| 训练流水线（TKE 侧） | 在 TKE GPU 节点上部署 VERL，消费 tracing 数据进行 GRPO/PPO 训练 |
| 闭环验证 | 训练后的模型再次在 SandBox 中跑 SWE 题目，对比训练前后的 pass@1 |

## 3. 技术要求

- **SandBox 环境**：腾讯云 Agent SandBox 代码沙箱（CPU 实例即可），自定义镜像含 Git + Python + 各 SWE 题目仓库依赖。不使用 GPU 沙箱。
- **TKE GPU 环境**：腾讯云 TKE 集群，GPU 节点（≥1×T4/A10），部署 VERL + PyTorch + vLLM/SGLang。
- **训练框架**：<https://github.com/volcengine/verl>（版本 ≥0.3.0），算法 GRPO 或 PPO。
- **模型**：HuggingFace 开源模型，自选（需在方案中给出选型理由）。
- **SWE 题目**：SWE-bench Verified 子集 或 swesmith 等开源 SWE 数据集，取指定分支的指定 commit 构建 SandBox。
- **Tracing 格式**：记录每步的 `(action, observation, reward, done)`，与 VERL 的 `DataProto` 对齐。
- **奖励函数**：执行反馈奖励 —— 在 SandBox 中跑 `python -m pytest` 或项目自带测试套件，以
  `fail→pass 测试数 / 总相关测试数` 作为 reward。
- **通信方式**：SandBox → TKE 通过 COS/CFS 传递 tracing 数据；TKE 训练完成后更新模型，触发下一轮 SandBox rollout。

## 4. 验收标准

- [ ] SandBox 能批量拉起 SWE 题目环境（≥10 题），Agent 在沙箱中完成解题并输出结构化 tracing
- [ ] 单条 tracing 包含完整的操作序列（≥3 步操作）和最终测试通过情况
- [ ] TKE GPU 集群上 VERL 环境可正常运行，训练至少 50 step
- [ ] reward 曲线呈上升趋势（提供 wandb 截图或 matplotlib 图表）
- [ ] 至少完成 1 轮完整闭环：SandBox 产出 tracing → TKE 训练 → 新模型回 SandBox 评估
- [ ] 训练后 pass@1 相比训练前有可观测提升（提升幅度不做硬性要求，但需有对比数据）
- [ ] 提供 README.md，包含：SandBox 环境构建方式、TKE 部署步骤、模型选型理由、训练超参、结果分析

## 5. 本方案的关键决策（与原始要求的差异说明）

1. **SWE 题目来源**：不重新从 SWE-bench/swesmith 选题构建镜像，而是**直接复用课题三已交付
   的 19 道 ACCEPTED 题目**（`课题三-数据合成/dist/swe-synth-delivery-20260821/data/tasks.jsonl`）。
   这些题目本质上就是"取指定 repo 指定 commit 挖空/新增/重构 + 判据测试"的 SWE 任务，镜像已推送
   TCR、`verify.sh` 判据契约已验证可用（空解 fail、golden pass、确定性通过）。**这是本方案节省
   时间的最大来源**——无需重新选题、无需重新构建/推送镜像。
2. **训练架构：在线 GRPO + 自定义 reward function**（2026-08-21 修正，取代初版的"离线 tracing 训练"）。
   - **初版设计的硬伤**：原计划"SandBox 先离线采一批 tracing → TKE 拿这批固定数据训 50 step"。
     该设计无法满足验收标准第 4 条"reward 曲线呈上升趋势"——固定数据集上跑梯度步时，
     每条样本的 reward 是数据里写死的常量，曲线必然是平线。reward 上升的本质是"当前策略在
     环境中的真实表现随训练变好"，**必须每步用最新策略采样并当场用沙箱打分**（on-policy）。
   - **修正后的做法**：让 VERL 跑它原生的在线 GRPO 循环，我们**只替换 reward function**
     （`custom_reward_function`，VERL 的标准扩展点），在其中调 AGS 沙箱执行模型输出的修复、
     跑 `verify.sh`、按 F2P/P2P 算分。
   - **依然不使用** VERL 原生 `BaseTool` + AsyncRollout 多轮工具集成：那需要让沙箱网络延迟被
     VERL 异步调度稳定吃住，配置项多、排查成本不可控。自定义 reward function 的接口面极小，
     工程量小一个量级，失败可定位。原生多轮集成留作加分项（plan.md Phase 6）。
   - **两条产出线分工**：线 A（SandBox 侧多轮 ReAct 采集 tracing 传 COS）满足"≥10 题 +
     单条 tracing ≥3 步"与"经 COS/CFS 传递"的要求；线 B（在线 GRPO）满足"≥50 step +
     reward 曲线上升 + 闭环 pass@1 对比"。详见 plan.md 第 2 节。
3. **模型选型**：`Qwen2.5-Coder-1.5B-Instruct`（HuggingFace 开源，Apache-2.0）。选型理由见
   plan.md 及最终 README.md。

## 6. 详细执行计划

见同目录 `plan.md`。执行进度见同目录 `PROGRESS.md`。
