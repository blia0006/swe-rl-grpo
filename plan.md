# 题目四执行计划（plan.md）

> 配套文件：`TASK-SPEC.md`（需求整理）、`PROGRESS.md`（进度日志）。
> 原则：**先跑通闭环最小可用版本，再谈优化**。每个 Phase 都有明确的"门禁"（Gate），
> 门禁不过不进入下一阶段，避免像课题三一样在某个环节反复耗时。

## 0. 总体时间预算（目标：GPU 计费窗口 < 2.5h，全流程 < 6h，而不是十几个小时）

| 环节 | 预估耗时 | 是否计 GPU 费 | 说明 |
|---|---|---|---|
| Phase 0 资源探测与准备 | 30 min ✅已完成 | 否 | 只读探测，结论见 1.1 |
| Phase 1 本地离线打通（mock）| 1.5~2 h | 否 | 本地 mock 沙箱 + mock LLM，把 tracing schema / reward 解析 / VERL 接入代码全部跑通 |
| Phase 1.5 云上预热（不开 GPU）| 40 min | 否 | ① 模型权重经 ModelScope 下好并传 COS ② VERL 官方镜像同步到自己 CCR ③ 沙箱实例复用 + 并发上限实测（关键数字） |
| Phase 2 SandBox 真实 rollout 采 tracing | 30~40 min | 否* | 14 题多轮 ReAct，产出交付用 tracing.jsonl → COS |
| Phase 3 TKE GPU 在线 GRPO 训练 | 60~75 min | **是** | 建节点池(10min) → 起 vLLM+VERL(10min) → 训练 50 step(30~40min) |
| Phase 4 闭环验证 | 20 min | **是** | 紧接 Phase 3 不关节点，直接跑训练后 5 题 pass@1 |
| Phase 5 复盘产出 README | 40 min | 否 | 本地整理 + 出 reward 曲线图 |

\* Phase 2 的生成侧（vLLM）**只走 Phase 3 的 GPU 窗口**，不使用本机推理，见 5.Phase2 说明与 2.3 硬性约束。

### 0.1 三条铁律（防止重演课题三的十几小时）

1. **GPU 节点只开一次**：baseline 评测 → 训练 → 训练后评测，全部塞进同一个节点生命周期，
   做完立刻删节点池。绝不"开了关、关了再开"。
2. **一切能在不计费时做的准备，都提前做完**：模型权重、Docker 镜像、代码调试、沙箱参数实测，
   全部在 Phase 1/1.5 完成。GPU 节点起来后只做"跑训练"这一件事。
3. **每个 Phase 都有门禁**，门禁不过不进下一阶段；任一阶段超出预算 1.5 倍立即停下重新评估，
   不允许"再试一次说不定就好了"式的无限重试。

---

## 1. 复用课题三资产清单

| 资产 | 路径 | 用途 |
|---|---|---|
| 题目元数据 + golden patch（2026-08-23 复核：29 道，原 `dist/` 快照目录已被课题三重构移除，
现直接读实时数据） | `课题三-数据合成/data/tasks.jsonl` | 本课题的 SWE 数据集，覆盖验收标准"≥10 题"；
题目内容/质量不在本课题关注范围内，只借用其"沙箱可跑、可判分"的基础设施 |
| 题目镜像（TCR，`swe-synth-000X:v1` / `:v1-sol`） | TCR 个人版 `ccr.ccs.tencentyun.com/tcb-100008634787-zbaf/` | 每题的隔离 repo + 依赖 + 测试环境，直接注册为 AGS 沙箱工具 |
| `verify.sh` 判据契约 | 每题镜像内 | 直接作为本课题的 reward 计算器（FAIL_TO_PASS / PASS_TO_PASS） |
| AGS SDK 封装 | `课题三-数据合成/swe_synth/clients/ags.py` | 沙箱创建/执行命令/销毁的调用代码，直接复用/精简 |
| 云资源探测脚本 | `课题三-数据合成/scripts/probe_cloud.py` | Phase 0 直接跑，探 AGS 沙箱工具配额、TKE 权限、TCR 现状 |

**题目划分**：从课题三现有的题目池中选 14 题作为训练集（rollout 产 tracing 喂 GRPO），
5 题作为评测集（训练前后 pass@1 对比，不参与训练，避免"考试抄答案"）。达标验收标准要求的
"≥10 题"和"闭环 pass@1 对比"两者都满足；具体选哪 19 题留到 Phase 1.5/2 开工前再定
（题目内容质量不是本课题的关注点，只要沙箱能起、`verify.sh` 能判分即可）。

### 1.1 Phase 0 探测结果（2026-08-21 已确认，见 `PROGRESS.md` 详细记录）

**重大发现：账号里已经有为本课题预留好的 TKE 集群 + COS bucket**，命名精确匹配 `swe-rl-*`：

| 预留资源 | 地域 | 现状 | 决策 |
|---|---|---|---|
| TKE 集群 `swe-rl-cluster`（`CLUSTER_ID`） | **ap-shanghai** | Running，0 节点、0 节点池（空壳） | **直接复用**，Phase 3 只需新建一个 GPU 节点池，不用重新申请集群 |
| COS bucket `COS_BUCKET` | **ap-shanghai** | 已存在，创建于 2026-07-08 | **直接复用**作为 tracing.jsonl 的传递通道，无需新建 bucket |

**架构更新（2026-08-23 复核，好消息）**：课题三已把 AGS SandBox 的地域从 ap-guangzhou
迁移到了 **ap-shanghai**（`.env` 里 `E2B_DOMAIN=ap-shanghai.tencentags.com`，AGS 管理面
`TENCENTCLOUD_REGION=ap-shanghai` 同步生效，已用 `AGSClient().list_tools()` 实测连通，
返回 11 个已有沙箱工具，含课题三自己的 `swe-synth-shared-runner`）。这意味着**不再需要
跨地域架构**：SandBox / TKE / COS 现在全部统一在 ap-shanghai，`driver.py` 与 TKE 训练进程
之间不存在跨地域访问，COS 传递也是同地域读写，链路比原计划更简单。

其余探测结论：
- **CFS**：ap-shanghai 现有 10 个文件系统均为其他同事项目所建，无 `swe-rl` 专属预留 → 决策：
  **不使用 CFS**，tracing 全部走 COS，简化链路（与 2.1 节原方案一致）。
- **GPU 机型库存（ap-shanghai，可售）**：`GN6S.LARGE20`（1 卡，最便宜，可用区 ap-shanghai-4）、
  `PTX1.7XLARGE116`（1 卡）、`PTX2.8XLARGE96`（1 卡）等均有货，Phase 3 建节点池前再核实具体
  GPU 芯型号与单价后选定。
- **AGS 沙箱工具配额**：当前账号（团队共享）已有 8/10 个工具在用，**只剩 2 个名额**，比课题三
  收尾时预期的更紧张（说明配额是全团队共享，会被其他人的项目动态占用）。Phase 2 必须严格执行
  "用前 create_tool、验证完立即 delete_tool"，**每次注册前先重新探测剩余配额**，不能假设课题三
  留下的余量还在。
- **权限**：当前子账号已挂载 `AdministratorAccess`，TKE/COS/CFS/CVM 相关操作均无权限阻塞。

---

## 2. 整体架构

本课题有两条并行的产出线，**分别对应两组验收标准**，不要混为一谈：

- **线 A（SandBox 侧交付）** → 满足"≥10 题批量拉起 + 单条 tracing ≥3 步操作"
  真正的多轮 ReAct：模型看到 observation 再决定下一步，产出高信息量 tracing.jsonl 上传 COS。
- **线 B（训练侧交付）** → 满足"VERL ≥50 step + **reward 曲线上升** + 闭环 pass@1 对比"
  VERL 原生**在线 GRPO** 循环，reward 由 AGS 沙箱实时打分，所以 reward 曲线天然上升。

两条线共用同一套：题目集、沙箱执行器、reward 解析器、tracing schema。

```
┌───────────── 线 A：Agent 在沙箱内跑多轮 ReAct（Phase 2，纯 CPU）─────────────┐
│                                                                            │
│  driver.py（本机，只做"投放 + 收取"，不参与决策）：                           │
│    1. 起沙箱实例（题目镜像 = 干净的题目环境）                                 │
│    2. sbx.files.write("/task/agent.py", ...)  ← 运行时注入 Agent，零镜像重建 │
│    3. sbx.commands.run("python3 /task/agent.py")  ← Agent 在沙箱内自主执行   │
│    4. sbx.files.read("/task/tracing.jsonl")  ← 收取 Agent 产出的 tracing    │
│                                                                            │
│  ┌────────────── 沙箱实例内部（Agent 的完整生命周期都在这里）─────────────┐   │
│  │  agent.py（ReAct 主循环，最多 M=10 步）：                             │   │
│  │    while not done and step < M:                                     │   │
│  │      组装 prompt（题目描述 + 历史 action/observation）                 │   │
│  │      ├─HTTPS→ vLLM (TKE GPU, 带 API Key 鉴权) 拿 action              │   │
│  │      │        ↑ 唯一的出网调用：大脑在 GPU 上，因为沙箱是纯 CPU        │   │
│  │      └─本地执行→ read_file / bash / apply_patch / run_tests          │   │
│  │               ↑ 零网络开销，直接操作沙箱内的题目仓库                   │   │
│  │      追加 (action, observation, reward, done) 到 tracing.jsonl        │   │
│  │    收尾：bash /task/verify.sh → 解析 F2P/P2P → final_reward           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  driver 汇总所有 episode → 上传 COS COS_BUCKET（ap-shanghai） │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ tracing.jsonl（交付物）
┌──────────────────────────────────▼─────────────────────────────────────────┐
│                  线 B：TKE GPU 在线 GRPO（Phase 3/4，1×GPU）                 │
│                                                                            │
│  VERL RayPPOTrainer（GRPO 配置）—— 用它原生的在线循环，不改 rollout：         │
│    每个 step：                                                              │
│      ① vLLM 就地生成：同一道题 × group_size=16 个采样（GRPO 组内比较）         │
│      ② 自定义 reward function（唯一需要我们写的钩子）：                       │
│           把每个 response 送进 AGS 沙箱执行 → 跑 verify.sh → 算 reward        │
│           · 并发 8 路、实例复用、reward 缓存（见 2.2，这是耗时的命门）         │
│           · ⚠️ 此处必然是"从 TKE 侧调沙箱"，因为 VERL 进程就在 GPU 节点上，    │
│             沙箱在这里是被调用的执行器 —— 见 2.3 的辨析，这是正确分工          │
│      ③ GRPO 用组内 reward 差异算 advantage → 反向传播更新 actor（无 critic）   │
│      ④ 记录该 step 的 mean reward 到 CSV → Phase 5 用 matplotlib 画上升曲线   │
│    每 5 个 step 轮换下一道题（50 step 覆盖 10 题，只切 10 次沙箱工具）         │
│                                                                            │
│  训练完不关节点，直接接 Phase 4：新权重跑 5 题评测集 pass@1，与 baseline 对比   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 关键修正：为什么必须用「在线 GRPO」而不是「离线 tracing 训练」

**这是本 plan 的一次重要修正（2026-08-21）。** 初版设计是"Phase 2 离线采一批 tracing →
Phase 3 拿这批固定数据训 50 step"。该设计有一个会**直接导致验收不通过**的硬伤：

> 在固定数据集上跑 50 个梯度步，每条样本的 reward 是**数据里写死的常量**，
> 训练过程中不会变化。画出来的 reward 曲线是一条平线（或纯随机抖动），
> **不可能呈现"上升趋势"**，而这正是验收标准明确要求的一条。

reward 曲线上升的本质是"当前策略在环境里的真实表现随训练变好"，因此**必须每个 step 都用
当前最新策略去采样、并当场用沙箱打分**（on-policy 在线 RL）。VERL 本身就是为在线 RL 设计的
框架，初版设计反而绕开了它最擅长的部分。

**修正后仍然保留的克制**：只替换 VERL 的 **reward function**（一个打分函数），
**不碰它的 rollout 调度**，也就是**依然不使用** `BaseTool` + AsyncRollout 那套多轮工具集成。
原因不变：多轮工具集成需要让沙箱网络延迟被 VERL 异步调度稳定吃住，配置项多、排查成本不可控。
而自定义 reward function 是 VERL 的标准扩展点（`custom_reward_function.path`），
接口面极小、失败可定位，工程量比多轮工具集成小一个量级。

**"多轮操作"的验收要求由线 A 承担**：线 A 跑的是真正的多轮 ReAct，
单条 tracing 含 ≥3 步模型自主决策的操作，完整满足该条标准。线 B 的沙箱执行器同样会记录
执行序列 tracing（read → patch → test，≥3 步），两条线格式统一。

原生 `BaseTool` 多轮集成仍留作 Phase 6 加分项。

### 2.2 沙箱调用的三层降本设计（决定成败的耗时命门）

在线 GRPO 意味着沙箱调用量 = `step × group_size`，若按"每次都新建沙箱实例"的朴素做法：
50 step × 16 = 800 次，每次含 7.5GB 镜像冷启 + verify，串行约 60s/次 → **13 小时，直接爆炸**。
三层优化把它压到 25 分钟量级：

| 层 | 做法 | 效果 |
|---|---|---|
| ① 实例复用 | 每道题起 N 个**常驻**沙箱实例，每次打分前 `git checkout -- . && git clean -fd` 还原到干净题目态，跑完不 kill，留给下一个 sample 用 | 省掉每次的镜像冷启动，单次成本从 ~60s 降到 ~15s |
| ② 并发 | 同一 step 内 group 的 16 个 sample 分发给 N 个实例并行打分（`asyncio`/线程池） | 墙钟时间 ÷ N |
| ③ reward 缓存 | 以 `(task_id, patch内容hash)` 为 key 缓存 verify 结果；训练早期模型输出重复度高，命中率可观 | 命中即 0 成本 |

**配额适配**：AGS 沙箱工具（=镜像）配额只剩 2 个，但 GRPO 的 group **本来就是"同一道题的多个采样"**，
所以每个 step 只需要 1 道题的环境在线 → **1 个工具足够**。每 5 step 轮换一道题，
50 step 覆盖 10 题，全程只做 10 次"注册工具→等 ACTIVE→用完删"，与配额限制天然兼容。

**Phase 1.5 必须实测的两个关键数字**（决定并发度 N，不实测就上等于赌命）：
1. 一个沙箱工具最多能同时起几个实例？（决定 N 的上限）
2. 实例复用 + `git` 还原 + `verify.sh` 一次打分的实测耗时？（决定单次成本）

### 2.3 关键辨析：Agent 到底跑在哪？（2026-08-21 修正）

**这是本 plan 的第二次重要修正。** 初版把 ReAct 控制循环放在本机、沙箱当"远程执行器"，
虽然"修复操作"确实在沙箱内执行，但不符合课题"Agent 在沙箱中执行修复操作"的架构本意。
修正为 **Agent 完整跑在沙箱内**。

**先把 Agent 拆成三部分，才能说清哪部分该在哪：**

| Agent 的组成 | 需要什么 | 能放进 CPU 沙箱吗 | 本方案放在 |
|---|---|---|---|
| ① **大脑**：LLM 推理，决定下一步动作 | GPU | ❌ 课题明确要求"CPU 沙箱，不使用 GPU 沙箱"，课题自身就排除了 | TKE GPU 的 vLLM |
| ② **手脚**：读文件 / 打 patch / 跑 pytest | 隔离的题目环境 | ✅ **必须**在沙箱内，这是隔离性的核心价值 | 沙箱内 |
| ③ **控制循环**：ReAct 调度（几十行 Python） | 几乎无算力 | ✅ 可以 —— **这才是真正需要抉择的部分** | **沙箱内**（已修正） |

课题原文"Agent 在沙箱中执行修复操作"，字面指 ②，但把 ③ 也放进沙箱，才是完整意义上的
"Agent 在沙箱里跑"，也更贴近生产形态（真实 agent 部署就是这样：agent 与工作区同机，
只有模型推理是远程 API 调用）。

**为什么初版放在了外面？—— 一次判断失误**
初版潜意识里把"Agent 进沙箱"等同于"要把 agent 代码打包进镜像 → 重建 19 个 7.5GB 镜像"，
那确实是灾难（课题三踩过 arm64→amd64 跨架构构建的坑）。但查证课题三实测记录后确认：
**`sbx.files.write()` / `files.read()` 已验证可用**（`check_env.py` 与 `sandbox_runner.py`
都在用），所以完全可以**运行时注入**几十 KB 的 `agent.py`，**零镜像重建**。
成本比原先设想的低一个数量级，因此采纳"Agent 进沙箱"。

**修正后的收益**（不只是"更符合要求"）：
1. Agent 的每一步文件操作变成沙箱内的**本地调用**，不再是每步一次 SDK 网络往返
   （原方案 10 步 = 10 次跨地域往返；现在 10 步 = 0 次，只剩 10 次 vLLM 调用）
2. 沙箱是"真·执行环境"而非"远程 shell"，隔离语义更干净
3. tracing 由沙箱内的 Agent 自己写，是第一手记录，不经中转

**一个必须说明的例外：线 B（训练时）无法也不该这样做。**
Phase 3 的 reward function 由 VERL 在 GPU 节点上调用 —— VERL 进程就在 GPU 里，
它拿到模型生成的 patch 后需要打分，这个调用**必然从 TKE 侧发起**，沙箱在此处是
"被调用的验证执行器"。这不是妥协，而是符合课题规定的分工原文：
> "SandBox 只管执行与采集（纯 CPU），TKE GPU 只管训练。"

训练时的主控是训练框架（在 GPU 上），沙箱提供执行反馈；采集时的主控是 Agent（在沙箱内）。
两者都正确，因为角色不同。

**硬前置：沙箱必须能出网访问 vLLM —— 已实测确认 ✅（2026-08-21）。**
Agent 在沙箱内跑，就必须能从沙箱内发 HTTPS 请求到 TKE 上的 vLLM。
`experiments/probe_sandbox_outbound.py` 用内置 `code-interpreter-v1` 工具（零占配额）实测：
出网 GET（`api.github.com` 200，0.3~0.5s）、出网 POST 带 `Authorization` header（
`postman-echo.com/post` 正确回显，0.9s）、Python 3.12.11 + requests 2.32.4 已预装、
后台长时命令组合正常。**结论：主方案成立，2.3 节描述的架构直接执行，不需要兜底形态。**
（曾观察到 `qq.com`→501、`httpbin.org`→503，均为目标站点侧限流/风控，非沙箱侧问题——
换 `api.github.com`/`postman-echo.com` 后干净通过，已排除误判。）

**再次收紧（2026-08-21，用户明确要求）：Agent 解题过程中不允许有任何环节在本地运行。**
逐一核对 Agent 的三个组成部分：
- ② 手脚（读文件/打patch/跑测试）→ 沙箱内 ✅
- ③ 控制循环（ReAct 调度）→ 沙箱内 ✅
- ① 大脑（LLM 推理）→ 此前 Phase 2 还留了"本机 Mac MPS 推理"作为可选项，**现已删除**，
  硬性收紧为**只能是 TKE GPU 上的 vLLM**（见 Phase 2）。
三部分全部确认不落在本地，"agent 在沙箱里解题、本地不参与"这条要求完全满足。
`driver.py`（本机跑，负责"起沙箱→注入→运行→收取→传COS"）不算例外：它不读题、不生成
修复代码、不跑测试、不参与任何解题判断，纯粹是调用云端 API 的编排脚本，性质上等同于
本地敲一条 `kubectl apply` 去启动云端任务——命令是本地发起的，但"干活"的是云端。

**附带的安全要求（不可省略）**：vLLM 要被沙箱访问，就得暴露到沙箱可达的网络。
绝不允许裸暴露推理服务：
- 必须开启 vLLM 的 `--api-key` 鉴权，key 经环境变量注入，不写进代码或镜像
- 优先用内网/VPC 打通；若必须走公网，安全组只放通必要端口，并在训练结束后立即回收
- 沙箱内注入的 `agent.py` 通过环境变量读取 key，tracing 落盘前做脱敏，不记录 key

---

### 3.1 单条 episode 的 tracing 结构（JSON，一行一个 episode）

```json
{
  "episode_id": "swe-synth-0007_ep3",
  "task_id": "swe-synth-0007",
  "model_version": "qwen2.5-coder-1.5b-step0",
  "steps": [
    {
      "step": 0,
      "action": {"tool": "read_file", "args": {"path": "src/foo.py"}},
      "observation": "```python\n...\n```（截断至 4k 字符）",
      "reward": 0.0,
      "done": false
    },
    {
      "step": 1,
      "action": {"tool": "bash", "args": {"cmd": "grep -n bar src/foo.py"}},
      "observation": "12: def bar(...):",
      "reward": 0.0,
      "done": false
    },
    {
      "step": 2,
      "action": {"tool": "apply_patch", "args": {"diff": "--- a/src/foo.py\n+++ ..."}},
      "observation": "patch applied cleanly",
      "reward": 0.0,
      "done": false
    },
    {
      "step": 3,
      "action": {"tool": "run_tests", "args": {}},
      "observation": "FAIL_TO_PASS: 3/3 passed; PASS_TO_PASS: 5/5 passed",
      "reward": 1.0,
      "done": true
    }
  ],
  "final_reward": 1.0,
  "fail_to_pass_rate": "3/3",
  "pass_to_pass_rate": "5/5",
  "num_steps": 4
}
```

满足验收标准"单条 tracing 包含 ≥3 步操作序列 + 最终测试通过情况"。

### 3.2 奖励函数

```
reward = FAIL_TO_PASS 中变为 pass 的用例数 / FAIL_TO_PASS 总用例数
       （若 PASS_TO_PASS 中有回归 fail，则整体 reward = 0，防止 reward hacking）
```
直接复用课题三 `verify.sh` 的判据输出解析，无需重新设计测试基础设施。

超过最大步数 M（如 15 步）未 submit，视为 done=True，reward=0。

### 3.3 与 VERL `DataProto` 的对齐关系

技术要求写明"tracing 格式与 VERL 的 `DataProto` 对齐"。在**在线 GRPO** 架构下，这个对齐分两层：

**(1) 线 B 训练时**：`DataProto` 由 VERL 自己构造，我们**不手工拼**（这正是改用在线方案后省掉的
一大块风险 —— 初版设计要手工构造 `prompt_ids`/`response_ids`/`token_level_rewards`，
接口一旦随版本变动就得现场翻源码）。我们只需实现 reward function 返回标量分数，
VERL 内部会把它填到 `token_level_rewards` 的最后一个有效 token 上（outcome-reward 标准做法）。

**(2) 线 A 的 tracing 文件**：按下表与 `DataProto` 字段一一对应标注，
使其既是人可读的交付物，也能被直接喂给离线训练/SFT（Phase 6 备用）：

| tracing 字段 | 对应 `DataProto` 字段 | 说明 |
|---|---|---|
| system + 题目描述 + 历史 action/observation 拼接 | `prompts` / `input_ids` 前缀 | 每步重新拼接完整上下文 |
| 各 step 的 `action`（模型生成部分） | `responses` | 多步拼为一条完整 response |
| 有效 token 掩码 | `attention_mask` / `response_mask` | observation 段不计入 loss |
| `final_reward`（仅落在最后一个 response token） | `token_level_rewards` | 其余位置为 0，outcome-reward |
| `final_reward` 在同题 group 内的相对高低 | GRPO advantage 输入 | 组内归一化，无需 critic |

实现为 `pipeline/schema.py` + `pipeline/tracing_to_dataproto.py`（后者在线方案下非必需，
作为 Phase 6 离线实验的备用工具保留）。**具体字段名以 Phase 1 实测的 VERL 版本为准，
实测结论写入 `PROGRESS.md`。**

---

## 4. 模型选型

**候选**：`Qwen/Qwen2.5-Coder-1.5B-Instruct`

理由：
1. **代码任务专精**：Qwen2.5-Coder 系列在代码理解/修复类任务上专门优化，起点能力比同尺寸通用模型强，
   便于在有限 step 内观察到 reward 上升趋势。
2. **尺寸适配单卡 T4/A10**：1.5B 参数，bf16 权重 ~3GB，vLLM 推理 + VERL 训练（actor+critic 或
   GRPO 无 critic）能在单卡 16~24GB 显存内跑起来，避免申请多卡/大显存机型，压缩 Phase 3 的资源
   准备时间。
3. **GRPO 天然适配**：选 GRPO（而非 PPO）可以不训 critic 网络，进一步降低显存占用和训练复杂度，
   适合"单卡、≥50 step"这种轻量验收目标。
4. **开源协议宽松**（Apache 2.0），HuggingFace 直接下载，无需额外授权流程，不占用准备时间。

若 Phase 3 实测 1.5B 在 T4 上仍吃紧，备选降级 `Qwen2.5-Coder-0.5B-Instruct`；若显存充裕可升到
`Qwen2.5-Coder-7B-Instruct`（需 A10 24GB 或以上，视 Phase 0 探测到的可用机型而定）。

---

## 5. 分阶段执行计划与门禁

### Phase 0：资源探测与准备（不产生长时间占用）
- [x] 跑 `scripts/probe_cloud.py`（题目四扩展版，加了 tke/cfs/cos/gpu 探测项），结论见 1.1 节：
  - AGS 沙箱工具配额：8/10 已用，剩 2 个，团队共享会动态变化，Phase 2 前需重新探测
  - TKE：发现预留空集群 `swe-rl-cluster`（ap-shanghai），直接复用，只建 GPU 节点池
  - COS：发现预留 bucket `COS_BUCKET`（ap-shanghai），直接复用
  - CFS：无预留资源，决策不用，简化为纯 COS 传递
  - 权限：`AdministratorAccess`，无阻塞
- [ ] Phase 3 前：核实 ap-shanghai 可售 GPU 机型（`GN6S.LARGE20` / `PTX1.7XLARGE116` 等）具体
      芯片型号与单价，选定按量计费机型
- [ ] 本地环境：安装 `verl>=0.3.0`、`vllm`、`torch`（可先在本地/CPU 环境装好但不跑训练，
      验证 import 和 CLI 可用即可，避免第一次在 GPU 节点上装环境浪费 GPU 计费时间）
- **门禁**：AGS 可用 + TKE 权限确认 + 本地 verl 环境 import 通过 → 进入 Phase 1 ✅（资源确认已达成，
  本地 verl 环境安装验证见 Phase 1）

### Phase 1：本地离线打通（mock，零云成本）
- [ ] `pipeline/schema.py`：tracing 数据结构（dataclass + 校验），线 A / 线 B 共用
- [ ] `pipeline/reward.py`：解析 `verify.sh` 输出的 `result.json` → reward（含 P2P 回归判 0 的逻辑），
      用课题三 `data/proofs/*/verification.json` 的真实输出做单元测试，**不依赖云端**
- [ ] `sandbox_agent/agent.py`：**这就是要注入沙箱内运行的 Agent 本体**（ReAct 主循环）。
      设计约束：① 只依赖沙箱内已有的 python3.12 标准库 + `requests`（无 requests 则用 `urllib`），
      **不得依赖需要联网 pip 安装的包** ② 通过环境变量读 vLLM 地址与 API Key，不硬编码
      ③ tracing 直接追加写 `/task/tracing.jsonl`（崩溃也不丢已完成的步骤）
- [ ] 本地先"假装在沙箱里"跑通 `agent.py`：用本地临时目录模拟题目仓库 + mock LLM，
      验证 action 解析、多轮拼接、超步终止、tracing 落盘
- [ ] `driver.py`：注入 + 启动 + 收取（`files.write` → `commands.run` → `files.read`），
      本阶段先对着 mock 沙箱跑
- [ ] `pipeline/verl_reward_fn.py`：**VERL 自定义 reward function**（线 B 的唯一钩子），
      先用 mock 沙箱打分，本地跑通
- [ ] **对着实际安装的 VERL 版本核实接口**：`custom_reward_function` 的函数签名、GRPO 配置项名、
      单题 batch + group_size 的配置写法、reward 日志字段名 → **实测结论写入 PROGRESS.md**
      （这一步是防止"到了 GPU 节点上现场翻源码"的关键）
- [ ] 本地跑一次 VERL 官方 toy 示例（GSM8K，CPU 或极小规模），确认框架本身无坑
- **门禁**：mock 全流程跑通一条 ≥3 步 tracing + reward 计算正确 + VERL toy 示例能跑
  + VERL 接口写入 PROGRESS.md → 进入 Phase 1.5

### Phase 1.5：云上预热（不开 GPU，把 GPU 窗口的活儿提前干完）
- [x] ✅ **【最高优先级门禁】沙箱出网能力实测** —— **已通过**（2026-08-21，见 PROGRESS.md）：
      `experiments/probe_sandbox_outbound.py`，用内置 `code-interpreter-v1` 工具（零占配额）
      实测：① 出网 GET（`api.github.com` 200，0.3~0.5s）② 出网 POST 带 Authorization
      header（`postman-echo.com/post` header/body 正确回显，0.9s）③ Python 3.12.11 +
      requests 2.32.4 已预装 ④ 后台长时命令 + files 读写组合正常
      → **结论：按"Agent 注入沙箱内运行"执行**（2.3 节主方案成立，兜底形态不需要启用）
- [ ] **沙箱内运行时环境核对**：确认沙箱内 `python3` 版本、是否有 `requests`、
      能否 `files.write` 后直接 `commands.run` 起长时进程（Agent 要跑十几步，可能几分钟）
- [ ] **沙箱关键数字实测**（决定并发度，见 2.2）：注册 1 个题目工具，实测
      ① 单工具可同时起几个实例 ② 实例复用 + git 还原 + verify.sh 一次打分耗时
      ③ 并发打分是否稳定 → 结论写入 PROGRESS.md，据此定 `group_size` 与并发数
- [ ] **模型权重预取**：经 **ModelScope**（国内直连）下 `Qwen2.5-Coder-1.5B-Instruct`，
      传到 COS `COS_BUCKET`。⚠️ 绝不在 GPU 节点上直连 HuggingFace 拉权重
- [ ] **VERL 镜像预取**：把 VERL 官方镜像（预装 torch/vllm/flash-attn）同步到自有 CCR，
      TKE 内网拉取。⚠️ 绝不在 GPU 节点上 `pip install verl vllm torch`
- [ ] 核实 ap-shanghai 可售 GPU 机型的芯片型号与单价，选定按量计费机型
- **门禁**：沙箱出网能力已确认通过 ✅ + 权重与镜像已在云上就位
  + 沙箱并发/耗时数字已实测 → 才允许开 GPU 节点

### Phase 2：Agent 在沙箱内跑多轮 ReAct（线 A 交付，纯 CPU）
- [ ] `driver.py` 接真实 AGS（复用课题三 `clients/ags.py`），完成"注入 → 运行 → 收取"闭环
- [ ] **大脑（vLLM）位置：硬性只用 TKE GPU，不设本机选项**（2026-08-21 收紧）：
      Agent 的控制循环、读文件/打patch/跑测试全部已确定跑在沙箱内（2.3 节），
      为保证"agent 在沙箱里解题、本地不承担任何解题逻辑"这条要求彻底落实到底，
      **LLM 推理这个环节也不允许放在本机**（哪怕本机 MPS 能跑得动 1.5B、能省 GPU 费）——
      本机运行推理服务本质上是把"解题的大脑"放到了本地，与要求相悖，且需要内网穿透暴露给
      沙箱访问，可靠性也差。**因此 Phase 2 采集必须并入 Phase 3 的 GPU 窗口**：
      GPU 节点起来后先做 Phase 2 的 tracing 采集，再做 Phase 3 的训练，全程模型推理只在 TKE 上发生。
      `driver.py` 本身只做"起沙箱、注入 agent.py、启动、收 tracing、传 COS"这类编排/胶水工作，
      不读题、不生成代码、不跑测试、不做任何解题判断，因此本机运行 `driver.py` 不违反本条约束
      （类比本地敲 `kubectl apply` 不代表应用逻辑跑在本地）。
- [ ] 对 14 题训练集跑多轮 ReAct（Agent 在沙箱内自主执行），产出 tracing.jsonl（每条 ≥3 步）
- [ ] 人工抽查 3~5 条 tracing：步骤是否真实、observation 是否来自沙箱、reward 是否与 verify 一致
- [ ] 上传 COS（满足"SandBox → TKE 经 COS 传递"的技术要求）
- **门禁**：≥10 题成功产出结构化 tracing（每条 ≥3 步）且已上传 COS → 进入 Phase 3

### Phase 3：TKE GPU 在线 GRPO 训练（线 B，**GPU 计费开始**）
- [ ] 在预留集群 `swe-rl-cluster`（ap-shanghai）新建 GPU 节点池（按量，1 卡）
- [ ] 从 CCR 拉 VERL 镜像起 Pod，从 COS 挂载/下载模型权重
- [ ] 起 vLLM 服务（**必须带 `--api-key` 鉴权**，见 2.3 安全要求），暴露给沙箱访问
- [ ] **若 Phase 2 选了方案 (b)**：此处先完成 Phase 2 的 tracing 采集，再继续训练
- [ ] **先跑 baseline**：base 模型对 5 题评测集跑 pass@1（训练前基准，必须在同一节点窗口内做）
- [ ] 跑在线 GRPO ≥50 step：每 step 单题 × group_size 采样，reward 由沙箱实时打分，
      每 5 step 轮换题目；每 step 的 mean reward 落 CSV
- [ ] 存 checkpoint
- **门禁**：≥50 step 完成 + reward CSV 有可见上升趋势 → **不关节点**直接进 Phase 4

### Phase 4：闭环验证（复用 Phase 3 节点，不重开）
- [ ] 新 checkpoint 加载到 vLLM，对 5 题评测集重跑 pass@1，与 baseline 对比
- [ ] 对比数据落盘
- [ ] ✅ **立即删除 GPU 节点池**（保留空集群本身，不产生持续计费），清理沙箱工具与实例
- **门禁**：有训练前后 pass@1 对比数据 + 云资源已清理 → 进入 Phase 5

### Phase 5：复盘产出
- [ ] 整理 README.md：SandBox 环境构建方式、TKE 部署步骤、模型选型理由、训练超参、结果分析
- [ ] 整理最终产物目录结构，清理云资源确认无遗留计费项（对照课题三"沙箱工具配额清理"的经验，
      逐一 delete 本课题创建的沙箱工具/实例/GPU 节点池）

### Phase 6（加分项，视时间余量）
- [ ] 尝试 VERL 原生 `BaseTool` 多轮 rollout 集成，对比与自定义方案的效果/工程量差异

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| AGS 沙箱工具配额（上限 10，**探测实测团队已占 8/10，仅剩 2 个**） | 利用 GRPO「group=同题多采样」的特性，每 step 只需 1 道题环境在线 → 1 个工具够用；每 5 step 轮换一题，全程 10 次滚动注册（用前建、用完删），每次注册前重新探测余量 |
| VERL 版本接口变动 | 改用在线方案后已**不需要手工构造 DataProto**（风险面大幅缩小）；仅剩 reward function 签名与 GRPO 配置项名需在 Phase 1 本地实测并写入 PROGRESS.md |
| TKE GPU 机型申请审批耗时 | 已探测确认有预留空集群 `swe-rl-cluster`（ap-shanghai）可直接建节点池，无需再走新建集群审批；机型库存已确认有货 |
| 1.5B 模型在少量 step 内 reward 上升不明显 | 在线 GRPO 下 reward 曲线本身就反映策略真实变化；若上升不明显，可提高 group_size 增强组内对比信号、或聚焦更少题目多采样。验收"提升幅度不做硬性要求" |
| GPU 计费失控 | GPU 节点只开一次：baseline→训练→训练后评测全在同一窗口做完立即删节点池（保留空集群不计费）。见 0.1 铁律 |
| 沙箱调用量爆炸导致跑十几小时 | 见 2.2 三层降本（实例复用 + 并发 + reward 缓存）与第 7 节炸弹清单；Phase 1.5 强制先实测并发与单次耗时才允许开 GPU |
| ~~沙箱无法出网访问 vLLM~~ | ✅ **已排除**（2026-08-21 实测确认沙箱可出网 GET/POST，见 Phase 1.5 与 PROGRESS.md），主方案成立，无需兜底 |
| vLLM 暴露给沙箱访问带来的未授权访问风险 | 强制 `--api-key` 鉴权（key 走环境变量，不入代码/镜像/tracing）；优先内网打通，走公网则最小化端口开放并在训练结束立即回收。见 2.3 安全要求 |
| 沙箱内缺少 Agent 所需的 Python 依赖（无法联网 pip） | `agent.py` 只用标准库 + 沙箱已预装的包；Phase 1.5 核对沙箱内 `requests` 是否存在，缺失则改用 `urllib.request` |

---

## 7. 耗时炸弹清单（"绝不重演十几小时"的专项对策）

| # | 炸弹 | 朴素做法的代价 | 规避手段 | 落地位置 |
|---|---|---|---|---|
| 1 | **在线 RL 的沙箱调用量** | 50 step × 16 sample = 800 次；每次新建 7.5GB 镜像实例并串行 → 约 **13 小时** | 实例复用 + 8 路并发 + reward 缓存 → **≈25 分钟** | 2.2 / Phase 1.5 实测 |
| 2 | **沙箱工具配额只剩 2 个** | 每 step 切题都要注册+等 ACTIVE（`wait_tool_active` 超时上限 180s）→ 纯等待数十分钟 | batch=单题×group 多采样，每 5 step 才换一题，全程仅 10 次切换 | 2.2 / Phase 3 |
| 3 | **GPU 节点上装环境** | `pip install verl+vllm+torch` 约 10GB，20~40 分钟，且 GPU 全程计费空转 | 用 VERL 官方 Docker 镜像，**提前同步到自有 CCR** 走内网拉取 | Phase 1.5 |
| 4 | **HuggingFace 拉模型权重** | 国内直连经常几十 KB/s 或直接卡死，白等一小时是常态 | 走 **ModelScope** 下载，**提前传 COS**；备选 `HF_ENDPOINT=hf-mirror.com` | Phase 1.5 |
| 5 | 跨架构镜像构建（课题三踩过） | 本机 arm64 → 沙箱需 amd64，跨架构构建极慢且易出玄学错误 | **本课题完全复用课题三已推送的镜像，不做任何镜像构建** | 1. 资产复用 |
| 6 | 反复开关 GPU 节点 | 每次建池+节点初始化（含 GPU 驱动）约 10 分钟，开关三次等于白烧半小时 | GPU 只开一次，三件事连做完再删 | 0.1 铁律 |

**兜底策略（时间墙）**：若 Phase 3 训练在 90 分钟内未能跑完 50 step，立即降规模保交付 ——
`group_size` 减半、题目数减到 10、`max_response_length` 收紧。**先保证"闭环跑通且曲线可见"，
再谈规模和效果**，绝不为追求更漂亮的数字而让 GPU 长时间空转。

---

## 8. 验收标准 ↔ 方案映射（自查表）

| 验收标准 | 由哪条线/哪个 Phase 保证 | 状态 |
|---|---|---|
| SandBox 批量拉起 SWE 环境（≥10 题）+ 结构化 tracing | 线 A / Phase 2（复用课题三 19 题镜像，取 14 训练 + 5 评测） | 设计就绪 |
| **Agent 在沙箱中执行修复操作** | **Agent 本体（ReAct 循环）运行时注入沙箱内执行**（`files.write` + `commands.run`，零镜像重建）；LLM 推理在 TKE，因课题规定沙箱纯 CPU。见 2.3 | **已修正**（初版把控制循环放在本机） |
| 单条 tracing ≥3 步操作 + 最终测试通过情况 | 线 A：沙箱内 Agent 多轮 ReAct，tracing 由 Agent 第一手追加写盘 | 设计就绪 |
| TKE 上 VERL 可运行 + 训练 ≥50 step | 线 B / Phase 3（VERL 官方镜像 + 在线 GRPO） | 设计就绪 |
| **reward 曲线呈上升趋势** | 线 B **在线 GRPO**（每 step 用当前策略采样 + 沙箱实时打分）→ matplotlib 出图 | **已修正**（初版离线方案画不出上升，见 2.1） |
| ≥1 轮完整闭环 | Phase 2 采集 → COS → Phase 3 训练 → Phase 4 回沙箱评估 | 设计就绪 |
| 训练前后 pass@1 对比 | Phase 3 开头跑 baseline、Phase 4 跑训练后，5 题评测集不参与训练 | 设计就绪 |
| README（环境构建/部署步骤/选型理由/超参/结果分析） | Phase 5 | 待产出 |
| 技术要求：SandBox→TKE 经 COS/CFS 传 tracing | 线 A 上传 COS `COS_BUCKET` | 设计就绪 |
| 技术要求：VERL ≥0.3.0 + GRPO/PPO | 官方镜像锁版本，算法用 GRPO（免 critic，省显存） | 设计就绪 |
| 技术要求：reward = fail→pass 数 / 相关测试数 | 复用课题三 `verify.sh` 的 F2P/P2P 输出（3.2 节公式） | 设计就绪 |
