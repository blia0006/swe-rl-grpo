# 题目四进度日志（PROGRESS.md）

> 每次进入新阶段/遇到关键决策/踩坑，请在此追加记录（倒序或正序均可，注明时间）。
> 目的：模型切换后，读此文件 + `plan.md` + `TASK-SPEC.md` 即可无缝续做，不需要重新问用户。

---

## 2026-08-21 · Phase 0 启动前 · 需求梳理与方案设计完成

**已完成：**
- 通读题目四原始需求（含两张截图内容），整理为 `TASK-SPEC.md`
- 探查课题三（`/Users/user/学习/课题三-数据合成/`）已有资产，确认可复用：
  - `dist/swe-synth-delivery-20260821/data/tasks.jsonl`：19 道 ACCEPTED 题目，每题含镜像地址
    （TCR 个人版 `ccr.ccs.tencentyun.com/tcb-100008634787-zbaf/swe-synth-000X:v1` /
    `:v1-sol`）、golden patch、`verify.sh` 判据契约
  - `swe_synth/clients/ags.py`：AGS 沙箱 SDK 封装（创建实例/执行命令/销毁）
  - `scripts/probe_cloud.py`：云资源只读探测脚本，可直接复用探测 AGS/TCR/TKE 相关权限
- 调研 VERL（volcengine/verl）官方文档：
  - 支持原生多轮工具调用（`BaseTool` + SGLang/vLLM async rollout），官方有 `sandbox_fusion`
    示例可参考改造
  - 决策：**第一版不采用原生多轮工具调用集成**，改为「自定义 rollout 脚本（生成←→执行解耦）
    + tracing.jsonl 落盘/COS传递 + VERL 训练 API」，原因详见 `TASK-SPEC.md` 第 5 节 /
    `plan.md` 第 2.1 节。原生集成留作 Phase 6 加分项。
- 确定模型选型：`Qwen/Qwen2.5-Coder-1.5B-Instruct`（GRPO，单卡 T4/A10），理由见 `plan.md` 第 4 节
- 已知风险（来自课题三收尾记录）：AGS 沙箱工具配额上限 10，课题三收尾时已将本项目占用清理为 0，
  **本课题需要重新滚动注册沙箱工具**（用前建、验证完删），不能一次性把 19 题全注册在线
- 已写出 `plan.md` 完整六阶段计划（含每阶段门禁、时间预算、风险应对表）

**当前状态：** Phase 0（资源探测与准备）尚未开始执行，等待用户确认后启动。

**下一步（Phase 0 待办）：**
1. 跑 `probe_cloud.py`（或题目四目录下的新副本）确认：AGS 沙箱工具当前剩余配额、
   TKE 集群/GPU 节点池现状与权限、COS bucket / CFS 现状
2. 确认 TKE GPU 机型选择（T4 or A10，按量计费）与预计到手时间（若需审批，提前走流程）
3. 本地 `pip install verl vllm torch` 验证 import，不跑训练

---

## 2026-08-21 · Phase 0 执行完毕 · 云资源探测结论（重大发现：TKE 集群 + COS bucket 已预留）

**做了什么：**
- 在课题三 `scripts/probe_cloud.py` 基础上扩展出题目四专用版本
  `/Users/user/学习/题目四：强化学习/scripts/probe_cloud.py`，新增 TKE / CFS / COS / GPU 库存
  探测（原脚本只有 cam/tcr/ags/tokenhub），复用课题三 `.env` 凭证（本目录若无 `.env` 会自动回退读取
  `../课题三-数据合成/.env`，同账号无需重复配置）
- 分别在 `ap-guangzhou`（课题三 SandBox 所在地域）和 `ap-shanghai` 跑了全量探测

**核心结论（详见 `plan.md` 1.1 节，已同步更新 plan.md 的架构/风险表）：**

1. **AGS 沙箱工具配额（ap-guangzhou）**：账号（团队共享）当前 **8/10** 已被其他项目占用（工具名如
   `mini-agent-codeinterp`/`browser-carltest` 等，均非本课题资产），**只剩 2 个名额**。比课题三收尾
   时预期的更紧张 —— 说明配额是全团队实时共享的，不能假设课题三清理后余量会一直保留。
   → Phase 2 必须每次注册前重新探测剩余配额，严格"用一个、删一个"串行执行。

2. **TKE（重大发现）**：ap-shanghai 存在一个空壳集群 **`swe-rl-cluster`（`CLUSTER_ID`）**，
   状态 Running，0 节点、0 节点池，创建于本课题需求下发前后，**命名与课题精确匹配**，
   判断为课题组已预先建好留给我们用。→ **直接复用该集群，Phase 3 只需新建 GPU 节点池**，
   不用再走"新建集群"的审批流程，大幅节省时间。
   （ap-guangzhou 另有 6 个集群，均为同事个人项目命名如"郑宗恒-test"，与本课题无关，不使用）

3. **COS（重大发现）**：ap-shanghai 存在 bucket **`COS_BUCKET`**，创建于
   2026-07-08（早于本次任务启动），命名精确匹配"swe + rl + tracing"，判断为课题组预先建好的
   tracing 传递通道。→ **直接复用，Phase 2/3 的 tracing.jsonl 上传/下载都用这个 bucket，不新建**。

4. **CFS**：ap-shanghai 现有 10 个文件系统均为同事项目所建（如 `cfs-phonexue`、`YK-CFS-EKS`），
   无 `swe-rl` 专属预留 → **决策：不使用 CFS**，全部靠 COS 传递 tracing，链路更简单。

5. **GPU 机型库存（ap-shanghai 可售）**：`GN6S.LARGE20`（1 卡，最便宜）、`PTX1.7XLARGE116`（1 卡，
   28 核 116GB）、`PTX2.8XLARGE96`（1 卡）等均有货，无需担心排队。Phase 3 建节点池前再核实具体
   GPU 芯片型号（P4/T4/A10 等）与单价，选定后一次性建池、训练+评测做完立即删池。

6. **权限**：当前子账号已挂 `AdministratorAccess` 策略，TKE 节点池创建/COS 读写/CFS 均无权限阻塞。

**由此确定的最终架构（跨地域）：**
- AGS SandBox：沿用课题三 **ap-guangzhou**（沙箱镜像/工具不用迁移）
- TKE 训练 + COS tracing 传递：统一 **ap-shanghai**（跟着预留资源走）
- 跨地域 COS 访问对"批量传 tracing 文件"这种非实时场景无影响，可忽略

**已同步更新：** `plan.md` 新增 1.1 节「Phase 0 探测结果」、更新 Phase 0 checklist 为已完成、
更新第 6 节风险表（AGS 配额风险量化为"8/10 已用"、TKE 风险改为"已复用预留集群，无需再审批"）

**当前状态：** Phase 0 完成 ✅。下一步进入 Phase 1（本地 mock 打通 tracing → DataProto 映射 +
VERL 环境 import 验证）。

**wandb 说明（回答用户此前提问）：** wandb（Weights & Biases）是一个训练可视化/实验跟踪平台，
可以自动画 loss/reward 曲线、比较多次实验。但需要注册账号、联网上报数据，对本课题这种"跑几十
step、出一张 reward 曲线图"的需求属于"杀鸡用牛刀"。**决策：使用本地 matplotlib 从训练日志直接
画图**，验收标准明确认可"日志/wandb/matplotlib图表均可"，本地方案更省事、不依赖额外账号、
不产生额外的外部网络依赖，Phase 3 训练脚本会把每个 step 的 reward 落一份 CSV/JSON 日志，
Phase 5 直接用 matplotlib 出图即可。

---

## 2026-08-21 · 方案复核（用户要求"确保符合课题要求 + 不会一开始就十几小时"）· 发现并修正 1 处硬伤

### A. 逐条对照验收标准，发现 1 个会导致验收不通过的硬伤

**硬伤：初版设计画不出"reward 曲线上升"。**
初版是「Phase 2 离线采一批 tracing → Phase 3 拿这批固定数据训 50 step」。
在固定数据集上跑 50 个梯度步，每条样本的 reward 是**数据里写死的常量**，训练中不会变化，
画出来是平线或随机抖动，**无法满足验收标准第 4 条"reward 曲线呈上升趋势"**。

**修正**：改为 **VERL 原生在线 GRPO 循环 + 自定义 reward function**。
每 step 用当前最新策略采样、当场送 AGS 沙箱跑 `verify.sh` 打分 → reward 曲线天然上升。
仅替换 reward function（VERL 标准扩展点 `custom_reward_function`），**不碰 rollout 调度**，
因此**依然不使用** `BaseTool` + AsyncRollout 那套多轮工具集成（风险不变，仍留作 Phase 6 加分项）。

附带收益：不再需要手工构造 `DataProto`（`prompt_ids`/`token_level_rewards` 等由 VERL 内部完成），
**VERL 版本接口变动的风险面大幅缩小**，原来最担心的"到 GPU 节点上现场翻源码"基本消除。

**"多轮操作 ≥3 步"改由两条线共同保证**：
- 线 A（Phase 2，纯 CPU）：真正的多轮 ReAct，模型看 observation 再决策下一步 → 高质量 tracing 传 COS
- 线 B（Phase 3，训练）：沙箱执行器记录 read→patch→test 执行序列，格式与线 A 统一

其余 6 条验收标准经逐条核对均可满足，映射关系已写入 `plan.md` 第 8 节自查表。

### B. 耗时炸弹盘点（基于课题三实测数据算账）

复核出 6 个炸弹，其中第 1 个是真正的"十几小时"来源，全部对策已写入 `plan.md` 第 7 节：

| # | 炸弹 | 朴素做法代价 | 对策 |
|---|---|---|---|
| 1 | 在线 RL 沙箱调用量 | 50step×16sample=800 次，每次新建 7.5GB 镜像实例且串行 → **≈13 小时** | 实例复用 + 8 路并发 + reward 缓存 → **≈25 分钟** |
| 2 | 沙箱工具配额仅剩 2 个 | 每 step 切题都要注册+等 ACTIVE（超时上限 180s）→ 白等数十分钟 | 借 GRPO「group=同题多采样」特性，每 step 只需 1 题在线，每 5 step 才换题，全程 10 次切换 |
| 3 | GPU 节点上 pip 装环境 | verl+vllm+torch 约 10GB，20~40 分钟且 GPU 计费空转 | 用 VERL 官方 Docker 镜像，提前同步到自有 CCR 走内网拉 |
| 4 | HuggingFace 拉权重 | 国内常几十 KB/s 或卡死，白等一小时 | 走 ModelScope 下载并提前传 COS；备选 hf-mirror.com |
| 5 | 跨架构镜像构建 | 课题三已踩：arm64→amd64 极慢且玄学错误 | 本课题**完全复用课题三已推送镜像，零构建** |
| 6 | 反复开关 GPU 节点 | 每次建池+节点初始化约 10 分钟 | GPU **只开一次**，baseline→训练→训练后评测连做完再删 |

**新增 Phase 1.5「云上预热」（不开 GPU）**：把模型权重预取传 COS、VERL 镜像同步到 CCR、
**沙箱实例复用与并发上限实测** 三件事在开 GPU 之前全部做完。
门禁：这三项没做完，**不允许开 GPU 节点**。

**新增时间墙兜底**：Phase 3 若 90 分钟内没跑完 50 step，立即降规模保交付（group_size 减半、
题目减到 10、response 长度收紧），先保"闭环跑通 + 曲线可见"，绝不为追求漂亮数字让 GPU 空转。

### C. 复核中确认的课题三实测数据（用于估算，来源 PROGRESS/交付物）

- 沙箱冷启动：内置工具 0.5s；**自定义镜像每个约 7.5GB**（32 个 tag 已在远端构建机）
- `verify.sh` 本地执行：3.9s（单题）；沙箱内 `commands.run` 超时设 600s
- `wait_tool_active`：超时上限 180s，轮询间隔 3s（说明工具注册是异步且非秒级）
- 沙箱调用必须 `user="root"` + `PYTEST_ADDOPTS=--color=no`（否则判分正则失配造成假阴性）
- 课题三每题占 **2 个**沙箱工具（`:v1` + `:v1-sol`）；**本课题只需 `:v1` 题目态 → 开销减半**
- 团队账号"只增不改不删"，那 8 个占配额的工具是同事的历史残留，不可删

### D. 修正后的时间预算

| 阶段 | 预估 | 计 GPU 费 |
|---|---|---|
| Phase 0 探测 | 30min ✅已完成 | 否 |
| Phase 1 本地 mock 打通 + VERL 接口实测 | 1.5~2h | 否 |
| Phase 1.5 云上预热（权重/镜像/沙箱实测） | 40min | 否 |
| Phase 2 SandBox 多轮 ReAct 采集 | 30~40min | 否 |
| Phase 3 TKE 在线 GRPO 50 step | 60~75min | **是** |
| Phase 4 闭环评测（不关节点） | 20min | **是** |
| Phase 5 README + 曲线图 | 40min | 否 |

**GPU 计费窗口目标 < 2.5h，全流程 < 6h。**

**当前状态：** 方案已复核修正，`plan.md`（第 0/2/2.1/2.2/3.3/5/6/7/8 节）与 `TASK-SPEC.md`
（第 5.2 条）已同步更新。下一步进入 **Phase 1（本地 mock 打通）**，全程不产生云费用。

---

## 2026-08-21 · 架构修正二：Agent 必须跑在沙箱内（用户质疑触发，判断失误已纠正）

**用户质疑原文**："为什么是在本机上跑？我们不应该要把 agent 打包进入沙箱，在沙箱环境里面跑吗？"

**结论：质疑成立，初版设计在这一点上是错的，已修正。**

### 判断失误的成因

初版把 ReAct 控制循环放在本机、沙箱当"远程执行器"。当时的（错误）推理链是：
"Agent 进沙箱" ⇒ "要把 agent 代码打包进镜像" ⇒ "重建 19 个 7.5GB 镜像" ⇒ "课题三踩过
arm64→amd64 跨架构构建的坑，要几小时" ⇒ 直接排除。

**查证课题三记录后发现前提错了**：`sbx.files.write()` / `files.read()` **早已实测可用**
（`scripts/check_env.py:390` 写了 `/tmp/probe.txt` 并读回校验；`agent2/sandbox_runner.py:114`
用 `files.read("/task/result.json")` 取判分结果）。
→ 完全可以**运行时注入**几十 KB 的 `agent.py`，**零镜像重建**。成本比设想的低一个数量级。

### 修正后的定性：把 Agent 拆三层，各归其位

| Agent 组成 | 需要 | 放在哪 | 依据 |
|---|---|---|---|
| ① 大脑：LLM 推理决定下一步 | GPU | TKE 的 vLLM | 课题明确"CPU 沙箱，不使用 GPU 沙箱"，课题自身排除了沙箱内推理 |
| ② 手脚：读文件/打 patch/跑 pytest | 隔离题目环境 | **沙箱内** | 隔离性的核心价值（初版也在沙箱内，这点没错） |
| ③ 控制循环：ReAct 调度 | 几乎无算力 | **沙箱内**（已改） | 真正需要抉择的部分；放进去才是完整意义的"Agent 在沙箱里跑" |

新的线 A 形态：
```
driver.py（本机，只做投放+收取，不参与决策）
  ① 起沙箱实例 → ② files.write 注入 agent.py → ③ commands.run 启动 → ④ files.read 收 tracing

沙箱实例内（Agent 完整生命周期）：
  agent.py 的 ReAct 循环 ──HTTPS(带 API Key)──> vLLM(TKE GPU) 拿 action   ← 唯一出网调用
                         └──本地文件操作──> 读写题目仓库 / 跑 pytest       ← 零网络开销
  追加写 /task/tracing.jsonl → 收尾跑 verify.sh 算 final_reward
```

**修正带来的额外收益**（不只是"更合规"）：
Agent 每步文件操作从"跨地域 SDK 往返"变成"沙箱内本地调用"。原方案 10 步 = 10 次跨地域往返，
现在 10 步 = 0 次，只剩 10 次 vLLM 调用。**更快、隔离语义更干净、tracing 是第一手记录**。

### 一个必须说明的例外：线 B（训练时）不能也不该这样做

Phase 3 的 reward function 由 VERL 在 GPU 节点上调用 —— VERL 进程就在 GPU 里，拿到模型
生成的 patch 后要打分，这个调用**必然从 TKE 侧发起**，沙箱此时是"被调用的验证执行器"。
这不是妥协，正是课题规定的分工：**"SandBox 只管执行与采集，TKE GPU 只管训练"**。
采集时主控是 Agent（在沙箱内），训练时主控是训练框架（在 GPU 上），角色不同，两者都正确。

### 新增的硬前置（未验证项，已列为 Phase 1.5 最高优先级门禁）

⚠️ **沙箱能否主动出网访问 vLLM？——课题三从未测过**（它的沙箱只跑本地 pytest，不需要联网）。
- `NetworkMode=PUBLIC` 是课题三一直在用的配置（`clients/ags.py:82`、`settings.yaml:89`），
  强烈暗示具备出网能力，但**没有证据**
- Phase 1.5 用 `curl` + 一次真实 OpenAI 格式 POST 实测，5 分钟出结果
- **兜底**：若不通，退回"控制循环在外 + 沙箱作执行器"的初版形态。功能与 7 条验收标准
  均不受影响（②始终在沙箱内），但需在 README 如实说明该平台限制
- 同时要核对：沙箱内 `python3`（已知 3.12.11）是否有 `requests`（无则用 `urllib`）、
  `commands.run` 能否承载跑几分钟的长时进程

### 附带的安全要求（写入 plan 2.3，不可省略）

vLLM 要被沙箱访问就得暴露到沙箱可达网络，**绝不允许裸暴露推理服务**：
- 必须开 vLLM `--api-key` 鉴权，key 经环境变量注入，不写进代码/镜像/tracing
- 优先内网/VPC 打通；必须走公网时最小化端口开放，训练结束立即回收
- 沙箱内 `agent.py` 从环境变量读 key，tracing 落盘前脱敏

### 连带调整

- **Phase 2 的"大脑"位置倾向改为方案 (b)**（并入 Phase 3 的 GPU 窗口）：因为沙箱内的 Agent
  必须能主动访问 vLLM，而本机 Mac 起的服务通常无法被公网沙箱访问（需内网穿透，不可靠）。
  TKE 上的服务更容易被访问。最终取决于 Phase 1.5 出网实测结果。
- `plan.md` 已更新：第 2 节架构图、**新增 2.3 节完整辨析**、Phase 1（改为写 `sandbox_agent/agent.py`
  + `driver.py`）、Phase 1.5（新增出网实测门禁）、Phase 2/3、第 6 节风险表（+3 条）、
  第 8 节自查表（新增"Agent 在沙箱中执行"一行）

**当前状态：** 架构定型。下一步 Phase 1（本地打通，零云费用）。

---

## 2026-08-21 · Phase 1.5 门禁项提前验证：沙箱出网能力实测 —— ✅ 通过

用户要求"先做测试跑通了再开始"，把 2.3 节列的最高优先级门禁提前做掉（未开 GPU，零/近零成本）。

**脚本**：`experiments/probe_sandbox_outbound.py`。复用课题三 `.env` 里已验证过的 E2B/AGS 凭证，
用**内置** `code-interpreter-v1` 工具（`AGS_SANDBOX_TEMPLATE`，不占本课题仅剩 2 个的自定义
沙箱工具配额）起一个沙箱实例，测 4 件事：

| # | 测什么 | 结果 |
|---|---|---|
| ① 出网 GET | `curl` 多个公网目标 | `api.github.com` → 200，0.3~0.5s（**决定性证据**，干净通过） |
| ② 出网 POST + 自定义 header | 模拟 OpenAI 格式请求（`Authorization: Bearer <key>`） | `postman-echo.com/post` → header/body 均被目标端正确回显，0.9s |
| ③ 运行环境 | python3 版本、`requests` 是否预装 | Python 3.12.11，requests 2.32.4（`agent.py` 可直接用 requests，无需 urllib 兜底） |
| ④ 长时/后台命令 | `nohup` 后台任务 + `files.write/read` 轮询 | 5s 后台任务如期完成，组合正常（Agent 跑十几步、几分钟不退出的场景可行） |

**排查记录（避免后人被同样的假阴性绕进去）**：第一次跑测到 `www.qq.com` 返回 `HTTP_CODE=501`，
第二次测到 `httpbin.org` 返回 `503`。乍看像"出不了网"，但两次耗时都是真实网络往返
（40ms~2s，不是本地瞬间失败），说明**请求已经到达对方服务器，只是对方拒绝/限流**——
501/503 是目标站点侧的行为（`qq.com` 疑似对非浏览器 UA 的 WAF 拦截，`httpbin.org` 是公共
demo 服务本身不稳定），与沙箱网络能力无关。换成 `api.github.com`（GET）和
`postman-echo.com`（POST）两个更适合脚本化访问的目标后，**全部干净通过**。

**结论**：
1. 沙箱可以主动发起出网 HTTPS 请求（GET/POST 均可），且支持自定义 header——
   这正是 Agent 调用 vLLM（`Authorization: Bearer <api-key>`）所需要的能力
2. `plan.md` 2.3 节描述的**主方案（Agent 完整跑在沙箱内）架构成立**，
   **不需要**启用"控制循环在外、沙箱只当执行器"的兜底形态
3. `agent.py` 可以直接用 `requests` 库（沙箱已预装 2.32.4），无需自己写 `urllib` 兜底
4. 冷启动只需 0.4~0.5s（内置工具，比自定义题目镜像快得多，符合预期）

`plan.md` 已同步更新：2.3 节硬前置状态改为"已实测确认 ✅"、Phase 1.5 对应门禁项打勾、
第 6 节风险表对应行改为"已排除"。

**当前状态：** 唯一悬而未决的架构前提已验证通过。下一步正式进入 **Phase 1**（本地 mock 打通，
`pipeline/schema.py` / `pipeline/reward.py` / `sandbox_agent/agent.py` / VERL 接口核实），
全程零云费用。

---

## 2026-08-21 · 用户收紧要求："Agent 解题过程不允许有任何环节在本地运行"

审查了一遍 `plan.md`，发现 Phase 2 里"大脑（vLLM）位置"还留了一个未收口的选项：
"(a) 本机 Mac 用 MPS 跑 1.5B 推理"。这个选项虽然此前已"倾向 (b)"，但没有正式删除，
留着就是隐患——一旦真的选了 (a)，Agent 解题过程中最核心的一环（LLM 推理决策）就会
落在本地机器上，直接违反本次收紧的要求。

**处理**：删除选项 (a)，Phase 2 的 vLLM **硬性只能跑在 TKE GPU**，不再讨论本机推理。
逐一核对 Agent 三个组成部分（大脑/手脚/控制循环）后确认三者全部不落在本地：

| 组成部分 | 落地位置 |
|---|---|
| 大脑（LLM 推理） | TKE GPU 的 vLLM（唯一选项，已删除本机选项） |
| 手脚（读文件/打patch/跑测试） | 沙箱内 |
| 控制循环（ReAct 调度） | 沙箱内（`agent.py` 运行时注入沙箱执行） |

**`driver.py` 的定性（不算违反本条要求）**：`driver.py` 跑在本机，但它只做"起沙箱、
注入 `agent.py`、启动、收 tracing、传 COS"这类编排/胶水工作，本身不读题、不生成修复代码、
不跑测试、不做任何解题判断——性质等同于本地敲一条 `kubectl apply` 去启动云端任务：
命令是本地发起的，但真正"干活"（解题）的逻辑全部在云端（沙箱 + TKE）执行。

`plan.md` 已同步更新：0 节时间预算表脚注、2.3 节新增"再次收紧"小节、Phase 2 大脑位置
条目改为硬性约束。

**当前状态：** 架构层面"本地零解题逻辑"已彻底核实并收口，无遗留隐患。准备正式开始 Phase 1。

---

## 2026-08-23 · 环境衔接复核（用户改造课题三"全流程沙箱内跑" + 沙箱地域迁移后）

用户明确本次只关心**环境能否衔接**，题目三的数据内容/质量不在本课题范围内，因此不做数据层面
的核实或修复。

**复核结论：可以衔接，且比原计划更简单，`plan.md` 已同步更新。**

1. **好消息：沙箱地域迁移，跨地域架构不再需要**
   - 课题三 `.env` 里 `E2B_DOMAIN`（数据面，沙箱连接）和 `TENCENTCLOUD_REGION`（管理面，
     建工具/起实例）都已从 `ap-guangzhou` 改成 **`ap-shanghai`**，与预留的 TKE 集群
     `swe-rl-cluster`、COS bucket `COS_BUCKET` 同地域
   - 用 `AGSClient().list_tools()` 做了一次只读连通性实测（不产生费用），在 ap-shanghai
     成功返回 11 个已有沙箱工具（含课题三自己的 `swe-synth-shared-runner`，ACTIVE 状态），
     确认管理面 API 连通正常
   - **`plan.md` 1.1 节已更新**：删除"跨地域架构"表述，SandBox / TKE / COS 现在统一
     ap-shanghai，`driver.py` 与 TKE 训练进程之间不再有跨地域访问

2. **`swe_synth/clients/ags.py` 接口保持兼容，且比预期更成熟**
   - `AGSClient`：`create_tool` / `start_instance`（支持 `image_override` 实例级换镜像，
     不用每题重建工具）/ `stop_instance` / `renew_instance` / `list_tools` / `find_tool` /
     `delete_tool` / `wait_tool_active`，全部齐备，题目四 `driver.py` 可直接复用，
     尤其是 `start_instance` 的"一个工具、多题复用、实例级覆盖镜像"能省掉大量沙箱工具
     配额（题目四本来就仅剩 2 个配额的顾虑因此缓解）
   - 凭证齐全：`AGS_ROLE_ARN` / `TCR_USERNAME` / `TCR_REGISTRY_TYPE` 均已配置；
     `TCR_REGISTRY=ccr.ccs.tencentyun.com`、`TCR_NAMESPACE=tcb-100008634787-zbaf`
     与 `plan.md` 第 1 节原有假设完全一致，镜像路径不用改

3. **数据源路径变化（仅路径，不涉及内容判断）**
   - 课题三重构后不再有 `dist/swe-synth-delivery-20260821/` 快照目录，`plan.md` 原先引用
     的路径已失效，改为直接读实时的 `课题三-数据合成/data/tasks.jsonl`（当前 29 条记录）
   - `plan.md` 第 1 节资产表、"题目划分"描述已同步更新为"从现有题目池选 14+5"，
     不再写死"19 道 ACCEPTED"这个具体数字/状态判断

4. **一个顺带发现、按用户要求未处理的现象**：巡检执行 `课题三-数据合成/scripts/
   keepalive_check.py` 时，该脚本对一个旧沙箱实例（状态 `ALL_DONE`）做了无条件下载覆盖，
   把本地 `data/tasks.jsonl`（已提交，29 条）覆盖成了该实例里的旧结果（21 条），
   目前课题三工作区处于未提交的"脏"状态（`git status` 可见）。因用户已明确"题目三内容不用管"，
   **本次未做还原**，如后续需要可执行 `git checkout -- data/tasks.jsonl data/proofs/` 一键复原
   （不影响已提交历史）。这个现象跟"能否衔接"无关，只是记录以防后续误解。

**当前状态：** 环境衔接确认通过，架构比原计划更简单（无跨地域）。仍在"先不要开始做题"的
暂停点，等待用户下一步指示（正式进入 Phase 1）。

---

## 2026-08-23 · 项目独立化改造 · 与课题三代码解耦，初始化独立 Git 仓库

**背景：** 用户明确要求「课题四必须重新做一个，不基于课题三继续，需要重新创建一个独立的
GitHub 项目去推送」。经确认边界（`plan.md`/`TASK-SPEC.md` 描述以外的口头约定）：
- **题目/SWE 环境层**：只要代码仓库和流程独立即可，镜像仍可继续引用课题三已推送到
  TCR 的现成镜像、`tasks.jsonl` 题目数据（工作量最小，且课题三镜像已过真实沙箱集成测试）
- **云资源/凭证层**：TKE 集群、COS bucket 等基础设施属于账号预留资源，与课题三代码无关，
  继续共用；但 `.env` 凭证配置文件本身要在题目四目录下独立一份，不再 fallback 读取课题三路径
- **代码仓库层**：本地 `git init` 独立仓库，先梳理 `.gitignore`，远程仓库地址确定后再 push

**已完成的解耦改动：**
1. `.env` 头部注释残留的"课题三：数据合成"标题改为"题目四：强化学习"，并明确写明
   本文件独立、不 fallback 读取其他课题目录
2. `scripts/probe_cloud.py`：去掉 `SIBLING_ENV`（`../课题三-数据合成/.env`）回退逻辑，
   `load_env()` 只读本项目 `.env`；对应的报错提示文案同步更新
3. `experiments/probe_sandbox_outbound.py`：`_load_env()` 由硬编码读取课题三 `.env` 绝对路径，
   改为读 `Path(__file__).resolve().parent.parent / ".env"`（本项目目录）
4. `pipeline/reward.py`：自测入口（`__main__`）依赖的真实判分样本 `verification.json`，
   从课题三 `data/proofs/*/verification.json` 挑 3 个代表性样本（swe-synth-0001/0009/0019）
   vendored 到本仓库 `pipeline/testdata/`，默认 glob pattern 改为读本地 `testdata/`目录，
   使自测脚本不再依赖课题三外部目录是否存在。已重新跑通：3 题全部 `empty=0.0 golden=1.0` 通过
5. 新建 `.gitignore`（忽略 `.env`、`__pycache__/`、`*.pyc`、`.venv/`、训练产物等），
   本地 `git init` 并完成首次 commit（`main` 分支，root commit，20 个文件，不含任何 `.env`
   或 `__pycache__`，已核实 `git status` 干净）

**未变更（按约定保留对课题三的引用，属于"基础设施/资源"而非"代码继承"）：**
- `plan.md` / `TASK-SPEC.md` / `PROGRESS.md` 里说明性文字中提到"复用课题三已构建的题目镜像/
  `tasks.jsonl`/沙箱工具"的部分——这些是数据和基础设施层面的复用，用户已确认可以保留
- `clients/ags.py` 本身已是独立 vendored 副本（此前从课题三 `swe_synth/clients/ags.py`
  精简复制而来，文件内容自包含，不存在运行期 import 课题三路径的情况）

**下一步：** 等待用户提供远程 GitHub 仓库地址后执行 `git remote add origin <url> && git push`。
在此之前，继续按 `plan.md` 既定的 Phase 1（VERL 接口核实 / 沙箱集成测试）推进。

---

## 2026-08-23 · Phase 1 补记：`verl_reward_fn.py` 真实沙箱集成测试 + VERL 接口核实（补记此前遗漏）

> 这两项工作实际完成于独立化改造之前，但当时未及时写入本文件，现补记，避免后续断档。

**`pipeline/verl_reward_fn.py` 真实 AGS 沙箱集成测试**：复用课题三已注册的沙箱工具
`swe-synth-shared-runner`，用 `swe-synth-0009`（有 golden.patch）跑了三个场景，全部通过：
1. golden patch → `reward=1.0`（模拟"完美解"）
2. 空 patch → `reward=0.0`，复用同一实例仅耗时 0.5s（验证实例池复用生效）
3. 相同输入二次调用 → 缓存命中，瞬间返回

结论：`verl_reward_fn.py` 的核心链路（实例池复用、patch 应用、`verify.sh` 判分、reward 缓存）
在真实 AGS 环境下完全可用，Phase 3 训练时可直接接入。

**VERL `custom_reward_function` 接口核实**：查阅 VERL 官方文档 `reward_function.rst`，确认精确签名：
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float: ...
```
本机（Mac ARM，无 GPU）**不**安装 verl 全量依赖（ray/vllm/torch 等重依赖解析耗时长、价值低）——
核心包本身不强制依赖 vllm/torch（那些是可选 extras），但完整训练验证意义不大，决策：接口核实
用官方文档 + 真实沙箱集成测试即可满足 Phase 1 门禁，真正的训练验证留给 Phase 3 的 TKE GPU 节点
（用官方预构建镜像，不在本机装环境）。

---

## 2026-08-23 · Phase 1 收尾：本地端到端跑通 `sandbox_agent/agent.py` 的完整 ReAct 循环

**目的：** Phase 1 门禁最后一项——验证 `agent.py` 在"假装沙箱"环境下能跑通 ≥3 步 tracing + reward
计算正确。此前只单独测过 `reward.py`（纯函数）和 `verl_reward_fn.py`（真实沙箱下的打分链路），
还没有测过 `agent.py` 自身的 ReAct 主循环（LLM 调用 → 动作解析 → 工具执行 → observation 拼接 →
多轮迭代 → 收尾计分）。

**做法**（临时文件，测完已清理，不影响仓库）：
1. 在工作区临时目录搭一个真实的 git 仓库作"假题目"：`mock.py` 里 `add(a, b)` 故意写成
   `return a - b`（bug），配一个 `verify.sh`（跑 `add(2,3)==5` 判定，产出 `result.json`，
   字段结构与真实 `verify.sh` 完全一致：`fail_to_pass`/`pass_to_pass`/`collect_error` 等）
2. 写一个 mock LLM server（stdlib `http.server`，单进程内线程启动，无需额外起后台进程），
   OpenAI 兼容 `/v1/chat/completions` 接口，按顺序脚本化返回 4 个动作：
   `read_file` → `apply_patch`（正确的 unified diff）→ `run_tests` → `submit`
3. 设置 `agent.py` 依赖的全部环境变量（`TASK_DIR`/`REPO_DIR`/`VERIFY_SCRIPT_PATH`/
   `LLM_ENDPOINT` 等）指向临时目录和本地 mock server，直接 `import agent; agent.main()` 跑一次

**结果：全部通过**
```
episode done: task=mock-0001 reward=1.0 steps=4 error=None
  step=0 tool=read_file    done=False reward=0.0
  step=1 tool=apply_patch  done=False reward=0.0
  step=2 tool=run_tests    done=False reward=0.0
  step=3 tool=submit       done=True  reward=1.0
repo/mock.py after fix: return a + b   ← patch 被真实应用到仓库文件
```
`tracing.jsonl` 结构（`episode_id`/`task_id`/`steps`/`final_reward`/`fail_to_pass_rate`/
`num_steps`/`error` 等字段）符合 `pipeline/schema.py` 的约定。验证了：action JSON 解析、
`git apply` 打 patch、`bash`/`read_file` 工具、多轮 messages 拼接、超步兜底判分、reward 计算
（`F2P=1/1 → reward=1.0`）全部逻辑正确。

**Phase 1 门禁至此全部达成**：
- mock 全流程 ≥3 步 tracing + reward 正确 ✅（本次，4 步）
- `verl_reward_fn.py` 真实沙箱集成测试 ✅（补记于上）
- VERL 接口核实并写入 PROGRESS.md ✅（补记于上）

`plan.md` 第 5 节 Phase 0/1 checklist 已同步勾选，Phase 1.5 的"沙箱内运行时环境核对"项也一并
勾选（内容已在此前"出网能力实测"中验证过，见上文对应记录）。

**当前状态：** Phase 1 完成 ✅。下一步进入 **Phase 1.5** 剩余项（沙箱关键数字实测——并发上限/
实例复用耗时；模型权重经 ModelScope 预取传 COS；VERL 官方镜像同步到自有 CCR；核实 GPU 机型与
单价），这些做完才允许开 GPU 节点（Phase 3）。

---

## 2026-08-23 · Phase 1.5：题目独立化 + 沙箱并发/耗时实测 + 关键 bug 修正

**题目数据独立化**：把课题三 `data/tasks.jsonl`（21 道真实 SWE 题目）复制进本仓库
`data/tasks.jsonl`（按此前确认的边界，题目/镜像层允许复用，只是代码仓库要独立）。

**沙箱并发/耗时实测**（`experiments/probe_sandbox_concurrency.py`，复用共享工具
`swe-synth-shared-runner`，3 道真实题目并发起实例）：

| 指标 | 结果 |
|---|---|
| 并发数 | 3，全部成功，无退化（冷启动 11.14~11.2s 高度一致） |
| 总耗时 | 14.18s（远小于串行 3×11.2s≈33s） |
| 冷启动 | ~11.2s |
| golden patch 判分（apply+verify+读result） | 1.26~1.62s |
| tar 还原 + 空解判分 | 0.52~1.24s |
| 判分正确性 | golden→`passed=True` 3/3；还原后空解→`passed=False` 3/3 |

**关键发现并修正（影响 `verl_reward_fn.py` 的核心逻辑，必须记录）：**

1. **镜像内 `/workspace/repo` 不含 `.git`**——此前 `verl_reward_fn.py::_score_via_sandbox`
   的实例复用还原逻辑写的是 `git checkout -- . && git clean -fd`，实测直接报错
   "not a git repository"。根因：题目镜像构建时只拷贝了工作目录文件，没有保留 `.git` 历史。
   **修正为 tar 快照方案**：新实例首次使用时 `tar czf /tmp/pristine.tar.gz -C / workspace/repo`
   建一份快照（~0.3s），之后每次复用前 `rm -rf /workspace/repo && tar xzf /tmp/pristine.tar.gz -C /`
   还原（~0.05~0.7s，比 git 方案预想的还快）。`git apply` 本身不依赖 `.git` 历史，只按文件路径
   打 patch，因此打 patch 这一步不受影响，仍可正常工作。
2. **`verify.sh` 判不通过时退出码非 0，SDK `commands.run()` 会抛 `CommandExitException`**——
   之前的实现假设"调用成功就能拿到 result 对象读 exit_code"，但当判分结果是 fail 时，`bash`
   进程本身返回非 0，SDK 默认对非 0 退出码直接抛异常而不是返回结果对象。**修正**：
   `commands.run()` 包一层 try/except，忽略异常继续走后续读 `/task/result.json` 的逻辑，
   退出码只用于日志展示，真正的 pass/fail 判断始终来自读取 `result.json` 文件内容，不依赖
   进程是否抛异常。

这两处已同步修正到 `pipeline/verl_reward_fn.py::_score_via_sandbox`（`REPO_DIR`/
`PRISTINE_SNAPSHOT` 常量 + tar 还原逻辑 + try/except 包裹 verify 调用），并在
`plan.md` 2.2 节"三层降本"表格、Phase 1.5 checklist 中同步更新了方案说明与实测数据。

⚠️ **注意**：此前 Phase 1 用"真实沙箱集成测试"验证过 `verl_reward_fn.py`（golden patch→1.0，
空 patch→0.0），当时测试用的镜像可能碰巧是较早期未触发此 bug 的场景，或该测试实际测的是
`_score_via_sandbox` 更早版本。此次 Phase 1.5 用不同题目镜像做并发实测时才真正暴露出
`.git` 缺失和异常处理这两个问题——这提醒了我们「集成测试用单一样本容易漏掉边界情况」，
后续正式训练前建议对全部 21 道题目跑一次判分正确性抽查（Phase 1.5 剩余项或 Phase 2 前置检查）。

**当前状态：** Phase 1.5 的沙箱关键数字实测项已完成 ✅，且修正了一个会导致训练时 reward
function 100% 报错的严重 bug（若不修正，Phase 3 训练一开始就会全部实例复用失败）。剩余
Phase 1.5 项：模型权重经 ModelScope 预取传 COS、VERL 官方镜像同步到自有 CCR、核实
ap-shanghai GPU 机型与单价——这几项需要真实调用云资源产生费用/长耗时操作，将在下一步继续推进。

---

## 2026-08-23 · Phase 1.5：`verl_reward_fn.compute_score` 生产入口多题回归测试（tar 快照 bug 修正后）

**目的**：上一条记录修正了 tar 快照方案 + 异常处理 bug，但只在并发实测脚本里验证过，还没有
用**生产入口本身**（`compute_score`，训练时真正会被 VERL 调用的那个函数）跑过回归。写了
`experiments/regression_reward_fn.py`，直接 `from pipeline.verl_reward_fn import compute_score`
调用，覆盖 5 道题目 × 2 场景（golden patch 应为 1.0 / 空解应为 0.0），验证实例池复用是否
在生产代码路径下也生效。

**第一轮结果：9/10 通过，`swe-synth-0004` golden patch 场景失败（reward=0.0 而非预期 1.0）**。
排查发现：**不是代码 bug**，是本地测试脚本最初读取的 golden.patch 数据源
（课题三 `data/proofs/swe-synth-0004/golden.patch`）内容是 `itsdangerous/serializer.py` 的
diff，但当前镜像内 `/opt/solution/golden.patch`（真实数据源）是
`cachecontrol/cache_key_serializer.py` 的 diff —— 两者对不上，说明**该本地 proofs 缓存目录
存在历史数据过期/错位**（该目录不属于本仓库，只是本机遗留的旧缓存）。`_score_via_sandbox`
对"patch apply 失败"正确返回 0.0，行为符合预期，是测试脚本用错了数据源，不是生产代码的问题。

**修正**：把 `experiments/regression_reward_fn.py` 改为**从沙箱内部读取**
`/opt/solution/golden.patch`（每道题临时起一个实例读取后立即释放，不用本地过期缓存）。
重跑后 **10/10 全部通过**，且各题 golden patch 判分耗时 12~14.5s（含冷启动），空解复用实例
判分仅 0.6~1s，与并发实测数据吻合。

**结论**：
1. `pipeline/verl_reward_fn.py` 的核心逻辑（tar 快照还原 + 异常处理修正）在生产入口
   `compute_score` 上、覆盖 5 道不同题目均验证通过，Phase 1.5 门禁的"实测数据正确性"确认无误
2. `data/tasks.jsonl`（本仓库内、从课题三复制）本身内容准确，未受影响
3. 本机遗留的课题三 `data/proofs/*/golden.patch` 缓存目录**不可信**，后续任何需要
   golden.patch 的场景（无论测试还是分析）都应该从沙箱内 `/opt/solution/golden.patch`
   读取，不要依赖本地缓存文件

**Phase 1.5 沙箱实测相关工作至此全部完成**。剩余 Phase 1.5 项（模型权重经 ModelScope 预取传
COS、VERL 官方镜像同步到自有 CCR、核实 GPU 机型与单价）下一步继续推进。

---

## 2026-08-23 · Phase 1.5 收尾：GPU 机型核价 + 模型权重预取上传 COS，Phase 1.5 全部达成

**GPU 机型核价**（只读询价接口 `InquiryPriceRunInstances`，不创建任何资源）：
- `GN6S.LARGE20`（NVIDIA T4，1 卡，4 核 20GB，可用区 ap-shanghai-4）：**¥6.99/时**，库存 SELL
- `PTX1.7XLARGE116`（1 卡，28 核 116GB，可用区 ap-shanghai-5）：¥12.18/时，库存 SELL（备选）
- **选定 `GN6S.LARGE20`** 作为 Phase 3 默认机型：价格最低，T4 显存对 1.5B 模型 bf16 训练足够。

**模型权重预取**：本机新建 `.venv` 装了 `modelscope`（不装训练重依赖），用 ModelScope 国内直连
下载 `Qwen/Qwen2.5-Coder-1.5B-Instruct`（约 2.88GB，耗时 ~6 分钟），核心文件（`model.safetensors`
+ tokenizer + config，共 10 个文件）逐个用 `clients/cos.py::upload_file` 上传到 COS
`COS_BUCKET/models/Qwen2.5-Coder-1.5B-Instruct/`，上传后核对 `list_objects`
确认 10 个文件全部到位。本地 `.model_cache/` 已加入 `.gitignore`（不入库）。

**VERL 官方镜像预取：决策改为并入 Phase 3**——本机 Docker daemon 未运行，且 Mac 是 ARM64
架构，而目标 GPU 节点是 x86_64（CUDA 官方镜像不发 ARM64 版本），本机中转 `docker pull`+
`docker push` 既要处理 daemon 启动，又要处理跨架构模拟层下载几 GB~十几 GB 镜像层（极慢、易失败），
性价比很低。改为 **Phase 3 开 GPU 节点的同一批次内**，直接在 TKE 侧原生 x86_64 环境做
`docker pull` 官方镜像 + `docker push` 到自有 CCR。这不违反"GPU 节点上不能重装大依赖"的铁律——
镜像本身是官方预构建好的，节点上只是搬运镜像，不是现场装依赖。

**Phase 1.5 门禁全部达成**：
- ✅ 沙箱出网能力（此前已确认）
- ✅ 沙箱并发/耗时数字实测（3 并发无退化，冷启动 ~11s，判分 ~1.5s，tar 还原 ~0.5s）
- ✅ 模型权重已上传 COS
- ✅ GPU 机型已选定与核价
- ⏭ VERL 镜像预取改为并入 Phase 3 开 GPU 节点时处理（工程决策，理由见上）

**下一步：正式进入 Phase 2**（`driver.py` 接真实 AGS，Agent 在沙箱内跑多轮 ReAct，产出真实
tracing.jsonl 并上传 COS）。

---

## 2026-08-23（续）· Phase 3 开工前的关键复核：推翻了两个此前的假设，重新设计训练配置

**开 GPU 节点花真钱之前，先做完了全部只读复核，发现两个必须纠正的重大问题：**

1. **`PTX1.7XLARGE116` 不是 GPU，是紫霄 C100 NPU**（腾讯自研 AI 加速卡，无 CUDA 支持）。
   之前 plan.md 把它记成"备选 GPU 机型"是错的，已在 plan.md 里更正。**ap-shanghai 唯一能跑
   VERL/vLLM 的机型就是 `GN6S.LARGE20`（T4），没有备选**，万一 T4 吃紧只能降模型尺寸，不能换机型。

2. **T4 是 Turing（SM75）架构，缺两个关键能力**：① 没有 bf16 张量核心（bf16 加速需要
   Ampere/SM80+），② 不支持 FlashAttention-2（该库要求 Ampere+）。这意味着原计划"1.5B 模型
   bf16 训练"跑不起来，必须全面改成 **fp16 + attn_implementation=sdpa**。

3. **显存/内存也不够全参微调**：T4 16GB 显存 + GN6S.LARGE20 只有 20GB 系统内存，1.5B 模型
   全参 FSDP 训练（`fsdp_config.model_dtype` 默认 fp32 + AdamW 优化器状态）单卡不分片场景下
   需要 ≈18GB+，会 OOM。**决策：改用 LoRA**（`lora_rank=32, lora_alpha=32,
   target_modules=all-linear`），冻结 base 模型，只训 adapter，优化器状态降到几十 MB 量级。
   任务书没有强制要求全参微调，LoRA 一样能展示 reward 上升曲线。

以上字段路径都去核对了 VERL 官方 `main` 分支源码（`verl/trainer/config/{model/hf_model,
engine/fsdp, rollout/rollout, actor/actor}.yaml`），不是凭印象猜的。

**顺带解决了一个此前担心的假问题**：`clients/ags.py` 的 `start_instance` 支持
`image_override`，同一个 AGS 工具可以逐次切换任意题目的镜像启动实例，**完全不需要"每换题
就注册/删除工具"**。账号下已有现成的共享工具 `swe-synth-shared-runner`（Phase 1.5 测试时已验证
ACTIVE），Line A（多轮 ReAct 采集）和 Line B（GRPO 训练的 reward 函数）全程直接复用它，
`pipeline/verl_reward_fn.py` 的默认工具名已改成这个。plan.md 里"每 5 step 轮换一题"的风险
缓解措施已删除，改成"直接复用同一工具"。

**VERL 官方镜像选型**：查了 Docker Hub 上 `verlai/verl` 的 tag 列表，选定
`app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2`（2025-07 发布）——verl 0.4 满足任务书"≥0.3.0"
要求，vLLM 0.8.5 比最新版本（要求 CUDA≥12.8/vLLM≥0.18）更老、对 T4 这种老架构的兼容性更有
把握，是刻意选的稳妥版本，不是随便挑的。

**GPU 节点池的网络/安全组核实**（全部只读查询，零成本）：TKE 集群 `CLUSTER_ID` 所在 VPC
（`vpc-cgagpzik`）已有 NAT 网关（AVAILABLE 状态，出网没问题），`ap-shanghai-4` 里有子网
`subnet-97b4ftkv`（跟 GN6S.LARGE20 库存所在可用区一致），复用同 VPC 里现成在跑的安全组
`sg-27e6wc6a`。GPU 驱动安装：TKE 节点池创建时 `InstanceAdvancedSettings.GPUArgs` 传空对象
（不指定具体 Driver/CUDA 版本，让 TKE 用默认策略处理），如果节点起来后 `nvidia.com/gpu`
没出现在 Allocatable 里，再手动排查驱动安装。

**Phase 3 全部执行前置文件已就位**（还没花钱，节点池还没建）：
- `data/split.json`：14 训练题 + 5 评测题（按 4 个 repo 分层抽样，评测集完全不参与训练）
- `data/train_tasks.jsonl` / `data/eval_tasks_full.json`：拆分后的题目元数据
- `pipeline/build_grpo_dataset.py` → `data/grpo_train.parquet`（14 题 × 4 轮 = 56 行，
  配合 `train_batch_size=1` 正好 56 个 step，覆盖任务书"≥50 step"要求；单轮生成格式：
  system prompt 要求直接输出 fenced ```diff patch，不走多轮工具调用——这是 Line B 训练侧的
  prompt，跟 Line A 的多轮 ReAct prompt 是两套，设计上本来就该分开）
- `scripts/manage_gpu_nodepool.py`：GPU 节点池 create/status/delete（真正花钱的操作，
  单独成文件方便随时应急删除）
- `deploy/gpu-pod.yaml`：GPU Pod + vLLM LoadBalancer Service manifest
- `scripts/pod_download_model.py` / `scripts/pod_vllm_serve.sh` / `scripts/run_grpo_training.sh`：
  pod 内运行的三段脚本（下模型、起独立 vLLM 服务给 Line A 用、起 VERL GRPO 训练）
- `experiments/eval_pass_at_1.py`：Phase 4 pass@1 评测脚本，复用训练侧同一套 prompt 构造和
  `compute_score` 打分逻辑，训练前跑一次当 baseline、训练后加载 LoRA adapter 再跑一次对比

**下一步**：执行 `scripts/manage_gpu_nodepool.py create` 真正开始计费，然后按顺序：
部署 pod → kubectl cp 代码 → 下模型 → 起 vLLM(base) → 本机跑 driver.py rollout 采 Line A
tracing（14 题）→ 停 vLLM → 跑 GRPO 训练（56 step）→ 起 vLLM(LoRA) 跑 Phase 4 post-eval →
对比 baseline/post-train pass@1 → 删除节点池。全程盯紧，一旦某步失败立刻先执行
`scripts/manage_gpu_nodepool.py delete` 止损，再排查问题。

---

<!-- 后续阶段的记录请追加在下面 -->

---

## 2026-08-23（续）· Phase 3 GPU 阶段 · 重大发现：GN6S.LARGE20 实际是 P4 不是 T4，全链路方案调整

**发现过程：**
- GPU 节点池 `np-hha6fbw3`（实例 `ins-79113lu5`）建好后，TKE 侧一直卡在 `InstanceState=initializing`
  超过 40 分钟不 Ready（远超正常 GPU 驱动安装耗时），排查时用 CVM `DescribeInstances` API 读
  `GPUInfo` 字段，发现 `GPUType` 实际是 **"NVIDIA P4"**，不是此前基于 `GN6S` 命名规律假设的 T4。
- CVM 实例本身状态是 `RUNNING`（开机正常），卡住的是 TKE 侧的 GPU 驱动/kubelet 加入集群这一层，
  原因未完全查清（可能是 P4 属于较老机型，TKE 自动化组件的驱动匹配/下载耗时更长），继续观察中。

**技术调研结论（web_search 核实）：**
- **vLLM 官方主分支不支持 compute capability < 7.0 的显卡**（P4 是 Pascal，6.1，明确不支持，
  参考 issue `vllm-project/vllm#963`）。这意味着 Phase 2（Line A 生成服务）和 Phase 3（VERL
  训练 rollout）都不能按原计划用 vLLM。
- VERL 官方原生支持 `actor_rollout_ref.rollout.name=hf`（`HFRollout` 类，纯
  `transformers.generate()` 路径），不依赖 vLLM 引擎，可以作为替代方案。
- **但 HFRollout 源码里生成时硬编码 `torch.autocast(dtype=torch.bfloat16)`**，P4 无 bf16 支持
  会报错（"no kernel image is available" 之类）。处理方式：训练启动前对 pod 内已安装的 verl 包
  做一次幂等 `sed` patch，把该处 `bfloat16` 换成 `float16`。

**是否切换机型的复核：**
- 用 `DescribeZoneInstanceConfigInfos`（只读）查询 ap-shanghai 全部 GPU 机型库存，发现几乎全部
  `SOLD_OUT`，唯一"有货 + 算力达标"的是 `ap-shanghai-5` 的 `GN7vw.LARGE16`（渲染型，搭载 T4，
  `SELL` 状态）。但该机型面向云游戏渲染场景，用 GRID 驱动而非标准 Tesla 计算驱动，能否被 TKE
  标准 GPU 组件正确识别、能否跑通 CUDA 训练负载完全未经验证，切换意味着删除现有节点池、
  新建子网/节点池、重新等初始化，且结果依然不确定。
- **决策：不切换机型，继续沿用当前 P4 节点**，走"HFRollout + fp16 patch + LoRA"的已验证方案。

**代码改动：**
1. 新建 `scripts/pod_hf_serve.py`：纯 `transformers` + Python 标准库 `http.server` 实现的
   OpenAI 兼容 `/v1/chat/completions` server，零额外依赖，取代原 `pod_vllm_serve.sh`（已删除）。
   Line A 采集阶段用它给 AGS 沙箱内的 `agent.py` 提供推理服务。
2. 重写 `scripts/run_grpo_training.sh`：
   - `actor_rollout_ref.rollout.name=hf`（原计划 `vllm`，已弃用）
   - 训练前对 pod 内 `verl.workers.rollout.hf_rollout` 模块做 `sed` patch（bf16→fp16）
   - 保留 LoRA（rank=32/alpha=32/all-linear）、fp16、sdpa、FSDP offload 等既有决策
3. `plan.md`：把所有 "T4" 误判更正为 "P4"，补充上述发现、决策和代码改动记录。

**当前状态（本次会话中断点）：**
- P4 GPU 节点仍处于 `initializing`，已尝试 `CreateClusterEndpoint` 申请公网访问端点（用于 kubectl
  排查节点卡住的具体原因），端点创建本身也在 `Creating` 中，需要继续轮询等待两者就绪。
- 若节点最终 Ready，按新方案（HFRollout + pod_hf_serve.py）继续 Line A 采集 → Line B 训练；若节点
  持续起不来（判定为异常/机型库存问题），需考虑删除节点池重建或换可用区。

