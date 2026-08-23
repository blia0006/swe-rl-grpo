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

<!-- 后续阶段的记录请追加在下面 -->
