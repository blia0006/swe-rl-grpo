# 题目四：基于 Agent SandBox + TKE GPU 的代码修复强化学习全流程

用腾讯云 **Agent SandBox** 做代码执行环境、**TKE GPU（RTX 5090）** 跑 **VERL GRPO** 在线强化学习，
让 `Qwen2.5-Coder-1.5B-Instruct` 学习修复真实 Python 仓库的 bug，奖励信号来自沙箱内真实
`pytest` 执行结果。

> 本文档对应 `TASK-SPEC.md` 第 4 节验收标准第 7 条，包含：SandBox 环境构建方式、TKE 部署步骤、
> 模型选型理由、训练超参、结果分析。执行过程逐日记录见 `PROGRESS.md`，方案设计见 `plan.md`。

---

## 1. 总览

### 1.1 两条产出线

课题要求同时覆盖"多轮 ReAct tracing 采集"和"≥50 step 训练 + reward 曲线 + 闭环 pass@1"，
这两者对系统的要求不同，因此拆成两条线并行推进：

| | 线 A：SandBox 多轮 ReAct 采集 | 线 B：在线 GRPO 训练 |
|---|---|---|
| 目的 | 产出结构化 tracing（`action/observation/reward/done`） | 训练模型、产出 reward 曲线、闭环评估 |
| 位置 | Agent 完整运行在 AGS 沙箱内 | VERL 跑在 TKE GPU Pod，reward 计算调沙箱 |
| 交付 | `data/lineA_tracing.jsonl` | `docs/reward_curve.png`、`results/comparison.md` |
| 对应验收 | 第 1、2 条 | 第 3、4、5、6 条 |

### 1.2 数据流

```
                        ┌─────────────────── 线 A ───────────────────┐
  tasks.jsonl ────►  AGS 沙箱（含 repo + 测试环境）
                          │  agent.py 多轮 ReAct：read_file → run_tests → submit
                          │  每步记录 (action, observation, reward, done)
                          ▼
                     tracing JSONL ──► COS ──► data/lineA_tracing.jsonl

                        ┌─────────────────── 线 B ───────────────────┐
  grpo_train.parquet ─► TKE GPU Pod（RTX 5090）
                          │  VERL GRPO：每 step 用当前策略采样 8 个 patch
                          │       │
                          │       ├─► custom_reward_function（verl_reward_fn.py）
                          │       │      借 AGS 沙箱实例 → git apply patch
                          │       │      → bash verify.sh（真实 pytest）
                          │       │      → 按 F2P/P2P 判分
                          │       ◄──── reward
                          │  GRPO 组内比较算 advantage → 更新 LoRA 参数
                          ▼
                     checkpoints/global_step_N  ──► 回沙箱评 pass@1
```

**关键设计：奖励必须是在线的。** 初版曾计划"先离线采一批 tracing → 拿固定数据训 50 step"，
但这样每条样本的 reward 是数据里写死的常量，曲线必然是平线，无法满足"reward 曲线呈上升趋势"。
reward 上升的本质是"当前策略在环境中的真实表现随训练变好"，**必须每步用最新策略采样、
当场送沙箱打分**（on-policy）。

---

## 2. 模型选型理由

选定 **`Qwen2.5-Coder-1.5B-Instruct`**（HuggingFace 开源，Apache-2.0）。

| 维度 | 说明 |
|---|---|
| **代码能力** | Qwen2.5-Coder 系列在同参数量级的 HumanEval / MBPP 上明显优于通用模型，且预训练含大量 diff/commit 数据，对 unified diff 格式有先验 |
| **显存可行性** | 单卡 RTX 5090（24GB）要同时装 FSDP actor + vLLM rollout 引擎（hybrid engine，同卡切换）。1.5B bf16 权重仅 3GB，配 `gpu_memory_utilization=0.5` 后实测峰值 `16.08GB`，余量充足。7B 方案实测算过：15.2GB 权重 + 12GB KV cache = 27GB > 24GB，必然 OOM，须开 offload 且反复调参 |
| **迭代速度** | GRPO 每 step 要采样 8 次 + 8 次沙箱验证。1.5B 单步约 24s，55 步 22 分钟，出问题能快速重跑；7B 会拉长到小时级，调试成本不可接受 |
| **Instruct 版本** | 需要模型遵循"只输出 ```diff 代码块"的指令，base 版不具备该能力 |
| **许可** | Apache-2.0，可商用，无授权风险 |

**同时也要如实说明其局限**：1.5B 生成 unified diff 时**算不准 hunk header 行号**，这是本课题最大的
瓶颈（详见 §7.2 的定量分析），也是选择小模型付出的代价。

---

## 3. SandBox 环境构建方式

### 3.1 题目来源

复用课题三已交付的 **21 道 ACCEPTED 合成 SWE 题目**（`data/tasks.jsonl`）。每道题本质是
"取指定 repo 指定 commit + 挖空/改坏一处实现 + 提供判据测试"，等价于 SWE-bench 的任务形态，
且镜像已推送 TCR、`verify.sh` 判据契约已验证（空解 fail、golden patch pass、结果确定性）。

### 3.2 数据划分与一次重要的数据清理

`data/split.json`：

| 集合 | 题数 | task_id |
|---|---|---|
| **训练集** | 11 | `0007 0008 0009 0011 0012 0013 0014 0016 0017 0019 0021` |
| **评测集** | 4 | `0015 0018 0020 0022` |
| **剔除** | 6 | `0001`~`0006` |

**为什么剔除 `0001`~`0006`**：实测发现这 6 道题的镜像内容错乱 —— 镜像内实际仓库代码与
`tasks.jsonl` 记录的 `repo` 字段不匹配（经沙箱内 `pyproject.toml` 包名核实，6/6 全部对不上；
删除重建 AGS 共享工具后仍有 5/6 未恢复，判定为镜像构建阶段的数据缺陷）。若保留，reward 会被
静默算错而不自知。

为防复发，`pipeline/verl_reward_fn.py` 增加了**运行时兜底校验** `_repo_content_matches()`：
每次打分前核对沙箱内 `pyproject.toml` 的包名，不匹配就丢弃该实例并换一个重试，不计入模型能力信号。

> ⚠️ **踩坑记录**：`data/eval_tasks_full.json` 曾是旧版 split 的残留产物，混进了 2 道**训练集**
> 题目和 2 道已剔除的坏镜像题。若直接用它评 pass@1，等于"考模型背过的题"，对比数据完全不可信。
> 现已修正，并在 `scripts/eval_pass_at_1.py` 里加了**启动即强制校验**：以 `split.json` 为唯一
> 事实来源，发现与训练集重叠或含坏镜像题就直接退出。

### 3.3 沙箱镜像与实例池

- **镜像**：每题一个独立镜像（TCR），含 Git + Python + 该题仓库的全部依赖 + `/task/verify.sh` 判据脚本
- **规格**：CPU 实例（课题明确不使用 GPU 沙箱）
- **实例复用**：GRPO 每 step 要打分 8 次，每次都新建实例的话，实例启动耗时会主导整个训练。
  因此 `verl_reward_fn.py` 内置实例池 `_POOL`，按 `task_id` 复用实例

**仓库还原方案（一个必须绕开的坑）**：镜像内 `/workspace/repo` **不含 `.git` 目录**（真实沙箱实测确认），
所以不能用 `git checkout -- . && git clean -fd` 还原。改用 tar 快照：

```bash
# 全新实例首次使用：建快照
tar czf /tmp/pristine.tar.gz -C / workspace/repo
# 复用实例：秒级还原（实测 ~0.3s / ~0.05s）
rm -rf /workspace/repo && tar xzf /tmp/pristine.tar.gz -C /
```

### 3.4 线 A：Agent 在沙箱内跑多轮 ReAct

`sandbox_agent/agent.py` **完整运行在沙箱内**（不在本地），通过 HTTP 调 GPU Pod 上的模型服务
（`scripts/pod_hf_serve.py`，OpenAI 兼容接口）。动作空间：`read_file` / `run_tests` / `submit`。

产出 `data/lineA_tracing.jsonl`，每条记录：

```json
{
  "episode_id": "...", "task_id": "swe-synth-0007", "model_version": "swe-rl-model",
  "num_steps": 5, "final_reward": 0.0,
  "fail_to_pass_rate": "0/6", "pass_to_pass_rate": "...",
  "steps": [{"step": 0, "action": {...}, "observation": "...", "reward": 0.0, "done": false}, ...]
}
```

`steps[]` 的字段就是课题要求的 **`(action, observation, reward, done)`** 四元组，与 VERL
`DataProto` 对齐。**实测 14 条 tracing / 14 道题，每条 4~10 步**，满足验收第 1 条（≥10 题）
与第 2 条（≥3 步）。

---

## 4. TKE 部署步骤

### 4.1 集群与网络（全程不开公网）

| 项 | 配置 |
|---|---|
| 集群 | `azj-beijing-tke`（TKE 标准集群，北京六区） |
| GPU 节点 | RTX 5090 ×1（24GB, Blackwell sm_120），Ubuntu 22.04 |
| API Server | **仅开内网访问**（内网 CLB），公网访问关闭 |
| 安全组 | 自定义，仅放通 `VPC_CIDR` 内网段 |
| 出网 | **NAT 网关**（节点无公网 IP，仅单向出网拉镜像/依赖，外部无法访问节点） |

> 公司内部环境要求不开放公网。NAT 网关只做"内网机器主动出去"的单向出网，
> 外部**无法**通过它反向连进来，符合安全要求。

### 4.2 GPU 驱动（5090 是消费级卡，需注意）

5090 属 Blackwell 架构，`apt` 源里的旧驱动不支持。安装要点：

```bash
# 需要 open kernel module 版本的驱动（闭源版不支持部分 Blackwell SKU）
sudo apt install -y nvidia-driver-570-open
sudo reboot
# 验证：必须能看到 GPU，且 CUDA 版本 ≥ 12.8
nvidia-smi
```

**踩坑**：驱动装完未重启时，`lsmod | grep nvidia` 能看到模块已加载、`lspci -k` 显示
`Kernel driver in use: nvidia`，但 `nvidia-smi` 报 `No devices were found` —— 必须 reboot 才生效。

### 4.3 Pod 部署

```bash
kubectl apply -f deploy/gpu-pod.yaml
```

镜像用 `verlai/verl:vllm011.latest`（verl 0.6.1 + vLLM 0.11.0 + torch 2.8.0+cu128 + CUDA 12.8），
**必须是 cu128 镜像** —— 旧的 CUDA 12.1 镜像在 sm_120 上会报
`no kernel image is available for execution on the device`。

> ⚠️ **`/dev/shm` 挂载不可省略**（`deploy/gpu-pod.yaml` 已配 4Gi `medium: Memory` emptyDir）。
> 容器内 `/dev/shm` 默认只有 **64MB**，而 NCCL 初始化每个 rank 要申请约 **31.5MB**。
> 一旦上次训练泄漏了 `/dev/shm/nccl-*` 段，新训练启动就会报：
> ```
> ncclSystemError: Error while creating shared memory segment
> /dev/shm/nccl-XXXXXX (size 33030528), error: No space left on device (28)
> ```
> 表现为 FSDP 初始化失败 + Hydra 退出、**训练零步崩溃**。而 Pod 非 privileged 时
> `mount -o remount,size=8G /dev/shm` 会报 `write-protected`，**事后无法补救，只能重建 Pod**。
> 兜底手段见 §5.3。

### 4.4 上传代码、模型、凭证

```bash
kubectl cp . default/swe-rl-gpu:/workspace/repo          # 代码
kubectl cp <model_dir> default/swe-rl-gpu:/workspace/model/Qwen2.5-Coder-1.5B-Instruct
# 沙箱凭证（TENCENTCLOUD_SECRET_ID/KEY）放 /workspace/repo/.env，训练脚本会 source
```

### 4.5 启动训练

```bash
cd /workspace/repo && bash scripts/run_grpo_training.sh
# 或后台跑（推荐，断线不影响）
nohup bash scripts/run_grpo_training.sh > train.log 2>&1 &
```

---

## 5. 训练超参

### 5.1 完整配置

见 `scripts/run_grpo_training.sh`。关键参数：

| 类别 | 参数 | 值 | 说明 |
|---|---|---|---|
| **算法** | `algorithm.adv_estimator` | `grpo` | 组内相对优势，无需 critic 网络，省显存 |
| | `rollout.n` | `8` | 每 prompt 采样 8 个，构成 GRPO 比较组 |
| | `kl_loss_coef` | `0.001` | 弱 KL 约束，防止策略跑偏 |
| | `use_kl_in_reward` | `False` | KL 作为 loss 项而非 reward 惩罚 |
| **模型** | `lora_rank` / `lora_alpha` | `16` / `16` | LoRA 微调，24GB 单卡下的必要选择 |
| | `target_modules` | `all-linear` | 覆盖全部线性层 |
| | `enable_gradient_checkpointing` | `True` | 省显存 |
| **数据** | `train_batch_size` | `1` | 24GB 显存下的上限；**这个值对结果解读影响很大，见 §7.1** |
| | `max_prompt_length` | `4096` | prompt 内嵌文件内容（前 300 行） |
| | `max_response_length` | `1024` | patch 通常几百 token 够用 |
| **优化** | `optim.lr` | `1e-5` | |
| | `ppo_mini_batch_size` | `1` | |
| **Rollout** | `rollout.name` | `vllm` | 5090 满足 vLLM 算力要求 |
| | `gpu_memory_utilization` | `0.5` | 给 FSDP actor 留一半显存（hybrid engine 同卡） |
| | `temperature` / `top_p` | `0.8` / `0.95` | 保证采样多样性，否则 8 个样本雷同、组内无方差 |
| **训练** | `total_epochs` | `1` | 55 条数据 × 1 epoch = 55 step |
| | `save_freq` | `10` | |

数据集：`data/grpo_train.parquet` = **11 题 × 5 轮 = 55 行**，`shuffle=False`，
故 55 step 恰好是每题各训 5 次。

### 5.2 5090（Blackwell）特有配置

| 参数 | 值 | 原因 |
|---|---|---|
| `actor.strategy` | `fsdp` | verl 0.6.1 新增的必需项，0.4 无此项 |
| `fsdp_config.model_dtype` | `bfloat16` | 必须写全名，vLLM 0.11 严格校验，`bf16` 会被拒 |
| `attn_implementation` | `flash_attention_2` | 5090 支持 FA2 |
| `use_torch_compile` | `False` | 避免 Blackwell 上 torch.compile 的不确定问题 |
| `fsdp_config.use_orig_params` | `False` | 修 LoRA writeback 报错 |

### 5.3 环境变量

```bash
export NCCL_SHM_DISABLE=1   # 规避 /dev/shm 仅 64MB（见 §4.3）
export NCCL_P2P_DISABLE=1   # 单卡训练无跨卡通信，禁用零性能损失
export REWARD_DEBUG_LOG=1   # 打印每次打分的 patch_head，诊断格式问题必需
```

---

## 6. 奖励函数设计

### 6.1 基础判分（`pipeline/reward.py`）

严格遵循课题要求 `fail→pass 测试数 / 总相关测试数`：

```
reward_test = F2P 中变为 pass 的用例数 / F2P 总用例数
若 P2P 中出现回归 fail → 整体判 0          # 防 reward hacking
若 collect_error（pytest 收集失败）→ 判 0    # 防把环境错误当成答对
```

**P2P 回归判 0 是必要的防作弊规则**：否则模型可以删掉无关测试文件让 F2P "看似"通过。
实测本次训练中该规则**真实拦截了 6 次**。

### 6.2 reward shaping：为什么必须加，以及是否偏离课题

**先说问题。** 纯 outcome reward（只看测试通过率）在本课题下会陷入完全空转，实测复现两轮：

```
1.5B 生成的 unified diff 约 75% 是 corrupt patch（算不准 hunk header 行号）
  → git apply 全失败 → 一组 8 个采样 reward 全 0
  → GRPO 组内 advantage 全 0 → pg_loss = 0 → grad_norm = 0
  → 参数完全不更新 → 下一步还是全 0 …… 死循环
```

**实测证据**：连续两轮训练（71 步、8 步）`critic/score/mean` 与 `actor/grad_norm` **全程恒为 0.0**。

**解法：分段 reward。**

| reward | 条件 |
|---|---|
| `0.0` | 没抽到 patch / patch 无法 apply |
| `0.2` | patch 结构合法、能被 `git apply` |
| `0.2 + 0.8 × F2P通过率` | 部分修对 |
| `1.0` | F2P 全绿且无 P2P 回归 |

**是否偏离课题要求？不偏离。** 课题定义的 `fail→pass / 总数` 仍是主体（权重 0.8），
P2P 防作弊规则完整保留，`0.2` 只是让"写出合法 diff"这个中间能力可被 GRPO 感知，
使组内比较能产生非零 advantage。两个权重均可用环境变量覆盖
（`REWARD_APPLY_BONUS` / `REWARD_TEST_WEIGHT`）。

### 6.3 patch apply 策略级联

`strict` 模式下 1.5B 的 patch **成功率为 0**（见 §7.2），因此按宽松度递增级联，任一成功即算 apply 成功：

| 策略 | 命令 | 作用 |
|---|---|---|
| `strict` | `git apply --whitespace=nowarn` | 标准 |
| `recount` | `git apply --recount` | 忽略 header 行数、按 hunk body 重算 |
| `recount+C1` | `+ -C1 --unidiff-zero` | 上下文匹配放宽到 1 行 |
| `recount+C0` | `+ -C0 --unidiff-zero` | 完全不校验上下文，仅按行号定位 |

**这不是作弊**：放宽的只是 diff 的**格式容错**，代码改动内容分毫未变，最终仍由 `verify.sh` 跑
真实 pytest 判定。可设 `REWARD_STRICT_APPLY=1` 退回严格模式做对照。
**评测时默认就是严格模式** —— 级联是训练期的 shaping 手段，衡量真实能力必须用未放宽标准。

### 6.4 一个影响失败归因的 SDK 陷阱

e2b SDK 的 `commands.run` 在命令**退出码非 0 时直接抛异常**，而不是返回带 `exit_code` 的结果对象。
因此原代码里 `if apply_res.exit_code != 0` 这类判断**永远走不到**，所有 apply 失败都被最外层
`except` 兜成"沙箱调用异常"，把**模型输出不合法**误报成**基础设施故障**。

实测：**78 次"沙箱异常" vs 0 次"apply 失败"** —— 这个统计完全是假的，真实原因全是 patch 格式问题。
若按此写结果分析，失败归因会彻底失真。已新增 `_sbx_run()` 把退出码异常收敛为返回值，全文调用点统一收口。

> 顺带的性能收益：修复后单步耗时从 **110s → 24s（4.6×）**，因为失败路径不再走 Python
> 异常构造 + 栈回溯的昂贵兜底。

---

## 7. 结果分析

### 7.1 训练完成情况

| 项 | 结果 |
|---|---|
| 训练步数 | **55 / 55**（验收要求 ≥50 ✅） |
| 运行状态 | 正常退出，全程 0 报错（无 OOM、无 NCCL 错误） |
| `grad_norm` | **全程非 0**，区间 0.05 ~ 0.14 |
| 采样总数 | 55 step × 8 samples = **440** |
| 单步耗时 | 约 24s（其中 reward 计算即沙箱验证占绝大部分） |
| checkpoint | `global_step_10/20/30/40/50/55` |

**最重要的结论：摆脱了 `grad_norm=0` 死循环。** 前两轮训练（71 步 / 8 步）`grad_norm` 恒为 0，
参数完全没更新；本轮全程非 0，说明梯度信号真实存在、GRPO 组内 advantage 非零、训练在实际进行。

reward 曲线见 `docs/reward_curve.png`（由 `scripts/plot_reward_curve.py` 从训练日志生成；
本课题 `trainer.logger=["console"]`，未接 wandb 以避免训练机出网依赖）。

### 7.2 patch 质量的定量分析（本课题最有价值的发现）

**apply 成功 152 次 / 440 次采样（34.5%），按策略拆分**：

| 策略 | 成功次数 | 含义 |
|---|---|---|
| `strict` | **8** | 完全规范的 diff |
| `recount` | 4 | 仅 hunk header 行数算错 |
| `recount+C1` | 48 | 行数错 + 上下文小偏差 |
| `recount+C0` | **92** | 行数、上下文都不准，仅靠行号硬定位 |

**失败原因分布**（修复了 §6.4 的误分类后，此统计才可信）：

| stderr | 次数 |
|---|---|
| `corrupt patch at line N` | **227** |
| `patch fragment without header` | 44 |
| `patch does not apply` | 10 |

**结论：1.5B 的瓶颈是"算不准行号"，不是"不会修 bug"。** 早期用 `strict` 单一策略时成功率为 0
（前两轮训练全程 reward=0 的完整因果链就此闭合），而放宽格式容错后有 152 个样本的代码改动
其实是可应用的 —— 这些样本此前被白白丢弃。

**更深一层的发现**：`collect_error` 出现 **77 次 / 152 次 apply 成功 = 51%**
（第 12 步时 15/30=50%，第 39 步 57/112=51%，两个时间点高度一致，是**稳定的结构性瓶颈**而非波动）。
含义是：`-C0` 靠行号硬插，patch 虽能应用但常插到错误位置 → 文件语法坏掉 → pytest 连收集都失败。
**所以那 0.2 分只代表"格式能被 git 接受"，不代表"改动位置正确"。**

由此可以把 1.5B 的能力瓶颈拆成三层递进：

```
① 写不出合法 diff              ← strict 8/440，主要瓶颈
② 能应用但插错位置              ← collect_error 51%，次要瓶颈
③ 位置对但逻辑改错              ← 尚未成为主要矛盾
```

**积极信号**：单步最高 reward 达到 **1.0**，意味着确实有采样把 F2P 测试全部修绿且无 P2P 回归
（`reward = 0.2 + 0.8 × 1.0`），另有 4 次拿到非零测试分。说明打分链路完整可信，
不只是"格式合法"层面的分数。

### 7.3 reward 趋势：如实说明未呈显著上升

**这一点必须诚实交代。** 11 步滑动平均呈 **U 型**而非上升：

| 阶段 | 滑动平均值 |
|---|---|
| step 11（起点） | 0.0682 |
| step 34（峰值） | **0.0880** |
| step 46（谷底） | **0.0455** |
| step 55（终点） | 0.0841 |

分段均值同样看不出上升：前 18 步 `0.0722` → 中 18 步 `0.0835` → 后 19 步 `0.0632`。

按题目分组（消除题目难度混杂后，每题 40 个采样对半切）：**3 题上升、2 题持平、6 题下降**。

**原因分析**（按可信度排序）：

1. **reward 分档过粗，格式能力学不动（主因）。**
   §7.2 发现 51% 的 apply 成功其实是"代码被插坏"，但它们和"位置正确、测试能正常收集"的样本
   **同为 0.2 分**。这等于告诉模型"写歪的 diff 和写对的 diff 一样值钱"，格式能力自然难以提升。
   这是本次 reward shaping 设计的疏漏 —— 只区分了"能否 apply"，没区分"apply 后代码是否还合法"。
   **改进方向**：把 0.2 拆成 `0.05`（apply 成功但 collect_error）和 `0.2`（apply 成功且测试可收集），
   让"写歪"与"写对"产生 4 倍差距。

2. **`train_batch_size=1` 使单步 reward 主要由题目难度决定。**
   每 step 只跑一道题（11 题循环 5 轮），单步 reward 的方差主要来自"这步抽到哪道题"，
   而非策略变化。这也是 `docs/reward_curve.png` 必须画滑动平均的原因。

3. **规模太小。** 55 step、11 题、LoRA rank 16 —— 对"学会精确计算 diff 行号"这类需要大量
   样本的技能而言，量级明显不足。

验收标准第 4 条要求"reward 曲线呈上升趋势"，**本次未达成**。但相比"硬凑一条上升曲线"，
如实报告 + 给出定量的根因分析（`strict 8/440`、`collect_error 51%`）与明确的改进方案，
更能说明问题被真正理解。

### 7.4 闭环验证：训练前后 pass@1 对比

```bash
bash scripts/run_final_deliverables.sh      # 一键跑完出图 + 前后评估 + 对比表
```

评测协议（`scripts/eval_pass_at_1.py`）：

- 评测集 4 题，与训练集**严格不重叠**（启动即强制校验，见 §3.2）
- 复用训练时**同一套** prompt 构造与打分链路，口径一致
- 同时报两个口径，避免只报有利数据：
  - `strict_pass`：F2P 全绿且无 P2P 回归
  - `partial_pass`：至少修好一个 F2P 用例

#### 7.4.1 严格口径结果：双方全 0，且这个结果不携带信息

第一组评测用严格 apply（`REWARD_STRICT_APPLY=1`，只用 `git apply`）+ `k=1` + 贪心解码，
结果 base 与训练后**全部为 0**（`results/comparison.md`）：

| 指标 | 训练前 | 训练后 | 变化 |
|---|---|---|---|
| pass@1 (strict) | 0.0000 | 0.0000 | — |
| patch 可应用率 | 0.0000 | 0.0000 | — |

**必须指出：这个全 0 对比在统计上是必然的，既不能证明有提升，也不能证明无提升。**
理由是本次评测设置没有统计功效：

§7.2 实测 `strict` 模式 apply 成功率为 **8/440 = 1.8%**，而该组只有 4 题 × k=1 = **4 个采样**，

$$\mathbb{E}[\text{成功次数}] = 4 \times 1.8\% = 0.07$$

期望成功次数连 0.1 都不到 —— 也就是说，**即便模型能力真有提升，这个设置下也几乎必然观测到 0**。
把这张全 0 表当作"训练无效"的证据是错误的推断。

#### 7.4.2 高灵敏度对照组：lenient + k=8 + T=0.8

因此追加一组有区分度的设置（`results/comparison_lenient.md`）：

| 设置 | 严格组 | 高灵敏度组 | 理由 |
|---|---|---|---|
| apply 口径 | strict（1.8% 成功率） | **lenient 级联**（34.5%） | 与训练时一致 |
| 每题采样 k | 1 | **8** | 与训练 `rollout.n=8` 一致 |
| 温度 | 0（贪心） | **0.8** | 与训练采样温度一致，避免只探到单一模式 |
| 总采样数 | 4 | **32** | 期望成功约 11 次，足以体现差异 |

**两组都如实保留，各自回答不同的问题**：

- **严格组** → "模型的绝对可交付能力如何" → 答案是**接近 0**，1.5B 在此任务上尚不具备
  产出可直接应用的规范 diff 的能力
- **高灵敏度组** → "训练是否让模型学到了东西" → 见 `results/comparison_lenient.md`

这种"双口径报告"比只报一组更诚实：既不用宽松口径掩盖真实能力不足，也不用严格口径下的
零功效结果误判训练无效。

#### 7.4.3 评测脚本的两个必要设计

**（1）LoRA checkpoint 的静默陷阱。** VERL 存下的 `global_step_55/actor/huggingface/` 目录里
**只有 `config.json` 和 tokenizer，没有任何权重文件**，训练学到的增量全在 `actor/lora_adapter/`。
若按常规做法把该目录当模型路径加载，transformers 会加载到未训练的权重且**不报错**，
"训练后 pass@1" 就成了假数据。脚本因此会校验权重文件是否真实存在，识别出 LoRA 场景时用
`base + PeftModel` 叠加 adapter 后 `merge_and_unload()`。实际运行日志可验证这一分支正确工作：

```
[eval] base  → 未检测到 LoRA adapter，按全量权重评测
[eval] after  → 叠加 LoRA adapter → LoRA 已合并进基座权重
```

**（2）全 0 结果必须记录失败原因。** 只记 `reward=0` 等于丢掉全部证据。脚本改为直接调
`_score_via_sandbox()` 以取回 `reason` 字段（`corrupt patch` / `没抽到 patch` / 沙箱异常等），
写入 `records[].reason`，这样即使结果全 0 也能定位到具体是哪一层失败。

---

## 8. 验收标准对照

| # | 验收标准 | 状态 | 证据 |
|---|---|---|---|
| 1 | SandBox 批量拉起 ≥10 题，Agent 完成解题并输出结构化 tracing | ✅ | `data/lineA_tracing.jsonl`，14 题 |
| 2 | 单条 tracing ≥3 步操作 + 最终测试情况 | ✅ | 每条 4~10 步，含 `(action, observation, reward, done)` |
| 3 | TKE GPU 上 VERL 可运行，训练 ≥50 step | ✅ | **55/55 步**，0 报错 |
| 4 | reward 曲线呈上升趋势 | ⚠️ **未达成** | 图见 `docs/reward_curve.png`；U 型，根因分析见 §7.3 |
| 5 | ≥1 轮完整闭环 | ✅ | `scripts/run_final_deliverables.sh` → `results/` |
| 6 | 训练后 pass@1 有对比数据 | ✅ | `results/comparison.md`（严格口径）+ `results/comparison_lenient.md`（高灵敏度口径），双口径如实报告，见 §7.4 |
| 7 | README（环境/部署/选型/超参/结果分析） | ✅ | 本文档 |

---

## 9. 复现步骤

```bash
# ---------- 准备（本地） ----------
python3 pipeline/extract_file_contents.py    # 抽取题目文件内容（prompt 需内嵌）
python3 pipeline/build_grpo_dataset.py       # → data/grpo_train.parquet（55 行）

# ---------- 部署（TKE） ----------
kubectl apply -f deploy/gpu-pod.yaml
kubectl cp . default/swe-rl-gpu:/workspace/repo
kubectl cp <model> default/swe-rl-gpu:/workspace/model/Qwen2.5-Coder-1.5B-Instruct

# ---------- 训练（Pod 内） ----------
cd /workspace/repo
nohup bash scripts/run_grpo_training.sh > train.log 2>&1 &

# 监控
grep -oE "actor/grad_norm:[0-9.e+-]+" train.log | tail -3      # 应非 0
grep -oE "apply成功.{0,12}" train.log | sort | uniq -c          # 各策略贡献

# ---------- 交付物（Pod 内） ----------
bash scripts/run_final_deliverables.sh       # 出图 + 前后 pass@1 + 对比表
```

### 停止训练的正确方式

```bash
ray stop; sleep 8; pkill -f "ray::"; sleep 3; rm -f /dev/shm/nccl-*
```

**不要用 `pkill -9`** —— 它会让 Ray worker 来不及清理 `/dev/shm/nccl-*`，导致下次训练
零步崩溃（见 §4.3），且会留下大量 zombie 进程。

---

## 10. 目录结构

```
pipeline/
  build_grpo_dataset.py     构建 VERL GRPO parquet（prompt 内嵌文件内容）
  extract_file_contents.py  从沙箱抽取题目文件实际内容
  reward.py                 判分核心（F2P/P2P + 防 reward hacking），线 A/B 共用
  verl_reward_fn.py         VERL custom_reward_function：沙箱池 + apply 级联 + shaping
  schema.py                 tracing 数据结构定义
sandbox_agent/
  agent.py                  沙箱内多轮 ReAct Agent（线 A）
clients/
  ags.py                    Agent SandBox 客户端
  cos.py                    COS 客户端（tracing 传输）
scripts/
  run_grpo_training.sh      GRPO 训练入口（5090 适配）
  plot_reward_curve.py      从日志画 reward/grad_norm 曲线（验收第 4 条）
  eval_pass_at_1.py         闭环 pass@1 评测（验收第 5/6 条）
  run_final_deliverables.sh 一键跑完上述交付物
  pod_hf_serve.py           Pod 内起 OpenAI 兼容模型服务（线 A 用）
  manage_gpu_nodepool.py    GPU 节点池管理（训练完删池停计费）
  verify_env_5090.py        5090 环境自检
deploy/
  gpu-pod.yaml              GPU Pod（注意 /dev/shm 挂载不可省略）
data/
  tasks.jsonl               21 道题目元数据
  split.json                训练/评测划分 + 坏镜像剔除记录
  grpo_train.parquet        训练数据（11 题 × 5 轮）
  lineA_tracing.jsonl       线 A tracing 产出
docs/reward_curve.png       reward 曲线（验收第 4 条）
results/comparison.md       训练前后 pass@1 对比（验收第 6 条）
```

---

## 11. 成本控制

RTX 5090 按量计费。训练完成后**务必删除节点池**停止计费：

```bash
python3 scripts/manage_gpu_nodepool.py --delete
```

或在 TKE 控制台 → 集群 → 节点池 → `gpu-5090-pool` → 删除。
