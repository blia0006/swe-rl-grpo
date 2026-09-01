#!/usr/bin/env bash
# Phase 3：GPU Pod 内运行的 VERL GRPO 训练脚本 ——【5090 适配版】。
#
# 环境：RTX 5090 (Blackwell sm_120, 24GB) + verl 0.6.1 + vllm 0.11.0
#       + torch 2.8.0+cu128 + transformers 4.57.1 + numpy 1.26.4 + CUDA 13.0
#
# 相比 P4/V100 版本的全部改动（5090 是全新架构，老卡 hack 全部删除）：
#   1. 删除所有 sed/python patch（fp16 降级、flash_attn try/except、sdpa 降级、
#      triton CE 禁用、position_ids 移除、empty_cache 等）—— 这些是 P4/V100 老卡
#      兼容性补丁，5090 原生支持 bf16 + FlashAttention-2 + Triton，全部不需要。
#   2. rollout.name 改回 vllm（5090 满足 vLLM 最低算力要求，且 vllm 0.11 支持 sm_120）
#   3. attn_implementation 用 flash_attention_2（5090 支持 FA2，性能最好）
#   4. dtype 全程 bf16（5090 原生支持 bf16 张量核心）
#   5. actor.strategy=fsdp（verl 0.6.1 新增的必需项，0.4 没有）
#   6. use_torch_compile=False（避免 Blackwell 上 torch.compile 的玄学问题）
#
# 用法（pod 内，repo 代码已上传到 /workspace/repo）：
#   cd /workspace/repo && bash scripts/run_grpo_training.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# --- 加载 .env（TENCENTCLOUD_SECRET_ID/KEY 等，reward function 调 AGS 沙箱需要）。
#     Ray 单机模式下会继承 driver 进程的 env，所以必须在 shell 层 export。
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "已加载 .env（TENCENTCLOUD_SECRET_ID=${TENCENTCLOUD_SECRET_ID:0:4}...）"
else
  echo "警告：未找到 .env，沙箱相关凭据可能缺失"
fi

MODEL_PATH="${MODEL_PATH:-/workspace/model/Qwen2.5-Coder-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$(pwd)/data/grpo_train.parquet}"
REWARD_FN="$(pwd)/pipeline/verl_reward_fn.py"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/swe-rl-grpo}"
mkdir -p "$OUTPUT_DIR"

# NCCL 关闭共享内存 / P2P 通道。
# 【为什么必须加】容器内 /dev/shm 只有 64MB（K8s 默认），而 NCCL 初始化时每个
# rank 要申请约 31.5MB 的共享内存段。一旦上次训练残留了未回收的 shm 段
# （`pkill -9` 强杀 Ray worker 就会泄漏 /dev/shm/nccl-* ），新训练启动就会报：
#   ncclSystemError: Error while creating shared memory segment
#   /dev/shm/nccl-XXXXXX (size 33030528), error: No space left on device (28)
# → FSDP 的 _sync_module_params_and_buffers 失败 → Hydra 直接退出，训练零步启动失败。
# 而 Pod 非 privileged 时 `mount -o remount,size=8G /dev/shm` 会报 write-protected，
# 无法扩容，因此改从 NCCL 侧规避。
# 【为什么无性能损失】本课题是单卡训练（world_size=1 / n_gpus_per_node=1），
# 不存在任何跨卡通信，shm 与 P2P 通道本就用不到，禁用只是省掉无用的初始化。
# 【彻底修法】在 Pod YAML 里挂 medium=Memory 的 emptyDir 到 /dev/shm（见 deploy/gpu-pod.yaml）。
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

# reward function 调试日志：打印每次打分的 patch_head（诊断 patch 格式问题用）
export REWARD_DEBUG_LOG=1

# --- 5090 不需要任何源码 patch，直接跑 verl 0.6.1 的标准 GRPO 训练。
#     注意几个 5090/verl0.6.1 特有的配置项：
#       - actor.strategy=fsdp          （0.6.1 新增必需项）
#       - use_torch_compile=False      （Blackwell 上编译玄学，关闭求稳）
#       - attn=flash_attention_2       （5090 支持 FA2，比 sdpa 快）
#       - rollout.name=vllm            （5090 支持 vLLM；若报 sm_120 内核错误改回 hf）
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.train_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.return_raw_chat=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  custom_reward_function.path="${REWARD_FN}" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  trainer.test_freq=-1 \
  trainer.save_freq=10 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl \
  trainer.experiment_name=qwen2.5-coder-1.5b-grpo-5090 \
  2>&1 | tee "${OUTPUT_DIR}/../train.log"
