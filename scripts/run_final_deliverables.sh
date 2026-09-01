#!/usr/bin/env bash
# 一键补齐 TASK-SPEC 验收标准第 4/5/6 条：reward 曲线图 + 训练前后 pass@1 闭环对比。
#
# 设计目标：**一条命令跑完，中途所有已知阻碍自动处理**，无需人工介入。
# 已内置的自愈项（每一项都是实际踩过的坑）：
#   1. 缺 matplotlib / peft   → 自动 pip install（失败也不中断，绘图退化为纯文本统计）
#   2. Ray/训练进程残留占显存 → 评测前自动优雅清理（ray stop 而非 pkill -9，
#                                避免再次泄漏 /dev/shm/nccl-*）
#   3. /dev/shm 残留段         → 自动清理
#   4. LoRA checkpoint 识别    → eval 脚本自动叠加 base + adapter（见其 _resolve_model_dir）
#   5. 评测集数据泄漏          → eval 脚本启动即强制校验，发现即退出
#   6. 单步失败不影响整体      → 各阶段独立执行并记录状态，最后统一汇总
#
# 用法（GPU Pod 内）：
#     bash scripts/run_final_deliverables.sh
#     bash scripts/run_final_deliverables.sh train_final_55steps.log
#
# 产出：
#     docs/reward_curve.png / .csv     reward + grad_norm 曲线（验收第 4 条）
#     results/eval_before.json         base 模型 pass@1（验收第 5/6 条）
#     results/eval_after.json          训练后 pass@1
#     results/comparison.md            markdown 对比表格，可直接贴进 README

set -uo pipefail   # 故意不用 -e：单个阶段失败要继续跑后面的，最后汇总

cd "$(dirname "$0")/.." || exit 1
REPO_ROOT="$(pwd)"

TRAIN_LOG="${1:-train_final_55steps.log}"
BASE_MODEL="${BASE_MODEL_DIR:-/workspace/model/Qwen2.5-Coder-1.5B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/checkpoints/swe-rl-grpo}"
CKPT="${CKPT:-}"

mkdir -p docs results

# 加载 .env：评测要调 AGS 沙箱执行 patch，需要 TENCENTCLOUD_SECRET_ID/KEY。
# run_grpo_training.sh 里是 `set -a; source .env` 在 shell 层 export 的，
# 而评测脚本此前未加载 .env，导致 _POOL.acquire() 阶段抛
# "AGSError: 未配置 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY"。
# 现在 eval_pass_at_1.py 内部也会自行加载，这里是双保险。
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*"; }

STATUS_PLOT="未执行"
STATUS_BEFORE="未执行"
STATUS_AFTER="未执行"
STATUS_CMP="未执行"
STATUS_BEFORE_L="未执行"
STATUS_AFTER_L="未执行"
STATUS_CMP_L="未执行"

# ---------------------------------------------------------------- 0. 环境自愈
log "阶段 0/5：环境准备"

python3 -c "import matplotlib" 2>/dev/null || {
  warn "matplotlib 缺失，尝试安装"
  pip install -q matplotlib 2>&1 | tail -3
}
python3 -c "import matplotlib; print('matplotlib', matplotlib.__version__)" 2>/dev/null \
  && ok "matplotlib 可用" || warn "matplotlib 仍不可用，绘图将跳过（统计摘要仍会输出）"

python3 -c "import peft" 2>/dev/null || {
  warn "peft 缺失（加载 LoRA adapter 必需），尝试安装"
  pip install -q peft 2>&1 | tail -3
}
python3 -c "import peft; print('peft', peft.__version__)" 2>/dev/null \
  && ok "peft 可用" || err "peft 不可用，LoRA checkpoint 无法评测"

# 清理训练残留：必须先优雅 ray stop，再 pkill（不带 -9），否则会泄漏 shm
if pgrep -f "ray::|raylet|main_ppo" >/dev/null 2>&1; then
  warn "检测到 Ray/训练进程残留，清理中（评测需要独占显存）"
  ray stop >/dev/null 2>&1
  sleep 8
  pkill -f "ray::"   2>/dev/null
  pkill -f raylet    2>/dev/null
  pkill -f main_ppo  2>/dev/null
  sleep 3
fi
rm -f /dev/shm/nccl-* 2>/dev/null
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo 'N/A')"
ok "显存占用：${GPU_USED}；/dev/shm：$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $3" used"}')"

# 沙箱凭证前置检查：缺了的话两个评测阶段必然失败，提前告知比跑到一半才报错好
if [[ -n "${TENCENTCLOUD_SECRET_ID:-}" && -n "${TENCENTCLOUD_SECRET_KEY:-}" ]]; then
  ok "沙箱凭证已就绪（SECRET_ID=${TENCENTCLOUD_SECRET_ID:0:4}...）"
else
  err "缺少 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY —— 评测阶段会失败"
  warn "请确认 $(pwd)/.env 存在且含这两个变量（reward 计算需调 AGS 沙箱跑 pytest）"
  if [[ -f .env ]]; then
    warn ".env 存在，但未解析出凭证。文件中的 key 列表如下（值已隐藏）："
    grep -oE '^[A-Za-z_][A-Za-z0-9_]*' .env 2>/dev/null | sed 's/^/    /'
  else
    warn ".env 文件不存在"
  fi
fi

# ---------------------------------------------------------------- 1. reward 曲线
log "阶段 1/5：生成 reward 曲线（验收第 4 条）"
if [[ -f "$TRAIN_LOG" ]]; then
  if python3 scripts/plot_reward_curve.py "$TRAIN_LOG" -o docs/reward_curve.png; then
    STATUS_PLOT="成功"
    ok "曲线图：docs/reward_curve.png"
  else
    STATUS_PLOT="失败"
    err "绘图脚本返回非 0"
  fi
else
  STATUS_PLOT="跳过（日志不存在）"
  err "找不到训练日志：$TRAIN_LOG"
  ls -la ./*.log 2>/dev/null | head
fi

# ---------------------------------------------------------------- 2. 定位 checkpoint
log "阶段 2/5：定位训练后 checkpoint"
if [[ -z "$CKPT" ]]; then
  # 取 global_step 编号最大的那个（而非按时间，避免 mtime 被 cp 等操作干扰）
  CKPT="$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
          | sed 's/.*global_step_//' | sort -n | tail -1 \
          | sed "s|^|$CKPT_ROOT/global_step_|")"
fi
if [[ -n "$CKPT" && -d "$CKPT" ]]; then
  ok "checkpoint：$CKPT"
  [[ -d "$CKPT/actor/lora_adapter" ]] \
    && ok "检测到 LoRA adapter，评测将叠加 base + adapter" \
    || warn "未见 lora_adapter，按全量权重处理"
else
  err "找不到 checkpoint，跳过评测阶段"
  ls -la "$CKPT_ROOT" 2>/dev/null | head
fi

# ---------------------------------------------------------------- 3. 评 base
log "阶段 3/5：评测 base 模型（训练前 pass@1）"
if [[ -d "$BASE_MODEL" ]]; then
  if python3 scripts/eval_pass_at_1.py \
        --model "$BASE_MODEL" --tag before --out results/eval_before.json; then
    STATUS_BEFORE="成功"
  else
    STATUS_BEFORE="失败"
  fi
else
  STATUS_BEFORE="跳过（基座模型目录不存在）"
  err "基座模型不存在：$BASE_MODEL"
fi

# ---------------------------------------------------------------- 4. 评训练后
log "阶段 4/5：评测训练后模型（GRPO 55 步）"
if [[ -n "$CKPT" && -d "$CKPT" ]]; then
  if python3 scripts/eval_pass_at_1.py \
        --model "$CKPT" --base-model "$BASE_MODEL" \
        --tag after --out results/eval_after.json; then
    STATUS_AFTER="成功"
  else
    STATUS_AFTER="失败"
  fi
else
  STATUS_AFTER="跳过（无 checkpoint）"
fi

# ---------------------------------------------------------------- 5. 对比
log "阶段 5/7：生成训练前后对比表格 · 严格口径（验收第 6 条）"
if [[ -f results/eval_before.json && -f results/eval_after.json ]]; then
  if python3 scripts/eval_pass_at_1.py --compare \
        results/eval_before.json results/eval_after.json \
        | tee results/comparison.md; then
    STATUS_CMP="成功"
    ok "对比表格：results/comparison.md"
  else
    STATUS_CMP="失败"
  fi
else
  STATUS_CMP="跳过（缺少评测结果）"
  warn "需要 results/eval_before.json 与 results/eval_after.json 都存在"
fi

# ---------------------------------------------------------------- 6-7. 高灵敏度对照组
# 【为什么必须补这一组】严格口径 + k=1 在本课题下没有统计功效：
# 训练日志实测 strict apply 成功率仅 8/440 = 1.8%，评测 4 题 × k=1 = 4 个采样，
# 期望成功次数 = 4 × 1.8% = 0.07 —— 期望值连 0.1 都不到，**双方全 0 是统计必然**，
# 既不能证明有提升也不能证明无提升，该对比表实际上不携带任何信息。
# 因此追加一组高灵敏度设置：
#   --lenient  用与训练一致的 apply 级联口径（实测成功率 34.5%）
#   -k 8       每题采样 8 次，与训练时 rollout.n=8 一致
#   温度 0.8   与训练采样温度一致，避免贪心解码只探到单一模式
# 4 题 × 8 次 = 32 采样，期望成功约 11 次，才足以体现训练前后差异。
# 两组结果都如实保留：严格组反映"真实可交付能力"，宽松组反映"训练是否学到东西"。
if [[ "${SKIP_SENSITIVE:-0}" != "1" ]]; then
  log "阶段 6/7：高灵敏度评测 · base（lenient + k=8 + T=0.8）"
  if [[ -d "$BASE_MODEL" ]]; then
    python3 scripts/eval_pass_at_1.py --model "$BASE_MODEL" \
      --tag before_lenient --lenient -k 8 --temperature 0.8 \
      --out results/eval_before_lenient.json \
      && STATUS_BEFORE_L="成功" || STATUS_BEFORE_L="失败"
  else
    STATUS_BEFORE_L="跳过（基座模型不存在）"
  fi

  log "阶段 7/7：高灵敏度评测 · 训练后（lenient + k=8 + T=0.8）"
  if [[ -n "$CKPT" && -d "$CKPT" ]]; then
    python3 scripts/eval_pass_at_1.py --model "$CKPT" --base-model "$BASE_MODEL" \
      --tag after_lenient --lenient -k 8 --temperature 0.8 \
      --out results/eval_after_lenient.json \
      && STATUS_AFTER_L="成功" || STATUS_AFTER_L="失败"
  else
    STATUS_AFTER_L="跳过（无 checkpoint）"
  fi

  if [[ -f results/eval_before_lenient.json && -f results/eval_after_lenient.json ]]; then
    log "生成高灵敏度对比表格"
    python3 scripts/eval_pass_at_1.py --compare \
      results/eval_before_lenient.json results/eval_after_lenient.json \
      | tee results/comparison_lenient.md \
      && { STATUS_CMP_L="成功"; ok "对比表格：results/comparison_lenient.md"; } \
      || STATUS_CMP_L="失败"
  else
    STATUS_CMP_L="跳过（缺少高灵敏度评测结果）"
  fi
else
  STATUS_BEFORE_L="跳过（SKIP_SENSITIVE=1）"
  STATUS_AFTER_L="$STATUS_BEFORE_L"
  STATUS_CMP_L="$STATUS_BEFORE_L"
fi

# ---------------------------------------------------------------- 汇总
log "全部阶段汇总"
printf '  %-34s %s\n' "reward 曲线（第4条）"            "$STATUS_PLOT"
printf '  %-34s %s\n' "base pass@1 · strict（第5条）"    "$STATUS_BEFORE"
printf '  %-34s %s\n' "训练后 pass@1 · strict（第5条）"  "$STATUS_AFTER"
printf '  %-34s %s\n' "前后对比 · strict（第6条）"       "$STATUS_CMP"
printf '  %-34s %s\n' "base pass@1 · lenient k=8"       "$STATUS_BEFORE_L"
printf '  %-34s %s\n' "训练后 pass@1 · lenient k=8"     "$STATUS_AFTER_L"
printf '  %-34s %s\n' "前后对比 · lenient k=8"          "$STATUS_CMP_L"
echo
echo "产出文件："
ls -la docs/reward_curve.png docs/reward_curve.csv \
       results/eval_before.json results/eval_after.json \
       results/comparison.md 2>/dev/null | awk '{print "  "$0}'
echo
echo "若有阶段失败，把上面完整输出发回即可定位（各阶段互不影响，已跑完的产出仍有效）。"
