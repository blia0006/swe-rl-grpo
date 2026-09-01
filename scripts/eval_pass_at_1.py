#!/usr/bin/env python3
"""闭环验证：训练前后模型在 SandBox 中跑 SWE 题目，对比 pass@1。

对应 TASK-SPEC.md 验收标准：
  - 第 5 条：至少完成 1 轮完整闭环（SandBox 产出 tracing → TKE 训练 → 新模型回 SandBox 评估）
  - 第 6 条：训练后 pass@1 相比训练前有可观测提升，需有对比数据

【评测协议】
  · 评测集：`data/split.json` 的 `eval_task_ids`，与训练集**严格不重叠**
    （脚本启动时会强制校验，发现泄漏直接退出——避免"考背过的题"导致数据无效）
  · 对每道题采样 k 次（默认 k=1，即 pass@1），模型输出 unified diff patch
  · 用与训练**完全相同**的打分链路 `verl_reward_fn.compute_score` 送沙箱执行、
    跑 verify.sh、按 F2P/P2P 判分，保证训练与评测口径一致
  · pass@1 判定口径（两个都报，避免只报对自己有利的那个）：
      strict_pass  —— F2P 全绿且无 P2P 回归（即 compute_reward 的 passed=True）
      partial_pass —— 至少修好一个 F2P 用例（测试分 > 0）

【为什么用 REWARD_STRICT_APPLY=1】
评测时默认开启严格 apply（只用 `git apply --whitespace=nowarn`，不走训练时的
recount/-C0 级联容错）。因为级联容错是**训练期的 reward shaping 手段**（让格式
瑕疵不至于把梯度信号打成全 0），而评测要回答的是"模型真实能力如何"，必须用
未放宽的标准。可加 `--lenient` 关掉，用与训练一致的宽松口径再报一组做对照。

用法（在 GPU Pod 内跑，需要能访问沙箱的凭证）：
    # 1) 评 base 模型（训练前）
    python3 scripts/eval_pass_at_1.py \
        --model /workspace/model/Qwen2.5-Coder-1.5B-Instruct \
        --tag before --out results/eval_before.json

    # 2) 评训练后的 checkpoint
    python3 scripts/eval_pass_at_1.py \
        --model /workspace/checkpoints/swe-rl-grpo/global_step_55 \
        --tag after --out results/eval_after.json

    # 3) 生成对比表格
    python3 scripts/eval_pass_at_1.py --compare results/eval_before.json results/eval_after.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 训练时使用的基座模型。LoRA checkpoint 评测时需要它作为底座叠加 adapter。
DEFAULT_BASE_MODEL = os.environ.get(
    "BASE_MODEL_DIR", "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"
)


# --------------------------------------------------------------- 评测集加载与校验

def load_eval_tasks() -> list[dict]:
    """加载评测题目，并强制校验与训练集无重叠、不含已知坏镜像题。

    历史教训：`data/eval_tasks_full.json` 曾是旧版 split 的残留产物，里面混进了
    2 道训练集题目和 2 道已剔除的坏镜像题，若直接拿来评 pass@1，对比数据完全
    不可信。因此这里不信任该文件，一律以 `split.json` 为准重新取。
    """
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    eval_ids: list[str] = list(split["eval_task_ids"])
    train_ids = set(split["train_task_ids"])
    bad_ids = set(split.get("excluded_bad_task_ids") or [])

    leaked = [t for t in eval_ids if t in train_ids]
    if leaked:
        raise SystemExit(f"❌ 评测集与训练集重叠，评测无效：{leaked}")
    bad = [t for t in eval_ids if t in bad_ids]
    if bad:
        raise SystemExit(f"❌ 评测集含已知坏镜像题：{bad}")

    all_tasks: dict[str, dict] = {}
    with open(ROOT / "data" / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            all_tasks[d["task_id"]] = d

    missing = [t for t in eval_ids if t not in all_tasks]
    if missing:
        raise SystemExit(f"❌ 评测题在 tasks.jsonl 中缺失：{missing}")

    return [all_tasks[t] for t in eval_ids]


# --------------------------------------------------------------- 生成

class HFGenerator:
    """用 transformers 直接加载模型生成 patch。

    评测量很小（4 题 × k 次），不值得为此启动 vLLM 引擎（初始化 + CUDA graph
    capture 的固定开销比生成本身还大），直接用 transformers 最省事也最稳。
    """

    def __init__(self, model_path: str, max_new_tokens: int = 1024,
                 temperature: float = 0.0, base_model: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        weights_dir, lora_dir = self._resolve_model_dir(model_path, base_model)

        print(f"[eval] 加载基座权重：{weights_dir}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()

        if lora_dir:
            # 【为什么必须这样加载】本课题用 LoRA 训练（lora_rank=16），VERL 的
            # checkpoint 里 `actor/huggingface/` 只有 config.json + tokenizer，
            # **不含任何权重文件**；训练学到的增量全部在 `actor/lora_adapter/`。
            # 若直接把 `actor/huggingface/` 当模型目录加载，transformers 会因缺
            # 权重而随机初始化（或报错），评测到的根本不是训练后的模型，
            # 训练前后对比数据会完全失真——这是必须避免的静默错误。
            try:
                from peft import PeftModel
            except ImportError:
                raise SystemExit(
                    "❌ 需要 peft 才能加载 LoRA adapter：pip install peft"
                ) from None
            print(f"[eval] 叠加 LoRA adapter：{lora_dir}", flush=True)
            self.model = PeftModel.from_pretrained(
                self.model, lora_dir, torch_dtype=torch.bfloat16
            )
            # merge 后推理更快，且避免 PeftModel 包装层影响 generate 行为
            self.model = self.model.merge_and_unload().eval()
            print("[eval] LoRA 已合并进基座权重", flush=True)
        else:
            print("[eval] 未检测到 LoRA adapter，按全量权重评测", flush=True)

        print("[eval] 模型加载完成", flush=True)

    @staticmethod
    def _has_weights(d: Path) -> bool:
        """判断目录内是否真的有模型权重文件（而非只有 config/tokenizer）。"""
        if not d.is_dir():
            return False
        pats = ("*.safetensors", "*.bin", "*.pt", "*.pth")
        return any(any(d.glob(p)) for p in pats)

    @classmethod
    def _resolve_model_dir(cls, model_path: str,
                           base_model: str | None) -> tuple[str, str | None]:
        """解析模型路径，返回 (基座权重目录, LoRA adapter 目录或 None)。

        兼容三种情况：
          1) 普通 HF 目录（有 config.json + 权重文件）→ 直接用
          2) VERL LoRA checkpoint：`global_step_N/actor/` 下有 `lora_adapter/`，
             而 `huggingface/` 只有 config + tokenizer 无权重
             → 基座取 --base-model（或默认的训练基座），adapter 单独返回
          3) VERL 全量 checkpoint：`actor/huggingface/` 含权重 → 直接用
        """
        p = Path(model_path)
        if not p.exists():
            raise SystemExit(f"❌ 路径不存在：{p}")

        # 情况 1：本身就是完整 HF 目录
        if (p / "config.json").exists() and cls._has_weights(p):
            return str(p), None

        # 找 LoRA adapter
        lora_dir: str | None = None
        for cand in (p / "actor" / "lora_adapter", p / "lora_adapter"):
            if (cand / "adapter_config.json").exists() or cls._has_weights(cand):
                lora_dir = str(cand)
                break

        # 找含权重的 HF 目录
        hf_dir: str | None = None
        for cand in (p, p / "actor" / "huggingface", p / "huggingface"):
            if (cand / "config.json").exists() and cls._has_weights(cand):
                hf_dir = str(cand)
                break

        if lora_dir and not hf_dir:
            # LoRA 训练场景：必须显式提供基座
            base = base_model or DEFAULT_BASE_MODEL
            if not (Path(base) / "config.json").exists():
                raise SystemExit(
                    f"❌ 检测到 LoRA adapter（{lora_dir}），但找不到基座模型。\n"
                    f"   请用 --base-model 指定训练时的基座权重目录。\n"
                    f"   已尝试：{base}"
                )
            return base, lora_dir

        if hf_dir:
            return hf_dir, lora_dir

        # 都没找到，把目录结构打出来便于排查
        listing: list[str] = []
        for sub in (p, p / "actor"):
            if sub.is_dir():
                listing.append(f"{sub}: {sorted(x.name for x in sub.iterdir())}")
        raise SystemExit(
            f"❌ 在 {model_path} 下找不到可用的模型权重。\n"
            f"   既无含权重的 HF 目录，也无 LoRA adapter。\n"
            + "\n".join("   " + s for s in listing)
        )

    def generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------- 评测主流程

def build_messages(task: dict, file_contents: dict[str, str]) -> list[dict]:
    """复用训练时**完全相同**的 prompt 构造逻辑，保证评测与训练输入同构。"""
    from pipeline.build_grpo_dataset import SYSTEM_PROMPT, build_user_prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task, file_contents)},
    ]


def evaluate(model_path: str, tag: str, k: int, lenient: bool,
             temperature: float, base_model: str | None = None) -> dict:
    from pipeline.verl_reward_fn import compute_score

    # 严格 apply：评测衡量真实能力，不用训练期的格式容错级联
    if lenient:
        os.environ.pop("REWARD_STRICT_APPLY", None)
    else:
        os.environ["REWARD_STRICT_APPLY"] = "1"

    tasks = load_eval_tasks()
    print(f"[eval] 评测集 {len(tasks)} 题（与训练集无重叠已校验）："
          f"{[t['task_id'] for t in tasks]}", flush=True)

    fc_path = ROOT / "data" / "file_contents.json"
    file_contents: dict[str, str] = {}
    if fc_path.exists():
        file_contents = json.loads(fc_path.read_text(encoding="utf-8"))
    else:
        print(f"⚠️ {fc_path.name} 不存在，prompt 不含文件内容（与训练输入不一致，"
              f"评测会明显偏低）。先跑 pipeline/extract_file_contents.py", flush=True)

    gen = HFGenerator(model_path, temperature=temperature, base_model=base_model)

    records: list[dict] = []
    t_start = time.time()
    for task in tasks:
        tid = task["task_id"]
        ground_truth = json.dumps(
            {"task_id": tid, "image": task["image"], "repo": task.get("repo", "")},
            ensure_ascii=False,
        )
        for attempt in range(k):
            t0 = time.time()
            solution = gen.generate(build_messages(task, file_contents))
            reward = compute_score(
                data_source="swe_rl",
                solution_str=solution,
                ground_truth=ground_truth,
                extra_info={"task_id": tid},
            )
            # reward = APPLY_BONUS + TEST_WEIGHT × 测试分，反解出纯测试分
            from pipeline.verl_reward_fn import APPLY_SUCCESS_BONUS, TEST_WEIGHT
            test_score = max(0.0, (reward - APPLY_SUCCESS_BONUS) / TEST_WEIGHT) \
                if reward > APPLY_SUCCESS_BONUS else 0.0
            rec = {
                "task_id": tid,
                "attempt": attempt,
                "reward": round(float(reward), 4),
                "test_score": round(float(test_score), 4),
                "applied": bool(reward >= APPLY_SUCCESS_BONUS - 1e-9),
                "strict_pass": bool(test_score >= 1.0 - 1e-9),
                "partial_pass": bool(test_score > 0),
                "elapsed_s": round(time.time() - t0, 1),
                "solution_len": len(solution),
            }
            records.append(rec)
            print(f"  {tid} #{attempt}: reward={rec['reward']} "
                  f"test={rec['test_score']} applied={rec['applied']} "
                  f"strict_pass={rec['strict_pass']} ({rec['elapsed_s']}s)", flush=True)

    n_task = len(tasks)
    # pass@1：每题只要有任一次尝试成功即算该题通过（k=1 时就是标准 pass@1）
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)

    strict_solved = sum(1 for rs in by_task.values() if any(r["strict_pass"] for r in rs))
    partial_solved = sum(1 for rs in by_task.values() if any(r["partial_pass"] for r in rs))
    applied_any = sum(1 for rs in by_task.values() if any(r["applied"] for r in rs))

    summary = {
        "tag": tag,
        "model": model_path,
        "n_tasks": n_task,
        "k": k,
        "apply_mode": "lenient(级联容错，同训练)" if lenient else "strict(仅 git apply)",
        "temperature": temperature,
        "pass_at_1_strict": round(strict_solved / n_task, 4) if n_task else 0.0,
        "pass_at_1_partial": round(partial_solved / n_task, 4) if n_task else 0.0,
        "apply_rate": round(applied_any / n_task, 4) if n_task else 0.0,
        "mean_reward": round(sum(r["reward"] for r in records) / len(records), 4)
                       if records else 0.0,
        "mean_test_score": round(sum(r["test_score"] for r in records) / len(records), 4)
                           if records else 0.0,
        "total_elapsed_s": round(time.time() - t_start, 1),
        "records": records,
    }

    print()
    print(f"===== {tag} =====")
    print(f"  模型            : {model_path}")
    print(f"  apply 口径      : {summary['apply_mode']}")
    print(f"  pass@1 (strict) : {summary['pass_at_1_strict']} "
          f"（{strict_solved}/{n_task} 题 F2P 全绿且无 P2P 回归）")
    print(f"  pass@1 (partial): {summary['pass_at_1_partial']} "
          f"（{partial_solved}/{n_task} 题至少修好 1 个 F2P 用例）")
    print(f"  patch 可应用率  : {summary['apply_rate']} （{applied_any}/{n_task} 题）")
    print(f"  平均 reward     : {summary['mean_reward']}")
    return summary


def compare(before_path: Path, after_path: Path) -> None:
    b = json.loads(before_path.read_text(encoding="utf-8"))
    a = json.loads(after_path.read_text(encoding="utf-8"))

    rows = [
        ("pass@1 (strict)", b["pass_at_1_strict"], a["pass_at_1_strict"]),
        ("pass@1 (partial)", b["pass_at_1_partial"], a["pass_at_1_partial"]),
        ("patch 可应用率", b["apply_rate"], a["apply_rate"]),
        ("平均 reward", b["mean_reward"], a["mean_reward"]),
        ("平均测试分", b["mean_test_score"], a["mean_test_score"]),
    ]

    print(f"\n## 训练前后对比（评测集 {a['n_tasks']} 题，k={a['k']}，"
          f"apply 口径：{a['apply_mode']}）\n")
    print("| 指标 | 训练前 (base) | 训练后 (GRPO) | 变化 |")
    print("|---|---|---|---|")
    for name, bv, av in rows:
        delta = av - bv
        arrow = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "—")
        print(f"| {name} | {bv:.4f} | {av:.4f} | {arrow} {delta:+.4f} |")

    print("\n### 逐题明细\n")
    print("| task_id | 训练前 reward | 训练后 reward | 训练前 strict_pass | 训练后 strict_pass |")
    print("|---|---|---|---|---|")

    def by_task(d: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in d["records"]:
            cur = out.get(r["task_id"])
            if cur is None or r["reward"] > cur["reward"]:
                out[r["task_id"]] = r
        return out

    bt, at = by_task(b), by_task(a)
    for tid in sorted(set(bt) | set(at)):
        rb, ra = bt.get(tid, {}), at.get(tid, {})
        print(f"| {tid} | {rb.get('reward', '-')} | {ra.get('reward', '-')} "
              f"| {rb.get('strict_pass', '-')} | {ra.get('strict_pass', '-')} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="闭环验证：训练前后 pass@1 对比")
    ap.add_argument("--model", help="模型路径（HF 目录或 VERL checkpoint 目录）")
    ap.add_argument("--base-model", default=None,
                    help=f"LoRA 评测时的基座权重目录（默认 {DEFAULT_BASE_MODEL}）")
    ap.add_argument("--tag", default="eval", help="本次评测标签，如 before / after")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("-k", type=int, default=1, help="每题采样次数（默认 1 = pass@1）")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="生成温度，默认 0 贪心（评测需可复现）")
    ap.add_argument("--lenient", action="store_true",
                    help="用与训练一致的宽松 apply 级联（默认严格，只用 git apply）")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE_JSON", "AFTER_JSON"),
                    help="对比两份评测结果并输出 markdown 表格")
    args = ap.parse_args()

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return 0

    if not args.model:
        ap.error("需要 --model，或用 --compare 对比已有结果")

    summary = evaluate(args.model, args.tag, args.k, args.lenient,
                       args.temperature, args.base_model)

    out_path = Path(args.out) if args.out else ROOT / "results" / f"eval_{args.tag}.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
