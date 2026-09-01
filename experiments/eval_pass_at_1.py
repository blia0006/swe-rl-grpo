"""Phase 4：pass@1 评测脚本，在 GPU Pod 内运行（训练前跑一次算 baseline，
训练后加载 LoRA adapter 再跑一次算 post-train，两次结果做对比）。

复用训练侧一模一样的 prompt 构造（`pipeline/build_grpo_dataset.py`）和打分逻辑
（`pipeline/verl_reward_fn.py::compute_score`），保证"训练怎么打分，评测就怎么打分"，
数字才有意义。

用法（pod 内）：
    # baseline（训练前，不传 --lora-path）
    python3 experiments/eval_pass_at_1.py --tag baseline

    # 训练后（传 LoRA checkpoint 路径）
    python3 experiments/eval_pass_at_1.py --tag post_train \
        --lora-path /workspace/checkpoints/swe-rl-grpo/global_step_56/actor/lora_adapter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/model/Qwen2.5-Coder-1.5B-Instruct")


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="baseline / post_train，用于输出文件命名")
    ap.add_argument("--lora-path", default=None, help="LoRA adapter 目录，不传则评测 base 模型")
    ap.add_argument("--n-samples", type=int, default=1, help="每题采样几次，pass@1 用 1 即可")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    load_env()
    os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
    os.environ.setdefault("AGS_REWARD_TOOL_NAME", "swe-synth-shared-runner")

    from pipeline.build_grpo_dataset import SYSTEM_PROMPT, build_user_prompt
    from pipeline.verl_reward_fn import compute_score

    eval_tasks = json.loads((ROOT / "data" / "eval_tasks_full.json").read_text(encoding="utf-8"))
    print(f"评测集：{len(eval_tasks)} 题")

    # ⚠️ 不用 vLLM：实测 GPU 机型是 NVIDIA P4（Pascal, compute capability 6.1），
    # vLLM 官方主分支不支持 compute capability < 7.0（vllm-project/vllm#963），
    # 改用纯 transformers + peft 路径，与训练侧 HFRollout（`rollout.name=hf`）保持
    # 同一套推理栈，兼容任意算力。
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # P4/Pascal 无 bf16 支持
        attn_implementation="sdpa",  # P4 不支持 FlashAttention-2（需 Ampere+）
        trust_remote_code=True,
    ).to("cuda").eval()

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.eval()

    def generate_one(prompt_text: str) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=max(args.temperature, 1e-5) if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    results = []
    n_pass = 0
    for task in eval_tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ground_truth = json.dumps(
            {"task_id": task["task_id"], "image": task["image"], "repo": task.get("repo", "")},
            ensure_ascii=False,
        )

        best_score = 0.0
        best_text = ""
        for _ in range(max(args.n_samples, 1)):
            text = generate_one(prompt_text)
            score = compute_score(
                data_source="swe_rl",
                solution_str=text,
                ground_truth=ground_truth,
                extra_info={"task_id": task["task_id"]},
            )
            if score > best_score:
                best_score = score
                best_text = text

        passed = best_score >= 1.0
        n_pass += int(passed)
        print(f"  [{task['task_id']}] score={best_score:.2f} passed={passed}")
        results.append({
            "task_id": task["task_id"], "score": best_score, "passed": passed,
            "response_preview": best_text[:500],
        })

    pass_at_1 = n_pass / len(eval_tasks)
    print(f"\npass@1 ({args.tag}) = {n_pass}/{len(eval_tasks)} = {pass_at_1:.3f}")

    out_path = ROOT / "data" / f"eval_result_{args.tag}.json"
    out_path.write_text(json.dumps({
        "tag": args.tag, "lora_path": args.lora_path, "pass_at_1": pass_at_1,
        "n_pass": n_pass, "n_total": len(eval_tasks), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存到 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
