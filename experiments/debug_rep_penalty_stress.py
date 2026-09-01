import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)

with open("/workspace/repo/data/tasks.jsonl") as f:
    tasks = [json.loads(line) for line in f]

system = (
    "你是一个资深 Python 工程师，负责修复代码仓库中的 bug。\n\n规则：\n"
    "1. 你会看到一段问题描述（issue）和需要修改的文件路径。\n"
    "2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` `+++ b/` 格式），"
    "用一个 ```diff 代码块包裹，不要输出多余的解释性文字。\n"
    "3. patch 必须能被 `git apply` 直接应用，路径要和仓库里的路径完全一致。\n"
    "4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()
model.eval()


def has_degenerate_repeat(text, ngram=4, min_repeat=6):
    words = text.split()
    if len(words) < ngram * min_repeat:
        return False
    from collections import Counter

    grams = [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram)]
    c = Counter(grams)
    return c.most_common(1)[0][1] >= min_repeat if grams else False


n_degenerate = 0
n_total = 0
for t in tasks[:6]:
    user = (
        f"仓库：{t.get('repo')}\n\n问题描述：\n{t.get('problem_statement', '')}\n\n"
        f"需要修改的文件：{', '.join(t.get('modified_files') or [])}\n\n请直接给出修复该问题的 unified diff patch。"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=700,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=6,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        degenerate = has_degenerate_repeat(resp)
        n_total += 1
        n_degenerate += int(degenerate)
        print(f"[{t['task_id']} seed={seed}] degenerate={degenerate} resp_len={len(resp)}", flush=True)
        if degenerate:
            print("  head:", resp[:200], flush=True)

print(f"\n=== SUMMARY: {n_degenerate}/{n_total} degenerate ===", flush=True)
