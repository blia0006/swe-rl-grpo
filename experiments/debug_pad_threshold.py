import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)

with open("/workspace/repo/data/tasks.jsonl") as f:
    tasks = {json.loads(line)["task_id"]: json.loads(line) for line in f}
t = tasks["swe-synth-0001"]
system = (
    "你是一个资深 Python 工程师，负责修复代码仓库中的 bug。\n\n规则：\n"
    "1. 你会看到一段问题描述（issue）和需要修改的文件路径。\n"
    "2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` `+++ b/` 格式），"
    "用一个 ```diff 代码块包裹，不要输出多余的解释性文字。\n"
    "3. patch 必须能被 `git apply` 直接应用，路径要和仓库里的路径完全一致。\n"
    "4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"
)
user = (
    f"仓库：{t.get('repo')}\n\n问题描述：\n{t.get('problem_statement', '')}\n\n"
    f"需要修改的文件：{', '.join(t.get('modified_files') or [])}\n\n请直接给出修复该问题的 unified diff patch。"
)
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
raw_inputs = tok(text, return_tensors="pt", add_special_tokens=False)
input_ids = raw_inputs.input_ids
attention_mask = raw_inputs.attention_mask
real_len = input_ids.shape[1]
print("real_len", real_len, flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()
model.eval()

for pad_len in [0, 100, 300, 500, 692]:
    padded_input_ids = torch.cat(
        [torch.full((1, pad_len), tok.pad_token_id, dtype=input_ids.dtype), input_ids], dim=1
    ).cuda()
    padded_attention_mask = torch.cat(
        [torch.zeros((1, pad_len), dtype=attention_mask.dtype), attention_mask], dim=1
    ).cuda()
    position_ids = (padded_attention_mask.cumsum(dim=-1) - 1).clamp(min=0).cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            position_ids=position_ids,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    resp = tok.decode(out[0][padded_input_ids.shape[1] :], skip_special_tokens=True)
    print(f"=== pad_len={pad_len} (total_len={padded_input_ids.shape[1]}) greedy ===", flush=True)
    print(resp[:400], flush=True)
    print("---", flush=True)
