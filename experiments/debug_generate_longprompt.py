import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

model_path = "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16, target_modules="all-linear", bias="none"
)
model = get_peft_model(model, lora_config)
model.to(torch.bfloat16)
model.eval()
print("MODEL LOADED", flush=True)
print("sliding_window config:", getattr(model.config, "sliding_window", "N/A"), flush=True)
print("use_sliding_window:", getattr(model.config, "use_sliding_window", "N/A"), flush=True)
print("max_window_layers:", getattr(model.config, "max_window_layers", "N/A"), flush=True)

# load the REAL task prompt (same as used in training) from tasks.jsonl
system = (
    "你是一个资深 Python 工程师，负责修复代码仓库中的 bug。\n\n规则：\n"
    "1. 你会看到一段问题描述（issue）和需要修改的文件路径。\n"
    "2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` `+++ b/` 格式），"
    "用一个 ```diff 代码块包裹，不要输出多余的解释性文字。\n"
    "3. patch 必须能被 `git apply` 直接应用，路径要和仓库里的路径完全一致。\n"
    "4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"
)

with open("/workspace/repo/data/tasks.jsonl") as f:
    tasks = {}
    for line in f:
        d = json.loads(line)
        tasks[d["task_id"]] = d
t = tasks["swe-synth-0001"]
user = (
    f"仓库：{t.get('repo')}\n\n问题描述：\n{t.get('problem_statement', '')}\n\n"
    f"需要修改的文件：{', '.join(t.get('modified_files') or [])}\n\n请直接给出修复该问题的 unified diff patch。"
)

messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
print("prompt tokens:", inputs.input_ids.shape[1], flush=True)

for cfg_name, cfg in [
    ("greedy", {"do_sample": False}),
    ("temp0.8_topp0.95_topk50", {"do_sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 50}),
]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=400,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            **cfg,
        )
    resp = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"=== {cfg_name} (real long prompt, {inputs.input_ids.shape[1]} tokens) ===", flush=True)
    print(resp[:1000], flush=True)
    print("--- END ---", flush=True)
