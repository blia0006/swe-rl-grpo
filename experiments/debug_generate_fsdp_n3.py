import json
import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29512")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

dist.init_process_group(backend="nccl", rank=0, world_size=1)
torch.cuda.set_device(0)

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

mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp,
    use_orig_params=True,
    device_id=torch.cuda.current_device(),
)
fsdp_model.eval()
print("FSDP MODEL LOADED", flush=True)

system = (
    "你是一个资深 Python 工程师，负责修复代码仓库中的 bug。\n\n规则：\n"
    "1. 你会看到一段问题描述（issue）和需要修改的文件路径。\n"
    "2. 你必须直接输出修复该问题的 unified diff patch（`diff --git` / `--- a/` `+++ b/` 格式），"
    "用一个 ```diff 代码块包裹，不要输出多余的解释性文字。\n"
    "3. patch 必须能被 `git apply` 直接应用，路径要和仓库里的路径完全一致。\n"
    "4. 只修改与问题直接相关的文件，不要删除或修改测试文件。"
)
with open("/workspace/repo/data/tasks.jsonl") as f:
    tasks = {json.loads(line)["task_id"]: json.loads(line) for line in f}
t = tasks["swe-synth-0001"]
user = (
    f"仓库：{t.get('repo')}\n\n问题描述：\n{t.get('problem_statement', '')}\n\n"
    f"需要修改的文件：{', '.join(t.get('modified_files') or [])}\n\n请直接给出修复该问题的 unified diff patch。"
)
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
print("prompt tokens:", inputs.input_ids.shape[1], flush=True)

param_ctx = FSDP.summon_full_params(fsdp_model, writeback=False, recurse=False)
with param_ctx, torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = fsdp_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
for i in range(out.shape[0]):
    resp = tok.decode(out[i][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"=== FSDP+LoRA+n=3, real long prompt, sample {i} (len={len(resp)}) ===", flush=True)
    print(resp[:1000], flush=True)
    print("--- END ---", flush=True)
