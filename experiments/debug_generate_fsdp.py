import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29511")
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
    "你是一个资深 Python 工程师，负责修复代码仓库中的 bug。你必须直接输出修复该问题的 "
    "unified diff patch，用一个```diff代码块包裹。"
)
user = (
    "仓库：psf/cachecontrol\n\n问题描述：\n实现 CacheControl 函数：为传入的 requests.Session "
    "配置并挂载缓存适配器，然后返回该 Session。\n\n需要修改的文件：cachecontrol/wrapper.py\n\n"
    "请直接给出修复该问题的 unified diff patch。"
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
        max_new_tokens=400,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        num_return_sequences=3,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
for i in range(out.shape[0]):
    resp = tok.decode(out[i][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"=== FSDP+LoRA generation sample {i} ===", flush=True)
    print(resp[:800], flush=True)
    print("--- END ---", flush=True)
