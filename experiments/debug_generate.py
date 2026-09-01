import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/model/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()
model.eval()
print("MODEL LOADED", flush=True)

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
inputs = tok(text, return_tensors="pt").to("cuda")
print("prompt tokens:", inputs.input_ids.shape[1], flush=True)

configs = [
    {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    {"do_sample": False},
    {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1},
]

for cfg in configs:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, **cfg)
    resp = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("=== CONFIG:", cfg, "===", flush=True)
    print(resp[:1200], flush=True)
    print("--- END ---", flush=True)
