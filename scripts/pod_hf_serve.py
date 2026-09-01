#!/usr/bin/env python3
"""在 GPU Pod 内启动一个纯 HuggingFace transformers（不依赖 vLLM）的 OpenAI 兼容
HTTP server，暴露给 AGS 沙箱内的 agent.py 通过公网 HTTP 调用（Line A 多轮 ReAct
采集用）。

⚠️ 为什么不用 vLLM（对照原 `pod_vllm_serve.sh` 的重大变更）：
本课题实际到手的 GPU 机型是 GN6S.LARGE20，实测其硬件是 **NVIDIA Tesla P4**
（Pascal 架构，compute capability 6.1），而不是命名规律暗示的 T4（Turing, 7.5）。
vLLM 官方主分支明确不支持 compute capability < 7.0 的显卡
（见 https://github.com/vllm-project/vllm/issues/963），P4 的 6.1 不满足要求。
因此 Line A 生成 与 VERL 训练 rollout（`actor_rollout_ref.rollout.name=hf`）
统一改为纯 transformers 路径，全程不初始化 vLLM 引擎（镜像里仍打包了 vllm，只是
不调用，import 本身不受 GPU 算力影响，不会报错）。

依赖：仅用 torch + transformers + Python 标准库 http.server（GPU Pod 内 VERL
官方镜像已自带，不需要额外 pip install）。

用法（pod 内，repo 代码已 kubectl cp 到 /workspace/repo）：
    MODEL_DIR=/workspace/model/Qwen2.5-Coder-1.5B-Instruct python3 scripts/pod_hf_serve.py
    MODEL_DIR=/workspace/checkpoints/swe-rl-grpo/merged-lora PORT=8000 python3 scripts/pod_hf_serve.py

训练开始前必须先 Ctrl-C / kill 掉这个进程，把显存腾给 VERL 的训练/rollout。
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_PATH = os.environ.get("MODEL_DIR", "/workspace/model/Qwen2.5-Coder-1.5B-Instruct")
API_KEY = os.environ.get("LLM_API_KEY", "swe-rl-secret-key")
PORT = int(os.environ.get("PORT", "8000"))
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "swe-rl-model")

print(f"[pod_hf_serve] loading tokenizer/model from {MODEL_PATH} (fp16, sdpa, P4-safe) ...", flush=True)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # P4/Pascal 没有 bf16 张量核心，全程用 fp16
    attn_implementation="sdpa",  # P4 不支持 FlashAttention-2（需 Ampere+）
    trust_remote_code=True,
).to("cuda").eval()
print("[pod_hf_serve] model loaded, ready to serve.", flush=True)

_lock = threading.Lock()  # 单卡场景下串行生成，避免并发请求抢显存 OOM


def _generate(messages: list[dict], temperature: float, max_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with _lock, torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        auth = self.headers.get("Authorization", "")
        if API_KEY and auth != f"Bearer {API_KEY}":
            self._send_json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            temperature = float(payload.get("temperature", 0.7))
            max_tokens = int(payload.get("max_tokens", 1024))
            content = _generate(messages, temperature, max_tokens)
            self._send_json(200, {
                "id": "chatcmpl-hf-local",
                "object": "chat.completion",
                "model": SERVED_NAME,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
            })
        except Exception as e:  # noqa: BLE001 - 顶层兜底，返回结构化错误而不是让连接挂死
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        print("[pod_hf_serve] " + (fmt % args), flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)  # noqa: S104 - Pod 内部端口，靠 K8s Service/安全组控制暴露面
    print(f"[pod_hf_serve] serving on 0.0.0.0:{PORT} (served_model_name={SERVED_NAME})", flush=True)
    server.serve_forever()
