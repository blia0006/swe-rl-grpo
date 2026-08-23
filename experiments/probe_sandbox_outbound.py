"""Phase 1.5 最高优先级门禁：沙箱能否主动出网访问外部服务（模拟访问 vLLM）。

背景（见 plan.md 2.3）：线 A 方案要把 Agent 的 ReAct 控制循环整体放进沙箱内运行，
Agent 每一步都要用 HTTPS 调 TKE 上的 vLLM 拿下一步动作，因此需要先验证「沙箱能否主动出网」。

本脚本使用内置 code-interpreter 沙箱工具（AGS_SANDBOX_TEMPLATE，免去注册/删除自定义工具），
在沙箱内部执行：
  1. curl 一个公网 HTTPS 地址，看能不能出网、延迟量级
  2. 真实发起一次 OpenAI 兼容格式的 POST（打到一个公开的 echo/测试端点），
     验证的是「POST + 自定义 header（模拟 Authorization: Bearer <key>）」这条路径通不通，
     不依赖真实 vLLM 已经部署（此时 GPU 还没开）
  3. 核对沙箱内 python3 版本、是否有 requests（决定 agent.py 用 requests 还是 urllib）
  4. 核对能否 files.write 后用 commands.run 跑一个「长时后台命令」（模拟 Agent 跑十几步、
     几分钟不退出的场景）

用法：
    cd /Users/user/学习/题目四：强化学习
    source .venv/bin/activate
    python3 experiments/probe_sandbox_outbound.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _load_env() -> None:
    """读取本项目目录下的 .env 里的 E2B / AGS 凭证。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print(f"[FATAL] 找不到 {env_path}，请先 cp .env.example .env 并填入真实值")
        sys.exit(1)
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

    # e2b 2.x 默认强制校验 API Key 必须是 "e2b_" 前缀，AGS 的 Key 是 "ark_xxx"
    # 会被客户端拦截；必须在 import e2b 系列包之前设置这个开关。
    os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")


def _ok(title: str, notes: list[str]) -> None:
    print(f"\n[OK] {title}")
    for n in notes:
        print(f"     - {n}")


def _fail(title: str, err: str) -> None:
    print(f"\n[FAIL] {title}")
    print(f"     - {err}")


def main() -> int:
    _load_env()
    from e2b_code_interpreter import Sandbox

    template = os.environ.get("AGS_SANDBOX_TEMPLATE", "")
    if not template:
        print("[FATAL] .env 缺少 AGS_SANDBOX_TEMPLATE")
        return 1

    print(f"=== Phase 1.5 门禁：沙箱出网能力实测（工具={template}）===")

    sbx = None
    all_pass = True
    try:
        t0 = time.time()
        sbx = Sandbox.create(template=template, timeout=600)
        boot = time.time() - t0
        _ok("沙箱实例创建", [f"冷启动 {boot:.1f}s", f"sandbox_id={getattr(sbx, 'sandbox_id', '?')}"])

        # ---------- 1. 基础出网：分别探测多个公网目标，避免单一目标被 WAF/UA 拦截误判 ----------
        targets = [
            "https://httpbin.org/get",
            "https://api.github.com",
            "https://www.qq.com",
        ]
        outbound_ok_any = False
        notes1 = []
        for url in targets:
            t0 = time.time()
            r = sbx.commands.run(
                f"curl -sS -m 10 -o /dev/null -w 'HTTP_CODE=%{{http_code}} TIME=%{{time_total}}s' '{url}'",
                timeout=15,
            )
            dt = time.time() - t0
            out = r.stdout.strip()
            reached = out.startswith("HTTP_CODE=") and "HTTP_CODE=000" not in out
            if reached:
                outbound_ok_any = True
            notes1.append(f"{url} → {out or '(空)'}（墙钟 {dt:.1f}s，{'到达服务器' if reached else '未到达/超时'}）")
        if outbound_ok_any:
            _ok("① 基础出网（多目标探测）", notes1)
        else:
            all_pass = False
            _fail("① 基础出网（多目标探测）", "; ".join(notes1))

        # ---------- 2. 模拟 OpenAI 格式 POST（带 Authorization header）----------
        # httpbin.org 公共实例偶发 503（服务方不稳定，非沙箱问题，见首次探测已验证走通）；
        # 这里换用另一个独立的公开 echo 服务，两者互为交叉验证，任一成功即视为通过。
        post_targets = ["https://postman-echo.com/post", "https://httpbin.org/post"]
        post_ok = False
        notes2 = []
        for url in post_targets:
            t0 = time.time()
            r2 = sbx.commands.run(
                f"curl -sS -m 10 -X POST '{url}' "
                "-H 'Content-Type: application/json' "
                "-H 'Authorization: Bearer sk-fake-test-key' "
                "-d '{\"model\":\"probe\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}' "
                "-w '\\nHTTP_CODE=%{http_code}'",
                timeout=15,
            )
            dt2 = time.time() - t0
            out2 = r2.stdout.strip()
            success = "HTTP_CODE=200" in out2 and "sk-fake-test-key" in out2
            notes2.append(f"{url} → {'成功，header/body 均正确回显' if success else out2[:150]}（{dt2:.1f}s）")
            if success:
                post_ok = True
                break
        if post_ok:
            _ok("② 模拟 OpenAI 格式 POST（带 Authorization header）", notes2)
        else:
            all_pass = False
            _fail("② 模拟 OpenAI 格式 POST", "; ".join(notes2))

        # ---------- 3. 沙箱内运行环境核对 ----------
        r3 = sbx.commands.run(
            "python3 -V; python3 -c 'import requests; print(\"requests_ok\", requests.__version__)' "
            "2>&1 || echo 'NO_REQUESTS'",
            timeout=15,
        )
        _ok("③ 沙箱内运行环境", r3.stdout.strip().split("\n"))

        # ---------- 4. 长时后台命令能力（模拟 Agent 跑几分钟不退出）----------
        sbx.files.write("/tmp/agent_probe_marker.txt", "not_yet")
        r4 = sbx.commands.run(
            "nohup bash -c 'sleep 5 && echo done > /tmp/agent_probe_marker.txt' "
            ">/tmp/nohup.log 2>&1 &",
            timeout=10,
        )
        time.sleep(7)
        marker = sbx.files.read("/tmp/agent_probe_marker.txt").strip()
        if marker == "done":
            _ok("④ 长时/后台命令能力", ["5s 后台任务如期完成，files.write/read 与 commands.run 组合正常"])
        else:
            all_pass = False
            _fail("④ 长时/后台命令能力", f"marker={marker!r}（预期 'done'）")

    except Exception as e:  # noqa: BLE001
        all_pass = False
        _fail("沙箱链路", f"{type(e).__name__}: {e}")
    finally:
        if sbx is not None:
            try:
                sbx.kill()
                print("\n[cleanup] 沙箱实例已回收")
            except Exception as e:  # noqa: BLE001
                print(f"\n[cleanup][WARN] 回收失败，可能需要手动清理: {e}")

    print("\n" + "=" * 60)
    if all_pass:
        print("[RESULT] 全部通过 → 沙箱可主动出网，线 A「Agent 整体跑在沙箱内」架构成立")
        return 0
    else:
        print("[RESULT] 存在失败项 → 需要判断走兜底形态（控制循环在外 + 沙箱作执行器）")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
