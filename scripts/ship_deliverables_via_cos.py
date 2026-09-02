#!/usr/bin/env python3
"""交付物摆渡：GPU Pod ──COS──► 本地，用于 Pod 无法直连 GitHub 时回传产出。

【为什么需要】训练交付物（reward 曲线图、pass@1 对比表、评测 JSON、训练日志）
只存在于 GPU Pod 内，需要进 Git 仓库供评审查看。正常路径是在 Pod 内 git push，
但 Pod 走 NAT 出网访问 github.com:443 可能超时不可用（实测出现过 129s 超时），
而腾讯云 COS 走内网/就近接入通常仍然可达 —— 课题本身也要求
"SandBox → TKE 通过 COS/CFS 传递数据"，这里复用同一条通路回传产出。

用法：
    # 1) 在 GPU Pod 内打包上传
    python3 scripts/ship_deliverables_via_cos.py push --bucket <bucket>

    # 2) 在本地拉回（本地能访问 GitHub，再正常 commit & push）
    python3 scripts/ship_deliverables_via_cos.py pull --bucket <bucket>

凭证取自仓库根目录 `.env`（TENCENTCLOUD_SECRET_ID / SECRET_KEY / COS_REGION）。
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 需要回传的交付物。缺失的条目会被跳过并提示，不影响其余文件。
DELIVERABLES = [
    "docs/reward_curve.png",
    "docs/reward_curve.csv",
    "results/comparison.md",
    "results/comparison_lenient.md",
    "results/eval_before.json",
    "results/eval_after.json",
    "results/eval_before_lenient.json",
    "results/eval_after_lenient.json",
    "train_final_55steps.log",
]

REMOTE_KEY = "deliverables/swe-rl-deliverables.tar.gz"


def load_dotenv() -> None:
    """加载 .env（已存在的环境变量不覆盖）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def do_push(bucket: str, key: str) -> int:
    from clients.cos import upload_file

    present, missing = [], []
    for rel in DELIVERABLES:
        (present if (ROOT / rel).exists() else missing).append(rel)

    if missing:
        print("⚠️ 以下文件不存在，将跳过：")
        for m in missing:
            print(f"    {m}")
    if not present:
        print("❌ 没有任何交付物可打包，先跑 scripts/run_final_deliverables.sh")
        return 1

    tar_path = ROOT / "deliverables.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in present:
            tar.add(ROOT / rel, arcname=rel)
            print(f"  + {rel} ({(ROOT / rel).stat().st_size} bytes)")

    size = tar_path.stat().st_size
    print(f"\n已打包 {len(present)} 个文件 → {tar_path.name}（{size / 1024:.1f} KB）")

    url = upload_file(bucket, key, str(tar_path))
    print(f"已上传：{url}")
    print("\n下一步在**本地**执行：")
    print(f"    python3 scripts/ship_deliverables_via_cos.py pull --bucket {bucket}")
    return 0


def do_pull(bucket: str, key: str) -> int:
    from clients.cos import download_file

    tar_path = ROOT / "deliverables.tar.gz"
    print(f"从 cos://{bucket}/{key} 下载 ...")
    download_file(bucket, key, str(tar_path))
    print(f"已下载 {tar_path.name}（{tar_path.stat().st_size / 1024:.1f} KB）")

    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        # 安全检查：拒绝解压到仓库目录之外（防路径穿越）
        for n in names:
            target = (ROOT / n).resolve()
            if not str(target).startswith(str(ROOT.resolve())):
                print(f"❌ 压缩包含越界路径，已中止：{n}")
                return 1
        tar.extractall(ROOT)

    print(f"\n已解压 {len(names)} 个文件：")
    for n in names:
        p = ROOT / n
        print(f"  {n}  ({p.stat().st_size} bytes)" if p.exists() else f"  {n}  (缺失)")

    tar_path.unlink(missing_ok=True)
    print("\n下一步：")
    print("    git add -f docs/ results/ train_final_55steps.log")
    print("    git commit -m 'chore: 提交训练交付物' && git push origin main")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="交付物摆渡：Pod ──COS──► 本地")
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", required=True, help="COS bucket 名（形如 name-appid）")
    ap.add_argument("--key", default=REMOTE_KEY, help="COS 对象键")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("TENCENTCLOUD_SECRET_ID"):
        print("❌ 缺少 TENCENTCLOUD_SECRET_ID / SECRET_KEY（见 .env）")
        return 1

    key = args.key
    if args.action == "push":
        # 加时间戳避免覆盖历史交付物
        key = key.replace(".tar.gz", f"-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz")
        print(f"本次对象键：{key}\n")
        rc = do_push(args.bucket, key)
        return rc
    return do_pull(args.bucket, key)


if __name__ == "__main__":
    sys.exit(main())
