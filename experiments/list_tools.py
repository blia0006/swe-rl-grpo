"""列出当前全部沙箱工具，评估哪些可以清理。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")

from clients.ags import AGSClient  # noqa: E402


def main() -> int:
    ags = AGSClient()
    tools = ags.list_tools()
    print(f"共 {len(tools)} 个工具：")
    for t in tools:
        print(f"  name={t['name']:35s} tool_id={t['tool_id']:15s} status={t['status']:10s} image={t['image']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
