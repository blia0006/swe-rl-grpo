"""从 VERL 训练日志解析指标，输出 reward / grad_norm 曲线图 + 统计摘要。

对应 TASK-SPEC.md 验收标准第 4 条："reward 曲线呈上升趋势（提供 wandb 截图或
matplotlib 图表）"。本课题 `trainer.logger=["console"]`（未接 wandb，避免训练机
出网依赖），因此从 console 日志反解指标画图。

用法：
    python3 scripts/plot_reward_curve.py train_final_55steps.log
    python3 scripts/plot_reward_curve.py train.log -o docs/reward_curve.png

日志中每个训练 step 会打印一行形如：
    step:1 - actor/grad_norm:0.0 - critic/score/mean:0.0 - critic/score/max:0.0 ...
本脚本按 `step:N` 分行解析，逐 step 抽取所需字段。

【为什么必须画滑动平均而不能只画原始曲线】
本课题 `train_batch_size=1`，即**每个 step 只跑一道题**（10~11 题循环 5 轮）。
单步 reward 主要由"这一步恰好抽到哪道题"决定，题目难度差异远大于策略变化带来的
差异，原始曲线是剧烈锯齿状，无法反映训练趋势。因此同时画：
  - 原始逐步值（浅色散点/细线，示意波动幅度）
  - 滑动平均（粗线，窗口默认 11 = 约一个完整题目轮次，消除题目轮转造成的周期性）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 需要从日志里抽取的指标 → 图上的展示名
METRICS = {
    "critic/score/mean": "reward (mean)",
    "critic/score/max": "reward (max)",
    "actor/grad_norm": "grad_norm",
    "actor/pg_loss": "pg_loss",
}

STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")


def parse_log(log_path: Path) -> dict[str, list[float]]:
    """解析日志，返回 {指标名: [每步的值]}。

    VERL console logger 每步输出一行 `step:N - k1:v1 - k2:v2 - ...`。
    同一 step 可能因日志交错重复出现，用 dict 按 step 去重（后写覆盖前写）。
    """
    per_step: dict[int, dict[str, float]] = {}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            body = m.group(2)
            vals: dict[str, float] = {}
            for key in METRICS:
                # 指标值可能是 0.0 / 1e-05 / 0.12499 等形式
                km = re.search(rf"{re.escape(key)}:([0-9.eE+-]+)", body)
                if km:
                    try:
                        vals[key] = float(km.group(1))
                    except ValueError:
                        pass
            if vals:
                per_step[step] = vals

    if not per_step:
        return {}

    steps = sorted(per_step)
    out: dict[str, list[float]] = {"step": [float(s) for s in steps]}
    for key in METRICS:
        out[key] = [per_step[s].get(key, float("nan")) for s in steps]
    return out


def moving_average(values: list[float], window: int) -> tuple[list[float], list[float]]:
    """返回 (x 轴索引, 滑动平均值)。window 大于序列长度时自动收缩。"""
    n = len(values)
    w = min(window, n)
    if w <= 0:
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    for i in range(w - 1, n):
        seg = [v for v in values[i - w + 1: i + 1] if v == v]  # 过滤 nan
        if seg:
            xs.append(float(i + 1))
            ys.append(sum(seg) / len(seg))
    return xs, ys


def segment_means(values: list[float], n_seg: int = 3) -> list[tuple[str, float]]:
    """把序列等分 n_seg 段，返回每段均值，用于判断整体趋势。"""
    vals = [v for v in values if v == v]
    n = len(vals)
    if n == 0:
        return []
    size = max(1, n // n_seg)
    out: list[tuple[str, float]] = []
    for i in range(n_seg):
        lo = i * size
        hi = n if i == n_seg - 1 else min(n, (i + 1) * size)
        if lo >= hi:
            continue
        seg = vals[lo:hi]
        out.append((f"step {lo + 1}-{hi}", sum(seg) / len(seg)))
    return out


def print_summary(data: dict[str, list[float]], window: int) -> None:
    steps = data["step"]
    scores = data["critic/score/mean"]
    grads = data["actor/grad_norm"]

    print(f"解析到 {len(steps)} 个训练 step")
    print()

    valid_scores = [v for v in scores if v == v]
    print("【reward (critic/score/mean)】")
    print(f"  总体均值 : {sum(valid_scores) / len(valid_scores):.4f}")
    print(f"  最大单步 : {max(valid_scores):.4f}")
    print(f"  非零步数 : {sum(1 for v in valid_scores if v > 0)}/{len(valid_scores)}")
    print("  分段均值 :")
    for label, mean in segment_means(scores):
        print(f"    {label:<16} {mean:.4f}")

    ma_x, ma_y = moving_average(scores, window)
    if ma_y:
        print(f"  滑动平均(w={window}) 首={ma_y[0]:.4f} 峰={max(ma_y):.4f} "
              f"谷={min(ma_y):.4f} 末={ma_y[-1]:.4f}")
        trend = "上升" if ma_y[-1] > ma_y[0] else ("下降" if ma_y[-1] < ma_y[0] else "持平")
        print(f"  首末对比 : {trend}（{ma_y[0]:.4f} → {ma_y[-1]:.4f}）")

    print()
    valid_grads = [v for v in grads if v == v]
    if valid_grads:
        nonzero = [v for v in valid_grads if v > 1e-8]
        print("【grad_norm】（判断参数是否真的在更新）")
        print(f"  有效步数 : {len(nonzero)}/{len(valid_grads)} 步 grad_norm > 1e-8")
        if nonzero:
            print(f"  区间     : {min(nonzero):.4g} ~ {max(nonzero):.4g}")
            print(f"  均值     : {sum(nonzero) / len(nonzero):.4g}")


def plot(data: dict[str, list[float]], out_path: Path, window: int, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境（训练节点/容器）必需
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ 未安装 matplotlib，跳过绘图。安装：pip install matplotlib", file=sys.stderr)
        return

    steps = data["step"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ---- 上图：reward ----
    ax = axes[0]
    ax.plot(steps, data["critic/score/mean"], color="tab:blue", alpha=0.28,
            lw=1.0, marker="o", ms=2.5, label="reward/step (raw)")
    ma_x, ma_y = moving_average(data["critic/score/mean"], window)
    if ma_y:
        ax.plot(ma_x, ma_y, color="tab:red", lw=2.4,
                label=f"moving average (window={window})")
    if "critic/score/max" in data:
        ax.plot(steps, data["critic/score/max"], color="tab:green", alpha=0.35,
                lw=0.9, ls="--", label="reward/step (max of 8 samples)")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 下图：grad_norm（对数轴，因为跨好几个数量级）----
    ax = axes[1]
    grads = data["actor/grad_norm"]
    ax.plot(steps, grads, color="tab:purple", lw=1.2, marker="o", ms=2.5,
            label="actor/grad_norm")
    positive = [v for v in grads if v == v and v > 0]
    if positive and max(positive) / max(min(positive), 1e-12) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("grad_norm (log)")
    else:
        ax.set_ylabel("grad_norm")
    ax.set_xlabel("training step")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n已保存曲线图：{out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="从 VERL 训练日志画 reward / grad_norm 曲线")
    ap.add_argument("log", nargs="?", default="train.log", help="训练日志路径")
    ap.add_argument("-o", "--out", default="docs/reward_curve.png", help="输出图片路径")
    ap.add_argument("-w", "--window", type=int, default=11,
                    help="滑动平均窗口（默认 11 ≈ 一个完整题目轮次）")
    ap.add_argument("-t", "--title", default=None, help="图标题")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path) if not log_path.exists() else log_path
    if not log_path.exists():
        print(f"❌ 日志不存在：{log_path}", file=sys.stderr)
        return 1

    data = parse_log(log_path)
    if not data:
        print(f"❌ 未从 {log_path} 解析到任何 `step:N - ...` 指标行。", file=sys.stderr)
        print("   确认这是 VERL console logger 的输出日志。", file=sys.stderr)
        return 1

    print_summary(data, args.window)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    title = args.title or (
        f"SWE-RL GRPO · Qwen2.5-Coder-1.5B-Instruct · {len(data['step'])} steps"
    )
    plot(data, out_path, args.window, title)

    # 同时导出 CSV，便于二次绘图或写进 README 表格
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["step"] + list(METRICS)
        f.write(",".join(cols) + "\n")
        for i in range(len(data["step"])):
            f.write(",".join(
                f"{data[c][i]:.6g}" if data.get(c) and data[c][i] == data[c][i] else ""
                for c in cols
            ) + "\n")
    print(f"已保存指标 CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
