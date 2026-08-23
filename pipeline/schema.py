"""Tracing 数据结构（线 A / 线 B 共用）。

对应 plan.md 3.1 节的 episode 格式，纯 stdlib 实现（dataclass + json），
不依赖 pydantic / verl 等第三方包 —— 这份 schema 既要在沙箱内（agent.py，
只有标准库）用，也要在本机 / TKE 侧（driver.py、reward.py）用，必须保持
零第三方依赖才能在最受限的沙箱环境里直接跑。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


# 单条 tracing 最多记录的 observation 字符数，超出截断，防止 tracing.jsonl 爆炸
# （比如 pytest 全量输出、超大文件内容）
MAX_OBSERVATION_CHARS = 4000


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(truncated, total {len(text)} chars)"


@dataclass
class Action:
    """模型这一步产出的动作。"""
    tool: str  # read_file | bash | apply_patch | run_tests | submit
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}


@dataclass
class Step:
    """单步 (action, observation, reward, done)，对齐 VERL DataProto 的最小单元。"""
    step: int
    action: Action
    observation: str
    reward: float = 0.0
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action.to_dict(),
            "observation": _truncate(self.observation),
            "reward": self.reward,
            "done": self.done,
        }


@dataclass
class Episode:
    """一道题一次完整的解题过程（一条 tracing）。"""
    episode_id: str
    task_id: str
    model_version: str
    steps: list[Step] = field(default_factory=list)
    final_reward: float = 0.0
    fail_to_pass_rate: str = "0/0"
    pass_to_pass_rate: str = "0/0"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def add_step(self, action: Action, observation: str, reward: float = 0.0, done: bool = False) -> Step:
        s = Step(step=len(self.steps), action=action, observation=observation, reward=reward, done=done)
        self.steps.append(s)
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "model_version": self.model_version,
            "steps": [s.to_dict() for s in self.steps],
            "final_reward": self.final_reward,
            "fail_to_pass_rate": self.fail_to_pass_rate,
            "pass_to_pass_rate": self.pass_to_pass_rate,
            "num_steps": self.num_steps,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def validate(self) -> list[str]:
        """校验交付约束：episode_id/task_id 非空、steps 非空、done 语义自洽。
        返回错误信息列表，空列表代表通过。"""
        errs: list[str] = []
        if not self.episode_id:
            errs.append("episode_id 不能为空")
        if not self.task_id:
            errs.append("task_id 不能为空")
        if not self.steps:
            errs.append("steps 不能为空（验收要求单条 tracing ≥3 步）")
        else:
            for i, s in enumerate(self.steps):
                if s.step != i:
                    errs.append(f"steps[{i}].step 应为 {i}，实际为 {s.step}（必须严格递增且从 0 开始）")
            if not self.steps[-1].done:
                errs.append("最后一步 done 应为 True（正常结束或超步终止）")
            for i, s in enumerate(self.steps[:-1]):
                if s.done:
                    errs.append(f"steps[{i}] 提前把 done 置为 True，但它不是最后一步")
        if not (0.0 <= self.final_reward <= 1.0):
            errs.append(f"final_reward 应在 [0,1] 区间，实际为 {self.final_reward}")
        return errs

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        steps = [
            Step(
                step=sd["step"],
                action=Action(tool=sd["action"]["tool"], args=sd["action"].get("args", {})),
                observation=sd.get("observation", ""),
                reward=sd.get("reward", 0.0),
                done=sd.get("done", False),
            )
            for sd in d.get("steps", [])
        ]
        return cls(
            episode_id=d["episode_id"],
            task_id=d["task_id"],
            model_version=d.get("model_version", "unknown"),
            steps=steps,
            final_reward=d.get("final_reward", 0.0),
            fail_to_pass_rate=d.get("fail_to_pass_rate", "0/0"),
            pass_to_pass_rate=d.get("pass_to_pass_rate", "0/0"),
            started_at=d.get("started_at", 0.0),
            finished_at=d.get("finished_at"),
            error=d.get("error"),
        )


def load_episodes(path: str) -> list[Episode]:
    """读取 tracing.jsonl（一行一个 episode）。"""
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(Episode.from_dict(json.loads(line)))
    return episodes
