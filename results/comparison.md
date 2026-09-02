
## 训练前后对比（评测集 4 题，k=1，apply 口径：strict(仅 git apply)）

| 指标 | 训练前 (base) | 训练后 (GRPO) | 变化 |
|---|---|---|---|
| pass@1 (strict) | 0.0000 | 0.0000 | — +0.0000 |
| pass@1 (partial) | 0.0000 | 0.0000 | — +0.0000 |
| patch 可应用率 | 0.0000 | 0.0000 | — +0.0000 |
| 平均 reward | 0.0000 | 0.0000 | — +0.0000 |
| 平均测试分 | 0.0000 | 0.0000 | — +0.0000 |

### 逐题明细

| task_id | 训练前 reward | 训练后 reward | 训练前 strict_pass | 训练后 strict_pass |
|---|---|---|---|---|
| swe-synth-0015 | 0.0 | 0.0 | False | False |
| swe-synth-0018 | 0.0 | 0.0 | False | False |
| swe-synth-0020 | 0.0 | 0.0 | False | False |
| swe-synth-0022 | 0.0 | 0.0 | False | False |
