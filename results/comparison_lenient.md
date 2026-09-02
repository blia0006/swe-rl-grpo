
## 训练前后对比（评测集 4 题，k=8，apply 口径：lenient(级联容错，同训练)）

| 指标 | 训练前 (base) | 训练后 (GRPO) | 变化 |
|---|---|---|---|
| pass@1 (strict) | 0.0000 | 0.0000 | — +0.0000 |
| pass@1 (partial) | 0.2500 | 0.0000 | ↓ -0.2500 |
| patch 可应用率 | 1.0000 | 1.0000 | — +0.0000 |
| 平均 reward | 0.0875 | 0.0813 | ↓ -0.0062 |
| 平均测试分 | 0.0156 | 0.0000 | ↓ -0.0156 |

### 逐题明细

| task_id | 训练前 reward | 训练后 reward | 训练前 strict_pass | 训练后 strict_pass |
|---|---|---|---|---|
| swe-synth-0015 | 0.2 | 0.2 | False | False |
| swe-synth-0018 | 0.2 | 0.2 | False | False |
| swe-synth-0020 | 0.2 | 0.2 | False | False |
| swe-synth-0022 | 0.5 | 0.2 | False | False |
