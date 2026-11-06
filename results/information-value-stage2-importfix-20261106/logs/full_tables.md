## 总览

- code_sha: `0cefd1f2507c355067f3c4f56d2dd43d919f65ef`
- status: `COMPLETE_FRESH_FIXED_SIX_ON_POLICY`
- actual_math_calls: 1152 = 576 initial + 576 online
- aircraft/RCAIDE: 0/0
- trials: 96

### 每臂总计

| arm | trials | online | failures | final feasible |
|---|---:|---:|---:|---:|
| soft_gp_eligible | 48 | 288 | 84 | 45 |
| soft_gp_all_support | 48 | 288 | 88 | 44 |

### 逐任务 arms

| case | arm | final feasible | mean NF | NF sum | cold start | forecast | support collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| branin_ordinary | soft_gp_eligible | 12/12 | 0.417 | 5 | 0 | 72 | 0 |
| branin_ordinary | soft_gp_all_support | 12/12 | 0.417 | 5 | 0 | 72 | 0 |
| branin_sparse | soft_gp_eligible | 10/12 | 2.833 | 34 | 0 | 65 | 3 |
| branin_sparse | soft_gp_all_support | 10/12 | 2.833 | 34 | 0 | 68 | 3 |
| rosenbrock_ordinary | soft_gp_eligible | 12/12 | 0.667 | 8 | 0 | 72 | 0 |
| rosenbrock_ordinary | soft_gp_all_support | 12/12 | 0.667 | 8 | 0 | 72 | 0 |
| rosenbrock_sparse | soft_gp_eligible | 11/12 | 3.083 | 37 | 3 | 64 | 3 |
| rosenbrock_sparse | soft_gp_all_support | 10/12 | 3.417 | 41 | 2 | 65 | 4 |

### 配对（all_support − eligible）

| case | W/L/T | both feasible | mean Δ objective | median Δ | path changed |
|---|---:|---:|---:|---:|---:|
| branin_ordinary | [0, 0, 12] | 12 | 0.0 | None | 0/12 |
| branin_sparse | [0, 0, 12] | 10 | 0.0 | None | 2/12 |
| rosenbrock_ordinary | [0, 0, 12] | 12 | 0.0 | None | 0/12 |
| rosenbrock_sparse | [1, 2, 9] | 9 | 0.0 | None | 3/12 |

### support 统计

| arm | receipts | model | ablation | EI true | median eligible | min eligible | median all | official collapse | model eligible=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_gp_eligible | 288 | 273 | 0 | 0 | 125 | 3 | 128 | 6 | 0 |
| soft_gp_all_support | 288 | 277 | 277 | 277 | 125 | 1 | 128 | 7 | 7 |
