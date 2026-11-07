# PR77 Phase B0：冻结风险决策回放的只读归档

本目录只归档 PR77 已完成的两个平台 CI artifact，不重新运行实验、不回放、不拟合模型、不调用真值评价器，也不触发新的 CI。

## 1. 固定身份

| 项目 | 值 |
|---|---|
| 仓库 | `Luyi1235/RA-LSBO` |
| PR | 77 |
| 固定代码 SHA | `634ff13c07d11e4df4748199806726a3eb14fd70` |
| CI run | `42370899496` (`frozen-risk-decision-value`) |
| run 结论 | `success` |
| 输入 PR75 结果 HEAD | `fba37629d003e2b528b7c3e3c5f50d3f245eb21d` |
| Windows job | `128398623401`，success |
| Ubuntu job | `128398623562`，success |

归档前检查了本机和远程：没有同 run/artifact 的既有完整归档；本目录是新的。

## 2. 两个 artifact

| 平台 | 名称 | artifact ID | ZIP 大小 | ZIP SHA256 | ZIP 完整性 |
|---|---|---|---:|---|---|
| Windows | `phase-b0-windows-latest` | `13574020196` | 8,827,127 B | `dadc5327...504f` | 通过 |
| Ubuntu | `phase-b0-ubuntu-latest` | `13573651527` | 8,827,127 B | `98255cd4...766f` | 通过 |

两个 ZIP 的字节哈希不同（压缩元数据不同），但解压后的 5 个文件逐字节相同。

## 3. manifest 与跨平台一致性

每个平台包含 `manifest.json` + 4 个数据文件；manifest 自身的 4 条 SHA256 全部匹配：

| 文件 | SHA256（Windows = Ubuntu） |
|---|---|
| `plan.json` | `7f7aa5ab...a3f5` |
| `report.json` | `287d9508...69b2` |
| `decisions.json` | `e8fcca0c...f4de` |
| `label_coverage.json` | `1c70ad9f...2a21` |
| `manifest.json` | `4fcbcc51...a979` |

两个平台解压后的 5 个文件 **5/5 逐字节相同**，跨平台字节核验通过。

## 4. report 核验

`report.json`：

- `status = COMPLETE_FROZEN_RISK_DECISION_REPLAY`
- `source_model_rows = 360`
- `replay_decision_rows = 1800`
- `input_result_sha = fba37629d003e2b528b7c3e3c5f50d3f245eb21d`
- `new_bill`：`new_true_evaluations=0`、`new_gp_fits=0`、`new_vae_fits=0`、`new_fantasy_fits=0`、`aircraft_calls=0`、`rcaide_calls=0`
- `input_verification.status = VERIFIED_LEARNABILITY`
- `input_verification.files = 402`
- `input_verification.model_rows = 360`
- `historical_bill.math_started = math_settled = 12,672`
- `historical_bill.gp_head_started = gp_head_completed = 540`，0 errors
- `historical_bill.gp_kernel_started = gp_kernel_completed = 4,320`，0 errors
- `failed_model_rows = 0`

**402 个文件 / 12,672 次评价 / 540 个 GP head / 4,320 次核构造属于被核验的 PR75 历史记录，不是本轮新增计算。** 本轮归档只读文件、校验哈希，新增真值/GP/VAE/幻想/飞机/RCAIDE 调用全部为 0。

## 5. CI 日志与元数据

- `metadata/run.json`：GitHub Actions run 级元数据。
- `metadata/jobs.json`：两个平台 job 的 id、名称、结论。
- `metadata/artifacts.json`：两个 artifact 的 id、名称、大小、过期状态。
- `logs/ci-logs.zip`：GitHub API 下载的完整 run logs（28 个条目）。
- `logs/ci-logs/`：解压后的逐 job/step 日志，包括两个平台的完整控制台输出。

## 6. 目录

```
artifacts/                 原始两个 ZIP
windows/                   Windows 解压的 5 个文件
ubuntu/                    Ubuntu 解压的 5 个文件
metadata/                  run/jobs/artifacts JSON
logs/ci-logs.zip           完整 CI 日志 ZIP
logs/ci-logs/              解压后的日志
verification_summary.json  机器可读核验结论
fixed_sha.txt              固定身份与哈希
instructions.txt           本任务指令副本
README.md                  本说明
```

证据目录使用 `.gitattributes` 固定 `* -text`，避免换行转换改变字节。

## 7. 边界

- 本归档不证明 PR77 回放之外的任何在线算法收益；它只证明 CI artifact 与 PR75 结果绑定、可读、可核验。
- 两个 ZIP 不必字节相同；本归档只要求解压后文件逐字节一致，已满足 5/5。
- 本轮没有代码 worktree 运行、没有模型拟合、没有真值调用，也没有修改 PR75 或 PR77 代码分支。
