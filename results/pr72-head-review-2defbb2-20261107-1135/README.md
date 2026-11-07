# PR72 head `2defbb2` CI 复核归档

固定 head：`2defbb275bac93c41cbcfbed9ca0b5f39f168e26`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/72>  
CI run：<https://github.com/Luyi1235/RA-LSBO/actions/runs/42242255386>  
工作流：`.github/workflows/quadrature-diagnostic.yml`

本目录是**只读归档**：没有修改 2defbb2 代码，没有安装/升级依赖，没有重新运行 smoke/full、PR70 历史诊断或任何飞机/RCAIDE 评价。所有 job、artifact 和日志均通过 GitHub API 下载。

## 1. 固定身份与提交链

`git log --reverse f46e268^..2defbb2`：

```
f46e268 Add fixed-prefix quadrature diagnostic for PR70 without new evaluations
1a460fb PR72 R1: attest LF/CRLF source identity before historical checkpoint replay
2defbb2 PR72: reconstruct exact feature identity from sealed codebook when native forward differs
```

- `f46e268` 是 PR72 初始诊断代码；
- `1a460fb` 是 LF/CRLF source-identity R1；
- `2defbb2` 是当前 head，新增 sealed codebook 的精确 feature identity 重建。
- `git merge-base --is-ancestor f46e268 2defbb2` 为真。
- 只读 worktree：`D:\LSBO\RA-LSBO-pr72-review-20261107-1135`，HEAD 为 `2defbb2`，干净。

## 2. run 的四组 job

run 本身 `conclusion=failure`，四个 job 中仅 Windows CRLF 成功。

| job | OS | EOL | conclusion | 测试 | 失败步骤 | artifact |
|---|---|---|---|---:|---|---|
| `127975257624` | windows-latest | CRLF | **success** | 24 + 64 OK | — | `quadrature-r1-windows-latest-crlf`（368,930 B） |
| `127975257049` | ubuntu-latest | LF | failure | 24 + 64 OK | step 10 | `quadrature-r1-ubuntu-latest-lf`（4,823 B） |
| `127975257652` | ubuntu-latest | CRLF | failure | 24 + 64 OK | step 10 | `quadrature-r1-ubuntu-latest-crlf`（4,823 B） |
| `127975257473` | windows-latest | LF | failure | 24 + 64 OK | step 10 | `quadrature-r1-windows-latest-lf`（4,824 B） |

- 每个 job 都先在 step 7/8 运行 `python -S -m unittest discover -s tests -p "test_pr72*.py"` 等测试；日志显示 **24 tests OK** 和 **64 tests OK**，**0 skip**。
- 三个失败 job 都在 step 10 “Same-state four-case checkpoint reconstruction (ZERO true calls)” 失败；step 11 read-only reconciliation 被跳过；step 12 仍保存 failure 证据。
- 失败 artifact 只有 `failure.json`、`identity_source_receipt.json`、`manifest.json`、`plan.json`、`source_receipt.json`；无 `report.json`、无 states。

> 注意：任务描述中的“23 tests”与本次 run 日志的 **24 + 64** 不一致；以日志和 artifact 为准。Windows 成功 artifact ZIP 实际为 **368,930 B**，与任务描述中的 139,624 B 不一致。

## 3. 三个失败 job 的首个异常

三个失败报告均在同一位置抛出同一个 `ValueError`：

```
run_pr70_quadrature_diagnostic.py:170 run()
  -> run_pr70_quadrature_diagnostic.py:103 restore()
  -> quadrature_checkpoint_features.py:83 restore_features()
ValueError: Archived codebook also fails EXACT historical feature identity: {...}
```

| job | failure type | completed_states | 首个异常位置 | native_objective_identity | 关键差异 |
|---|---|---:|---|---|---|
| ubuntu-latest, LF | ValueError | 0 | `quadrature_checkpoint_features.py:83` | `a34dd6be...aea4a` | 与 Ubuntu CRLF 相同 |
| ubuntu-latest, CRLF | ValueError | 0 | `quadrature_checkpoint_features.py:83` | `a34dd6be...aea4a` | 与 Ubuntu LF 相同 |
| windows-latest, LF | ValueError | 0 | `quadrature_checkpoint_features.py:83` | `8850826a...49f85` | Windows LF 独立 identity |

共同证据：

- `recorded_objective_identity = 5caa6360...aad8b5`；
- `restored_objective_identity = 5d6afa72...bbdbe`（三组一致，均不匹配 recorded）；
- `mode = sealed_archived_codebook`；
- `max_abs_native_forward_difference = 7.105427357601002e-15`；
- `vae_training = false`，`recorded_identity_assigned_without_reconstruction = false`；
- `failure.json.bill`：`gp_started=gp_completed=0`，`new_true_evaluations=0`，`new_vae_training=0`，`aircraft_calls=0`，`rcaide_calls=0`，`phase=prefix_gp_reconstruction`；
- `completed_states = 0`。

结论：**三个失败 job 并非在测试或 EOL source-contract 阶段失败，而是在历史 feature identity 重建阶段失败。** Ubuntu LF 与 CRLF 的首个异常相同；Windows LF 的异常类型/消息相同，但 `native_objective_identity` 与其余三组不同，说明不能把所有失败简单归为同一个 LF/CRLF 原因。

## 4. Windows CRLF 成功 artifact

成功 job 输出 `quadrature-r1-windows-latest-crlf`，`report.json`：

- `status = COMPLETE_SAME_STATE_QUADRATURE_DIAGNOSTIC`
- `profile = smoke`，`state_count = 4`（Branin/Rosenbrock × ordinary/sparse）
- 每个 state 的 `n4_r0`、`n16_r0`、`n64_r0`、`n64_r1` 四个 rule 状态均为 `OK`
- `bill.gp_started = bill.gp_completed = 8,736`，`gp_errors = 0`
- `elapsed_seconds ≈ 200.7`
- `new_true_evaluations = 0`，`new_vae_training = 0`，`aircraft_calls = 0`，`rcaide_calls = 0`
- `source_identity_schema = pr72_pinned_lf_crlf_identity_v1`
- 使用 `identity_source_receipt.json`；`identity_source_receipt_sha256 = 2c0a2793...bf99`
- artifact 共 25 个文件，其中 24 个文件列入 manifest 且全部通过 SHA256 校验。

### 成功 artifact 的 schema 与任务预期不一致

成功 artifact **不包含**独立的：

- `source_identity.json`
- `contract_binding.json`

它包含的是：

- 顶层 `identity_source_receipt.json`（历史/本地 LF、CRLF、blob SHA1 与 origin 证据）；
- 每个 `states/<case>_<seed>_<iter>/snapshot.json` 内的 `contract_identity` 与 `checkpoint_feature_replay`；
- 顶层 `report.json` 的 `identity_source_receipt_sha256` 与 `source_identity_schema`。

因此，“Windows 成功 artifact 是否包含 source_identity.json/contract_binding.json” 的答案是：**不包含**。如果验收要求这两个独立文件，则当前成功 artifact 不满足该 schema；2defbb2 的实现把身份信息放进了 `identity_source_receipt.json` 和 snapshot 的 `contract_identity` 中。

## 5. 判读

- R1 的 source-identity 测试在四个平台/EOL 组合上均通过（24 + 64 tests），但只有 Windows CRLF 走通了 checkpoint reconstruction。
- 三个失败 job 的 `failure.json` 都是同一个“archived codebook 仍不匹配 historical feature identity”的异常；不是哈希、下载、manifest 或测试失败。
- Ubuntu LF 与 Ubuntu CRLF 的 `native_objective_identity` 相同；Windows LF 不同。这个差异是当前最重要的未解释事实，不能仅靠“再跑一次 CRLF”消除。
- 成功 artifact 是 **smoke** 技术重建，不是 full；它没有 `source_identity.json`/`contract_binding.json`，不能替代验收 schema。
- 整个 run 的 `conclusion=failure`；Windows CRLF 单 job 成功不表示 PR72 head 已通过四组合验收。

## 6. 本轮没有执行的操作

- 没有修改代码、没有安装/升级依赖；
- 没有运行 smoke/full 或 PR70 历史诊断；
- 没有新增数学评价、GP 拟合、VAE 拟合、飞机或 RCAIDE 调用；
- 仅通过 GitHub API 下载 job、artifact 和日志。

## 7. 文件

- `api/run.json`、`api/jobs.json`、`api/artifacts.json`、`api/ci_summary.json`：API 原始响应与本地汇总。
- `jobs/*.json`：四个 job 的步骤、状态、时间。
- `logs/*.log`：四个 job 的完整 GitHub Actions 日志。
- `artifacts/*.zip`：四个原始 artifact ZIP。
- `artifacts/<name>/*`：解压内容；失败 job 为 failure/identity/manifest/plan/source，成功 job 另含 report 与 states。
- `instructions.txt`：本轮任务指令副本。
- `sha256_manifest.txt`：本目录全部文件的 SHA256。
- `.gitattributes`：`* -text`。
