# PR79：PR78 Phase B1 瓶颈只读审计的 CI 归档

本目录只归档 PR79 已完成的两平台只读诊断 artifact。没有重新运行 PR78/PR79，没有触发 CI，没有拟合模型，也没有调用真值评价器。

## 1. 固定身份

| 项目 | 值 |
|---|---|
| 仓库 | `Luyi1235/RA-LSBO` |
| PR | 79 |
| 固定审计代码 SHA | `27e9724150961b44268bd6672dc41abe32b8455b` |
| 输入 PR78 结果 HEAD | `5ff896b387a7506991ee356d9bfc981a8e080e6a` |
| 最终 CI run | `42496481864` |
| run 结论 | `success` |
| Ubuntu job | `readonly (ubuntu-latest, 3.13)`，success |
| Windows job | `readonly (windows-latest, 3.9)`，success |

归档前检查了本机与远端：没有同 run/artifact 的既有完整归档。

## 2. 最终 artifact

| 平台 | 名称 | artifact ID | ZIP 大小 | ZIP SHA256 | ZIP 完整性 |
|---|---|---|---:|---|---|
| Ubuntu | `b1-bottleneck-ubuntu-latest` | `13623334471` | 891,011 B | `1bea88e5...44ab` | 通过 |
| Windows | `b1-bottleneck-windows-latest` | `13623229838` | 891,011 B | `5b860d47...ff27` | 通过 |

两个 ZIP 的压缩字节不同，解压后的 4 个文件逐字节相同。

## 3. manifest 与跨平台一致性

每个平台包含 `manifest.json` + 3 个数据文件；manifest 的 3 条 SHA256 全部匹配：

| 文件 | SHA256（Windows = Ubuntu） |
|---|---|
| `plan.json` | `a242e764...6359` |
| `report.json` | `3a7345e0...97de` |
| `states.json` | `836962cb...96d9` |
| `manifest.json` | `a376f641...b4f3` |

两个平台解压后的 **4/4 文件逐字节相同**。

## 4. report 核验

`report.json`：

- `status = COMPLETE_READ_ONLY_B1_BOTTLENECK_AUDIT`
- `audit_code_sha = 27e9724150961b44268bd6672dc41abe32b8455b`
- `input_result_sha = 5ff896b387a7506991ee356d9bfc981a8e080e6a`
- `model_states = 144`
- `new_bill` 全部为 0：
  - `new_true_evaluations=0`
  - `new_gp_fits=0`
  - `new_vae_fits=0`
  - `new_fantasy_fits=0`
  - `aircraft_calls=0`
  - `rcaide_calls=0`

`historical_verification`：

- `status = VERIFIED_FIXED_OBJECTIVE_DECISIONS`
- `files = 438`
- `model_states = 144`
- `failed_states = 0`
- `math_started = math_settled = 21,504`
- `gp_head_started = gp_head_completed = 425`，0 errors
- `gp_kernel_started = 3,400`、`gp_kernel_completed = 3,399`、`gp_kernel_errors = 1`
- `kernel_candidates_rejected = 1`
- `rankings_sealed_before_truth = true`

**438 文件 / 21,504 评价 / 425 GP 头 / 3,400 核 / 1 次核错误都属于 PR78 历史记录，不是本次归档新增计算。**

## 5. 开发费用事件（单独保留）

首版审计曾意外触发旧技术集成：

- run：`42496109213`
- 代码 SHA：`8313122edfb96c13a6ba598fe10eb4128832feb7`
- 结论：success；三个 job（actual-shared-model windows/ubuntu、cross-python-verifier）均 success
- Windows artifact：`13623234410`，97,823 B
- Ubuntu artifact：`13623124553`，97,872 B

两平台各自的 `ci/report.json` bill 均为 160 次数学评价、11 个 GP head、88 次核构造；合计 **320 次数学评价、22 个 GP head、176 次核构造**。这属于开发费用事件，不是最终只读诊断的科学结果，也不进入 `new_bill`。

原始 ZIP、解压文件、CI 日志与 run 元数据在 `dev/` 和 `metadata/dev/`，与最终诊断隔离。

## 6. 目录

```
final/                 最终两平台 ZIP 与解压文件
  windows/             plan/report/states/manifest
  ubuntu/              同 4 个文件
dev/                   开发费用事件两平台 ZIP 与解压复现
metadata/final/        run/jobs/artifacts JSON
metadata/dev/          事件 run/jobs/artifacts JSON
logs/final-ci-logs.zip 最终 run 完整日志
logs/final/            解压后的最终 CI 日志
logs/dev-ci-logs.zip   开发费用事件完整日志
logs/dev/              解压后的事件日志
verification_summary.json
fixed_sha.txt
instructions.txt
README.md
sha256_manifest.txt
```

证据目录使用 `.gitattributes` 固定 `* -text`。

## 7. 边界

- 本归档不证明 PR79 回放之外的在线算法收益；只证明最终 artifact 与固定审计代码/PR78 结果绑定、可读、可核验。
- 开发费用事件被保留而不是删除；它不能混入最终 `new_bill`，也不能当作新的科学实验。
- 本轮工作机新增真值/GP/VAE/幻想/飞机/RCAIDE 调用均为 **0**。
