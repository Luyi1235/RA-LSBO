# PR56：PR55 报告聚合修复（只读审计）

固定审计代码：`15374a1d1af05e7461056b1f80ba9aff544d708f`  
源 PR55 实验 SHA：`d947db5f0b544739c9db286a75d99c0308713fdb`  
源 PR55 报告 SHA256：`583d21106615cf05d8ec6c0a96a20d5a55f01de341efe7af76ee441139aa23f9`  
修正报告 SHA256：`1b297db3db130245731e2dd17cb1ec20bc39ecc7cff2535ed099b56aa5a0ef9a`  
环境：`Rcaide_vsp` / Python 3.9.25。

## 命令

```bash
python -m ra_lsbo.experiments.risk_budget56_report_audit \
  --source "D:\LSBO\RA-LSBO-pr55-independent-budget-math-full-20261104\risk_budget55_report.json" \
  --output "D:\LSBO\RA-LSBO-pr56-corrected-20261104\risk_budget55_report_corrected_pr56.json"
```

exit 0。输出路径为全新目录，**未覆盖**原始 PR55 报告。

## 审计结果

| 任务/算法 | 有效预测收据 | 预算回退 | 回退中实际失败 |
|---|---:|---:|---:|
| Branin sparse / budgeted | **148** | **17** | **12 (70.6%)** |
| Rosenbrock sparse / budgeted | **149** | **22** | **16 (72.7%)** |
| Branin sparse / semantic_only | 123 | 0 | — |
| Rosenbrock sparse / semantic_only | 137 | 0 | — |
| 两个 ordinary 的 budgeted | 192 | 0 | — |
| legacy（全部任务） | 0 | 0 | — |

修正报告中的 `method_summary_by_case` 现在与原始 `pre_evaluation_risk_receipts` 一致：

- `predicted_risk_receipt_coverage`：Branin sparse budgeted 148、Rosenbrock sparse budgeted 149；
- `predicted_allowance_gate_fallbacks`：17 / 22；
- `observed_nonfeasible_in_predicted_gate_fallbacks`：12 / 16；
- `predicted_receipt_actual_nonfeasible_rate`：0.4662 / 0.4497；
- 两个 ordinary 任务的回退数均为 0。

根因已由模块记录：

```text
PR55 runner writes prediction_available_for_selected;
original summary incorrectly tested forecast_available
```

`report_aggregation_audit` 记录：

- `trial_details_count=192`
- `aggregation_only=true`
- `new_analytical_evaluations=0`
- `new_aircraft_evaluations=0`
- `new_rcaide_evaluations=0`
- `original_report_modified=false`
- `source_proposals_or_observations_modified=false`
- `summary_changed=true`

## 原始证据不变性

- 运行前后源报告 SHA256 均为 `583d2110...`，一致；
- 源报告路径、PR55 的 `trials/`、`shared_initial/` 均未被写入；
- 本次只读取 `risk_budget55_report.json` 的 `trial_details`，从 192 个 trial 的 pre-evaluation receipts 重新聚合；
- 没有运行 GP/VAE、数学评价器、飞机或 RCAIDE。

## 代码/测试异常

按要求工作机不修改代码。固定 SHA 的模块可以通过 `python -m ra_lsbo.experiments.risk_budget56_report_audit` 正常执行，但同 SHA 的测试文件存在导入名错误：

```text
tests/test_risk_budget56_report_audit.py:6
from ra_lsbo.experiments.risk_budget55_report_audit import audit
ModuleNotFoundError: No module named 'ra_lsbo.experiments.risk_budget55_report_audit'
```

实际模块文件名为 `risk_budget56_report_audit.py`，不存在 `risk_budget55_report_audit.py`；因此 pytest 在收集阶段失败，0 个测试执行。该问题不影响本次 `-m` 命令的聚合结果，但测试门禁在该固定 SHA 上不可用。完整 pytest 输出见 `pr56_logs_tests.log`。

## 文件

- `risk_budget55_report_corrected_pr56.json`：修正后的完整报告（1.8 MB）。
- `audit-console.log`：`-m` 命令完整 stdout/stderr。
- `pr56_logs_tests.log`：pytest 收集失败日志（异常记录）。
- `audit_metadata.json`：`report_aggregation_audit` 字段。
- `sha256sums.txt`：修正报告、两轮日志的 SHA256。
- `fixed_sha.txt`：固定 SHA 与关键身份字段。
- `README.md`、`.gitattributes`：说明与字节保真设置。
