# PR106 — X-57 archived feedback audit（只读）

## 固定身份

- 代码仓库：`Luyi1235/RA-LSBO`
- 分支：`research/aircraft-risk-x57-feedback-audit-20261115`
- 固定代码 SHA：`61c265c626957cb8d69761087270721b846025b5`
- 结果分支：`results/aircraft-risk-x57-feedback-audit-20261115-61c265c`
- 审计源：`D:/LSBO/RA-LSBO/runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025`

## 验收与可复现性

- Python：`C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`，3.9.25。
- 测试：`Ran 2 tests ... OK`。
- 审计：`AUDITED_ARCHIVED_X57_FEEDBACK`，37 条归档调用；`new_aircraft_calls=0`，`new_rcaide_mda_calls=0`。
- 独立重复运行：`report.json` 与 `report-repeat.json` 逐字节相同（`cmp` 通过）；因此结果可由归档证据重复导出。
- 报告为只读归档审计，不触发任何 RCAIDE/MDA 评价。

## 按距离的结果

| 距离 | 已计费调用 | Hidden failure | 其中带可判定非解析符号 | 未结算 | Infeasible | 可用解 |
|---|---:|---:|---:|---:|---:|---:|
| 55 nmi | 30 | 16 | 0 | 0 | 14 | 0 |
| 65 nmi | 7 | 5 | 0 | 1 | 1 | 0 |

`Hidden failure` 的 `raw_status` 均为 `evaluation_error:DomainSolveError`；`hidden_failure_modes` 均为 `mission_solver_nonconvergence`（55:16，65:5）。

## 非解析反馈可用性

- 55 nmi：16 条 hidden failure 中，`battery_current`、`battery_voltage`、`soc`、`stall_speed`、`structure`、`usable_energy` 六个非解析约束的 `kind` 全为 `missing`，`sign=null`；带任一可判定符号的 hidden failure 为 **0/16**。
- 65 nmi：5 条 hidden failure 中，同样六项全为 `missing`；带任一可判定符号为 **0/5**。
- 未结算：65 nmi 有 **1** 条 `unresolved_started`（`phase_call` 记录存在但没有完成状态）；55 nmi 无未结算。
- 所有 hidden failure 都没有可用的非解析符号反馈；因此本轮不能支持“按约束符号做 hidden-failure 条件采集”的证据主张。

## 源文件校验

审计报告记录的源文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `observations.jsonl` | `dd5d7e6234d759c7053d8500203d3a90c54391a777b6514341407daa8974e28a` |
| `plan.json` | `087f587efcec17f966ffb9723821919f7fd52860a3e3a356933f893e5b381bdf` |
| `submissions.jsonl` | `b289acf2b6b6fe3565af602ea572ce8297fa497231d6417af5ed099d3c194d6d` |
| `task_55/budget.jsonl` | `4b586435c31bd829848459a8fde282215aafb1e30e19fe2fd5213770de16729a` |
| `task_65/budget.jsonl` | `084e5e1bb9c6e7475051a9b76c0460adea1c83a9853736ea2eacf5d499a26301` |

这些哈希在两次独立审计中一致；源快照未被修改。

## 判读边界

- 这是归档证据的只读复用，不是新的 X-57/RCAIDE 评价，也不证明模型或算法效果。
- 55/65 nmi 的 hidden failure 均由 mission solver 不收敛触发，且没有可判定的非解析约束符号；只能说明当前归档不适合做“符号反馈”机制验证，不能说明该机制在真实可判符号数据上无效。
- 65 nmi 的 1 条未结算记录单独计数，不并入 hidden failure 或 infeasible。

## 目录

```
report.json             审计报告（37 calls + 两个 case 汇总）
report-repeat.json      独立重复审计报告（与 report.json 相同）
audit.log               第一次审计 stdout/stderr
audit-repeat.log        重复审计 stdout/stderr
tests.log               2 项专项测试
python-version.log      解释器与版本断言
fixed_sha.txt           固定代码 SHA
SHA256SUMS.txt          除自身外全部文件
```
