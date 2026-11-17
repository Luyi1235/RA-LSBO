# PR119 停止证据消融（reduced aircraft）

固定代码 SHA：`2e3ccf59749d89cb902493144820543e83835e73`
分支：`research/aircraft-ra-lsbo-stop-evidence-20261117`

## 设计

- 两臂：`ra_to_lsbo`（使用停止状态证据）与 `ra_stop_status_to_lsbo`（拟合时遮蔽停止状态证据）。
- 40 个种子/工况；每臂 4 个共享初始点 + 20 次在线评价；无 held-out 面板。
- 初始无解种子才进入 cold primary：headwind 28/40，payload 25/40。
- 0 次 RCAIDE/MDA。

## 调用与验收

| Profile | Calls | Initial | Online | Verify |
|---|---:|---:|---:|---|
| stop-evidence-smoke | 32 | 16 | 16 | VERIFIED_RA_LSBO_COMPARISON |
| stop-evidence-headwind | 1,760 | 160 | 1,600 | VERIFIED_RA_LSBO_COMPARISON |
| stop-evidence-payload | 1,760 | 160 | 1,600 | VERIFIED_RA_LSBO_COMPARISON |

## Primary（cold，capped first-feasible call）

负值表示“使用停止证据”更早。

| 工况 | n | mean | 95% paired |
|---|---:|---:|---|
| headwind | 28 | -2.6429 | [-4.6786, -0.8214] |
| payload | 25 | -2.5600 | [-4.9200, -0.4000] |

## 全 40 种子（参考）

| 工况 | first mean [95%] | discovery mean [95%] | W/L/T | energy mean Wh [95%, n] |
|---|---:|---:|---:|---:|
| headwind | -1.850 [-3.350,-0.525] | +0.075 [-0.025,+0.175] | 12/12/16 | +45.44 [-627.23,+641.30] (33) |
| payload | -1.600 [-3.200,-0.224] | +0.025 [0.000,+0.075] | 15/7/18 | -495.91 [-1376.20,+175.38] (38) |

## stop_evidence_steps

| 工况 | 臂 | fit_with_stop_evidence | fit_with_masked | scheduled_coverage |
|---|---|---:|---:|---:|
| headwind | ra_to_lsbo | 164 | 0 | 52 |
| headwind | ra_stop_status | 0 | 221 | 66 |
| payload | ra_to_lsbo | 138 | 0 | 46 |
| payload | ra_stop_status | 0 | 190 | 57 |

无 `fit_without_stop_evidence`（没有无停止证据的直接拟合回退）。

## 结论边界

- Primary 显示**使用**停止状态证据显著加快 cold 首次可行发现；遮蔽证据后变慢约 2.6 次调用。但终端 W/L/T 与能量区间不支持稳定的终端收益。
- discovery 数量差异很小（headwind +0.075，payload +0.025）。
- 这是 reduced-aircraft 描述性结果，不涉及 RCAIDE，也不支持默认方法替换或安全声明。
