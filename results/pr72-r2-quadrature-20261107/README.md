# PR72 R2：固定状态真实 GP 四节点/十六节点/六十四节点求积一致性诊断

固定代码 SHA：`932f7202cf81890a108e37266bbb639affba0ae7`  
输入结果 HEAD：`f859cca5adc56dd101d2f59665f7829a2b0331c4`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/72>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录只包含零新增真实评价的**固定状态求积一致性诊断**：读取 PR70 的已封存 32 个历史状态，在同一状态、同一候选池、同一 GP/VAE 前缀上比较 `n4_r0`、`n16_r0`、`n64_r0`、`n64_r1` 四个求积规则。没有重跑优化器，没有新增真实评价、VAE 训练或飞机/RCAIDE 调用。

## 1. 身份与完整性

- 代码 worktree：`D:\LSBO\RA-LSBO-pr72-r2-code-20261107-134400`，HEAD = 固定 SHA，`git status --porcelain` 为空。
- 证据 worktree：`D:\LSBO\RA-LSBO-pr72-r2-evidence-20261107-134400`，HEAD = `f859cca...`，干净。
- 输入历史：309 个文件、48 trials、480 次真实评价（192 initial + 288 online）、80 次未来池检查；运行前后 manifest 均为 `1f681abc...d9a0726`，输入未变化。
- 精确特征参考：`feature_reference.json` 1,331,282 B，SHA256 `0fc0a6b0...e4396`；生产代码 SHA `932f720`；`quadrature_feature_reference.py --task verify` 返回 `VERIFIED_EXACT_FEATURE_REFERENCE`，32 states。
- CI 成功 run `42336743192`，artifact `pr72-exact-feature-reference`（id `13560260956`，144,188 B，SHA256 `f312c4b4...a347`）。
- 预检：22 项 reference 测试、86 项 quadrature 测试、compileall、`--help` 全部 exit 0，0 skip。
- smoke：4 states，8,736 GP fits（started=completed，0 errors），状态 `COMPLETE_SAME_STATE_QUADRATURE_DIAGNOSTIC`；verify 25 files。
- full：32 states，83,582 GP fits（started=completed，0 errors），状态 `COMPLETE_SAME_STATE_QUADRATURE_DIAGNOSTIC`，1,389.4 s；verify 165 files，`manifest_sha256=ac444f60...596dce`。
- 新增真实评价 / VAE 训练 / 飞机 / RCAIDE：全部 **0**。

## 2. 四规则总览

| 规则 | 节点 | full GP fits | full 秒 | 相对 n4 的拟合倍数 |
|---|---:|---:|---:|---:|
| `n4_r0` | 4 | 2,234 | 37.9 | 1.0× |
| `n16_r0` | 16 | 8,936 | 144.7 | 4.0× |
| `n64_r0` | 64 | 35,710 | 567.8 | 16.0× |
| `n64_r1`（打乱节点） | 64 | 35,710 | 567.4 | 16.0× |

full 的 83,582 次 GP 拟合全部成功，0 次数值错误；另有 992 次前缀 GP 重建，0 错误。

## 3. 动作变化与分数漂移

每 case 8 个状态；`winner_changes` 是同一状态下首动作选择改变的次数；漂移为 `normalized_max_score_drift`。

| case | states / cold | n4→n16 改选 | 漂移 | n4→n64 改选 | 漂移 | n16→n64 改选 | 漂移 | n64→打乱 改选 | 漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Branin ordinary | 8 / 0 | 0 | 0.0648 | 0 | 0.0861 | 0 | 0.0228 | 0 | 0.0042 |
| Branin sparse | 8 / 4 | 0 | 0.1962 | 0 | 0.2161 | 0 | 0.0305 | 0 | 0.0098 |
| Rosenbrock ordinary | 8 / 0 | 0 | 0.0522 | 0 | 0.0534 | 0 | 0.0131 | 0 | 0.0103 |
| Rosenbrock sparse | 8 / 3 | **2** | 0.0993 | **2** | 0.1091 | 0 | 0.0204 | 0 | 0.0045 |

- **32/32 状态的候选短名单完全相同**（每种规则 `first_indices` 一致）。
- **30/32 状态四种规则选中同一个首动作**；仅 Rosenbrock sparse 的两个冷启动/无解状态不同。
- **`n16` 与 `n64` 在 32/32 状态选中相同动作，`n64` 的两个打乱复制也 32/32 相同**。
- n4 的两个改选都发生在 Rosenbrock sparse 的冷启动近邻状态：
  - seed `998901` iter 0：n4 选 68，n16/n64 选 101；n4 winner gap 0.00397，n64 0.00252；
  - seed `998901` iter 3：n4 选 11，n16/n64 选 113；n4 winner gap 3.3e-5，n64 8.0e-4。
- 出现改选时，较高节点规则对“低节点选择”的归一化损失为 `0.003829`、`0.001828`（n4→n16），`0.003793`、`0.000855`（n4→n64）；均为极小近邻差异，不是大幅动作翻转。

## 4. 分数与估计优势

`estimated_advantage_over_myopic` 是规则自己的分数估计，不是真实评价收益；每 case 8 状态的均值：

| case | n4 | n16 | n64 | n64 打乱 |
|---|---:|---:|---:|---:|
| Branin ordinary | 0.0059 | 0.0035 | 0.0041 | 0.0041 |
| Branin sparse | 0.0 | 0.0 | 0.0 | 0.0 |
| Rosenbrock ordinary | 2.2175 | 0.9818 | 0.9093 | 0.9178 |
| Rosenbrock sparse | 0.0056 | 0.0022 | 0.0021 | 0.0023 |

- 在 Rosenbrock ordinary，n4 的估计优势约为 n16/n64 的 2.2–2.4 倍；这只是分数估计随节点数变化，不能解释为 n4 的真实优化收益。
- 稀疏任务的估计优势接近 0，说明在这些已封存状态上四规则都没有明显“更好动作”的分数信号。
- 所有比较都是固定状态、固定候选池；没有新的真实后续评价，因此不能把这些数值当作在线算法收益。

## 5. 逐任务诊断

### Branin ordinary

- 8/8 状态均有可行 `best`，无冷启动。
- n4→n16、n4→n64、n16→n64 动作改变均为 0；最大漂移 0.086（n4→n64）。
- 四规则选点相同；节点数只改变分数尺度，没有改变动作。
- 估计优势很小（0.0035–0.0059），不存在可报告的求积收益。

### Branin sparse

- 8 个状态中 4 个无可行 `best`（冷启动）。
- 三种节点比较动作改变均为 0；但分数漂移最大（n4→n64 0.216），说明分数对节点数敏感、动作不敏感。
- 四规则都选中同一动作；没有因为 64 节点而避免冷启动。

### Rosenbrock ordinary

- 8/8 状态均有可行 `best`，无冷启动。
- 动作改变 0；最大漂移 0.053。
- n4 的估计优势（2.22）高于 n16/n64（0.98/0.91），但选点相同；这更像粗节点下的分数放大，而不是更好的决策。
- 没有可验证的动作或真实收益差异。

### Rosenbrock sparse

- 8 个状态中 3 个无解；是唯一出现动作变化的 case。
- n4 与 n16/n64 在 2/8 状态改选，n16 与 n64 仍然 0/8；两个改选都在 seed `998901` 的冷启动近邻状态。
- 改动很小：n4 的 winner gap 分别为 0.00397 和 3.3e-5；n64 分别为 0.00252 和 8.0e-4。较高节点规则对低节点选择的损失为 0.0009–0.0038。
- 因此“4 节点会改变动作”只在极近邻且无解的状态出现；在本样本中未改变最终任务可行性（这些状态本来就没有 `best`）。

## 6. 与真实后续搜索的关系

本实验只回答“同一已封存状态上，减少/增加求积节点是否改变首动作与分数”，**不回答**：

- 被改选的动作在真实后续 6 次评价中是否更好；
- 四节点估计是否在真实在线路径上更稳；
- 64 节点是否是收敛真值；
- 飞机/RCAIDE 任务的泛化。

因此不能把“n16 与 n64 动作一致”写成“16 节点已经足够”，也不能把“n4 只在 2 个近邻状态改选”写成“4 节点无损”。它们只是固定状态的一致性证据。

## 7. 文件

- `reference/`：artifact ZIP、`feature_reference.json`、manifest、CI artifact/run JSON。
- `smoke/`、`full/`：完整原始输出，包含 report、plan、source_receipt、identity_source_receipt、progress、states、manifest。
- `logs/`：环境、引用验证、22+86 测试、compileall、help、smoke/full 控制台与 verify。
- `analysis_summary.json`：机器可读汇总；`case_table.csv`：逐 case 表；`integrity_check.json`：完整性；`sha256_manifest.txt`：上传文件哈希。
- `protocol.txt`、`instructions.txt`：固定协议与任务指令副本。
- `.gitattributes`：`* -text`，避免换行转换破坏证据字节。

### 限制

- 仅 32 个已封存状态（4 case × 8），不是新的在线实验；没有新的真实后续评价。
- 64 节点不是 ground truth；分数漂移不能解释为收敛误差。
- 两个 64 节点打乱复制只能说明实现稳定性，不是置信区间。
- 只覆盖模型分数/动作，不覆盖真实后续搜索收益、飞机或 RCAIDE。
