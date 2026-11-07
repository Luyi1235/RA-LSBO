# PR78 Phase B1：共享模型下的目标决策价值检验

固定代码 SHA：`ab12650769e685966661648940301f330a84993e`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/78>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录是同数据、同目标 GP、同风险概率、同候选池下的**固定排名诊断**，不是新的在线优化算法，也不是 qEI、批量优化或安全方法。

## 1. 执行与身份

- 代码 worktree：`D:\LSBO\RA-LSBO-pr78-20261107-225607`，HEAD = 固定 SHA，干净。
- 结果根：`D:\LSBO\RA-LSBO-pr78-results-20261107-225607`；smoke/full 各自独立目录。
- 预检：34 项 core 测试、12 项 native 测试通过，0 skip；compileall、`--help`、`--task plan --profile full` 均 exit 0。
- smoke：12 states，672 calls = 288 train + 384 pool audit；状态 `COMPLETE_FIXED_OBJECTIVE_DECISIONS`，failed_states=0；verify 48 files；六类重封 manifest 篡改全部拒绝。
- full：144 states，21,504 calls = 9,216 train + 12,288 pool audit；状态 `COMPLETE_FIXED_OBJECTIVE_DECISIONS`，failed_states=0，509.2 s；verify 438 files；六类篡改全部拒绝。
- 全部状态：`rankings_sealed_before_truth=true`；math started=settled=21,504；0 未知/中断。
- 飞机/RCAIDE/VAE/幻想：全部 0；真实 GP 拟合按账本执行。

## 2. 冻结策略

四个策略在同一状态共享完全相同的训练前缀、目标预测、风险预测、候选池与目标归一化：

| 策略 | 排名规则 |
|---|---|
| `uniform_hash` | 与结果无关的 salted hash 排名，一个确定性 uniform-like 实现 |
| `risk_only` | 学到的联合可行概率降序 |
| `objective_ei` | 对最佳已观察可行目标的普通标量 GP-EI |
| `risk_ei` | 同一 EI 乘以同一联合可行概率 |

预算 1/2/4/6/12 是同一排名的前缀，不是独立重复。Cold 状态下 `objective_ei` 与 `risk_ei` 使用相同的 `uniform_hash` 回退。

## 3. 主分析：n=48，k=6，initial_warm

| 任务 | warm / cold | risk_ei vs objective_ei W/L/T | Δ 均值（risk−obj） | 改善次数 risk/obj | 有改进机会 / 零机会 | 排名改变 warm | 选中可行（risk/obj，共6×warm） | 初始目标均值 | 最终目标均值（risk） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tradeoff_s0 | 8 / 0 | 0 / 0 / 8 | 0 | 3 / 3 | 5 / 3 | 8/8 | 18/48 vs 12/48 | 0.5448 | 0.3422 |
| tradeoff_s1 | 8 / 0 | 1 / 0 / 7 | −0.007545 | 5 / 5 | 7 / 1 | 8/8 | 32/48 vs 13/48 | 1.0125 | 0.6500 |
| tradeoff_s2 | 7 / 1 | 1 / 0 / 6 | −0.082169 | 2 / 1 | 4 / 3 | 7/7 | 8/42 vs 2/42 | 1.0088 | 0.9130 |
| tradeoff_s3 | 7 / 1 | 2 / 0 / 5 | −0.167251 | 2 / 0 | 7 / 0 | 7/7 | 4/42 vs 0/42 | 1.5995 | 1.4323 |
| **合计** | **30 / 2** | **4 / 0 / 26** | — | **12 / 9** | **23 / 7** | **30/30** | **62/180 vs 27/180** | — | — |

- 主比较 **4 胜、0 负、26 平**（warm）；包含 2 个 cold 的全样本为 **4/0/28**，cold 两对均为 uniform 回退，不解释为 EI 能力。
- 四个任务的逐重复目标差（risk−objective；0 为平局）：
  - s0：`[0, 0, 0, 0, 0, 0, 0, 0]`
  - s1：`[0, 0, 0, 0, 0, 0, 0, −0.060357]`
  - s2：`[0, 0, 0, 0, −0.575186, 0, 0]`
  - s3：`[0, −0.788364, 0, 0, 0, −0.382395, 0]`
- 没有一次 risk_ei 的最终目标比 objective_ei 差；但 26/30 warm 对完全平局，改善集中在 s1 的 1 个、s2 的 1 个、s3 的 2 个重复。
- **风险筛选明显提高了“选中可行点”的比例**（62/180 vs 27/180，risk-only 为 146/180，uniform 为 7/180），但最终目标只在少数重复上改善。这符合“风险筛选改善、目标收益未建立”的谨慎表述。
- 30/30 warm 状态的排序都改变了，23/30 的池内存在正改进机会；**7 个零机会组仍保留在表中**，没有被删除或替换。

### 逐任务说明

**tradeoff_s0**：8/8 warm；所有重复最终目标相同；risk 选中 18/48 可行点 vs objective 12/48；5/8 池有正机会、3/8 零机会。风险信息只改善筛选，未改善最终目标。

**tradeoff_s1**：8/8 warm；1 胜 7 平；唯一改善 −0.060357；risk 选中 32/48 vs objective 13/48；7/8 有正机会。

**tradeoff_s2**：7 warm + 1 cold；warm 1 胜 6 平；改善 −0.575186；risk 选中 8/42 vs objective 2/42；4/7 有正机会，3/7 零机会。

**tradeoff_s3**：7 warm + 1 cold；warm 2 胜 5 平；改善 −0.788364、−0.382395；risk 选中 4/42 vs objective 0/42；7/7 有正机会。

## 4. 次要分析

### 训练量（k=6，tradeoff 四任务汇总）

| n | warm | W/L/T（risk vs objective） |
|---:|---:|---:|
| 12 | 25 | 7 / 0 / 18 |
| 48 | 30 | 4 / 0 / 26 |
| 192 | 32 | 11 / 0 / 21 |

### 预算前缀（n=48，tradeoff 四任务汇总）

| k | W/L/T（risk vs objective） |
|---:|---:|
| 1 | 2 / 2 / 26 |
| 2 | 5 / 2 / 23 |
| 4 | 5 / 0 / 25 |
| 6 | 4 / 0 / 26 |
| 12 | 7 / 3 / 20 |

k=1、2、12 出现少量负例；k=4、6 没有负例，但多数仍是平局。预算前缀不是独立实验。

### 两个 legacy 稀疏任务（n=48，k=6）

| 任务 | warm / cold | W/L/T | Δ 均值 | 选中可行 risk/obj |
|---|---:|---:|---:|---:|
| legacy_branin_sparse | 7 / 1 | 2 / 1 / 5 | −2.6221 | 29 vs 27（共42） |
| legacy_rosenbrock_sparse | 8 / 0 | 2 / 0 / 6 | −4.7453 | 24 vs 15（共48） |

legacy 任务上 risk_ei 有较小的目标优势，但样本与任务数很少，不能外推。

## 5. GP 与计算账本

| 项目 | full |
|---|---:|
| 预算上限 GP heads | 432 |
| 实际 GP head started / completed / errors | 425 / 425 / 0 |
| 预算上限 kernels | 3,456 |
| 实际 kernel started / completed / errors | 3,400 / 3,399 / 1 |
| 选核拒绝 | 1 |
| 失败模型状态 | 0 |
| 数学评价 started / settled | 21,504 / 21,504 |
| 端到端耗时 | 509.2 s |

- 425 个 GP head 少于 432，是因为部分状态没有可观测目标、目标头不拟合；实际数量以账本为准。
- 有 1 个 kernel 候选被拒绝并记录；对应状态仍成功，full 没有 failed state，因此状态为 `COMPLETE_FIXED_OBJECTIVE_DECISIONS`，不是 `COMPLETE_WITH_MODEL_FAILURES`。
- 原始报告未分别记录“选核耗时”；只记录端到端 509.2 s，不能编造更细的分项时间。

## 6. 机制与完整性

- 同一状态四策略共享训练前缀、目标预测、风险预测、候选池和目标归一化；`objective_proxy_shared=true`、`risk_proxy_shared=true`、`no_truth_or_target_input=true`。
- 所有 n/策略排名在 pool truth 之前封存；`rankings_sealed_before_truth=true`。
- 六类重封篡改（incumbent、ranking、risk_probability、ledger、seal_order、objective_labels）全部拒绝；原始文件未被修改。
- `uniform_hash` 不是精确 uniform-random 期望；`risk_ei` 是 EI×可行性，不是新方法或安全保证。
- 目标 GP 不使用隐藏目标；completed-but-resource-infeasible 的真实目标可作为训练标签；中止设计不插补目标。

## 7. 判读

**风险信息对“选出可行点”有清晰作用，但对最终可行目标只有小幅、集中在少数重复的增益；不能据此把 risk_ei 提升为默认方法或宣称新的两阶段规划机制。**

- 主比较无负例但 26/30 为平局；4 个改善对分别来自 s1、s2、s3 的少数重复。
- 7/30 零改进机会组全部保留，说明并非所有池都能从风险重排获益。
- risk_only 选中 146/180 可行点，但 risk_only 的最终目标没有作为主终点；可行点更多不等于最终目标更好。
- cold 两对使用相同 uniform 回退，不能解释为 EI 的首次可行发现能力。
- 训练量/预算次要结果方向不一致，且共享数据不是独立重复；不能做总体均值或显著性外推。
- 无飞机/RCAIDE，不能外推到 100/140 km 或安全约束。

## 8. 文件

- `smoke/`、`full/`：完整原始输出（plan、source_receipt、environment、datasets、states、scores、evaluations、timeline、manifest、report）。
- `logs/`：34+12 测试、compileall、help、plan、smoke/full 控制台、verify、六类篡改检查。
- `analysis_summary.json`：主/次分析机器可读汇总；`README.md`：本说明。
- `protocol.txt`、`instructions.txt`：固定协议与任务指令副本。
- `fixed_sha.txt`、`environment.txt`、`environment_packages.txt`、`sha256_manifest.txt`。
- `.gitattributes`：`* -text`，防止换行转换破坏证据字节。
