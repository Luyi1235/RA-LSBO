# X-57/RCAIDE 冷启动实验 v1

## 本机复现

在 `D:\RA-LSBO\x57-cold-start` 中执行。每次运行使用新的输出目录。

```powershell
$py = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
& $py -B -m pytest -q tests/test_x57_cold_start.py tests/test_aircraft_structured_risk.py
& $py -B run_x57_cold_start.py run --seed 0 --target-wh 41000 --online 4 --fast-mock --output D:/RA-LSBO/x57-mock-reproduction
& $py -B run_x57_cold_start.py qualification-plan --output D:/RA-LSBO/x57-qualification-reproduction
```

`41000 Wh` 是模拟测试的固定目标。正式任务的能耗目标和种子名单在资格试验后冻结。`run` 默认使用模拟 worker。真实后端需要显式执行参数，资格准备只做预筛、资产核验和名单冻结。

## 源码与环境身份

| 项目 | 核对结果 |
| --- | --- |
| RA-LSBO 开发基点 | `71f1efd8e532d1064acd9512304a8782d9c45cb2` |
| 原 LSBO 参考 | `61c265c626957cb8d69761087270721b846025b5` 中的 `frozen_base/lsbo`。开发基点的冻结优化器多出 `refine_surrogate` 注入接口，基线调用默认原生路径。 |
| 本机工作副本 | `D:\LSBO\LSBO\X57-LSBO-refactored`。`surrogate.py` 与冻结版逐字节一致。`optimizer.py`、`space.py`、`representation.py` 和 `problem.py`、`adapter.py`、`worker.py` 字节哈希有差异。 |
| 本机与采用版优化器 | 本机 `4b3d96b85525704bbe3e1ece4a2562c3142b8006dd8735aabebc8d7e03987a82`，采用版 `257c65b63b12ca94eb5f1c43eb6c4c31a6f3cbaf596141fae9d928ca4af23a92` |
| 评价器 | 冻结 v49，`workspace_v49_nasa_modII_geometry_v2`。模型清单验证 87 个原始文件及初始化函数指纹。Git 检出的 `X57_Analysis.py` 与 `NACA_63_412.dat` 换行字节发生变化。资格计划从本机副本恢复原字节，要求归一化后的内容与冻结树一致且原始 SHA256 符合清单。 |
| 资格配置 | `configs/x57_cold_start_qualification_55.json`，SHA256 `97fa05e6400c86ac71359ca3d53b77ca6dcdec64ffd6e1c4a93e6ebb03321325`。55 nmi 为 v4 任务的巡航段距离，保留储备航段。翼面积范围 9–17 m²，使用冻结版支持的 NASA 电池、审计电气约束与概念化推进模型。 |
| Python 和 RCAIDE | Python 3.9.25，`RCAIDE_LEADS` 1.2.1。RCAIDE 从指定 conda 环境的 `site-packages` 导入，入口模块 SHA256 `22a0372ec741a3d6bc743a00ed26e1e21b3c49d56b100e0e6046cfebc1e8266`。 |
| 代理依赖 | NumPy 2.0.2、SciPy 1.13.1、Torch 2.8.0、BoTorch 0.10.0、GPyTorch 1.11、pytest 8.4.2 |
| 定版风险参考 | `fddfe2e476b09db2a38e8b957646bac4d7017055` 的结构化核、贪心完成概率选点、回退和 `RiskWeightedRefinement`。 |

`plan.json` 和 `qualification_plan.json` 保存完整变量域、分析约束、依赖和采用版源码哈希。资格计划额外包含恢复后的 worker 哈希。已保存的原生 LSBO 观测来自工作机的 `comparison_300/comparison_20261012_201305_7d302a67/lsbo/run_20261012_201309_cb93cbd9`。回放读取前 8 条记录，核对原生状态更新和相同前缀下的下一次完整提案及诊断，调用数为 0。

```powershell
$saved = 'D:/LSBO/LSBO/X57-LSBO-refactored/runs/comparison_300/comparison_20261012_201305_7d302a67/lsbo/run_20261012_201309_cb93cbd9'
& $py -B run_x57_cold_start.py replay --source-run $saved --prefix 8 --output D:/RA-LSBO/x57-replay-reproduction
```

## 三臂和统计口径

- `lsbo_full_budget` 直接调用冻结原生编码、目标和约束模型、提案与局部搜索。账本在连续缺失目标达到原阈值时记录 `native_missing_objective_stop`，随后继续到预算或其他停止条件。对符合契约的完成和数值失败保留原生 worker 记录，包括其原有求解器诊断用途。
- `structured_sign` 和 `structured_interval` 共享输入和回退。结构化混合观测 GP 沿用定版的分组长度与 EP。层级项使用推进布局、激活的推进器数量以及完整离散构型。可用观测不足 3 个时在通过预筛的潜空间候选池中按完成概率选点。此后通过原生注入接口把对数完成概率加到性能采集分数。原生恢复或探索路径及风险后验回退原因记入提案。两臂的完成观测类型相同，区别在于是否采用同一风险指标的额外区间信息。
- 两个布局分别从 16 点 LHS 批次生成候选。在激活子集内平衡离散水平和连续分箱。静态约束会删除部分原始分箱，按同布局的额外 LHS 批次补足，在合格候选中优先覆盖离散水平和连续分箱。最终分箱按实际激活子集报告，允许重复和缺箱。种子 0 的翼尖与分布式合格候选供给分别为 65 和 64。最终各 16 点，分布式布局覆盖推进器数量 2、4、6、8、10、12。已用种子 0–5 离线验证配额。
- 第 32 次为共享初始化，其后每臂最多 288 次在线评价。每个种子的 32 个点只启动一遍。三臂共享相同的调用 ID 和顺序，后续各自更新。前 16 点为两个布局各 8 点。报告累计节点 32、48、64、128、256、320，分别对应在线节点 0、16、32、96、224、288。六种子调用上限 5,376 次。
- `calls.jsonl` 保存候选、激活变量、提案、时间、预筛、成功启动、结算、超时中断与原始响应。按成功启动计费。预筛拒绝和启动失败零费用。共享阶段基础设施故障停止该种子，在线故障停止当前臂。无可用解时能耗和首次事件留空，保留最后观测节点。报告全部种子及初始可用设计为 0、1–2、至少 3 个的组，达标率同时记录节点观测分母及全种子上下界。配对能耗差只在双方于该预算有可用解时计算。

## X-57 反馈字段

| 字段 | 来源和可观测状态 | 风险模型用途 |
| --- | --- | --- |
| `status`、`solver_success` | `worker.py`、隔离进程退出码和超时，按完成、数值失败、未验证终止及基础设施异常编码 | 正常完成给完成符号，可信物理停止给失败符号。普通非收敛和基础设施异常的完成风险观测为未知。 |
| `objective` / `mission_depletion_Wh` | `postprocess.py` 和 `energy_audit.py` 的完整任务储能消耗，且须满足 worker 完成契约 | 完成但违约也有精确 Wh。非完整任务目标为缺失。 |
| `flutter`、`span_min/max`、三项 `prop_*_clearance` | `constraints.py` / `propulsion_geometry.py` 的解析预筛，派发前已知。完成后核对结果一致性。 | 工程筛选和约束记录。 |
| `soc`、`stall_speed`、`structure`、`usable_energy`、`battery_current`、`battery_voltage` | 原模型任务摘要、`postprocess.py`、`energy_audit.py` 和 `operating_limits.py`。完整合约成功后可用。 | 工程约束满足条件为所有必需约束有限且 `<= 0`。完成标签独立于工程约束。 |
| `mission_solver_segments`、`minimum_terminal_power_W`、`solver_discharge_margin_source` | `mission_solver.py` 的收敛航段诊断，`worker.py` 在完整且验证原放电模型的任务上可导出最小末端功率。 | 用于验证完成符号与物理指标的一致性。当前风险模型使用完成符号。 |
| `seed_continuation`、`residual_inf`、`minimum_trial_terminal_power_W`、`original_discharge_verified` | 求解器延拓轨迹、试探点及残差，失败时可部分观测。 | 试探根的负功率、残差或数值非收敛不能界定已验证任务风险区间。 |
| `solver_error.json`、`energy_audit.json`、`solver.log`、`response.json`、部分能耗诊断 | worker 原始响应与工件，异常和提前停止时视工件写入进度而定。 | 保存以供复核。部分能耗与全任务目标分别记录。 |
| `completion_certificate` | v1 模拟 worker 可提供经过策略验证的示例。定义风险指标为所有任务航段上原模型可接受的最小末端放电功率取负，单位 W。 | 核对指标名、来源、原模型验证标志和同号有序边界后，区间臂使用区间。独立确认物理停止但证书无效时，两臂均只用失败符号。真实 worker 当前没有合格证书生产路径。 |

## 20 次资格试验

以配置中的 55 nmi 巡航任务起步，每个布局 10 次。每支包括 1 个配置参考点、4 个通过预筛的边界附近点、3 个设计空间分散点及 2 个重复点。边界覆盖颤振代理、最小翼展、翼面积及布局对应的最大翼展或桨盘净距。参考和颤振边界各重复 1 次。两布局交错，重复安排在末尾。每次 worker 使用单独的工件目录，1800 秒单次上限，20 次独占串行 worker 等待上限约 10 小时，另加准备与归档开销。

1. 复核两布局各至少存在 1 个有限精确能耗且所有约束满足的设计。比较参考点和边界重复的状态、目标差及工程约束。如果某一分支没有可用解，记录约束瓶颈，再冻结下一版任务配置。
2. 对每次提前停止检查最后已验证的航段、原放电模型验证标志、残差和物理约束。只有工件能够证明同一风险标量的上下界且方向一致时，才可增加真实区间证书生产路径。现有历史证据为 37 次启动、15 次精确目标但 0 个可用设计、21 次非收敛、1 次未结算。若资格试验仍无法提供区间证据，本试验只能检验可用设计获取及符号反馈流程。
3. 报告两个布局的单次耗时、中位数、P90 和超时数，同时核算实际启动数、总墙钟时间与重复差异。基础设施失败冻结当前批次进度供核查。资格工件及结果保存在独立目录，正式初始化使用其独立的先验名单和观测。
