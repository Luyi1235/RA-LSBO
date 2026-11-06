# PR66：信息价值路线第一阶段（候选机会诊断 + 有限两步决策内核）

固定代码 SHA：`1917f305b39f93d79f6b16dd83e56af365098500`  
分支：`research/ra-lsbo-information-value-stage1-20261106`  
结果分支：`results/information-value-stage1-20261106`  
环境：`Rcaide_vsp` / Python 3.9.25。

本交付是**诊断与构造性机制验证**，不是新的 GP 两步优化器，也不是飞机/RCAIDE 验证。所有在线轨迹先封存，再另付数学评价取得候选真值；审计标签未参与任何选点或拟合。

## 1. 技术核验

| 项目 | 结果 |
|---|---|
| 固定 SHA | `1917f305b39f93d79f6b16dd83e56af365098500` |
| 检出前 `git status --porcelain` | 空 |
| 单元测试 | **21 passed** in 0.624 s |
| `compileall` | exit 0 |
| mechanism | exit 0，`mechanism_report.json` 82K |
| smoke | exit 0，`COMPLETE_STAGE1_DIAGNOSTIC_NOT_NEW_BO` |
| smoke 费用 | 48 初始 + 8 在线 + 640 审计 = **696** |
| smoke 文件核验 | 47 个文件，manifest `88b7b355...` |
| full | exit 0，`COMPLETE_STAGE1_DIAGNOSTIC_NOT_NEW_BO` |
| full 费用 | 192 初始 + 96 在线 + 16,384 审计 = **16,672** |
| full 检查点 | **32**（16 轨迹 × 第 1、4 次零起算迭代） |
| 每检查点候选 | **512**（原池 128 + 扩展 384） |
| full 文件核验 | 215 个文件，manifest `0790955c...` |
| 封存顺序 | `all_traces_sealed_before_any_audit=true` |
| 标签隔离 | `audit_labels_never_available_to_policy=true` |
| 飞机 / RCAIDE | **0 / 0** |
| pack | smoke 5 ZIP、full 5 ZIP；bundle manifest 全部 SHA256 通过 |

运行目录（源 worktree 外）：

- 机制：`D:\LSBO\RA-LSBO-pr66-information-stage1-mechanism-20261106`
- smoke：`D:\LSBO\RA-LSBO-pr66-information-stage1-smoke-20261106`
- full：`D:\LSBO\RA-LSBO-pr66-information-stage1-full-20261106`
- smoke pack：`D:\LSBO\RA-LSBO-pr66-information-stage1-smoke-pack-20261106`
- full pack：`D:\LSBO\RA-LSBO-pr66-information-stage1-full-pack-20261106`

真实 GP/VAE 接口只在 smoke/full 中验证；GitHub CI 与本地替身测试不替代真实集成。

## 2. 四个任务的候选机会诊断

每任务 4 个 full 种子、2 个检查点，共 8 个检查点；下列为 8 个检查点均值/中位数。

| 任务 | 原池可行 /128 | 原池最大改进（中位） | 扩展池可行 /512 | 扩展增量可行 | 原池有正机会的检查点 | 扩展池更优的检查点 |
|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | 47.12 | +5.862 | 184.50 | +137.38 | 7/8 | 5/8 |
| Branin sparse | 12.12 | +1.101 | 47.25 | +35.13 | 4/8 | 7/8 |
| Rosenbrock ordinary | 43.75 | +2.075 | 186.62 | +142.88 | 7/8 | 5/8 |
| Rosenbrock sparse | 10.25 | +21.341 | 40.00 | +29.75 | 5/8 | 5/8 |

结论：四个任务的原池都存在真实可行改进机会，尤其 sparse 任务也有 4–5/8 个检查点存在正机会；扩展到 512 个候选后，可行数与最优目标都增加。**但扩展池没有 GP 分数，不能据此声称“扩大候选池后优化器胜出”**；它只说明候选生成层还有空间。

### 2.1 原效用前八名

| 任务 | top8 可行数 /8 | top8 最大改进（中位） | top8 含原池最优的检查点 |
|---|---:|---:|---:|
| Branin ordinary | 6.5 | +3.766 | 1/8 |
| Branin sparse | 2.38 | +0.472 | 2/8 |
| Rosenbrock ordinary | 7.25 | 0.000 | 1/8 |
| Rosenbrock sparse | 2.88 | +12.272 | 4/8 |

top8 经常“有可行点”，但往往不是原池最优改进点；Rosenbrock ordinary 的 top8 中位改进为 0，说明原效用排序没有把池内最大改进点放进前八。

### 2.2 eligible 支持域

| 任务 | eligible 候选（中位）/128 | eligible 可行损失/检查点 | eligible 排除池最优的检查点 |
|---|---:|---:|---:|
| Branin ordinary | 127.0 | 0.00 | 0/8 |
| Branin sparse | 124.5 | 2.88 | 0/8 |
| Rosenbrock ordinary | 127.0 | 0.00 | 0/8 |
| Rosenbrock sparse | 124.5 | 1.25 | 0/8 |

一般情况下 eligible 支持没有排除原池最优；但 **Branin sparse 的种子 998602、迭代 4 出现支持域坍塌**：

- 原池 128 个候选、11 个可行、最优可行 `1.660085`；
- eligible 只剩 **1 个候选、0 个可行**；
- `selection_mode=degenerate_gate_coverage`，实际选点不可行。

这是本阶段最明确的“支持域/门退化造成机会损失”实例；其余检查点未出现排除池最优的情况。

### 2.3 风险预测与同点真值

在同一批 128 个模型候选上（每检查点 512 真值审计中的原池部分）：

| 任务 | Brier（均值） | 预测失败率 | 实际失败率 | 预测 ≤0.2 的失败数/检查点 |
|---|---:|---:|---:|---:|
| Branin ordinary | 0.1465 | 0.560 | 0.632 | 1.25 |
| Branin sparse | 0.1385 | 0.750 | 0.905 | 1.00 |
| Rosenbrock ordinary | 0.1243 | 0.539 | 0.658 | 0.25 |
| Rosenbrock sparse | 0.1309 | 0.711 | 0.920 | 0.50 |

风险概率总体**低估实际失败**（sparse 低估约 0.15–0.21），且低风险区仍存在失败。实际选点的单点 Brier 在 0.048–0.198 间但只有 1 个样本/检查点，受选择条件影响，不能作为总体校准结论。

### 2.4 实际选点错过了哪些池内机会

| 任务 | 选中可行 | 选中不可行 | 可行选点相对原池最优的差距（中位） | 正差距的可行选点 |
|---|---:|---:|---:|---:|
| Branin ordinary | 6/8 | 2 | +13.855 | 5/6 |
| Branin sparse | 5/8 | 3 | +9.730 | 5/5 |
| Rosenbrock ordinary | 8/8 | 0 | **+157.782** | 8/8 |
| Rosenbrock sparse | 4/8 | 4 | 0.000 | 1/4 |

- Branin ordinary：多数选点可行，但 5/6 明显差于池内最优；2 次直接不可行。
- Branin sparse：3/8 不可行，所有可行选点都差于池内最优；包含上述支持坍塌。
- Rosenbrock ordinary：**8/8 可行但都显著差于池内最优**（中位差距 157.8），说明“可行”不等于“选到了有价值的可行点”。
- Rosenbrock sparse：4/8 不可行；一旦选到可行点，多数就是池内最优（3/4 差距为 0），主要损失是半数的不可行选择。

### 2.5 选择模式分布

| 任务 | probability_weighted | budgeted_boundary | feasible_discovery | degenerate_gate |
|---|---:|---:|---:|---:|
| Branin ordinary | 4 | 4 | 0 | 0 |
| Branin sparse | 3 | 3 | 1 | 1 |
| Rosenbrock ordinary | 4 | 4 | 0 | 0 |
| Rosenbrock sparse | 3 | 4 | 1 | 0 |

## 3. 有限两步决策内核

机制报告：`mechanism_report.json`，72 行、3 个构造场景 × 4 个世界 × 3 个臂 × 2 个深度；`FINITE_MECHANISM_VERIFICATION_ONLY`。没有 GP 拟合，也没有真实数学 BO 调用。

`Q_h(a) = -cost(a) + Σ_o p(o|D,a) V_{h-1}(D+o)`，`V_h = max(stop_value, max_a Q_h(a))`；深度不超过真实剩余调用数。

### 3.1 失败信息消融（h=2）

| 场景 | myopic 净效用 | two_step_full 净效用 | 去掉失败证据 净效用 | full 调用数 | 去掉证据 调用数 |
|---|---:|---:|---:|---:|---:|
| informative_abort | 4.25 | **8.00** | 4.50 | 2.00 | 1.75 |
| uninformative_abort | 4.25 | 4.50 | 4.50 | 1.75 | 1.75 |
| expensive_evaluations | 0 | 0 | 0 | 0 | 0 |

- **有信息失败**：full 先用探针观察失败，再选择目标岛；净效用 8.0，高于 myopic 的 4.25，也高于去掉失败证据的 4.5。差异来自“失败后仍保留的额外证据”，不是单纯多一次调用。
- **无信息失败**：full 与去掉失败证据相同（4.5），没有把无信息探针当成有价值信息；这是负对照通过。
- **高成本**：所有臂停止，净效用 0，没有为了信息而执行负期望调用。
- h=1 时三个臂都选择安全交付（净效用 4.0），说明两步收益依赖真实剩余预算，而不是无条件探索。

### 3.2 剩余预算与停止

- 只有 h=2 且存在可区分信息时，full 才使用第二次调用；h=1 时退化为安全选择。
- `expensive_evaluations` 在 h=1 与 h=2 均为 0 调用，验证了停止条件。
- 这些是构造的四世界验证，不能外推为真实 GP 的在线收益，也不能证明飞机安全。

## 4. 上传内容

结果目录仅包含只读证据与诊断，不包含代码修改：

- `mechanism/`：`mechanism_report.json`、`source_receipt.json`、`manifest.json`。
- `smoke/`、`full/`：完整 `report.json`、`plan.json`、`source_receipt.json`、`trace_seal.json`、`manifest.json`、`trace/`、`audit/`。
- `packs/smoke/`、`packs/full/`：逐任务 ZIP 与 `bundle_manifest.json`；每任务 ZIP 含 `shared_initial`、proposal、observation、representation checkpoint、审计真值与评价账本。
- `logs/`：21 项测试日志、机制/smoke/full 控制台、verify/pack 日志、环境与 pip freeze。
- `diagnosis/`：`diagnosis_summary.json`、`per_task_diagnostic.csv`。
- `fixed_sha.txt`：固定 SHA、费用计数、环境和分支信息。
- `artifact_sha256.txt`：本目录所有上传文件的 SHA256。

`full` 的 16,384 次是**后置审计**，96 次是在线选点；不得把 16,672 当作新算法在线评价数，也不得把扩展池的真值当作免费标签。

## 5. 边界

- 只有原 PR63 `soft_gp` 生成的 16 条状态；四个种子/任务，诊断性，不做优越性统计。
- 扩展池没有 GP 分数，不能推断“扩池后优化器更好”。
- 审计真值只用于事后诊断，未反馈到策略。
- 有限机制验证未接入 GP/VAE，不是新的两步 GP 优化器。
- 没有飞机/RCAIDE 验证，不能声称飞行安全或真实设计收益。
