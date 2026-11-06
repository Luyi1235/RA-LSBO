# PR67 R1：导入修复后的 Stage-2 all-support / eligible 支持消融

固定代码：`0cefd1f2507c355067f3c4f56d2dd43d919f65ef`  
旧失败代码：`488a6da414572e09709533ac7a96e17115e632c9`  
旧失败结果 HEAD：`ee8949687f404b4df24ced1699810e337c3b5771`（未修改）  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支从上述固定 SHA 建立，仅上传本次 R1 的新证据。没有修改旧失败目录，没有修改 PR45–PR66 的代码或结果。

## 导入修复与预检

唯一代码改动：
`from ..soc_revision...` → `from ...soc_revision...`。

| 检查 | 结果 |
|---|---|
| 协议测试 | 9 passed |
| 导入路径测试 | 3 passed |
| 真实依赖导入测试 | 4 passed |
| `compileall` | exit 0 |
| `run_ra_lsbo_information_stage2_pycharm.py --help` | exit 0 |
| 固定 SHA / `git status --porcelain` | 精确相等 / 空 |
| 新增飞机、RCAIDE | 0 / 0 |

全部预检通过后才执行 smoke 与 full。日志见 `logs/01-*.log` 到 `logs/07-full-console.log`。

## 预算与完整性

| 项目 | smoke | full |
|---|---:|---:|
| case-seed 组 | 4 | 48 |
| trial | 8 | 96 |
| 共享初始评价 | 48 | 576 |
| 在线评价 | 16 | 576 |
| 数学评价总额 | **64** | **1,152** |
| 每臂在线 | 2 | 6 |
| 状态 | `SMOKE_TWO_CALLS_NOT_FULL_STUDY` | `COMPLETE_FRESH_FIXED_SIX_ON_POLICY` |

full 完整性：

- 96 个 `result.json`、576 条 proposal、576 条 observation、96 个 representation checkpoint、48 个共享初始记录；
- 48/48 个 case-seed 组的两臂 `initial_prefix_sha256` 完全一致；
- 550 条有模型预测的在线决策；all-support 277 条支持消融记录全部为
  `EI_normalization_reference_unchanged=true`，规则 `all_legal_gp_candidates`；
- 全候选数恒为 128；
- 飞机/RCAIDE 为 0；没有 unknown/infrastructure 错误；
- 详细机器可读核验：`logs/full_integrity_check.json`。

## 主结果

### 逐任务工作量（每任务 12 个种子）

| case | arm | 最终可行 | 条件平均目标 | 在线失败总数 | 冷启动新可行 | 覆盖率 | 官方 support-collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| Branin ordinary | eligible | 12/12 | 2.052 | 5 | 0 | 72 | 0 |
| Branin ordinary | all-support | 12/12 | 2.052 | 5 | 0 | 72 | 0 |
| Branin sparse | eligible | 10/12 | 6.206 | 34 | 0 | 65 | 3 |
| Branin sparse | all-support | 10/12 | 6.206 | 34 | 0 | 68 | 3 |
| Rosenbrock ordinary | eligible | 12/12 | 9.325 | 8 | 0 | 72 | 0 |
| Rosenbrock ordinary | all-support | 12/12 | 9.325 | 8 | 0 | 72 | 0 |
| Rosenbrock sparse | eligible | 11/12 | 56.360 | 37 | 3 | 64 | 3 |
| Rosenbrock sparse | all-support | 10/12 | 72.482 | 41 | 2 | 65 | 4 |

合计：

- eligible：45/48 最终可行，84/288 次在线失败；
- all-support：44/48 最终可行，88/288 次在线失败；
- 两者在线评价数均为 288，成本相同。

### 配对结果（feasible-first，all-support 对 eligible）

| case | W / L / T | 双方可行对 | 平均目标差 | 中位目标差 | 选点路径变化 |
|---|---:|---:|---:|---:|---:|
| Branin ordinary | 0 / 0 / 12 | 12 | 0.0 | 0.0 | 0/12 |
| Branin sparse | 0 / 0 / 12 | 10 | 0.0 | 0.0 | 2/12 |
| Rosenbrock ordinary | 0 / 0 / 12 | 12 | 0.0 | 0.0 | 0/12 |
| Rosenbrock sparse | **1 / 2 / 9** | 9 | 0.0 | 0.0 | 3/12 |

- 43 个双方可行对的目标差全部为 **0.0**：没有一对在双方都找到可行解时改善了目标。
- 所有胜负都来自“是否找到可行解”，不是目标改进。
- 只有 5/48 个种子改变了选点路径，其中只有 3 个改变了结果；其余路径变化没有改变最终效果。

### 改变结果的 3 个种子（Rosenbrock sparse）

| seed | eligible | all-support | 判定 |
|---|---|---|---|
| 998701 | 无可行解，6 次失败 | 可行，目标 121.157，5 次失败 | all-support 胜 |
| 998704 | 可行，目标 10.322，3 次失败 | 无可行解，6 次失败 | all-support 负 |
| 998707 | 可行，目标 5.979，4 次失败 | 无可行解，6 次失败 | all-support 负 |

Branin sparse 的 998702、998712 虽然路径改变，但两臂都没有可行解，结果仍为平局。

## support 统计

| arm | receipts | 模型决策 | 支持消融记录 | EI 归一化未变 | 中位原 eligible | 最小原 eligible | 官方 collapse | 模型决策中 eligible=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eligible | 288 | 273 | 0 | — | 125 | 3 | 6 | 0 |
| all-support | 288 | 277 | 277 | 277/277 | 125 | 1 | 7 | **7** |

说明：

- 官方 `support_collapse_decisions` 按 runner 的 `(original_eligible_count or 999) <= 1` 统计，包含 scheduled/degenerate coverage；因此 eligible 的 6 次来自 coverage，不是模型决策。
- 更严格的“有模型预测且原 eligible=1”的模型决策：eligible 为 **0**，all-support 为 **7**（Branin sparse 3、Rosen sparse 4）。
- all-support 没有消除支持坍塌。它只是在支持很小时仍可离开 eligible 集合选点，因而**进入**了 7 个原 eligible=1 的决策状态；最小 eligible 从 3 降到 1。

## 判读

1. **导入修复有效，实验真实完成；但没有出现 all-support 优于 eligible 的证据。**  
   四个任务中三个完全同结果；Rosenbrock sparse 是 1 胜 2 负，最终可行 10/12 对 11/12，在线失败 41 对 37，冷启动 2 对 3。没有平均收益，成本相同。

2. **全部 43 个双方可行对的目标差都是 0。**  
   这表明本轮没有观察到“放宽支持域后找到更好可行设计”的收益；all-support 的影响主要是有无可行解，而不是目标质量。

3. **支持坍塌没有被修复，反而在 all-support 路径上出现。**  
   PR66 的 1-eligible 状态仍然存在；all-support 有 7 个模型决策落在原 eligible=1 的状态，eligible-only 为 0。它不是“去除一个误杀好点的过滤”，而是允许策略进入支持不足的区域。

4. **失败代价集中在 Rosenbrock sparse。**  
   总失败从 84 增至 88（+4），全部发生在 Rosenbrock sparse；all-support 失去 2 个原本可行的种子，只换回 1 个。条件平均目标 56.36→72.48 主要反映可行集合组成变化，不能当作目标变差或变好的因果效应。

5. **路径改变不等于收益。**  
   5/48 个种子改变路径，只有 3 个改变结果；Branin sparse 的两次变化完全没有改变可行性。以“发生路径变化”作为机制有效性的证据是不充分的。

## 边界

- 固定预算 6 次在线评价/臂，12 个种子/任务；这是短视支持域筛选消融，不是完整 GP 前瞻或长期收敛比较。
- 只测试 `soft_gp` 阶段策略；不是 PR66 的有限机制核，也不是两步 GP 前瞻。
- 没有飞机/RCAIDE 评价；没有安全或真实工程泛化结论。
- 没有做显著性检验；1 胜 2 负是小样本方向性结果。
- 不得据此报告“all-support 已胜出”或“支持坍塌已解决”；最多只能说：在本轮设置下，all-support 未带来收益，且增加了低支持状态与失败。

## 文件

- `smoke/`、`full/`：plan、source_receipt、report、progress、全部 `shared_initial/` 与 `trials/`（proposal、observation、result、representation checkpoint）。
- `logs/`：9/3/4 测试、compileall、--help、smoke/full 控制台、环境、pip freeze、完整性核验、分析与表格。
- `instructions/`：R1 指令原文与 Stage-2 协议文档副本。
- `fixed_sha.txt`：固定 SHA、旧失败 SHA/HEAD、环境与文档哈希。
- `artifact_sha256.txt`：上传文件 SHA256 清单。
