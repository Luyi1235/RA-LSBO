# PR57：PR54/修正 PR55 真实轨迹的等评价次数审计

固定分析代码 SHA：`b996ecca755a68d69caf52045e40a9e7419609bd`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传**只读、零新增评价**的审计结果。PR45–PR56 的原始证据未修改。

## 测试状态（先说明）

指定命令：

```text
python -m pytest -q tests/test_risk_budget57_anytime_audit.py tests/test_risk_budget56_report_audit.py
```

结果：**4 passed, 3 failed**（0.18 s）。3 个失败全部是测试侧问题：

1. `test_complete_240_paired_seed_rows_only_with_actual_outcomes`：测试把输入和输出都放在同一 `tmp_path`，触发生产代码“输出目录必须在输入目录之外”的守卫；
2. `test_byte_identity_failure_rejects_tampering_before_output`：同一 `tmp_path` 布局问题；
3. `test_audit_rejects_non_pr55_and_missing_trial_details`：测试正则 `not a PR55` 与实际消息 `Not a PR55 frozen report` 仅大小写不匹配。

完整 traceback 见 `tests.log`，逐项说明见 `test_failure_summary.txt`。
生产入口仍使用从 Git blob 精确取出的输入，身份校验通过并成功完成；未修改任何代码。

## 输入身份

| 来源 | result commit | source commit | Git blob SHA1 | SHA256 |
|---|---|---|---|---|
| PR54 | `41db5094f29645cf5afd17b08e65ff62544d3f41` | `fd5492d33b21421ff8f54ad8bba453543caf39fe` | `7f94ed040ee88c2ae552f704516fcea003193ded` | `3e4de729...a4b4` |
| PR56 修正的 PR55 | `cdd02b551d3233d8262d5adbf46db207d6703f93` | `d947db5f0b544739c9db286a75d99c0308713fdb` | `be839a224ef56868f048cfb161fc54065c2eea8b` | `1b297db3...ef9a` |

- 240 条原始 on-policy 轨迹（PR54 48 + PR55 192）；
- 240 条 paired method-seed 记录（80 个 seed-case 配对 × 3 个对照）；
- 新增解析、飞机、RCAIDE 评价：0；
- 新增 GP/VAE 拟合、候选提案：0；
- 输入运行前后 SHA256 完全一致（`pre_run_sha256.txt` / `post_run_sha256.txt`）。

## 口径

- **固定 k=6（主要口径）**：所有臂都有 12 个共享初值，且六次已观察失败停止规则保证每条轨迹至少执行六次在线评价。只使用真实发生的前六次在线评价及当时的已知最优可行目标，不补全任何停止后的未知评价。
- **共同实际长度 k=min(N_a,N_b)**：每条配对使用双方都实际达到的步数；k 是运行结束后才知道的，受早停影响，**不是随机分配的同成本因果实验**。
- **完整终点/k=8/10/12**：k>6 只保留双方都实际达到该步的种子，报告覆盖数与被早停排除数；完整终点的调用次数不同。

## 固定 k=6：Budgeted vs Semantic-only

| 来源 / 任务 | 配对数 | 双方可行 | 改善/持平/退化 | 平均目标差 | 中位差 |
|---|---:|---:|---:|---:|---:|
| PR54 / Branin ordinary | 8 | 8 | 0/8/0 | 0.0000 | 0.0000 |
| PR54 / Branin sparse | 8 | 6 | 1/4/1 | −1.9695 | 0.0000 |
| PR55 / Branin ordinary | 16 | 16 | 0/16/0 | 0.0000 | 0.0000 |
| PR55 / Branin sparse | 16 | 12 | 4/6/2 | −2.6172 | 0.0000 |
| PR55 / Rosenbrock ordinary | 16 | 16 | 0/15/1 | +0.2050 | 0.0000 |
| PR55 / Rosenbrock sparse | 16 | 13 | 2/11/0 | −0.1365 | 0.0000 |

差值定义为 `budgeted − semantic_only`，负值表示 budgeted 的目标更好。

**结论：固定六次在线评价时，budgeted 相对 semantic-only 的优势只出现在稀疏任务，且幅度有限；普通任务没有优势，Rosenbrock ordinary 甚至略差。**  
稀疏任务中的改善也没有提高双方可行覆盖：上表双侧可行数与完整路径相同，单侧可行基本为 0，因此早期价值主要是“已可行时的目标质量”，不是“更早发现可行解”。

## 其它对照（固定 k=6）

| 对照 | Branin sparse | Rosenbrock sparse | Branin ordinary | Rosenbrock ordinary |
|---|---:|---:|---:|---:|
| budgeted vs legacy | −3.19 / −2.39 | −1.32 | +0.11 / +0.82 | +1.41 |
| semantic vs legacy | 0.00 / +0.23 | −1.19 | +0.11 / +0.82 | +1.21 |

注意：这些是不同 study、不同 seed 规模下的描述性数值（Rosenbrock 仅 PR55 有），不能直接相加作总体效应。

## 完整终点与调用次数（PR55 Branin sparse）

| 臂 | 平均在线调用 | 平均失败数 | 触达六次失败停止 | 终点双方可行 |
|---|---:|---:|---:|---:|
| budgeted | 10.19 | 5.25 | 9/16 | 12 |
| semantic_only | 8.63 | 5.50 | 13/16 | 12 |
| legacy | 9.06 | 5.50 | 12/16 | 12 |

- 完整终点 `budgeted − semantic_only`：平均 **−6.2801**，中位 −2.5248，6/4/2（改善/持平/退化）；
- 但 budgeted 平均多 **1.56** 次在线调用，且更少触发六次失败停止；
- 共同实际长度（k=min）下平均差为 **−4.4608**（6/4/2，中位 −1.5908）；
- 代数分解：共同前缀差 −4.4608，budgeted 额外调用贡献 −1.8194，semantic 额外调用贡献 0.0，合计 −6.2801。

因此，完整终点的稀疏优势 **一部分来自前六次选点，一部分来自少触发停止后获得的额外真实调用**；后者不是等评价次数因果效应。

PR54 Branin sparse 的对应结论更弱：k=6 平均 −1.9695（1/4/1），完整终点 −2.2596（2/3/1），平均调用 9.0 vs 9.375。

## 解释与限制

1. 本审计是在看到 PR54/55 结果后设计的探索性分析，不是独立确认性实验；不作显著性、调整多重比较或安全保证声称。
2. k=6 是唯一所有臂都真实拥有六次在线评价的等努力口径；k=8/10/12 与完整终点都存在治疗依赖的截尾/覆盖差异。
3. k=min 的“等努力”是事后按实际停时选出的，不能当作随机化同成本因果实验。
4. 终点分解只是数值恒等式，不证明“风险预算导致额外调用带来了改进”。
5. Branin/Rosenbrock 是自定义稀疏变体，不是上游原始 benchmark 复刻；140 km 独立稀疏资格仍未建立。
6. 风险预算策略仍是未校准启发式，不构成安全 BO、真实飞机或安全保证。

## 文件

- `fixed_effort_anytime_report.json`：18 个 study/case/pair 的完整汇总、固定 k 子集、共同长度、终点与分解。
- `fixed_effort_paired_seeds.csv`：240 条 paired seed 明细（含每对实际调用、失败数、k=6/k=common/final 目标）。
- `console.log`：生产运行完整控制台。
- `tests.log`：指定 pytest 的完整失败 traceback。
- `test_failure_summary.txt`：3 个测试侧失败的逐项说明。
- `fixed_sha.txt`：固定 SHA、输入 blob/SHA256、运行计数。
- `pre_run_sha256.txt` / `post_run_sha256.txt`：两个输入运行前后哈希，完全一致。
