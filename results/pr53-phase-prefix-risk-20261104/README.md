# PR53：真实在线选点的阶段风险校准与前缀校准

固定分析代码：`a7151a7bb44e2570eaec64f1ce3125024deadffc`  
输入：PR52 结果 commit `694b098cacba75bf5168eec3a1c55dbf84289051` 的
`prequential_risk_predictions.csv` 与 `prequential_risk_report.json`  
环境：`Rcaide_vsp` / Python 3.9.25。

本分支只上传**只读诊断产物**；不修改 PR45–PR52 的任何原始证据。

## 输入身份与运行范围

PR53 逐个校验：

- PR52 CSV 的 Git BLOB SHA1：`d1a32240bfdd...`；
- PR52 report 的 Git BLOB SHA1：`0a254889624e...`；
- PR45 源 SHA `386082b`、SQLite SHA `a30b0dad...`；
- 9,216 条在线选点、64 校准 + 64 测试种子、每种子 18 次；
- 概率乘积恒等式、mission/SOC/feasible 关系和 seed/iteration 序列。

运行：

- smoke：`--per-block 1`，72 个 held-out calls、144 个事件行，exit 0；
- full：默认每块 64 个种子，**4,608 个 held-out calls × 2 个事件 = 9,216 个事件预测行**，exit 0；
- 新增飞机、RCAIDE、GP/VAE 拟合、优化器提案均为 **0**；
- PR52 输入 CSV/report 运行前后 SHA256 均为
  `b54141b...` / `624006d...`，未变化。

## 四种策略

| 名称 | 来源 | 是否可被原在线算法使用 |
|---|---|---|
| `raw` | PR45 选点前原始概率 | 是（原始记录） |
| `pooled_other_seeds` | 其他校准种子的单截距 | 否，回顾性 |
| `phase_other_seeds` | boundary / other 分组单截距，向 pooled 收缩 ridge=0.10 | 否，回顾性 |
| `prefix_only` | 本种子当前提案之前已结算的 ≥6 条记录拟合 ridge=0.25 截距 | 原理上在线可得，但 PR45 未采用 |

测试种子不参与任何拟合。`phase` 的分组来自 PR52 事后观察，测试种子与 PR52 相同，**不是独立确认性数据**。

## 主结果（held-out 原始已选点）

### Brier / LogLoss

| 任务 / 臂 | 事件 | raw B / LL | pooled B / LL | phase B / LL | prefix B / LL |
|---|---|---|---:|---:|---:|---:|
| ordinary / full_latent | feasible | 0.1962 / 0.5845 | 0.1891 / 0.5545 | 0.1888 / 0.5529 | 0.1921 / 0.5758 |
| ordinary / full_latent | mission | 0.0341 / 0.1165 | 0.0331 / 0.1130 | 0.0306 / 0.1049 | 0.0344 / 0.1174 |
| ordinary / ra_lsbo | feasible | 0.1879 / 0.5630 | 0.1789 / 0.5335 | 0.1789 / 0.5327 | 0.1835 / 0.5513 |
| ordinary / ra_lsbo | mission | 0.0302 / 0.1068 | 0.0301 / 0.1061 | 0.0284 / 0.1011 | 0.0302 / 0.1069 |
| sparse / full_latent | feasible | 0.1999 / 0.5856 | 0.1810 / 0.5378 | 0.1801 / 0.5358 | 0.1948 / 0.5713 |
| sparse / full_latent | mission | 0.1542 / 0.5042 | 0.1502 / 0.4755 | 0.1496 / 0.4743 | 0.1511 / 0.4913 |
| sparse / ra_lsbo | feasible | 0.1899 / 0.5530 | 0.1805 / 0.5320 | 0.1767 / 0.5242 | 0.1879 / 0.5498 |
| sparse / ra_lsbo | mission | 0.1326 / 0.4248 | 0.1322 / 0.4222 | 0.1291 / 0.4163 | 0.1357 / 0.4316 |

### ECE / p≥0.8 假可行数

| 任务 / 臂 | 事件 | raw ECE / FP8 | pooled ECE / FP8 | phase ECE / FP8 | prefix ECE / FP8 |
|---|---|---|---:|---:|---:|---:|
| ordinary / full_latent | feasible | 0.0997 / 105 | 0.0525 / 54 | 0.0463 / 49 | 0.0753 / 101 |
| ordinary / full_latent | mission | 0.0170 / 13 | 0.0124 / 19 | 0.0083 / 13 | 0.0175 / 13 |
| ordinary / ra_lsbo | feasible | 0.0970 / 96 | 0.0518 / 55 | 0.0491 / 49 | 0.0778 / 90 |
| ordinary / ra_lsbo | mission | 0.0078 / 12 | 0.0070 / 15 | 0.0145 / 9 | 0.0087 / 12 |
| sparse / full_latent | feasible | 0.1262 / 70 | 0.0383 / 38 | 0.0299 / 37 | 0.0979 / 66 |
| sparse / full_latent | mission | 0.0837 / 77 | 0.0724 / 51 | 0.0745 / 48 | 0.0728 / 62 |
| sparse / ra_lsbo | feasible | 0.0985 / 55 | 0.0508 / 44 | 0.0412 / 45 | 0.0884 / 53 |
| sparse / ra_lsbo | mission | 0.0546 / 49 | 0.0457 / 47 | 0.0415 / 43 | 0.0621 / 47 |

## 关键发现

1. **统一截距校准复现了 PR52 的收益，阶段校准在此基础上只增加很小但方向一致的改进。**  
   feasible Brier 从 raw 0.1879–0.1999 降到 pooled 0.1789–0.1810；phase 进一步到 0.1767–0.1801。  
   phase−pooled 的总体 Brier 差为 −0.00006 到 −0.00384；5/8 个 cluster 95% CI 不含 0，但绝对幅度很小，且 phase 是看到 PR52 结果后的探索性分组。

2. **边界阶段是本轮最明确的收益点。**  
   在 `budgeted_boundary_information` 子集上，phase 相对 pooled 的 Brier/ECE **8/8 全部改善**：

   - sparse/ra_lsbo feasible：Brier 0.2339→0.2092，ECE 0.1808→0.0899；
   - sparse/full_latent feasible：0.2057→0.2000，ECE 0.0834→0.0369；
   - ordinary/full_latent mission：0.1657→0.1568，ECE 0.0945→0.0490；pooled 原本比 raw 更差，phase 将其拉回并略优于 raw。

   这说明“统一截距在边界阶段校准不足”的判断成立；但这是同批种子的回顾性评估。

3. **前缀校准是弱替代，不是强方案。**  
   它只依赖本种子此前已结算记录：feasible 上约 64–66% 的 held-out 调用可启用前缀拟合，其余回退 raw；ordinary mission 只有 25–29% 可启用，因为早期缺少正负两类历史。  
   启用前缀的调用上，feasible Brier 通常下降（例如 ordinary/full_latent 0.1939→0.1877；sparse/full_latent 0.2067→0.1989），但整体被大量回退调用稀释；sparse/ra_lsbo feasible 的总体差 CI 仍含 0。  
   mission 上前缀并不可靠：ordinary/full_latent 0.0341→0.0344（CI 全在 0 以上），sparse/ra_lsbo 0.1326→0.1357（CI 0.0011–0.0050）。

4. **p≥0.8 的假可行问题没有被任何一种校准解决。**  
   feasible 的 FP8：raw 55–105，pooled 38–55，phase 37–49，prefix 53–101。phase 最接近，但仍意味着 37–49 个已知不可行设计会被 0.8 门槛接受。  
   mission 的 FP8：raw 10–77，pooled 15–51，phase 9–48，prefix 12–62。  
   因此任何“风险预算/门限”都仍只能作为观察性诊断，不能声称安全。

5. **预测风险预算持续低估实际失败。**  
   18 次原始已选点路径上，feasible 平均预测失败费用 raw 4.57–6.27、prefix 4.98–6.65，实际非可行 6.31–8.33。  
   B=8 时：

   | 任务/臂 | 实际超限 seeds | raw 预测超限 | prefix 预测超限 | raw 预测未超但实际超限 | prefix 预测未超但实际超限 |
   |---|---:|---:|---:|---:|---:|
   | ordinary/full_latent | 13 | 0 | 3 | 13 | 10 |
   | ordinary/ra_lsbo | 9 | 0 | 2 | 9 | 7 |
   | sparse/full_latent | 28 | 7 | 14 | 21 | 14 |
   | sparse/ra_lsbo | 23 | 6 | 11 | 19 | 13 |

   即使 B=8，原始路径中仍有 9–28/64 个种子实际超过预算，而预测超限只有 0–7/64。报告中的 `not_a_counterfactual_stop_policy=True` 和 `no_risk_or_failure_budget_guarantee=True` 是硬限制。

6. **阶段校准并非处处优于统一校准的 ECE。**  
   例如 ordinary/ra_lsbo mission 的 phase ECE 0.0145 差于 pooled 0.0070；sparse/full_latent mission 0.0745 对 0.0724。Brier 改善不等于概率完全校准；应按事件和目标分别报告。

## 限制

- 所有标签和选点都是 PR45 的回顾性已选设计，不是新策略结果；不能作在线优化收益或因果表示结论。
- pooled 与 phase 使用其他种子的已观察结果，原 PR45 在线算法从未获得；prefix 虽原理上在线可得，但只在固定原始路径上评估，未被原算法采用。
- phase 分组来自 PR52 事后观察，测试种子与 PR52 相同，不是独立确认。
- 0.8/0.5 门限计数与 4/6/8 预算诊断都是观察性统计，不是反事实停策略，也不构成失败数上界。
- 140 km 独立稀疏资格仍未建立；简化飞机结论不能直接外推 RCAIDE。

## 文件

- `phase_prefix_report.json`：完整 24 个 case/arm/event/phase 组合、校准支持情况、配对差、预算诊断。
- `phase_prefix_predictions.csv`：9,216 个 held-out 事件预测行（含 raw/pooled/phase/prefix 概率）。
- `smoke-console.log`、`full-console.log`：两轮完整日志。
- `fixed_sha.txt`：固定 SHA、PR52 输入 blob/SHA256、Python 与计数。
- `pre_run_sha256.txt`、`post_run_sha256.txt`：PR52 输入运行前后哈希。
- `overview.md`：含 boundary 子集和预算诊断的完整表格。
