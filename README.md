# RA-LSBO：从风险可学习性到预算约束的序贯信息获取

> **当前研究阶段：PR86-B — Sequential Risk Information Acquisition。**
> PR86-A 已完成工作机 full 实验和独立归档；PR86-B 正在独立代码开发与 CI 验收。当前分支不授权工作机执行 smoke/full。

## 研究转向依据

PR83/C2 的 64 个初始组全部为 warm，不能回答风险模型冷启动问题。PR84 对 C2 原始轨迹的只读审计确认了这一覆盖限制，并保留了原始汇总字段与独立重算之间的差异。

PR86-A 随后在六个数学任务上完成了离线风险学习曲线：12/24/48/96/192 个训练观测、四种嵌套采样顺序、native-censored 与 privileged-exact 两种监督契约。Full 共 12,672 次数学评价、720 个模型行；全部评价结算，GP/kernel 错误为零。结果归档于：
- 代码：`research/phase-pr86-cold-start-learnability-20261110`，SHA `a5d5f20a4854de30939b590cb69126d1d5690ed5`
- 结果：`results/phase-pr86-cold-start-learnability-20261110`，HEAD `38f2f3a469d833c176f1a6bbe2eab8257d59c6c7`
- 研究 Issue：[#86](https://github.com/Luyi1235/RA-LSBO/issues/86)，代码 PR：[#87](https://github.com/Luyi1235/RA-LSBO/pull/87)

PR86-A 的后验 prevalence-only 复核显示：在当前六任务/当前模型和数据分布下，label-blind IID/branch/coverage 方案的 native GP 到 n=48 尚未明显优于 Laplace 平滑训练先验率；到 n=96 才显示清楚的 Brier skill。该经验阈值不得外推为普遍样本量定律。类别平衡能提高排序但可能损害概率校准；单纯 maximin 覆盖也不能同时保证排序能力。

## PR86-B：研究问题与实现

**核心问题：** 在真实评价预算受限、初始样本稀少且存在删失反馈时，能否通过序贯选择风险信息更有价值的候选，使 held-out 风险概率预测优于随机锚点 prevalence-only 基线？

五个配对方法：
1. `iid_random`：随机评价；
2. `maximin`：层级特征空间覆盖；
3. `latent_uncertainty`：GP 潜在方差排序；
4. `boundary`：联合预测边界不确定性；
5. `calibration_aware_voi`：保留固定比例随机校准锚点，其余使用 BALD 风格互信息获取。

五臂共享 12 个随机初始锚点；之后每臂独立获取反馈，并在每次真实评价后重新拟合 native-censored GP。Held-out 测试标签只用于评分；未选候选标签不生成、不缓存、不暴露。校准截距只使用随机锚点标签和无标签查询特征。

主要指标：region/resource/joint Brier、相对锚点 Laplace prevalence baseline 的 Brier Skill Score、log loss、AUROC、AUPRC、ECE、false-feasible rate、分支与连续空间覆盖，以及 GP/kernel 真实费用。联合可行概率仍是两个边际头的乘积，是独立性近似而非联合依赖模型。

### 固定预算

| Profile | 任务×重复 | 每臂总观测 | 独立测试点 | 总数学评价 |
|---|---:|---:|---:|---:|
| CI | 2×1 | 24 | 64 | 272 |
| Smoke | 6×1 | 48 | 128 | 1,920 |
| Full | 6×3 | 48 | 512 | 12,672 |

Full 预计 3,330 次 GP refit、6,660 个 GP heads、53,280 次 kernel construction。飞机/RCAIDE/VAE/fantasy/目标优化均为零。本轮是风险学习与反馈获取研究，不是飞行安全或端到端飞机优化证据。

## 当前开发身份

- Issue：[ #88 — PR86-B 序贯风险信息获取](https://github.com/Luyi1235/RA-LSBO/issues/88)
- 开发分支：`research/phase-pr86b-sequential-risk-voi-20261110`
- 基础代码：PR86-A 固定 SHA `a5d5f20a4854de30939b590cb69126d1d5690ed5`
- 执行入口：`run_pr86b.py`
- 冻结协议：`docs/PR86_B_SEQUENTIAL_RISK_INFORMATION.txt`

当前实现包含协议、五类采集规则、anchor-only prevalence/截距校准、序贯 runner、独立 verifier、任务分层分析、测试与 Windows/Ubuntu + Python 3.13 CI。**只有同一最终代码 SHA 的全部 CI 任务通过后，才会下发工作机 smoke 指令。** Smoke 通过独立 verify、analysis、预算闭合和零 GP/kernel 错误门槛后，才允许 Full。

计划命令（默认只显示计划，不执行实验）：

```powershell
python -S run_pr86b.py --task plan --profile full
```

PR86-A、C2、PR74 cold-coverage 及其历史结果均保持原样，不重跑、不覆盖、不合并。
