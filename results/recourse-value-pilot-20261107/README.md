# PR73：可观测反馈自适应价值消融

固定代码 SHA：`1949c3139d2256b35e1734af184248f551912ba0`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/73>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录保存 PR73 的完整原始证据与只读分析。所有评价都是真实数学评价；不是 PR72 的历史重建，也没有新增飞机/RCAIDE 调用。

## 1. 执行与完整性

- 代码 worktree：`D:\LSBO\RA-LSBO-pr73-20261107-152452`，HEAD 等于固定 SHA，`git status --porcelain` 为空。
- 结果根目录：`D:\LSBO\RA-LSBO-pr73-results-20261107-152452`；日志在外部 `logs/`。
- 预检：25 项 core 测试、18 项 runtime 测试通过，0 skip；compileall 与 `--help` exit 0。
- smoke：4 tasks × seed `999091`，48 initial + 32 online = **80** calls，16 trials，状态 `COMPLETE_SMOKE_TECHNICAL_ONLY`；verify 105 files，`future_pool_checks=8`。
- full：4 tasks × seeds `999101–999108`，384 initial + 768 online = **1,152** calls，128 trials，状态 `COMPLETE_RECOURSE_SCORING_PILOT`，1,165.5 s；verify 805 files，`future_pool_checks=320`。
- 评价账本：1,152 started / 1,152 settled，0 unknown；飞机/RCAIDE = 0 / 0。
- smoke+full 工作机合计 **1,232** 次数学评价；CI 的 80 次未在工作机重复。
- 真实在线历史按各 arm 自己的观测独立更新 GP/VAE；幻想计算固定当前表示与核参数，不用幻想数据训练 VAE。原始程序没有单独的 VAE 训练计数字段，本报告不编造该数字。

## 2. 四臂总览

| arm | 最终可行 /32 | 无解 | 冷启动发现 | 在线失败 /768 | future_pool_checks | 幻想拟合 | 回退 | 提议耗时 s | 自身单步改选 | 同状态评分分歧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `soft_gp_eligible` | 28 | 4 | 1 | 58 | 0 | 0 | 0 | 174.3 | — | — |
| `cei_myopic` | 27 | 5 | 0 | 65 | 0 | 0 | 0 | 183.8 | — | — |
| `cei_two_step_adaptive` | 27 | 5 | 0 | 64 | 160 | 11,528 | 0 | 392.8 | 41 | 22 |
| `cei_two_step_fixed_second` | 27 | 5 | 0 | 61 | 160 | 11,526 | 0 | 393.8 | 35 | 18 |

- adaptive 与 fixed 的最终可行数相同（27），但 adaptive 实际在线失败 **64**，fixed 为 **61**。
- soft_gp 以 28/32 可行、58 次失败为该批最佳；它不是主消融的等价准则，但作为旧参照不能被 adaptive 超过。
- 两个两步臂各 5 条无解轨迹；myopic 也 5 条；soft 4 条。

## 3. 主比较：adaptive vs fixed_second

每 case 8 个种子；W/L/T 为 feasible-first 胜负平；Δ 只在双方均可行时统计，单位为目标值。

| case | W/L/T | 双方可行 | 双方无解 | 路径改变 | Δ 均值 | Δ 中位数 | 最终可行（adaptive/fixed） | 在线失败（adaptive/fixed） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Branin ordinary | 3/1/4 | 8 | 0 | 7/8 | −0.204 | 0 | 8/8 | 1/0 |
| Branin sparse | 0/0/8 | 6 | 2 | 3/8 | 0 | 0 | 6/6 | 25/25 |
| Rosenbrock ordinary | 2/1/5 | 8 | 0 | 4/8 | +0.693 | 0 | 8/8 | 3/3 |
| Rosenbrock sparse | 0/1/7 | 5 | 3 | 3/8 | +1.351 | 0 | 5/5 | 35/33 |
| **合计** | **5/3/24** | **27** | **5** | **17/32** | **+0.395** | **0** | **27/27** | **64/61** |

**主消融没有显示正收益。** adaptive 只在 Branin ordinary 有 −0.204 的小目标优势；在 Rosenbrock ordinary/sparse 分别为 +0.693/+1.351，且 sparse 失败更多。32 个种子对里 24 对打平，5 胜 3 负；最终可行完全相同。

## 4. 次比较

### adaptive vs myopic

| case | W/L/T | 双方可行 | 路径改变 | Δ 均值 |
|---|---:|---:|---:|---:|
| Branin ordinary | 3/1/4 | 8 | 7/8 | +0.602 |
| Branin sparse | 0/0/8 | 6 | 7/8 | 0 |
| Rosenbrock ordinary | 1/2/5 | 8 | 7/8 | +3.802 |
| Rosenbrock sparse | 0/1/7 | 5 | 6/8 | +0.557 |
| **合计** | **4/4/24** | **27** | **27/32** | **+1.408** |

adaptive 没有超过 myopic；最终可行 27/27，失败 64/65，目标均值反而更高。

### fixed_second vs myopic

合计 **3/5/24**，双方可行 27，路径改变 26/32，Δ 均值 **+1.013**；final 27/27，失败 61/65。fixed 也没有超过 myopic。

### adaptive vs soft_gp

合计 **7/10/15**，双方可行 27，双方无解 4，路径改变 32/32，Δ 均值 **+7.903**；final 27/28，失败 64/58。adaptive 没有超过 soft_gp。

### myopic vs soft_gp

合计 **5/8/19**，Δ 均值 **+6.495**；final 27/28，失败 65/58。soft 在这一批仍更好。

## 5. 逐任务诊断

### Branin ordinary

- 四臂最终都 8/8 可行；失败 `soft 3 / myopic 0 / adaptive 1 / fixed 0`。
- adaptive vs fixed：3/1/4，7/8 路径改变，Δ 均值 −0.204；adaptive 略有目标优势，但多 1 次在线失败。
- adaptive vs myopic：3/1/4，Δ +0.602；fixed vs myopic 2/2/4，Δ +0.806。
- 冷启动无新可行；首次可行均在第一次在线评价后。
- 本任务没有候选池真值审计，不能把路径改变解释为“错过了池内好点”。

### Branin sparse

- 最终可行 `soft 7 / myopic 6 / adaptive 6 / fixed 6`；无解 `1/2/2/2`；失败 `21/27/25/25`。
- adaptive vs fixed：0/0/8，双方可行 6，双方无解 2，路径改变 3/8，目标差全 0。
- adaptive vs myopic：0/0/8，Δ 0；fixed vs myopic：0/0/8。
- 两个两步臂都未超过 myopic；soft 多恢复 1 个可行种子。
- 首次可行：soft 的 1 个种子到第 6 次在线才首次可行；其余已解运行均在第一次在线评价后。

### Rosenbrock ordinary

- 四臂 8/8 可行；失败 `soft 2 / myopic 3 / adaptive 3 / fixed 3`。
- adaptive vs fixed：2/1/5，4/8 路径改变，Δ +0.693（adaptive 更差）。
- adaptive vs myopic：1/2/5，Δ +3.802；fixed vs myopic 0/2/6，Δ +3.109；两个两步臂均未超过 myopic。
- 无冷启动；首次可行都在第一次在线评价后。

### Rosenbrock sparse

- 最终可行全部 5/8；无解全部 3/8；失败 `soft 32 / myopic 35 / adaptive 35 / fixed 33`。
- adaptive vs fixed：0/1/7，双方可行 5，双方无解 3，路径改变 3/8，Δ +1.351（adaptive 更差）。
- adaptive vs myopic：0/1/7，Δ +0.557；fixed vs myopic 1/1/6，Δ −0.794（fixed 略好）。
- soft 的唯一冷启动发现帮助其达到 5/8；两个两步臂都没有冷启动发现。
- 该任务与 Branin sparse 一起说明：两步臂没有提高可行性，且 adaptive 失败更多。

## 6. 机制

### 同状态评分分歧

- 两个两步臂共有 **122** 个相同 pre-outcome 状态；`Q_adaptive` 与 `Q_fixed` 的首根不同 **17/122**，最终选中根也不同 **17/122**。
- 逐臂决策计数：adaptive **22/160**，fixed **18/160**（包含非共享状态，故高于 122 状态中的 17）。
- 说明评分口径确实会改变一部分动作，但频率不高（约 14% 共享状态）。

### 相对自身单步最优的改选

- adaptive 在 41/160 个 depth=2 决策改选了自身 myopic 首动作；fixed 为 35/160。
- 两个两步臂最后一步都是 depth=1，0 次幻想拟合；三个 CEI 臂最后一步均无幻想（各 32 条轨迹）。
- 同状态下 adaptive 与 fixed 的最终首动作不同 17 次；这与上面的共享状态根差异一致。

### recourse_premium

- adaptive：160 个 depth=2 决策中 131 个 premium 非零；均值 **1.959**，中位数 **0.0836**，最大 **30.668**，最小 0；无负值。
- fixed：123 个非零；均值 **1.776**，中位数 **0.0497**，最大 **30.668**，最小 0；无负值。
- 正 premium 是有限节点下 `max` 与平均交换次序的代数差值，不是真实信息价值或安全保证。

### 分数可重算与下一轮候选

- 1,280 个 root（320 个 depth=2 决策 × 4 root）逐项重算 adaptive/fixed continuation、premium、scenario mass、fixed second index 和 Q1：**0 错误**；0 个负 premium。
- 320 个非末次两步决策的 4 个 root support 均与实际下一轮 `pool` **完全一致**；`future_pool_checks=320`，0 mismatch。smoke 为 8/8。
- 幻想 GP：adaptive 11,528 次 started=completed，fixed 11,526 次；fallback=0；幻想总耗时 375.5 s。
- 提议总耗时：adaptive 392.8 s、fixed 393.8 s，约为 myopic 183.8 s 的 2.14 倍。

## 7. 判读

- **主消融失败**：adaptive 相对 fixed 为 5/3/24，均值 +0.395（adaptive 更差），final 27/27，失败 64/61。没有证据表明“反馈后改选第二点”带来收益。
- **没有超过原 soft_gp**：adaptive 对 soft 为 7/10/15，final 27/28，失败 64/58；不能提升为默认方法。
- **也没有超过 myopic**：adaptive/fixed 对 myopic 分别为 4/4/24 与 3/5/24，均值 +1.408/+1.013。
- **机制作用弱**：122 个共享状态只有 17 个首根不同；premium 虽非零但中位数很小，且未转化为可行或目标收益。
- 唯一局部正向是 Branin ordinary 的 −0.204；它在其余任务被 +0.693/+1.351 抵消，不能作为总体结论。
- 正 premium 的代数非负性与真实收益无关；`future_pool_checks` 只验证候选支持一致，不验证未执行动作的反事实收益。

## 8. 限制

- 4 任务 × 8 种子 × 6 次在线，属于探索性 pilot；一次轨迹的多个决策不是独立种子复制。
- 没有飞机/RCAIDE，不能外推到工程任务或安全。
- 没有候选池真值审计；不能回答“池内机会/eligible 是否排除了好点/前八名”。
- 64 节点求积未使用；本实验固定 4 节点，不是求积收敛结论。
- 程序没有单独的 VAE 训练计数字段；不补造该计数。

## 9. 文件

- `smoke/`、`full/`：plan、source_receipt、report、progress、evaluations、manifest、shared_initial、trials（含 proposals/observations/result、representation checkpoints、future_values/future_supports/recourse_decomposition）。
- `logs/`：25+18 测试、compileall、help、smoke/full 控制台与 verify。
- `analysis_summary.json`、`case_table.csv`：汇总与逐 case 表。
- `integrity_check.json`：计数、分数重算、支持一致性、最后一步和账本核验。
- `fixed_sha.txt`、`commands.txt`、`environment.txt`、`environment_packages.txt`、`instructions.txt`、`sha256_manifest.txt`、`protocol.txt`。
- `.gitattributes`：`* -text`，避免换行转换破坏证据。
