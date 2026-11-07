# PR75 Phase A：风险代理监督学习诊断

固定代码 SHA：`bc0164882faa6ff6390d62d0e1452717af00416b`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/75>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

本目录是 Phase A 监督学习的完整封存结果。它不是在线 BO 对比，也不重新训练 VAE：每个任务/重复只生成一次训练与测试数据，跨样本量和模型复用；测试真值也计入评价成本。

## 1. 执行、状态与费用

| 项目 | 结果 |
|---|---|
| 34 项核心测试 | 通过，0 skip |
| 5 项原生测试 | 通过，0 skip |
| compileall / `--help` / `--task plan --profile full` | exit 0 / 0 / 0 |
| smoke | 912 次数学评价 = 144 train + 768 test；48 model rows；`COMPLETE_LEARNABILITY` |
| smoke verify | 66 files，`VERIFIED_LEARNABILITY`；五类篡改检测通过 |
| full | 12,672 次数学评价 = 3,456 train + 9,216 test；360 model rows；`COMPLETE_LEARNABILITY`；694.0 s |
| full verify | 402 files，`VERIFIED_LEARNABILITY`；五类篡改检测通过 |
| 工作机正式评价合计 | **13,584** |
| 失败模型行 | 0 |
| 飞机 / RCAIDE / VAE / 幻想拟合 | 0 / 0 / 0 / 0 |
| 数学账本 | 12,672 started / 12,672 settled；0 unknown |
| GP heads / kernels | 540 / 4,320；0 errors；0 kernel rejections |

CI 的 912 次单独计费，工作机没有重复执行 `--profile ci`。训练数据、测试数据和真值只生成一次；跨 n 的训练集是嵌套前缀。

## 2. 任务与标签覆盖

`initial` 行是训练集前 n 个点；测试集固定 512 点。测试集的 `exact/interval` 是每个 512 点测试集的平均值（3 个重复合计除以 3）。

| task | n | branches | joint feasible | region exact / interval | objective exact / missing | test pos / neg | test exact / interval |
|---|---:|---|---:|---:|---:|---:|---:|
| legacy_branin_sparse | 12 | 4/4/4/0 | 0 | 0 / 12 | 0 / 12 | 40 / 472 | 51.3 / 460.7 |
| legacy_branin_sparse | 192 | 59/42/52/39 | 10 | 13 / 179 | 13 / 179 | 40 / 472 | 51.3 / 460.7 |
| legacy_rosenbrock_sparse | 12 | 3/4/4/1 | 2 | 2 / 10 | 2 / 10 | 54 / 458 | 51.7 / 460.3 |
| legacy_rosenbrock_sparse | 192 | 43/49/53/47 | 16 | 18 / 174 | 18 / 174 | 54 / 458 | 51.7 / 460.3 |
| tradeoff_s0 | 12 | 5/1/5/1 | 1 | 2 / 10 | 2 / 10 | 57 / 455 | 57.3 / 454.7 |
| tradeoff_s0 | 192 | 53/45/55/39 | 17 | 20 / 172 | 20 / 172 | 57 / 455 | 57.3 / 454.7 |
| tradeoff_s1 | 12 | 3/5/2/2 | 1 | 1 / 11 | 1 / 11 | 33 / 479 | 53.3 / 458.7 |
| tradeoff_s1 | 192 | 52/42/43/55 | 18 | 23 / 169 | 23 / 169 | 33 / 479 | 53.3 / 458.7 |
| tradeoff_s2 | 12 | 2/1/4/5 | 2 | 2 / 10 | 2 / 10 | 31 / 481 | 48.0 / 464.0 |
| tradeoff_s2 | 192 | 51/57/44/40 | 14 | 17 / 175 | 17 / 175 | 31 / 481 | 48.0 / 464.0 |
| tradeoff_s3 | 12 | 2/3/3/4 | 0 | 0 / 12 | 0 / 12 | 35 / 477 | 53.7 / 458.3 |
| tradeoff_s3 | 192 | 35/59/41/57 | 12 | 16 / 176 | 16 / 176 | 35 / 477 | 53.7 / 458.3 |

- 12 点训练集里，Branin sparse 和 s3 没有任何 exact region 标签；所有任务都只有 0–2 个 joint feasible。
- 到 192 点，exact region 标签也只有 13–23 个，objective 仍以 missing 为主（169–179/192）。
- 12 点的分支覆盖明显不均（例如 Branin branch 3=0，s0 branch 1=1）；192 点后四个分支均在 35–59 之间。
- 测试集固定且类别不平衡：正例只有 31–57/512；AUROC 缺失时按 null 保留，不补 0。

## 3. Learning curves：joint feasibility Brier

每格为 3 个重复的均值；越小越好。`joint_prevalence` 是常数基线，`native_sign` 和 `privileged_exact` 为诊断参照（后者使用完整训练约束值，不是公平信息条件）。

| task | model | n=12 | n=24 | n=48 | n=96 | n=192 |
|---|---|---:|---:|---:|---:|---:|
| legacy_branin_sparse | joint_prevalence | .0861 | .0855 | .0864 | .0865 | .0869 |
|  | native_censored | .0926 | .1006 | .0909 | .0440 | **.0253** |
|  | native_sign | .0847 | .0834 | .0821 | .0785 | **.0646** |
|  | privileged_exact | .0749 | .0469 | .0282 | .0148 | **.0073** |
| legacy_rosenbrock_sparse | joint_prevalence | .0823 | .0792 | .0794 | .0785 | .0783 |
|  | native_censored | .1153 | .0903 | .0529 | .0348 | **.0202** |
|  | native_sign | .0875 | .0764 | .0723 | .0689 | **.0625** |
|  | privileged_exact | .0811 | .0388 | .0225 | .0099 | **.0060** |
| tradeoff_s0 | joint_prevalence | .0888 | .0872 | .0865 | .0865 | .0864 |
|  | native_censored | .1716 | .1086 | .0690 | .0434 | **.0326** |
|  | native_sign | .0929 | .0839 | .0808 | .0760 | **.0597** |
|  | privileged_exact | .0887 | .0584 | .0425 | .0156 | **.0082** |
| tradeoff_s1 | joint_prevalence | .0757 | .0743 | .0738 | .0720 | .0718 |
|  | native_censored | .1281 | .1152 | .0788 | .0742 | **.0387** |
|  | native_sign | .0725 | .0722 | .0690 | .0635 | **.0535** |
|  | privileged_exact | .0818 | .0605 | .0290 | .0201 | **.0090** |
| tradeoff_s2 | joint_prevalence | .0872 | .0767 | .0743 | .0684 | .0676 |
|  | native_censored | .1708 | .1010 | .0734 | .0746 | **.0572** |
|  | native_sign | .0850 | .0758 | .0685 | .0625 | **.0551** |
|  | privileged_exact | .0795 | .0663 | .0295 | .0180 | **.0085** |
| tradeoff_s3 | joint_prevalence | .0805 | .0759 | .0758 | .0758 | .0759 |
|  | native_censored | .1015 | .1090 | .0989 | .0542 | **.0363** |
|  | native_sign | .0873 | .0745 | .0729 | .0703 | **.0619** |
|  | privileged_exact | .0800 | .0642 | .0358 | .0191 | **.0081** |

主要形态：

- `native_censored` 在 12 点明显差于常数基线；到 48–96 点才追上，192 点在 5/6 任务成为最好的非特权模型。
- `native_sign` 在 12 点就接近基线，但后期提升幅度较小；在 s2 的 192 点反而略优于 censored（.0551 vs .0572）。
- `privileged_exact` 在所有任务、所有 n 都最好；这是“拥有完整约束值”的诊断参照，不能当作公平信息条件下的新方法。
- s2 的 censored 在 n=96 出现轻微反弹（.0746 > .0734 at n=48），说明曲线并非严格单调；不能用单点波动挑样本量。

### 相对常数基线的 Brier 降低（n=192）

| task | native_censored | native_sign | privileged_exact |
|---|---:|---:|---:|
| legacy_branin_sparse | 70.9% | 25.7% | 91.6% |
| legacy_rosenbrock_sparse | 74.2% | 20.1% | 92.3% |
| tradeoff_s0 | 62.2% | 30.9% | 90.5% |
| tradeoff_s1 | 46.1% | 25.5% | 87.5% |
| tradeoff_s2 | 15.4% | 18.6% | 87.4% |
| tradeoff_s3 | 52.2% | 18.5% | 89.3% |

s2 是唯一 censored 低于 sign 的任务；其正例最少（31/512），且 censored 在 96 点已经接近平台。

## 4. n=192 的完整指标

| task | model | Brier | LogLoss | AUROC | P@10% | R@10% | FF | FI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_branin_sparse | joint_prevalence | .0869 | .3222 | .500 | .103 | .112 | .000 | 1.000 |
|  | native_censored | .0253 | .0932 | .987 | .789 | .865 | .017 | .196 |
|  | native_sign | .0646 | .2267 | .918 | .513 | .572 | .004 | .757 |
|  | privileged_exact | .0073 | .0275 | .9997 | .897 | .976 | .001 | .048 |
| legacy_rosenbrock_sparse | joint_prevalence | .0783 | .2912 | .500 | .115 | .126 | .000 | 1.000 |
|  | native_censored | .0202 | .0791 | .992 | .724 | .863 | .019 | .137 |
|  | native_sign | .0625 | .2097 | .893 | .372 | .439 | .000 | 1.000 |
|  | privileged_exact | .0060 | .0234 | .9998 | .827 | .981 | .003 | .012 |
| tradeoff_s0 | joint_prevalence | .0864 | .3146 | .500 | .090 | .094 | .000 | 1.000 |
|  | native_censored | .0326 | .1250 | .968 | .744 | .781 | .024 | .213 |
|  | native_sign | .0597 | .2072 | .919 | .551 | .579 | .004 | .864 |
|  | privileged_exact | .0082 | .0315 | .9998 | .897 | .958 | .002 | .045 |
| tradeoff_s1 | joint_prevalence | .0718 | .2745 | .500 | .096 | .134 | .000 | 1.000 |
|  | native_censored | .0387 | .1410 | .961 | .590 | .775 | .030 | .286 |
|  | native_sign | .0535 | .1902 | .938 | .526 | .692 | .001 | .993 |
|  | privileged_exact | .0090 | .0330 | .9992 | .750 | .985 | .001 | .067 |
| tradeoff_s2 | joint_prevalence | .0676 | .2627 | .500 | .064 | .091 | .000 | 1.000 |
|  | native_censored | .0572 | .2078 | .906 | .442 | .614 | .020 | .488 |
|  | native_sign | .0551 | .2007 | .854 | .346 | .491 | .004 | .886 |
|  | privileged_exact | .0085 | .0294 | .9990 | .705 | .992 | .002 | .094 |
| tradeoff_s3 | joint_prevalence | .0759 | .2860 | .500 | .071 | .086 | .000 | 1.000 |
|  | native_censored | .0363 | .1336 | .949 | .609 | .747 | .024 | .323 |
|  | native_sign | .0619 | .2201 | .845 | .365 | .449 | .004 | .912 |
|  | privileged_exact | .0081 | .0303 | .9996 | .814 | 1.000 | .001 | .051 |

`FF` 为假可行率，`FI` 为假不可行率。native_sign 的假可行率极低，但假不可行率接近 0.76–1.00；censored 在两者之间更平衡。特权模型同时压低两者，但不具公平信息条件。

## 5. 边界与构型分层（n=192，region_boundary 的 joint Brier）

| task | joint_prevalence | native_censored | native_sign | privileged_exact |
|---|---:|---:|---:|---:|
| legacy_branin_sparse | .4330 | .2143 | .3370 | .1004 |
| legacy_rosenbrock_sparse | .3046 | .1862 | .2445 | .1045 |
| tradeoff_s0 | .3566 | .2171 | .2753 | .0984 |
| tradeoff_s1 | .3111 | .2586 | .2431 | .1014 |
| tradeoff_s2 | .3054 | .1632 | .2578 | .0962 |
| tradeoff_s3 | .3582 | .2206 | .2930 | .0842 |

- 边界样本明显比总体更难；四模型的边界 Brier 都高于总体。
- censored 在 5/6 任务的边界优于 sign（s1 例外，.2586 vs .2431）。
- 按四个构型 branch 0–3 比较，censored 在 19/24 个 branch 上 Brier 低于 sign；s2 只有 1/4，s0/s1 各 3/4。s2 的构型异质性最强。

## 6. GP 与计算

- 540 个 GP heads（每个 model row 3 个 head）全部 started=completed，0 errors。
- 4,320 个核构造 started=completed，0 errors，`kernel_candidates_rejected=0`。
- 选核候选全部记录在 `fits/.../receipt.heads`；本次没有拒绝候选，也没有选核失败。
- 模型失败行 0；没有 `COMPLETE_WITH_MODEL_FAILURES`。
- 评价账本 12,672 started / 12,672 settled，0 unknown。
- 训练数据跨 n 复用；没有为每个 n 重新购买评价。

## 7. 判读

- **样本量瓶颈是真实存在的**：censored 在 12 点几乎不可用，48–96 点才接近或超过常数基线，192 点才明显领先。不能把 12 点的失败解释为“反馈无用”。
- **反馈/标签类型仍有明显差异**：仅用符号的 native_sign 比 censored 更早接近基线，但后期上限较低；censored 在 5/6 任务最终更好，s2 例外。
- **完整约束值带来巨大但不等价的优势**：privileged_exact 在 6/6 任务、所有 n 最优；这支持“约束值本身重要”，不支持“公平条件下风险模型已解决”。
- **s2 是最困难的反馈情形**：正例最少、censored 在 96 点反弹、最终被 sign 略超；不能用一个总体均值掩盖。
- 这不是在线优化收益：全部是离线监督学习曲线，没有 BO 选点、VAE 训练或飞机/RCAIDE 评价。
- 三个 repeat 只是探索性稳定性检查；测试点数量不是独立实验重复数。

## 8. 文件

- `smoke/`、`full/`：完整 report、plan、source_receipt、datasets、fits、manifest、progress、evaluations。
- `full/datasets/<task>/r{0,1,2}/{train,test}.json`：训练/测试观测与 truth；测试 truth 已计费。
- `full/fits/<task>/r*/n*/<model>.json`：360 个模型行，含预测、标签计数、三 head 指标、核 receipt、费用。
- `logs/`：34+5 测试、compileall、help、plan、smoke/full、verify、五类篡改检测。
- `learning_curves.csv`、`label_coverage.csv`、`n192_metrics.csv`、`boundary_n192.csv`、`branch_n192.csv`、`analysis_summary.json`。
- `protocol.txt`、`RESEARCH_DIRECTION.md`、`instructions.txt`、`fixed_sha.txt`、`environment.txt`、`integrity_check.json`、`sha256_manifest.txt`。
- `.gitattributes`：`* -text`，防止 Git 换行转换破坏证据字节。

### 限制

- 6 个任务 × 3 repeat 是探索性诊断，不是确认性学习曲线实验。
- 测试集固定且类别不平衡；AUROC 在某些分层可能不适用，缺失保持 null。
- privileged_exact 使用完整训练约束值，是诊断参照，不是公平信息条件下的算法。
- 监督误差改善不等于在线优化收益；反馈信息归因和更长预算验证仍未完成。
