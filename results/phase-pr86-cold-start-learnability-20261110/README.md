# PR86-A：冷启动风险可学习性与训练覆盖诊断

固定代码 SHA：`a5d5f20a4854de30939b590cb69126d1d5690ed5`  
PR：<https://github.com/Luyi1235/RA-LSBO/pull/87>  
环境：`Rcaide_vsp` / Python 3.9.25 / Windows 10。

这是**离线监督学习诊断**，不是在线优化实验；不选择候选、不改变 EI/risk 采集函数，也不声称算法提升。`privileged_exact` 仅是信息上界，不是可部署基线。

## 1. 执行与预算

| 项目 | Smoke | Full |
|---|---:|---:|
| 任务数 | 6 | 6 |
| 每任务重复 | 1 | 3 |
| 训练池评价 | 1,152 | 3,456 |
| 测试集评价 | 768 | 9,216 |
| 数学评价合计 | 1,920 | 12,672 |
| 模型记录 | 144 | 720 |
| GP heads | 288 | 1,440 |
| kernels | 2,304 | 11,520 |

- 六个任务：`legacy_branin_sparse`、`legacy_rosenbrock_sparse`、`tradeoff_s0–s3`；`s4–s7` 保留未用。
- 训练池固定 192 个候选；测试集 smoke 128、full 512；训练/测试候选 ID 不重叠。
- 同一 task/repeat 的四策略、两种监督、五个嵌套大小共享同一个已付费 192 标签池和同一个测试集。
- 采样策略：`iid_uniform`、`class_balanced`（使用已观察训练标签，不可部署）、`branch_stratified`、`coverage_maximin`（冻结特征空间贪心 maximin，不看标签）。
- 嵌套大小：12/24/48/96/192。

## 2. 完整性

- 预检：9 项 protocol 测试、5 项 metrics 测试、15 项 native 测试，全部通过，0 skip；compileall/help/plan exit 0。
- smoke：`COMPLETE_PR86_LEARNABILITY`，144 行，`failed_model_rows=0`，1920/1920 calls，288 heads/2,304 kernels，0 errors/rejects；verify 168 files `VERIFIED_PR86_LEARNABILITY`；analysis `COMPLETE_PR86_ANALYSIS`。
- full：`COMPLETE_PR86_LEARNABILITY`，720 行，`failed_model_rows=0`，12,672/12,672 calls，1,440 heads/11,520 kernels，0 errors/rejects，1,052.3 s；verify 780 files `VERIFIED_PR86_LEARNABILITY`；analysis 240 行 `COMPLETE_PR86_ANALYSIS`。
- `heldout_labels_fit_isolated=true`，`nested_prefixes_verified=true`；测试标签只在预测后评分。
- VAE / fantasy / aircraft / RCAIDE：全部 **0**。
- 所有 started/settled 闭合，0 unknown/unsettled。

## 3. 训练类别与覆盖

每个 task/repeat 的 192 训练池中，joint feasible 正例只有 10–22 个（约 5–11%）；测试集固定 512。

| n | iid 正例/12 | class-balanced 正例/12 | branch 正例/12 | coverage 正例/12 |
|---|---:|---:|---:|---:|
| 12（六任务范围） | 0–2 | 6 | 0–1 | 0–1 |

- `class_balanced` 在 n12 强制 6/6；到 n24 多为 12/12，但 Rosenbrock sparse 的第三重复只有 12/12/10；到 n48 少数类耗尽后不再是 50/50；n192 回到池的真实比例。
- 因此 n12 的 50/50 是对真实 5–11% 先验的强烈重加权；这直接放大了 Brier/ECE（见下）。
- 分支覆盖：240 条曲线中 234 条达到 4/4 分支；未达 4/4 的只有 s2/s3 的部分 n12 iid、class-balanced 曲线。
- 连续覆盖（冻结特征空间最近邻距离均值，n12）：iid 0.514、class 0.434、branch 0.477、coverage **0.975**；n192 四策略都降到 0.147（同一完整池）。
- 覆盖更广并不等于排序更好：coverage_maximin 的 NN 距离最大，但 AUROC 最低。

## 4. native_censored 学习曲线（六任务均值）

联合可行 Brier / AUROC / ECE，每格为 6 个任务的均值（任务内先平均 3 个重复）：

| n | iid B / AUC / ECE | class B / AUC / ECE | branch B / AUC / ECE | coverage B / AUC / ECE |
|---:|---|---|---|---|
| 12 | .1173 / .652 / .1548 | .1864 / **.839** / .3382 | .1113 / .644 / .1469 | **.0911** / .614 / **.0848** |
| 24 | .1059 / .710 / .1387 | .1647 / **.886** / .2953 | .1063 / .735 / .1457 | **.0889** / .664 / **.0877** |
| 48 | .0887 / .785 / .1146 | .1122 / **.917** / .2119 | .0891 / .831 / .1291 | **.0810** / .718 / **.0782** |
| 96 | **.0501** / .906 / .0664 | .0603 / **.945** / .1161 | .0539 / .901 / .0729 | .0553 / .874 / **.0572** |
| 192 | .0313 / .971 / .0446 | .0313 / .971 / .0446 | .0313 / .971 / .0446 | .0313 / .971 / .0446 |

**n=192 时四个策略完全相同**：训练池只有 192 个候选，n192 就是完整池，采样顺序不再有影响。任何“策略差异”只存在于 n≤96。

### 关键模式

- `class_balanced` 在 n12/24/48 **AUROC 全部 6/6 任务最高**（均值 .839/.886/.917），但 Brier 和 ECE 显著最差（n12 .186/.338）。它改善了排序，却严重破坏了概率校准。
- `coverage_maximin` 在 n12/24/48 **Brier 4–5/6 任务最好**，ECE 也最低；但 AUROC 最差（.614/.664/.718）。更广的连续覆盖没有带来更好的判别。
- `iid_uniform` 与 `branch_stratified` 居中；n12 时 branch 在 Branin 上略优于 coverage。
- 到 n96，四策略 Brier/AUROC 收敛（Brier .050–.060，AUROC .874–.945）；n192 完全一致。

## 5. 逐任务 native Brier

每格为该任务 3 个重复均值的联合 Brier；n192 所有策略相同。

| 任务 | 策略 | n12 | n96 | n192 |
|---|---|---:|---:|---:|
| Branin sparse | iid / class / branch / coverage | .1208 / .1906 / .0973 / .0992 | .0419 / .0593 / .0330 / .0372 | .0241 |
| Rosenbrock sparse | iid / class / branch / coverage | .1074 / .1821 / .1251 / .0888 | .0327 / .0317 / .0403 / .0551 | .0192 |
| tradeoff_s0 | iid / class / branch / coverage | .1094 / .1801 / .1272 / .0980 | .0667 / .0799 / .0633 / .0595 | .0402 |
| tradeoff_s1 | iid / class / branch / coverage | .1272 / .1933 / .1100 / .0853 | .0513 / .0732 / .0797 / .0569 | .0319 |
| tradeoff_s2 | iid / class / branch / coverage | .1464 / .1980 / .0985 / .0878 | .0546 / .0585 / .0551 / .0622 | .0424 |
| tradeoff_s3 | iid / class / branch / coverage | .0923 / .1746 / .1097 / .0875 | .0534 / .0592 / .0523 / .0610 | .0302 |

- s2 是 n192 最难任务（.0424），Rosenbrock sparse 最易（.0192）。
- class-balanced 在六个任务的 n12 Brier 都是最高（最差）；其 AUROC 优势没有转化为概率校准。
- coverage 在 n12 的五个任务最好，Branin 略逊于 branch。

## 6. privileged_exact 信息上界

`privileged_exact` 使用完整 region/resource margin，只在训练行可见，仅作上界：

| 任务 | n192 Brier | n192 AUROC | n192 ECE |
|---|---:|---:|---:|
| Branin sparse | .0039 | .9999 | .0094 |
| Rosenbrock sparse | .0037 | 1.0000 | .0117 |
| tradeoff_s0 | .0067 | .9995 | .0166 |
| tradeoff_s1 | .0091 | .9992 | .0173 |
| tradeoff_s2 | .0083 | .9995 | .0203 |
| tradeoff_s3 | .0099 | .9984 | .0140 |

六任务均值：n12 Brier .0853 / AUROC .797；n192 Brier .0069 / AUROC .999 / ECE .0149。  
native n192 为 .0313 / .971 / .0446；完整 margin 上界的 Brier 约好 **4–9 倍**，但不能部署。

## 7. 六个任务逐项结论

1. **Branin sparse**：n12 coverage .0992、branch .0973 接近；class .1906 最差；n192 .0241，四策略相同。正例 15/192。
2. **Rosenbrock sparse**：n12 coverage .0888 最好，class .1821 最差；n192 .0192（全局最易）。正例 10–18/192。
3. **tradeoff_s0**：n12 coverage .0980、iid .1094；class .1801；n192 .0402。
4. **tradeoff_s1**：n12 coverage .0853、branch .1100；class .1933；n192 .0319。
5. **tradeoff_s2**：n12 branch .0985、coverage .0878；class .1980；n192 .0424（最难）。正例 10–22/192。
6. **tradeoff_s3**：n12 coverage .0875、iid .0923；class .1746；n192 .0302。

所有任务在 n192 的四策略相同；差异来自小样本前缀，而不是最终池。

## 8. 判读

- **冷启动样本量是主因**：n12→n192，native Brier 从 .09–.20 降到 .019–.042；AUROC 从 .61–.84 升到 .935–.996。
- **类别平衡不是免费的校准改进**：它把 n12 的 50/50 先验强加给 5–11% 的真实先验，AUROC 上升但 Brier/ECE 恶化；不能据此推荐 class-balanced 训练。
- **连续覆盖也不是充分条件**：coverage_maximin 的 Brier/ECE 好，但 AUROC 差；覆盖广不等于判别好。
- **n192 的“完全一致”是预算定义**：n192 就是完整 192 池，不是策略收敛的独立证据。
- **privileged_exact 是上界，不是方法**；它说明完整 margin 有信息，但不能在线获得。
- 所有比较都是 6 任务、3 重复、嵌套前缀的描述性诊断；没有在线选择、没有飞机/RCAIDE、没有安全保证。

## 9. 文件

- `smoke/`、`full/`：完整原始输出（datasets、fits、evaluations、progress、manifest、report、plan、source_receipt）。
- `smoke-analysis/`、`full-analysis/`：`learning_curves.json` / `.csv`（240 行 full）、manifest。
- `logs/`：预检、smoke/full 运行、verify、analysis 日志和退出码。
- `analysis_summary.json`、`summary_tables.json`、`case_table.csv`：机器可读汇总。
- `environment.json`、`environment_packages.txt`、`fixed_sha.txt`、`instructions.txt`、`protocol.txt`。
- `sha256_manifest.txt`、`.gitattributes`（`* -text`）。
