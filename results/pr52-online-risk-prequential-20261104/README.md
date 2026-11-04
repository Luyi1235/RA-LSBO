# PR52：PR45 原生在线候选的逐轮风险预测审计

固定分析代码：`9a3879d45f05718d7356545d54d8835628d1bf99`  
数据来源：`D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z`（PR45，源 SHA `386082b`）  
环境：`Rcaide_vsp` / Python 3.9.25；SQLite 以 `mode=ro` 打开。

本分支只上传**只读诊断产物**，不修改 PR45/PR46/PR48/PR49/PR50/PR51 的原始证据。

## 本轮做了什么

读取 PR45 中 `ra_lsbo` 与 `full_latent` 两个原始主臂在 100 km / 140 km 两个任务上的在线选点记录：

- 校准种子索引 640–703（每块 64 个），测试种子 704–767（每块 64 个）；
- 每任务每臂 128 个种子 × 18 次在线评价 = 2,304 条，两个任务两个臂合计 **9,216 条在线 selected calls**；
- 逐条对齐 proposal 哈希、候选索引、settle ticket、设计哈希与物理契约；
- 使用 PR50 的真实任务完成/可用可行标签评分；
- **0 次新增飞机/RCAIDE 调用，0 次 GP/VAE 重新拟合**；
- 原 SQLite 运行前后 SHA256 不变（`a30b0dad...`），修改时间不变。

覆盖率：9,216 条中 4,608 条校准、4,608 条测试；测试中 4,568 条有模型概率，40 条（0.87%）因 `posterior_fit_skipped` 跳过（全部在 140 km 任务，覆盖模式 `scheduled_coverage`）。判定时保留这些调用在分母中，没有伪造概率。

## 主结果：原始在线概率与校准

`FP@.5` 是已知负例中原始概率 >0.5 的数量；`FP@.8` 是原始概率 >0.8 的假可行数量。校准为**离线单截距**，只在同任务同臂的其他种子上拟合，测试种子不参与，**不是 PR45 在线算法可获得的策略**。

### 100 km / ordinary

| 臂 | 代理 | 目标 | n | raw Brier | raw LogLoss | raw ECE | raw FP@.5 | raw FP@.8 | cal Brier | cal ECE | cal FP@.8 | baseline Brier | baseline ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_latent | constraint_product | feasible | 1152 | 0.1962 | 0.5845 | 0.0997 | 319/412 | 105 | 0.1891 | 0.0525 | 54 | 0.2595 | 0.1434 |
| full_latent | soc_mission | mission | 1152 | 0.0341 | 0.1165 | 0.0170 | 47/47 | 13 | 0.0331 | 0.0124 | 19 | 0.0775 | 0.1711 |
| ra_lsbo | constraint_product | feasible | 1152 | 0.1879 | 0.5630 | 0.0970 | 323/404 | 96 | 0.1789 | 0.0518 | 55 | 0.2601 | 0.1431 |
| ra_lsbo | soc_mission | mission | 1152 | 0.0302 | 0.1068 | 0.0078 | 45/45 | 12 | 0.0301 | 0.0070 | 15 | 0.0763 | 0.1717 |

### 140 km / sparse

| 臂 | 代理 | 目标 | n | raw Brier | raw LogLoss | raw ECE | raw FP@.5 | raw FP@.8 | cal Brier | cal ECE | cal FP@.8 | baseline Brier | baseline ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_latent | constraint_product | feasible | 1133 | 0.1999 | 0.5856 | 0.1262 | 343/516 | 70 | 0.1810 | 0.0383 | 38 | 0.3233 | 0.2420 |
| full_latent | soc_mission | mission | 1133 | 0.1542 | 0.5042 | 0.0837 | 235/270 | 77 | 0.1502 | 0.0724 | 51 | 0.2883 | 0.3115 |
| ra_lsbo | constraint_product | feasible | 1131 | 0.1899 | 0.5530 | 0.0985 | 305/484 | 55 | 0.1805 | 0.0508 | 44 | 0.3308 | 0.2616 |
| ra_lsbo | soc_mission | mission | 1131 | 0.1326 | 0.4248 | 0.0546 | 190/225 | 49 | 0.1322 | 0.0457 | 47 | 0.2893 | 0.3356 |

完整 4 个代理（含 `legacy_completion_product`、`record_completion_as_mission`）见 `metrics_overview.md` 与 `prequential_risk_report.json` 的 `test_scores`。

## 关键发现

1. **原生可行概率系统性偏高、校准很差。**  
   100/140 km 的 raw ECE 为 0.097–0.126；在 0.5 门限下，已知不可行设计中 305–343 个（共 404–516 个）被判为可行。0.8 门限仍有 55–105 个假可行。  
   这不是“偶尔失准”：ordinary 的 `first_online_step` 已偏高，`boundary` 阶段最严重——边界阶段已知负例中 102/102、105/105（ordinary）和 139/140、137/139（sparse）的原始概率都 >0.5，Brier 0.255–0.269。

2. **离线单截距校准确实降低 Brier，但不能当作在线可部署的救兵。**  
   feasible 目标的校准 Brier 下降 0.0071–0.0190（4 个“任务×臂”的 cluster 95% CI 均不含 0），ECE 大致减半，FP@.5 从 305–343 降到 126–159。  
   但截距是在**其他种子**的在线结果上拟合的，原在线算法从未获得；这只能说明“原概率存在可校正的常数偏移”，不能说明 PR45 在线选点会被改善，也不能说明风险门限可用于新候选。mission 目标用 `soc_mission` 时校准收益很小（Brier 差 −0.0002 到 −0.0043，CI 均含 0）。

3. **任务完成风险：SOC 代理明显优于错误使用完成头。**  
   ordinary 的 `soc_mission` raw Brier 0.030–0.034，`record_completion_as_mission` 0.039–0.040；sparse 为 0.133–0.154 对 0.173–0.209。`record_completion_as_mission` 在 sparse 上严重失准（ECE 0.129–0.170），校准后可改善到 0.157–0.179，但仍差于 SOC 代理。这与 PR50 的结论一致：**status_completion 恒为 1，不是任务完成标签**。

4. **历史前缀基线在 Brier 上更差，但在 0.5 门限下假可行更少。**  
   历史基线 Brier 0.076–0.331，均差于原始/校准概率；但它在 sparse 上几乎从不给出 >0.5 的可行概率（FP@.5 仅 14–21/484–516），所以“假可行少”主要来自“几乎全部拒绝”，不是更好的判别。任何风险预算评估都必须同时报告覆盖率与拒绝率。

5. **风险费用预测不能当作失败预算保证。**  
   feasible 原始期望失败数比实际失败数少 104–143 次（每调用 −0.09 到 −0.13）；mission 代理在 ordinary 略高估、sparse 低估。报告中的 `no_failure_budget_guarantee=True` 是明确的限制，不是安全声明。

6. **两臂不可作因果表示比较。**  
   `ra_lsbo` 与 `full_latent` 选出的设计不同，误差差异包含选点差异；本报告只能描述“原在线方法已选样本上的预测质量”。140 km 的独立稀疏资格仍未建立，结论均为简化飞机模型上的探索性结果。

## 运行命令

```bash
# smoke
python -u run_ra_lsbo_online_risk_prequential_pycharm.py \
  --per-block 1 \
  --output "D:\LSBO\RA-LSBO-native-risk-prequential-no-aircraft-20261104-smoke"

# full（新目录）
python -u run_ra_lsbo_online_risk_prequential_pycharm.py \
  --output "D:\LSBO\RA-LSBO-native-risk-prequential-no-aircraft-20261104-full"
```

两轮均 exit 0；`status=COMPLETE_NATIVE_PREQUENTIAL_RISK_AUDIT`。

## 文件

- `prequential_risk_report.json`：完整报告（24 个 case/arm/stratum 组合，含分阶段指标）。
- `prequential_risk_predictions.csv`：9,216 条在线选点的逐条预测、校准值、真值与跳过原因。
- `smoke-console.log`、`full-console.log`：两轮完整控制台日志。
- `fixed_sha.txt`：固定分析 SHA、源 SHA、运行环境与调用计数。
- `pre_run_sha256.txt`、`post_run_sha256.txt`：源 `plan.json`/`real_calls.sqlite` 运行前后哈希。
- `metrics_overview.md`：4 个代理的完整指标表。

## 限制（必须随结果一起引用）

- 样本是模型已选中的设计，不是 IID qA 真值，也不是未选候选的反事实。
- 校准截距使用其他种子，属于回顾性诊断，不能回放为 PR45 在线策略。
- 0.5/0.8 门限的假可行计数是**事后观察**，不能解释为“换门限会选哪个未评价候选”，更不能证明安全。
- 未重新拟合 GP/VAE，未新增飞机/RCAIDE 调用；原始 SQLite 只读且哈希不变。
