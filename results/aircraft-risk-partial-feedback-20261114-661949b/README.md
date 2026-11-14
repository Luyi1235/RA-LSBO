# PR103：部分/精确约束反馈对首可用解发现的影响

固定代码 SHA：`661949b53dac0213aa4d1dc68d8ea1eb9a83766d`  
分支：`research/aircraft-risk-partial-feedback-20261114`

## 1. 执行

| 配置 | 种子 | 评价数 | 状态 |
|---|---|---:|---|
| feedback-smoke | 28429000 | 32 | COMPLETE_AIRCRAFT_RISK_COLD_START |
| feedback-nominal | 28432100–28433600（16） | 1,600 | 同上 |
| feedback-headwind（4 m/s） | 28434000–28435500（16） | 1,600 | 同上 |
| **合计** | | **3,232** | |

17 项专项测试通过；三套 verify/analyze 全部通过；两套正式 discovery_pairs.csv 各 64 行；0 RCAIDE/MDA。四臂：feedback_exact、feedback_sign、feedback_bounds、feedback_coverage。

## 2. 初值无解人群的主比较

analysis.json.discovery_summary 的 feedback_bounds vs feedback_sign：

| 工况 | 初值无解 seed | W/L/T | 候选/对照发现 | 受限平均 Δcall | 中位 | 留一法范围 |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 13/16 | 5/3/5 | 13/13 | −1.31 | 0 | [−1.50, −0.83] |
| headwind | 15/16 | 2/6/7 | 13/14 | +1.07 | 0 | [0.29, 1.43] |

无风 bounds 略早；逆风 bounds 反而更慢，方向翻转。区间/符号相对 exact 明显更早：nominal sign−exact −6.62、bounds−exact −7.92；headwind sign−exact −6.87、bounds−exact −5.80。bounds 对 coverage：nominal −5.00、headwind −3.80。

## 3. 同一历史风险预测（n=20）

### nominal

| arm | joint Brier | joint recall | joint FF | SOC Brier | SOC recall | SOC FF |
|---|---:|---:|---:|---:|---:|---:|
| feedback_sign | .0927 | .379 | .0504 | .1468 | .652 | .1494 |
| feedback_bounds | .1014 | .324 | .0464 | .1655 | .552 | .1428 |
| feedback_coverage | .0940 | .357 | .0364 | .1635 | .518 | .1187 |
| feedback_exact | .1040 | .201 | .0194 | .2107 | .815 | .5364 |

### headwind

| arm | joint Brier | joint recall | joint FF | SOC Brier | SOC recall | SOC FF |
|---|---:|---:|---:|---:|---:|---:|
| feedback_bounds | .0532 | .262 | .0104 | .1454 | .438 | .0787 |
| feedback_sign | .0554 | .226 | .0081 | .1376 | .502 | .0835 |
| feedback_coverage | .0637 | .250 | .0263 | .1245 | .498 | .0539 |
| feedback_exact | .0532 | .071 | .0000 | .2158 | .887 | .7130 |

exact 的 SOC 召回最高但假可行率也最高；不能视为更好校准。

## 4. 判读

- 区间/符号反馈比 exact 更早发现首可用解，两个工况一致（约早 5.8–7.9 次调用），是最稳定信号。
- bounds vs sign 没有稳定胜负：无风 −1.31、逆风 +1.07；不能写成通用改进。
- n20 风险预测中 sign/bounds 的 SOC 假可行率明显低于 exact，但没有一个臂在两个工况都占优。
- 只评价首可用解发现和风险预测，不评价最终能量或优化闭环；16 seed/工况、简化飞机、无 RCAIDE，结果探索性。

## 5. 文件

feedback-smoke/、feedback-smoke-analysis/、feedback-nominal/、feedback-nominal-analysis/、feedback-headwind/、feedback-headwind-analysis/、logs/、fixed_sha.txt、environment.txt、environment_packages.txt、SHA256SUMS.txt、README.md；.gitattributes 为 * -text。
