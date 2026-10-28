# RA-LSBO 方法修订与两轮开发结果

日期：2026-10-21。用户授权保留四项创新方向、放开原实现路线。
本轮完成文献核查、新代码、两轮独立冻结的受控开发、真实数据回顾性诊断及核验。
新增真实MDA为0；旧全部版本累计500次真实调用仍已用完。

## 结论

框架有实质改进，但完整方法尚未达到“稳定优于强基线”的论文效果要求。
保留原默认和旧实验，不把预测评分改善写成优化效率已提高。

主要变化是：删失观测下的七类联合终止模型；带掩码余量和监督对比损失的辅助风险表示；
bootstrap熵差；固定参考温度校准与无延迟训练资格；自适应局部候选和双重风险记录。
第二轮进一步按条件概率链，将评估完成概率与完成后约束GP满足评分耦合。
GP约束联合项采用并集不等式的代数表达，不能因这一表达就声称GP概率有安全保证。

风险输入保留one-hot直接路径并添加VAE潜变量，因此这是混合表示研究，
并非已经证明纯潜空间优于one-hot。余量回归和对比损失同时消融，未分离二者贡献。
主要开发评分使用联合事件概率；第二轮使用完成分支的GP条件约束评分，
各失效事件保持归一化，避免重复乘入同一失败证据。

## 文献依据与创新边界

局部约束BO参考Eriksson与Poloczek，AISTATS2021，2002.08526；
潜空间局部搜索参考Maus等，2201.11872；监督度量学习参考Grosnit等，2106.03609；
集成熵差信息代理参考Houlsby等，1112.5745；约束边界探索参考Tian等，2402.07692。
这些组成技术已有先例。本项目当前可检验的方法假设是“部分标签机制表示、
完成条件下的约束余量耦合、参考校准与有预算选点的协调”，不能把组合实现视为
创新效果或原创性已获确认。详细核查见METHOD_REVISION_RESEARCH_20261021.md。

## 实验范围及账目

第一轮：adaptive_research_20261021T071809Z_6fba5696，种子61–63，66组。
第二轮：margin_research_20261021T072847Z_79e09810，种子71–73，66组。
补充6组直接联合分类器，使用第二轮完全相同的600条初始数据；
该补充在查看部分第二轮结果后设立，明确属于事后开发诊断。

累计138组，实际2853次廉价受控在线调用；
两轮共享离线初始/静态测试标签10800次另计，补充6组复用既有初始标签。
真值重放是廉价方程检查，也与在线费用分开记录。新增真实MDA为0。
所有提前停止、零改进及不利消融均保留，不能只报告正常完成的试验。
每运行上限24次、已观测不可行上限12次；全部新方法硬账本按1扣费，
预期风险是另记的软评分，不作为安全界。

累计1937个新方法选点的风险预测恢复回放通过；
两轮独立真值、初始配对、校准隔离、训练ID和硬账本审计通过，
6组补充控制也核对同一初始VAE/终止网络及逐次预测。
这不是全部GP采集值或原LSBO搜索路径的独立回放。
234项软件测试通过。原LSBO、物理模型、旧冻结源码和旧结果保持哈希一致。

第二轮原回顾性评分曾因float32 softmax求和略高于1而中止；
失败目录physical_retrospective保留。后在独立适配层中用float64成员归一化，
重新拟合当前参考温度并保存至physical_retrospective_normalized_v2。
此数值修订不改写66组在线试验及其970个已回放预测；不把修订后的历史评分
冒称为一个额外真实/受控优化试验。各模型最大修订量在diagnostics.json中记录。

## 优化结果

改进均相对每个配对运行自己的共同可行初值；不同种子初值不同。
两轮之间不能直接用平均改进差推断桥接效应，桥接效应另用同初始6组对照检查。

### 第一轮联合分类

| 任务 | 方法 | 实际调用 | 可行 | 平均改进% |
| --- | --- | ---: | ---: | ---: |
| five_mode | adaptive_full | 70 | 36 | 0.223066 |
| five_mode | adaptive_aggregate | 71 | 38 | 0.210709 |
| five_mode | adaptive_onehot | 72 | 41 | 0.227862 |
| five_mode | adaptive_no_auxiliary | 70 | 34 | 0.218730 |
| five_mode | adaptive_no_calibration | 70 | 36 | 0.225581 |
| five_mode | adaptive_no_information | 72 | 39 | 0.232426 |
| five_mode | adaptive_no_trust | 72 | 43 | 0.163338 |
| five_mode | old_ra | 72 | 54 | 0.168942 |
| five_mode | cbo | 72 | 72 | 0.236582 |
| five_mode | lsbo | 72 | 66 | 0.231567 |
| five_mode | random | 41 | 5 | 0.000000 |
| sparse_interior | adaptive_full | 57 | 23 | 0.115390 |
| sparse_interior | adaptive_aggregate | 56 | 20 | 0.106062 |
| sparse_interior | adaptive_onehot | 46 | 10 | 0.073739 |
| sparse_interior | adaptive_no_auxiliary | 40 | 4 | 0.073739 |
| sparse_interior | adaptive_no_calibration | 54 | 19 | 0.152006 |
| sparse_interior | adaptive_no_information | 49 | 13 | 0.108665 |
| sparse_interior | adaptive_no_trust | 60 | 24 | 0.137201 |
| sparse_interior | old_ra | 64 | 45 | 0.122423 |
| sparse_interior | cbo | 45 | 9 | 0.224252 |
| sparse_interior | lsbo | 57 | 21 | 0.200887 |
| sparse_interior | random | 38 | 2 | 0.000000 |

### 第二轮条件约束桥接

| 任务 | 方法 | 实际调用 | 可行 | 平均改进% |
| --- | --- | ---: | ---: | ---: |
| five_mode | adaptive_full | 72 | 66 | 0.278101 |
| five_mode | adaptive_aggregate | 72 | 64 | 0.278099 |
| five_mode | adaptive_onehot | 72 | 65 | 0.277796 |
| five_mode | adaptive_no_auxiliary | 72 | 66 | 0.277381 |
| five_mode | adaptive_no_calibration | 72 | 66 | 0.275705 |
| five_mode | adaptive_no_information | 72 | 72 | 0.274067 |
| five_mode | adaptive_no_trust | 72 | 63 | 0.268095 |
| five_mode | old_ra | 72 | 57 | 0.263900 |
| five_mode | cbo | 72 | 72 | 0.275541 |
| five_mode | lsbo | 72 | 65 | 0.277286 |
| five_mode | random | 43 | 7 | 0.000000 |
| sparse_interior | adaptive_full | 68 | 34 | 0.105914 |
| sparse_interior | adaptive_aggregate | 63 | 28 | 0.170687 |
| sparse_interior | adaptive_onehot | 68 | 33 | 0.145521 |
| sparse_interior | adaptive_no_auxiliary | 68 | 33 | 0.140360 |
| sparse_interior | adaptive_no_calibration | 68 | 34 | 0.105914 |
| sparse_interior | adaptive_no_information | 66 | 34 | 0.154360 |
| sparse_interior | adaptive_no_trust | 65 | 29 | 0.168717 |
| sparse_interior | old_ra | 56 | 34 | 0.000000 |
| sparse_interior | cbo | 42 | 6 | 0.137923 |
| sparse_interior | lsbo | 61 | 27 | 0.248644 |
| sparse_interior | random | 37 | 1 | 0.000000 |


![全部受控端点](../runs/method_revision_report_20261021/controlled_endpoints.png)

![第二轮与配对直接模型曲线](../runs/method_revision_report_20261021/paired_learning_curves.png)

普通任务第二轮完整模型与强基线接近，但稀疏任务仍不稳定；
不能声称完整RA稳定领先原LSBO。合并机制、one-hot、去信息探索等版本
在部分任务更好，当前证据不支持四模块均有独立正向贡献。
6组桥接配对诊断见报告JSON，初始网络相同，不把不同种子两轮均值误作配对。

## 真实数据回顾性预测

始终使用560训练、72参考；288条既有优化观测不进入训练。
它们包含262个唯一设计；以下按唯一设计等权，避免26次重复观测改变权重。
旧共同参考、第一轮、第二轮风险模型均在相同输入上比较。
这是已查看历史数据上的方法开发，不能据此声称新方法产生了更好的实际能耗。

| 模型 | 唯一设计原始Brier | 唯一设计校准Brier |
| --- | ---: | ---: |
| old / old_common_reference | 0.032574 | 0.052482 |
| direct / joint_full | 0.013915 | 0.013940 |
| direct / joint_aggregate | 0.012753 | 0.012870 |
| direct / joint_onehot | 0.011295 | 0.011499 |
| direct / joint_no_auxiliary | 0.018082 | 0.017488 |
| bridge / joint_full | 0.019463 | 0.020956 |
| bridge / joint_aggregate | 0.019406 | 0.020887 |
| bridge / joint_onehot | 0.019269 | 0.021006 |
| bridge / joint_no_auxiliary | 0.019531 | 0.020911 |

![唯一设计风险误差](../runs/method_revision_report_20261021/historical_risk.png)

首轮按全部288观测评分，完整模型原始Brier为0.015228，
旧共同参考为0.036512；合并机制为0.012997、one-hot为0.013080，
都比完整风险模型更好。完整模型虽改善预测，机制和潜变量优势仍未证实。
校准结果必须逐版本看，不能把校准自动视为增益。

全锚点族5、19各自留出时，新完整、one-hot、无辅助以及旧RF的任务失败召回均为0。
桥接只重分配完成后的可行/显式违规概率，不改变隐藏终止质量，
因此不声称它解决了失效区域迁移。其他四类真实隐藏失效仍无正例。

## 当前可用实现

- ra_lsbo/adaptive_risk.py：部分标签事件似然、掩码辅助监督、集成及温度参考校准。
- ra_lsbo/adaptive_system.py：固定参考在线学习、局部区域、风险价格和信息选点。
- ra_lsbo/margin_joint.py：完成条件下的约束GP桥接与全部权重恢复。
- ra_lsbo/normalized_margin_joint.py：独立的float32舍入修订，记录模型适配标识。
- tools/run_adaptive_research.py、tools/run_margin_research.py：两轮冻结与完整受控循环。
- tools/diagnose_adaptive_physical.py：禁用真实MDA的回顾性风险和全锚点族诊断。
- tools/audit_adaptive_research.py、tools/report_method_revision.py：隔离、账本、配对与结果汇总。

模型权重保存在各运行的encoder_*.pt、risk_*.pt；第二轮另有risk_*.terminal.pt。
桥接checkpoint包括目标/约束GP全部学习参数、缩放和终止网络，加载须绑定相同训练历史与源码。

## 下一步研究判断

已实现的联合表示和约束桥接可作为研究原型继续使用，当前不推荐直接替换
原LSBO或把完整组合推入昂贵真实确认。下一步优先解决阶段条件下的失效迁移、
小样本机制分解的负迁移及探索对目标改进的真实信息价值，而不是继续堆叠
无正例的独立头或增加校准复杂度。潜变量贡献必须继续作为待否证假设。

真实五机制的数据充分性仍需要合法物理工况下的实际失败证据，不能靠合成标签、
代码异常或修改物理模型补齐。本轮没有新增物理预算，也未启动新的真实优化计划。
应先在更多独立受控任务与预先冻结的更大种子组建立稳定优势，再设计新的真实验证。
