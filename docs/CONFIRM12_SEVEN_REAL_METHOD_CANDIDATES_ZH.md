# 七种实际 BO 策略候选：代码身份、差异与 PyCharm 执行

**阶段：仅算法代码和离线回归。所有正式论文确认资格仍为 0/7，不能直接执行 151,040 次飞机调用。**

本次在 PR #43 的真正 H1 接口和 PR #42 的结构化证据账本基础上开发，不重写飞机评价模型。

## 七种方法的真实源码身份（不是七个模拟标签）

| 确认计划方法名 | 实际复用/新增的实现 | 本阶段能做的事情 | 仍需独立审查 |
|---|---|---|---|
| ra_lsbo | H1 `hybrid_onehot`，目标潜空间/约束显式 one-hot | 原始 GP、VAE、18步状态机的入口 | 新主算法命名和创新归因，是否应另建原理贡献 |
| full_latent | H1 `corpus_stable` | 全潜空间实际 GP/VAE 优化器 | 同预算及信息公平 |
| full_onehot | H1 `onehot_stable` | 全显式 GP 和局部 SLSQP | 对照公平性 |
| reverse_representation | 基于 H1 的 `ReverseRepresentationCandidate` | 目标显式、约束潜空间的实际后验路由 | 回退/元数据/长期状态及独立消融资格 |
| interval_as_missing | 新 `IntervalMissingHybridCandidate` | 对 **GP 响应头** 抹除原始 exact/interval 证据中的 interval 项，保留 exact/原始物理历史、候选合法性、完成事件和可行 incumbent | 完整分离表示训练中的反馈、区间信息消融的科学估计量 |
| random | H1 原有 `random` | 真实合法候选随机选择策略 | 新共用初值分布和正式公平验收 |
| cbob_adapted | 新 `CBOBAdaptedResearchCandidate` 独立编写 EICB balanced feasibility + H1 EP 多约束头 | 实际 Gaussian exact/interval GP 训练与 EICB 选点 | 与 CBOB 作者模型、采集和域适配的独立数学核对与基线资格 |

**关于 CBOB。** 作者公开仓库 https://github.com/COLA-Laboratory/CBOB，论文 Wang & Li, AAAI 2024, DOI 10.1609/aaai.v38i14.29488。经 GitHub API 检查，仓库没有声明许可证；本项目不复制其源文件，仅依据公开数学定义独立实现适配。与作者原版的**重大差异**必须在论文中披露：(1) 原作者 GPy/GPflow 连续输入模型改为 H1 的显式层级表征/固定候选 Matérn 核与精确+区间 EP；(2) 作者固定 1000 Sobol 平衡集改为已解析预筛且同预算的 128 个合法全局候选池；(3) 原始 EICB 在部分失败反馈下模型表达与现有完成/多约束输出不同；(4) 原作者连续域的采集优化流程不同于当前离散混合全局池选择；(5) 冷启动无可行目标时以 balanced feasibility 代替 EI。故目前只能称**研究性 CBOB/EICB 适配候选**，不能写“官方 CBOB 原版实现”。

`CBOBAdaptedResearchCandidate` 采用单步提交-结算的真实状态机、动态平衡系数及分约束可行因子。最终融合模式为 EI×balanced feasibility；如果无可信可行目标则仅按后者选点。不会把 unknown/基础设施错误伪装成物理失败，任何未接受观测都不推进历史。

## 可运行的 PyCharm 零物理调用试验

在新工作树内选 `Rcaide_vsp` 解释器，打开根目录 `run_ra_lsbo_confirm_12workers.py`。默认右键 Run 仍仅输出 **BLOCKED** 预检。要执行**真实算法代码（但全部为内存构造反馈）**，在 PyCharm Run Configuration 的 Script parameters 填入：

`--mode offline_actual_seven --workers 12 --output D:\\LSBO\\confirm12-offline-new-001`

**输出必须是此前不存在的目录。** 该命令会运行两任务×七种真实算法候选的第一个 GP/VAE/局部搜索提案，最多使用 12 个 spawn 工人；在全局结构化离线账本上按任务各记录 12 个合成 shared initials、每算法 1 条合成 online：共 38 条，全部写有 `OFFLINE_SYNTHETIC_NO_PHYSICS`，既非飞机标签，也不参与论文结果或正式确认种子资格。程序主动禁止 `AircraftCase.evaluate` / `SimplifiedElectricAircraft.evaluate`。

结果：`offline_plan.json`、`offline_evidence.sqlite`、每方法 `offline_result.json`、`offline_summary.json`；有 worker 异常将保留真实日志与 INCOMPLETE，不自动重试。Windows CI 用 2 进程进行实际算法离线测试；工作机用 12 进程确认其运行能力/内存，不要求 24 小时完成。

## 仍未解决的正式执行闸门

1. **生产物理评价器+正式事务账本**：现有 SQLite 结构化 `DesignRecord` 测试账本仍然为 `RECORD_CONTRACT_TEST_ONLY`，不可证明真实物理调用来源。正式调用需同机签发唯一票据，获批准源/任务/环境 SHA，`problem.evaluate` 原始结束证据以及 started/settled/unknown/failed 不可变日志和独立审计。
2. **任务资格**：需要独立合法设计空间抽样器的 600 次资格调用与审查；旧资格计划 SHA 与 Windows LF/CRLF 原始字节冲突，不能改写既有证据或通过 normalize 充当原始哈希。
3. **七算法确认**：本次只证明有实际执行策略候选，不证明与原 CBOB 的算法等价、方法贡献或五个消融的科研可比性；种子 1,024/256 嵌套候选名单尚未完成全档案唯一性复核。
4. **Windows 12 工人**：需按本次真实 7 策略的离线启动回放检查同时常驻内存、线程限制与运行稳定性，不能用哈希吞吐作为替代。
5. **明确阶段批准**：以上资格均验收后单独登记 151,040-call 预算和执行源 SHA。未通过者应打印 BLOCKED 而不是删掉 `RuntimeError` 后启动模拟算法。

所有代码首次交付仅运行预检/离线试验。无新增飞机或 RCAIDE 模型调用。
