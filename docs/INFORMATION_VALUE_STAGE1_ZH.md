# RA-LSBO 信息价值路线：第一阶段协议与工作机执行说明

日期：2026-11-06。代码基线：`18b48e25dda6ff7d17296087981d0c4dc3ce99b2`（PR63）。
本阶段是用户授权的新研究路线第一批交付，不修改 main、PR54–65 的实现或历史结果。
代码由 ChatGPT 负责；工作机 LLM 只读取固定 SHA、运行、核验和上传原始结果。

## 1. 本轮解决什么，不解决什么

本轮不是再叠加风险惩罚，也不是宣称完整的新 GP 两步优化器已经有效。
交付两个相互独立的模块：

1. **真实 GP/VAE 的候选机会诊断**：运行未改变的 PR63 `soft_gp`，保存评价前候选、模型预测和实际选点；所有在线轨迹结束并封存后，另付数学评价成本取得候选池真值。定位候选生成、支持域/风险排序、选择决策各层的损失。
2. **有限假设模型下的两步决策内核**：显式枚举可观测反馈，进行后验更新及两步收益/成本决策。含失败信息消融、剩余一次/两次预算、无信息失败负对照、高成本停止。它是构造机制验证，不是拟合 GP、不是真实数学 BO 比较、更不是飞机验证。

完整 GP/VAE 的幻想后验更新与新策略在线比较尚未在本阶段接入。取得本轮诊断后，再决定优先改候选、模型还是决策，避免同时改变所有组件。

## 2. 冻结实验计划

任务：原自定义 Branin ordinary/sparse、Rosenbrock ordinary/sparse。
仅一个轨迹策略：原 PR63 `soft_gp`；原 12 步内部阶段安排保持不变。这是基线诊断，不能当成已实现匹配实际时域的新算法。
初始 12 点，在线原候选池 128，记录原始预测和每次真实反馈。无后验门槛调参，无跨运行标签共享。

| 项目 | smoke | full |
|---|---:|---:|
| 每任务种子 | 998591 | 998601–998604 |
| 真实在线轨迹 | 4 | 16 |
| 每轨迹初始评价 | 12 | 12 |
| 每轨迹在线评价 | 2 | 6 |
| 保存检查点（零起算） | 1 | 1、4 |
| 每检查点审计候选数 | 160 | 512 |
| 初始评价总数 | 48 | 192 |
| 在线评价总数 | 8 | 96 |
| 事后候选真值审计评价 | 640 | 16,384 |
| **数学评价总计/上限** | **696** | **16,672** |
| 飞机 / RCAIDE 新评价 | **0 / 0** | **0 / 0** |

smoke 与 full 种子不重叠，两次成功合计 17,368 次廉价数学评价；其中只有 104 次是在线选点调用。
禁止把 16,672 写成新算法在线评价数。审计重复某个既有选点时，仍然实际调用并计费，不把额外真值当免费标签。
检查点 1 和 4 避开原冷启动每 5 步覆盖日历的计划覆盖位置，但模型失败或门退化仍可能产生无预测覆盖，必须原样保留。
扩展池以前 128 个原候选为前缀，另外使用独立无标签随机流补足；扩展部分**不进入在线选点、不重算 GP 或 EI 归一化**。

## 3. 信息隔离和费用合同

- 每次 proposal、原池身份、检查点 snapshot 均先落盘，然后才评价实际选点。
- 一次记录为 `started` 即记一次启动；异常另记 `error`，不能补造 completed/不可行标签。
- 所有轨迹（含表示 checkpoint）完成后写 `trace_seal.json`，之后才启动第一个候选真值审计调用。
- 审计阶段创建新的问题实例，不保留优化器，不调用 `accept`，不拟合 GP/VAE。
- 每个审计检查点之前及所有审计之后验证封存文件 SHA256 不变。
- 无 GP 风险收据的实际覆盖点不获得伪造概率；候选池预测质量和实际选点预测质量分开报告。
- 不可行的可行目标列为 null，不能用大惩罚数补齐。已有有限目标但不可行也不当作可交付目标。
- 原父提交全部 Git 文件对象必须保持不变；源目录必须干净。输出和控制台日志放在源 worktree 外。
- 数学阶段显式封锁两个飞机 evaluator 的调用；不读取 PR45–65 历史结果或 SQLite。

## 4. 核心指标和解释边界

逐任务、逐种子、逐检查点报告，不跨目标尺度汇总均值：

- 原 128 池、扩展池、实际 GP 池、eligible 支持池、按原效用排序前 8 点中的可行数量与最优可行目标；
- 在有 incumbent 时报告池内最大改进机会；冷启动保留 null 并报告可行发现机会，不编造目标；
- 实际选点可行目标与原池最优可行目标之差，仅在实际选点可行时定义；
- 同一批 GP 候选上的 Brier、预测风险均值、实际失败率以及预测风险 <=0.2 的失败计数；
- 模型收据缺失、覆盖、候选支持筛除的机会损失；原始 GP/VAE 成本和 checkpoint 可在 proposal/representation 中追溯。

池内真值选择器是诊断参照，不是可部署算法；固定历史下的错过机会不是替代策略的完整在线收益。
扩展池没有 GP 分数，不能由此声称“扩大池后的优化器胜出”。四种任务每种仅四个种子，不做确认性统计优越结论。
这些状态由 `soft_gp` 自适应产生，诊断不能直接外推其他策略的状态分布。

## 5. 两步机制内核

给定有限假设集合及后验权重，对行动 a 的可观测反馈 o 分组：

`Q_h(a) = -cost(a) + sum_o p(o | D,a) V_(h-1)(D+o)`

`V_h = max(stop_value, max_a Q_h(a))`，实现深度最多 2，且不超过真正剩余调用数。
已交付效用是构造问题事先定义的终端效用，不是对缺失目标的填补。每次成功和失败调用都支付正成本；停止成本为零且无结果标签。
规划器只收到假设集合，不接收实际世界索引。实际世界仅在决定行动后产生反馈。
失败时 `delivered_value=None`，可保留 `region:(0,+inf)` 和已经测得的辅助响应；消融保留已知失败事件，但去掉失败后的额外证据。

刻意构造的四世界验证含：只有一个岛有高质量解；失败探针可能揭示是哪个岛；另一个同样必失败的探针不提供该信息。
这用于证伪代码是否真正使用信息和时域，不用来证明该思想首次提出或真实 GP 一定受益。
单步应选择安全交付；两步完整反馈可选择信息探针；没有额外失败证据时该偏好消失；高成本时实际停止。

## 6. 代码和运行

- `ra_lsbo/experiments/information_value_stage1/mechanism.py`：纯标准库决策内核。
- `audit.py`：候选绑定、机会和预测指标。
- `runner.py`：真实原 GP/VAE 轨迹与后置审计隔离，所有评价记账。
- `io.py`：父提交锁定、只写一次、manifest、校验与分任务 ZIP 分包。
- `run_ra_lsbo_information_stage1_pycharm.py`：工作机入口。
- `tests/test_information_value_stage1.py`：纯单元/替身集成测试，不运行真实 GP/VAE。

在全新 worktree 切换到交接消息中的**完整固定 SHA**。使用现有 `Rcaide_vsp` 环境和原 LSBO 底座。
不要安装/升级依赖、修改物理参数、迁移老数据、编辑代码或猜测缺失接口。源文件 BOM、换行和编码不应手工转换。
首先运行：

```text
python -m unittest discover -s tests -p test_information_value_stage1.py -v
python -m compileall -q ra_lsbo/experiments/information_value_stage1 run_ra_lsbo_information_stage1_pycharm.py
python run_ra_lsbo_information_stage1_pycharm.py --task mechanism --output <新机制结果目录>
python run_ra_lsbo_information_stage1_pycharm.py --profile smoke --output <新smoke目录>
python run_ra_lsbo_information_stage1_pycharm.py --task verify --input <smoke目录>
```

仅技术完整性通过后运行 full；是否有优化优势不是升级 smoke -> full 的条件，不允许按效果挑种子。

```text
python run_ra_lsbo_information_stage1_pycharm.py --profile full --output <新full目录>
python run_ra_lsbo_information_stage1_pycharm.py --task verify --input <full目录>
python run_ra_lsbo_information_stage1_pycharm.py --task pack --input <full目录> --output <新full打包目录>
python run_ra_lsbo_information_stage1_pycharm.py --task pack --input <smoke目录> --output <新smoke打包目录>
```

所有命令保存完整 stdout/stderr，明确检查退出码。出现导入、固定 SHA、断言或模型异常时停止并上传失败证据，不原目录重跑、不现场修代码。
真实环境 smoke 才验证 GP/VAE 接口；GitHub CI 与本地替身测试通过不等于真实集成通过。

## 7. 上传与汇报合同

结果提交到独立新分支，例如 `results/information-value-stage1-20261106`；如已存在则加唯一后缀，禁止 force push。
建议结果目录同名，保存 full/smoke 的 report、plan、source_receipt、trace_seal、manifest、mechanism_report、所有 ZIP 与 bundle_manifest；不要只传总结。
同时保存完整控制台日志、单元测试日志、环境信息、固定代码 SHA、文件 SHA256 清单。
ZIP 包含所有原始 proposal、observation、initial、模型表示 checkpoint、真值审计标签和调用账本；每包按原文件大小分块，校验不丢文件。
结果分支可从固定代码 SHA 建立，不能合并代码 PR 或修改旧结果。GitHub 上传出错时保留本地全部结果并报告具体失败，不谎称已上传。

回报：结果分支 URL、结果 HEAD、固定代码 SHA、技术状态、192/96/16384 的 full 费用核对、逐任务诊断表、失败信息消融表现、未知/中断记录和全部附件位置。
报告不得写“新两步 GP 优化器已获胜”或“飞机安全已验证”。

## 8. 下一阶段的进入条件

若原池/扩展池都缺机会，优先修候选生成；若机会存在但模型/支持域持续错排，优先修部分观测约束代理；若模型已有区分力但决策未利用信息，再接入 GP 幻想后验和两步决策。
未来幻想更新必须只传递真实评价机制允许的反馈，失败不泄露隐藏目标；所有额外 GP 拟合成本也应记录。
本协议不自动授权新的飞机/RCAIDE 实验，也不要求工作机自行开发下一阶段。
