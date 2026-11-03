# 151,040 次七算法正式确认：现存硬阻塞与必须补齐的生产代码

> 时间：2026-11-03；范围：**直接开发大实验**，不另做小规模研究性 Pilot。本文件不授权飞机模型或 RCAIDE 调用。仅有 GitHub 写权限，不等于能够远程点击用户工作机 PyCharm。

## 已完成的实质代码

新增 `confirm12/real_legacy_adapters.py`，直接使用冻结的 H1 `dualview_pilot.engine_for`，可**在不调用飞机模型、不拟合 GP/VAE**的前提下实例化四种真实历史优化器：
`hybrid_onehot`、`corpus_stable`、`onehot_stable`、`random`。它们是**历史方法原型**，尚不代表七方法确认资格，尤其不得将 `hybrid_onehot` 在未经科学方法冻结/映射核验前直接改名成新的 `ra_lsbo` 算法。实际优化器的 `propose/accept` 接口复用历史源码。

新增 `confirm12/nested_seed_plan.py`，只读/无标签生成 **2 任务 × 各 1,024 个主种子 + 每任务 256 个 SHA-256 预选辅助种子**，预算严格按 **102,400 主 + 48,640 辅 = 151,040**。种子 500001–501024、600001–601024 只是**待核验的备选编号**，并未通过历史完整排除或任务资格，不能用此文件自动授权任何评价。脚本以 `execution_authorized=false` 和所有资格标志 false 落盘，禁止结果后挑选。原 #36 600 个候选的计划和 SHA 不变。

现有 PyCharm 入口增加零模型调用的 `--mode proposed_plan --output <新目录>` 和 `--mode legacy_probe --output <新目录>`，分别生成**候选计划**、检查真实历史 H1 引擎是否能实例化。二者都不是 `execute`；默认预检和 `execute` 拦截不变。

## 工作机必须直接补齐的代码（不要再只交论文规划或 synthetic-only 测试）

1. **七个真实确认算法的科学资格和接口：** 制定精确的主 `ra_lsbo` 与 `full_latent`、`full_onehot` 的历史 H1 实现对应表（算法/参数/GP 头/反馈/池搜索/随机规则）。直接复用冻结 `hybrid_onehot`、`corpus_stable`、`onehot_stable` 实现的数学行为，但不能默默改变标识或内核。补全 `reverse_representation` 的真正目标显式、约束潜空间代理，补全 `interval_as_missing` 的真实部分证据删除逻辑且不能改成精确标签；`random` 使用统一合法候选生成、真实预算；`cbob_adapted` 核验作者来源、许可证、EP/EICB 和层级空间/部分反馈适配差异，不可把一个随机/合成/旧近似控制冒充 CBOB。为各臂提供带完整 `propose/accept`、前缀隔离、同一预算与失败契约的真实 adapter，逐臂以 **mock evaluator** 和历史离线观测验收，状态不合格即 BLOCKED。无资格臂不得偷偷从 7 方法中删去后仍声称完成原确认设计。
2. **全局真实调用账本和执行依赖：** 把 `confirm12/call_ledger.py` 的 `SIMULATED_ONLY` 转成独立经审计的生产 ledger，追加唯一 source/plan/phase/seed/candidate/ticket/role 身份与真实模型证据。12 个 Windows spawn 工作进程；每 task/seed **12 个 shared initial 真值一次评价/收费**，确保全部 settled 后才能分配该 seed 的两主臂及预选 5 辅臂；每臂严格 **18 online + 最多 1 同设计 5s 验证且不回灌**。每个工人写独占结果目录和 VAE checkpoint，先预算 ticket 再调用真实 evaluator，异常不重试，unsettled/unknown/中断保留为 INCOMPLETE，按冻结全试验清单做**独立审计**，不是只数 290 条模拟记录。
3. **确认计划及历史前置资格：** 不是直接采用备选种子。冻结两任务 iid 合法构型抽样/解析筛选及 20s 真实合同，完成 #36 原始 LF/CRLF 源文件原始字节一致性修复、先前不可读种子档案复核与全项目种子冲突排除；先取得单独资格阶段的有效实测结果（140km 是否 <10% 可行等）与相应实验预算批准，完成外部 CBOB 资格。确认新源码/运行环境、全部 seed 和 256 子集后，写 **source/hash/environment/seed/call-cap 完整 pre-registration**，但不能改 #36 封存旧 SHA。
4. **运行资源和启动安全门：** 当前工作机 Ryzen 9 9950X、约31GiB RAM、CUDA 不可用的 CPU 版 PyTorch，不能默认 12 个长时间 GP/VAE/Local 优化器都不争用。以已封存前缀覆盖真实七臂的 CPU/进程内存/慢端时延，再评估是否能在含审计的总墙钟 **24小时** 内完成；若不符合应提前停在预检，不得改原算法数值控制或静默从 1024 减种子。新确认 `phase receipt` 应绑定源 SHA、计划哈希、预算和资格结果，无凭证一律拒绝真实调用。该凭证须来自真实独立科学/预算批准，不得在代码里自制“通过”。
5. **PyCharm 最终入口：** `run_ra_lsbo_confirm_12workers.py` 需保留默认零真值预检，完成上面 1–4 后，获准且全条件满足时一键 Run 才调用真实七方法优化器。明确显示 **BLOCKED/READY**、最多 151,040、预计24小时、进度/剩余调用和完成审计状态；Windows CI 模拟各类断电/异常/重复调用、种子泄漏和资源不足。**禁止只删除 `execute` 的 RuntimeError 而运行一套模拟策略。**

## 立即实施及报告方式

本阶段只允许直接向独立 Draft PR 提交完整**真实代码和零模型调用验收**。无法在一次交付中解决的环节应按上面顺序逐模块提交，并给出代码级明确阻塞；不要让用户反复回答“优先哪个任务”。这不是“先小试再说”的要求：目标始终是原始**两层嵌套 151,040 次七方法**独立实验。

仅在各资格与明确阶段预算批准全部通过后，才能执行任何新简化飞机求值。不能使用 GitHub Actions 或 ChatGPT GitHub 连接代替用户本机的 PyCharm 启动；本助手当前不能远程控制该 Windows 工作机。

最后回复：代码 PR、7/7 真实适配和资格状态、真实账本审计、已资格化任务/种子和确认计划哈希、12 进程资源/24h结论、PyCharm 入口运行状态与如仍 BLOCKED 的具体代码/科研缺项。
