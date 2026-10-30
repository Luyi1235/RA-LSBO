# 表示信任试验验证回执（2026-10-30）

固定运行提交：`21505bba5f9485287a3e34f4ce50dc849edf4e04`。
代码树：`74f9ef060b4eac0e4dd0ad42ea9a8e6ef76f0b67`。
分支research/phase-evidence-20261030，PR #5。主分支和历史记录未改。
CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/41006408429
Job123902195779。实际下载的JUnit与完整smoke产物是本回执依据。

## 实际验收

280 tests，0 failures，0 errors，0 skipped。包含新增18项参数化测试及此前262项影响范围回归，不是全仓库全部历史测试。首次本轮代码CI即通过，没有隐藏失败重试。

transfer smoke：112/112真实数学调用；24trial、48在线提案；32独立面板、24共享初值、8锚点。运行、审计、报告均通过。下载后再次核验104项封存证据哈希与170项来源快照全部一致；CI来源字节与本地开发副本一致。

四算例一seed两步仅验证流程：Rosenbrock sparse中equal/prequential各有2个实际冷阶段选点，确实拟合双专家、保存评估前预测并进行结算后权重更新。其余smoke主要为暖路径。不能由此宣称冷发现优势或收敛性。finite-interval/unknown语义由人工单元测试覆盖，尚非真实飞机证据。

完整旧phase ZIP的固定哈希、1126项审计证据、166源码绑定在CI只读验证；新3108上限pilot仅prepare，没有calls.jsonl，没有执行完整pilot。新计划路径/环境独立绑定，旧计划未重写。

Linux/Python3.11、torch2.8.0+cpu、numpy2.0.2、scipy1.13.1、gpytorch1.11、botorch0.10.0、scikit-learn1.6.1、pytest8.3.5；完整依赖见artifact environment.txt。Windows工作机必须重新验证，不能擅自改原MDA环境。

## 费用及证据边界

本轮云端研究smoke累计112次；数据获取和离线phase复核0次新真实评估；单元测试内部人工/廉价求值另计，未统一计数；完整pilot0、新增RCAIDE MDA0。

Artifact ID12955895719：transfer-validation-21505bba5f9485287a3e34f4ce50dc849edf4e04。
ZIP SHA256 `db65c8c6bf8258b730b9e49fcc0cd483bb5898d3d628e0bc855018b08b943d24`，保留14天。
含JUnit、环境、提交、只读先决导入回执、smoke完整证据和仅准备的pilot计划。

权重审计只重算保存预测的评分与更新，不重新拟合GP或独立重放选点。权重不是编码器正确性后验或校准认证。新组合没有完整性能实验，尚未证明创新有效。

本回执之后的交付提交只增加评审、指南和只读分析工具，不改变固定运行版本。请固定运行SHA，不能使用移动HEAD继续已登记实验。
