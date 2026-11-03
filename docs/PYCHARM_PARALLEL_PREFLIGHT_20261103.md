# PyCharm 零调用入口与并行演练

本入口不是正式实验执行器。当前仅支持零飞机、零 RCAIDE 的环境检查、一次性演练计划和只读检查点负载；`execute` 永远拒绝，正式计算必须另行实现、审核和批准。不会调用优化器、训练 VAE、拟合 GP 或修改历史计划。

1. PyCharm 打开本工作树目录，在项目解释器选择现有 `C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe`。
2. 在项目根目录右键 `run_ra_lsbo_pycharm.py`，选择 Run。默认是 `preflight`，打印解释器、计划哈希、历史源文件原始字节哈希、种子核查状态和阻塞。预检结果不是执行许可。
3. 需要生成零调用演练计划时，在 PyCharm 的 Run/Debug Configurations 中将 Script parameters 设置为 `--mode plan --workers 4 --output D:\LSBO\new-rehearsal-plan`；路径必须不存在。它不更改原资格计划，也不授权任何模型调用。
4. 离线负载测试可设置 `--mode benchmark_offline --workers 4 --checkpoint <既有state.npz绝对路径> --output <全新目录>`。该负载只读取文件并做哈希，不执行 VAE/GP，不代表算法耗时。每个任务独立目录，失败停止派发、保留证据且不重试；只保证演练语义。
5. 参数 `--max-wall-seconds` 限制新任务派发与完成状态；已派发任务会安全等待，不强制终止，因此不是硬实时限。允许的进程数为 1、2、4、8、12、16，不得超过逻辑核心数。

正式实验需要另外实现并验证：唯一调用编号与原子预算账本、共享初值只收费一次、真实观测及 unknown 契约、不可变正式计划、审计合并、断电未结算处理、完整阶段审批凭据、GPU 互斥和可复现性。演练目录不是正式调用账本，不得用它证明正式试验公平或完整。
