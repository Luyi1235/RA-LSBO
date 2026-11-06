# RA-LSBO PR67 工作机执行指令

## 任务

执行 PR67：在真实 GP/VAE 在线闭环中，对比原 PR63 `soft_gp` 的 optimistic-violation eligible 支持筛选，与移除该支持筛选后的全候选短视软效用选择。

本任务只做数学任务，不做飞机/RCAIDE 仿真。工作机负责读取固定 GitHub 代码、运行、核验和上传原始结果；不得修改代码、模型参数、种子或协议。

## 固定身份

- 仓库：https://github.com/Luyi1235/RA-LSBO
- PR：https://github.com/Luyi1235/RA-LSBO/pull/67
- 代码分支：`research/ra-lsbo-information-value-stage2-20261106`
- **唯一固定代码 SHA：`488a6da414572e09709533ac7a96e17115e632c9`**
- 父基线：PR66 固定代码 `1917f305b39f93d79f6b16dd83e56af365098500`
- 入口：`run_ra_lsbo_information_stage2_pycharm.py`
- 协议：`docs/INFORMATION_VALUE_STAGE2_ZH.md`

## 先读清楚实验比较的内容

两个臂：

1. `soft_gp_eligible`：原 PR63 `LocalCredibilityValueBO(variant="soft_gp")`，不改变。
2. `soft_gp_all_support`：使用完全相同的 GP/VAE 预测、风险代理、EI、soft utility 和阶段安排，但移除 optimistic-violation eligible 候选支持掩码，让全部合法 GP 预测候选参与短视效用排序。

特别注意：这不是风险概率硬门实验。PR63 soft_gp 没有概率硬门，但它仍然使用 optimistic-violation eligible 支持筛选。PR67 精确测试的是该支持筛选是否丢失了有价值的候选。

全支持臂的 EI 归一化仍以原 eligible 集合为参考，不因新增候选而改变已有候选分数的归一化尺度。不要修改这一规则。

## 准备

1. 不要在旧工作区直接切分支或覆盖文件。创建全新 worktree。
2. 检出固定 SHA `488a6da414572e09709533ac7a96e17115e632c9`，使用 detached HEAD。
3. 确认 `git rev-parse HEAD` 精确相等，`git status --porcelain` 为空。
4. 使用现有 `Rcaide_vsp` 环境和项目原有 Python 依赖。不要安装/升级依赖，不修改物理模型或环境。
5. 源 worktree 外创建独立结果父目录。smoke、full 必须使用不同且全新的输出目录。
6. 保存 Python 路径、Python 版本、`pip freeze`、Git SHA、系统信息。

## 执行顺序

在固定 worktree 根目录执行，使用 `Rcaide_vsp` 对应的 Python：

```text
python -m unittest discover -s tests -p test_information_value_stage2.py -v
python -m compileall -q ra_lsbo/experiments/information_value_stage2 run_ra_lsbo_information_stage2_pycharm.py
python run_ra_lsbo_information_stage2_pycharm.py --profile smoke --output <全新smoke目录>
```

smoke 通过后核验报告和账本，再执行：

```text
python run_ra_lsbo_information_stage2_pycharm.py --profile full --output <全新full目录>
```

每条命令都保存完整 stdout/stderr，并明确检查退出码。不要只看 PyCharm 最后一行输出。

## 冻结预算

| 项目 | smoke | full |
|---|---:|---:|
| 任务数 | 4 | 4 |
| case-seed 组数 | 4 | 48 |
| 每组共享初始评价 | 12 | 12 |
| 每臂在线真实评价 | 2 | 6 |
| 共享初始评价总数 | 48 | 576 |
| 两臂在线评价总数 | 16 | 576 |
| 数学评价合计 | **64** | **1,152** |
| 飞机/RCAIDE | **0/0** | **0/0** |

- smoke 种子：`998701`。
- full 种子：`998701–998712`。
- 四个任务：Branin ordinary/sparse、Rosenbrock ordinary/sparse。
- 两臂必须共享同一组初始 12 个真实记录；之后各自根据自己的真实反馈独立训练/更新 GP/VAE。
- 两臂均须完成规定的真实在线评价次数，不得提前停止、跳过、补造标签或填充缺失目标。

## 必须核验的内容

1. 真实在线调用总数与 plan/report 完全一致：smoke 64，full 1,152。
2. 每个 case/seed 两臂 `initial_prefix_sha256` 一致。
3. 每条 proposal 对应唯一合法候选；实际观测的 candidate hash 与 proposal 一致。
4. `soft_gp_eligible` 的候选支持数来自原始 eligible mask。
5. `soft_gp_all_support` 的实际选择可在全部合法 GP 候选中产生，且记录原始 eligible 数、全候选数、原 mask 和 `EI_normalization_reference_unchanged=true`。
6. 所有风险/效用收据必须绑定实际最终选中候选；覆盖/无模型预测不能伪造风险概率。
7. 失败必须作为真实不可行结果记录；unknown、infrastructure error、numerical failure 不得归类为物理不可行。
8. `new_aircraft_calls=0`、`new_rcaide_calls=0`。
9. 对每个任务分别报告 feasible-first 胜/负/平、双方均可行时的目标差、最终可行数、实际失败数、冷启动发现数、支持坍塌次数和风险预测覆盖率。
10. 不跨不同任务目标尺度合并平均值，不做未经注册的显著性/安全声明。

## 出错处理

任何导入错误、固定 SHA 不符、原始 PR54 SHA 校验失败、候选绑定错误、GP/VAE 拟合错误、评价异常或账本不一致：

- 立即停止；
- 保留完整 traceback、日志和当前输出目录；
- 不现场修改代码，不调整种子，不在原目录重跑；
- 上传失败证据并报告准确停止位置。

## GitHub 上传

将结果上传到独立新分支：`results/information-value-stage2-20261106`。若已存在，添加唯一时间后缀；禁止覆盖、force push 或修改 PR66 结果分支。

至少上传：

- `plan.json`、`source_receipt.json`、`report.json`；
- 全部 shared initial 记录；
- 每条 trial 的 proposal、observation、result；
- `progress.jsonl` 和完整调用账本（若程序已有）；
- 完整 stdout/stderr、单元测试日志、环境信息；
- 固定代码 SHA、每个文件 SHA256 清单；
- 中文 README，逐任务报告实际结果和限制。

保留全部原始证据，不要只上传汇总表。上传后重新读取 GitHub 结果分支，核对结果 HEAD 和文件完整性。

## 最终回报

向统筹对话返回：

1. 结果分支 URL 和结果 HEAD；
2. 固定代码 SHA；
3. smoke/full 状态和实际调用次数；
4. 四个任务各自的 W/L/T、可行率、失败数和目标差；
5. all-support 是否真的改变选点路径；
6. 是否消除了 PR66 的 eligible 支持坍塌，以及是否带来更多高风险失败；
7. 所有失败、unknown 和异常证据位置。

不得声称 PR67 已实现 GP 两步前瞻、已证明新算法优越或已验证飞机安全。PR67 是 myopic eligible-support ablation；完整 GP fantasy update 的两步决策仍是后续独立阶段。
