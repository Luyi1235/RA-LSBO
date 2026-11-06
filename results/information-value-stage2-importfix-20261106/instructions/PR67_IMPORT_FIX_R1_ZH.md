# PR67 导入路径修复 R1 与重跑协议

## 已核对的失败

失败代码：`488a6da414572e09709533ac7a96e17115e632c9`。
失败证据固定提交：`ee8949687f404b4df24ced1699810e337c3b5771`，目录
`results/information-value-stage2-20261106/`。
工作机按合同停止；该次尝试没有数学、飞机或 RCAIDE 评价，full 没有启动。

原因是 `information_value_stage2/engine.py` 中的相对导入少了一个点：

```diff
-from ..soc_revision.acquisition import quantities
+from ...soc_revision.acquisition import quantities
```

包名为 `ra_lsbo.experiments.information_value_stage2`；旧写法查找不存在的
`ra_lsbo.experiments.soc_revision`，修复后使用已有 `ra_lsbo.soc_revision`。
这是提交者的代码错误，不是工作机依赖或环境问题。
原 9 项测试只导入 protocol，compileall 只检查语法，所以之前的绿色 CI 未验证真实引擎导入。

## 修复范围

生产代码仅改变上述导入行，不改变 runner、protocol、两个臂、种子、预算、候选数、
原始 eligible 参考集合、EI 归一化、GP/VAE、风险效用或历史文件。
原 PR67 支持域消融的解释边界继续有效；本修复不是新算法，也不是两步 GP 前瞻。

新增标准库导入路径测试 3 项和真实依赖导入测试 4 项，加原 9 项，共 16 项。
真实测试用独立 Python 子进程导入引擎、runner 和入口，检查 CLI help 不启动 run；
还在内存中恢复旧错误并要求出现精确的缺失包异常，不修改源文件。
不使用 sys.modules 假模块、不跳过缺失依赖，不执行 GP/VAE 拟合或 evaluator。

CI 保留 Windows/Ubuntu × Python 3.9/3.13 四个纯协议任务，另增加
Windows/Ubuntu × Python 3.9 的真实 CPU 依赖导入任务。
CI 安装依赖仅发生在临时 CI 机器，工作机不安装或升级任何依赖。
CI 导入通过仍不等于真实拟合和在线实验通过，后者必须由工作机 smoke 验证。

## 工作机重跑

使用 ChatGPT 交接消息提供的新完整固定 SHA；不要执行旧 SHA 或浮动分支 HEAD。
创建全新 detached worktree，确认 SHA 精确相等、源目录干净。
保留旧失败 worktree 和结果目录，使用现有 Rcaide_vsp 环境，不修代码、不改环境。
先读本文件和 `docs/INFORMATION_VALUE_STAGE2_ZH.md`。

在仓库根目录依次运行并保存退出码及完整 stdout/stderr：

```text
python -m unittest discover -s tests -p test_information_value_stage2.py -v
python -m unittest discover -s tests -p test_information_value_stage2_import_paths.py -v
python -m unittest discover -s tests -p test_information_value_stage2_runtime_imports.py -v
python -m compileall -q ra_lsbo/experiments/information_value_stage2 run_ra_lsbo_information_stage2_pycharm.py
python run_ra_lsbo_information_stage2_pycharm.py --help
python run_ra_lsbo_information_stage2_pycharm.py --profile smoke --output <全新R1-smoke目录>
```

测试数量分别为 9、3、4，全部通过且 help 退出码为 0 后才执行 smoke。
任何失败立即停止，上传完整日志，禁止现场修复、换种子或原目录重跑。

smoke 技术核验通过后才运行：

```text
python run_ra_lsbo_information_stage2_pycharm.py --profile full --output <另一全新R1-full目录>
```

预算保持原协议：smoke 48 初始 + 16 在线 = 64；full 576 共享初始 + 576 在线 = 1,152。
成功完成两轮实际共 1,216 次数学调用；import/help 是 0 次。
smoke 种子 998701 是 full 998701–998712 的子集，不能把两轮当作独立验证样本。
保留四个任务、两个臂、每臂六次 full 评价和全部原始证据要求；飞机/RCAIDE 均为 0。

## 证据上传

使用新结果分支 `results/information-value-stage2-importfix-20261106`，若已存在加唯一后缀。
不要覆盖或继续写入旧失败分支 `results/information-value-stage2-20261106`，不 force push，
不合并 PR67，不修改 main 或 PR66 结果。
新 README 引用旧失败 HEAD 和本次固定代码 SHA，明确旧尝试 0 调用。

上传：测试及 help 日志、smoke/full 控制台、环境版本、plan/source_receipt/report/progress、
shared_initial、全部 proposal/observation/result、representation checkpoint、SHA256 清单。
保留所有已存在的账本，不用人工填造程序未产生的标签或调用记录。
如入口导入再次失败而程序未生成 failure.json，在独立的外层失败证据目录保存记录和 traceback。

结果分支从新固定代码 SHA 创建；证据目录放专用 `.gitattributes`（`* -text`）以保留字节。
上传后核对远程 blob/哈希，回报新结果分支、结果 HEAD、新代码 SHA、测试/smoke/full 状态、
真实调用次数、逐任务结果和异常位置。不要以 CI 绿色或导入成功宣称算法效果。
