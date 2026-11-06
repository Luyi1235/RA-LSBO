# PR67 信息价值第二阶段：运行阻断记录

**状态：`FAILED_IMPORT`。未执行 smoke，未执行 full，未产生任何评价或原始轨迹。**

## 固定身份

- 代码 SHA：`488a6da414572e09709533ac7a96e17115e632c9`
- 代码分支：`research/ra-lsbo-information-value-stage2-20261106`
- 工作树：`D:\LSBO\RA-LSBO-pr67-stage2-20261106_01`（detached HEAD）
- 检出前 `git status --porcelain`：空
- 环境：`Rcaide_vsp` / Python 3.9.25 / SQLite 3.51.2

## 已通过的技术检查

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s tests -p test_information_value_stage2.py -v` | **9 passed**，exit 0 |
| `python -m compileall -q ra_lsbo/experiments/information_value_stage2 run_ra_lsbo_information_stage2_pycharm.py` | exit 0 |
| 代码/种子/环境/物理模型修改 | 无 |

## 阻断点

执行：

```text
python run_ra_lsbo_information_stage2_pycharm.py --profile smoke --output <新smoke目录>
```

在导入阶段失败，exit 1；smoke 输出目录未被创建，没有开始任何任务。

```text
Traceback (most recent call last):
  File "D:\LSBO\RA-LSBO-pr67-stage2-20261106_01\run_ra_lsbo_information_stage2_pycharm.py", line 9, in <module>
    from ra_lsbo.experiments.information_value_stage2.runner import run
  File "D:\LSBO\RA-LSBO-pr67-stage2-20261106_01\ra_lsbo\experiments\information_value_stage2\runner.py", line 16, in <module>
    from .engine import AllSupportSoftGPBO
  File "D:\LSBO\RA-LSBO-pr67-stage2-20261106_01\ra_lsbo\experiments\information_value_stage2\engine.py", line 10, in <module>
    from ..soc_revision.acquisition import quantities
ModuleNotFoundError: No module named 'ra_lsbo.experiments.soc_revision'
```

## 诊断

- `engine.py:10` 使用 `from ..soc_revision.acquisition import quantities`。
- 当前包位于 `ra_lsbo.experiments.information_value_stage2`，因此 `..soc_revision` 解析为 `ra_lsbo.experiments.soc_revision`。
- 固定树中存在 `ra_lsbo/soc_revision`（顶层），但不存在 `ra_lsbo/experiments/soc_revision`。
- 这是固定 SHA 的导入路径问题，不是环境缺少依赖；测试和 `compileall` 因未真正导入该模块而未暴露它。

## 处理

按错误处理要求：**停止，不修改代码，不在原目录重跑，不启动 full**。没有可上传的原始轨迹；本分支只上传阻断证据：

- `failure.json`
- `failure/traceback.txt`
- `logs/unit-tests.log`
- `logs/compileall.log`
- `logs/smoke-console.log`
- `git_receipt.txt`
- `environment.txt`
- `fixed_sha.txt`
- `artifact_sha256.txt`

## 后续

需要 ChatGPT 在 PR67 代码中修正导入路径（例如按实际包结构调整），给出新的固定 SHA；工作机随后在**全新 worktree 和全新输出目录**重跑 9 项测试、smoke、full。未启动任何算法开发、飞机或 RCAIDE 调用。

注：未在工作区或仓库中找到《RA-LSBO_PR67_工作机执行指令.md》；本记录依据 `docs/INFORMATION_VALUE_STAGE2_ZH.md` 与任务消息执行。
