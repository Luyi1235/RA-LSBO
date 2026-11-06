# PR70 同状态求积诊断：CI 阻断记录

固定代码包：`RA-LSBO_PR70_Quadrature_Diagnostic_Code.zip`  
代码包 SHA256：`ce691ee487f46569fad85e15b08c7a22f43c30f067413386d80aae7676fe4af5`  
父代码 SHA：`275aac2c5d5d6a8c8f6278c0d556de3de5867f74`  
输入证据 HEAD：`f859cca5adc56dd101d2f59665f7829a2b0331c4`  
诊断代码分支：`analysis/pr70-quadrature-resolution-20261106-223912`  
诊断代码 HEAD：`f46e268706aab1c9cefe19cdbd887408c424058b`  
草稿 PR：<https://github.com/Luyi1235/RA-LSBO/pull/72>（编号 72）

## 结论：诊断未执行，CI 在预检阶段失败

按指令，CI 出现错误后停止后续诊断。**没有执行 smoke、full 或 verify，也没有产生任何求积诊断结果。**

- 本地安装：`INSTALLED_REVIEWED_ADDITIONS_ONLY`，七文件哈希与包内 `PAYLOAD_SHA256.json` 一致。
- 本地测试：`26` 项数值/协议测试通过；`6` 项真实运行测试通过；0 skipped。
- `compileall` 与 `--help`：exit 0。
- PR 代码提交只新增七文件，无父版本修改/删除；分支已推送。
- CI（ubuntu-latest）：**失败**，job `127953372683`，run `42235921834`。
- CI（windows-latest）：上传时仍为 `in_progress`，状态以 `ci-checkruns.json` 为准。

## 失败点与 traceback

CI 在 `run_pr70_quadrature_diagnostic.py` 的 smoke 预检阶段失败：

```text
File ".../code/run_pr70_quadrature_diagnostic.py", line 160, in run
    posterior,diag=restore(problem,model_base,seed,initial,history,folder/'representation'/'iteration_000',meta)
File ".../code/run_pr70_quadrature_diagnostic.py", line 98, in restore
    obj=feature_snapshot(problem,base,history,session)
File ".../code/ra_lsbo/structure_corpus.py", line 242, in feature_snapshot
    raise ValueError('Training contract differs')
ValueError: Training contract differs
```

CI 的 `failure.json` 记录：

- `completed_states=0`
- `phase="prefix_gp_reconstruction"`
- `gp_started=0`、`gp_completed=0`、`gp_errors=0`
- `new_true_evaluations=0`、`new_vae_training=0`、`aircraft_calls=0`、`rcaide_calls=0`

失败发生在特征/训练合同重建，不是求积节点运行之后；没有产生可用的状态结果。

## 判断

这一失败与 PR71 分支标题所述问题同类：历史语料/源码的 **LF/CRLF 字节身份**在 Linux CI 上无法重建，导致 `Training contract differs`。本包只按固定父提交安装七个文件，没有包含跨平台源码身份修复；因此不能在 Ubuntu runner 上重建 PR70 的历史特征合同。

按指令，不能在工作机现场修改 `structure_corpus.py`、放宽 `Training contract differs` 检查、替换 checkpoint 或换一个 SHA 重跑。该失败必须由 ChatGPT 在新代码中修复，再由工作机重新获取新固定 SHA。

## 本轮实际费用

- 新增数学真值评价：0
- 新增 GP 构造：0（失败发生在 prefix GP 重建之前）
- 新增 VAE 训练：0
- 飞机：0；RCAIDE：0
- 诊断 smoke/full：未执行

## 证据

- `logs/01-install.log`：安装校验与七文件 SHA256。
- `logs/02-*`、`logs/03-*`：26 项数值/协议测试、6 项真实运行测试的完整 stdout/stderr/exit。
- `logs/04-*`、`logs/05-*`：compileall、`--help`。
- `logs/06-staged-files.txt`、`logs/06-code-sha.txt`：代码提交只含七个 A 文件。
- `pr-response.json`、`ci-checkruns.json`、`ci-ubuntu-annotations.json`、`ci-ubuntu-failure.json`、`ci-ubuntu.log`、`ci-ubuntu-artifact/`：PR 与 CI 原始记录。
- `environment.txt`、`fixed_sha.txt`、`commands.txt`、`instructions.txt`：环境与执行身份。

## 下一步

需要 ChatGPT 提供包含跨平台源码身份修复的新固定代码包；工作机不得修改本包或绕过 CI。新的诊断只能在新 SHA、新 worktree 和新结果目录上执行。
