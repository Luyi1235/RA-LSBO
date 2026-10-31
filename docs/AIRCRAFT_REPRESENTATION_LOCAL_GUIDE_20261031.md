# 工作机下一步：核对飞机pilot的表示身份与构型可分辨性

本任务不是重复旧transfer诊断，也不是重新运行飞机pilot。目的是核对Linux表示重拟合的异常
能否在原Windows/Rcaide_vsp环境复现，以及30epoch特征身份是否等于历史提案。
新增飞机/MDA评估均为0；64次编码器拟合+32次one-hot转换，GP拟合0。总计算不是零。

## 固定版本及输入

- 分支：`analysis/aircraft-nominal-review-20261031`
- 本机固定提交：`5f33e4b9174b23e232ebe5d5ce53f7859a58e1ed`
- 表示脚本此前在`a72c520043bd31782df802e8cb8b87119e703499`实际运行成功，两个提交之间该脚本和优化器不变。
- 原完整ZIP：`aircraft_nominal_20261030_01_full_records.zip`
- ZIP SHA256：`35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd`
- 输入是原始ZIP，不是public summary.json，也不需要再次下载全部档案。
- 保留e4fbcd37原工作区、全部计划/快照/锁/历史结果；不得覆盖或改写。

使用原执行Rcaide_vsp环境，核查Python、NumPy、PyTorch等记录；不要为诊断安装/升级软件或修改旧环境。
如果原环境不可用或依赖不满足，停止并报告，不能用另一个环境冒称原平台复现。

## 独立worktree示例（PowerShell）

以下变量由本地模型依据已经存在的仓库、ZIP和Python实际路径填写；不要假造绝对路径。
所有输出使用新的目录，目录不在旧研究记录内部。

```powershell
$Repo = '<现有RA-LSBO git仓库路径>'
$NewWorktree = '<尚不存在的新worktree路径>'
$Archive = '<已下载的aircraft_nominal_20261030_01_full_records.zip>'
$Py = '<原Rcaide_vsp/python.exe>'
$Out = '<新的诊断输出目录>'
$Commit = '5f33e4b9174b23e232ebe5d5ce53f7859a58e1ed'
$ErrorActionPreference = 'Stop'

function CheckExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed, exit $LASTEXITCODE" }
}
if (Test-Path $NewWorktree) { throw '新worktree必须不存在' }
if (Test-Path $Out) { throw '输出目录必须不存在' }
if (!(Test-Path $Archive) -or !(Test-Path $Py)) { throw '输入路径不存在' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant() -ne '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd') { throw '原始ZIP哈希不符' }

git -C $Repo fetch origin analysis/aircraft-nominal-review-20261031
CheckExit 'fetch'
git -C $Repo worktree add --detach $NewWorktree $Commit
CheckExit 'worktree'
Set-Location $NewWorktree
$Head = git rev-parse HEAD
CheckExit 'rev-parse'
if ($Head.Trim() -ne $Commit) { throw '未处于固定提交' }
$Dirty = git status --porcelain --untracked-files=no
CheckExit 'status'
if ($Dirty) { throw '受跟踪源码不是干净状态' }
New-Item -ItemType Directory -Path $Out | Out-Null
$JUnit = Join-Path $Out 'junit.xml'
$Diagnostic = Join-Path $Out 'diagnostic.json'
$Log = Join-Path $Out 'refit.log'

& $Py -m pip check
CheckExit 'pip check'
& $Py -m pip freeze | Set-Content -Encoding utf8 (Join-Path $Out 'environment.txt')
CheckExit 'pip freeze'
& $Py -c 'import sys,platform,torch,numpy; print(sys.version); print(platform.platform()); print(torch.__version__); print(numpy.__version__); print(torch.__config__.show())' | Set-Content -Encoding utf8 (Join-Path $Out 'platform.txt')
CheckExit 'runtime record'
& $Py -m pytest -q tests/test_aircraft_nominal_evidence_review.py --junitxml $JUnit
CheckExit '20 diagnostic tests'

$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'
& $Py tools/refit_aircraft_representation_diagnostic.py $Archive $Diagnostic 2>&1 | Tee-Object -FilePath $Log
CheckExit 'representation refit'
```

保存实际命令与完整控制台日志，非零退出码即停止，不通过改仓库源码绕过检查。

## 执行边界与如何读结果

脚本固定全部16个latent试验、提案0和29前缀。每个前缀比较30和120epochs，并有one-hot几何参照。
不选择有利种子、不追加更高epochs、不训练GP、不运行propose或evaluate、不重新选择设计。
120epochs仅训练预算敏感性，不是新的算法配置，也不意味着达到了模型资格。

每个结果包含`representation/epochs/iteration/saved_feature_identity/original_identity_matches`。
只对latent且epochs=30且原身份非空的29项统计身份匹配；3个初始覆盖步骤没有原身份，不能算失败。
one-hot及120epoch本来不是原身份，不应计入复现匹配率。

再分别汇总完整活跃构型重构率、各变量准确率、潜变量范围、归一化、同布局不同电池的距离及核相关。
若身份全匹配且重构/几何异常复现，可以说原表示已在这些前缀恢复；仍不能直接证明这是BO退化的唯一原因。
若身份不匹配，完整保留输出和环境，不篡改原记录、源码摘要或契约去强制匹配。
低训练重构不是独立预测精度；当前优化器不靠decoder生成设计，不能只凭decoder指标判算法错误。

## 回传并停止

回传`diagnostic.json`、`environment.txt`、`platform.txt`、20项JUnit、完整控制台日志和固定提交。
可附一张30/120epoch统计表，但不要改原JSON。小产物可放新结果分支或Release；原176MB档案无需重复上传。
不运行nominal_pilot、transition、transfer、MDA，不重做原352/3718调用，也不要求重跑原327项优化器回归。
本轮完成后先评审身份和表示证据，再决定有限修订；没有自动授权新的模型调用或大规模训练网格。
