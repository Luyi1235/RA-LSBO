# 机制数据与模型使用

本轮实验目录为 `runs/system_validation_v5_20261020T192312Z_5f6531eb`。
实验已经结束，独立审计、预测重建与两套完整权重恢复检查均通过。
以下数据和模型已实际生成；总结见 `SYSTEM_VALIDATION_FINAL_RESULTS_20261021.md`。

## 选择数据与模型

| 产物 | 路径 | 范围 |
| --- | --- | --- |
| 共同原始训练前缀 | 实验目录的 `precalibration_training.json` | 560条；包含累计140次训练补采，不含72条新校准 |
| 共同历史 | 实验目录的 `shared_history.json` | 632条；历史420、训练补采140、新校准72 |
| 共同参考模型 | `runs/system_shared_reference_v5_20261021` | 560训练、72校准、种子0；不使用任何优化结果 |
| 实验后数据与模型 | `runs/system_training_release_v5_20261021` | 894个唯一设计、822训练和72校准；无冲突，26条一致重复记录去除 |
| 最优已观察构型 | `runs/system_best_design_v5_20261021` | 实际可行响应、19个变量、12项约束及重复评估状态 |
| 实际选点预测评分 | `runs/system_risk_diagnostics_v5_20261021` | 当时已保存预测与之后实际结果配对，不在校准拟合段上评分 |
| 验收报告 | `runs/system_acceptance_v5_20261021/RESULTS.md` | 训练支持、方法对照、已通过核验和证据缺口 |

模型目录中的 `observations.json` 是带有效性掩码和来源的规范化观测。
`training_tables.json` 按目标、数值约束和分类头分别选择可用标签；
共同参考模型的该表只包含560条原始训练前缀。
未知标签保持 `null`，不能转换成负例。实际MDA费用按原始账本计费，
不会因为去重或隔离冲突而减少。

共同参考模型保留有20个真实任务失败的新校准段，最终可行性、总体隐藏成功、
任务失败及合并隐藏事件校准均为 `fitted_empirical`。其余四类原始头未拟合。
实验后数据由920条记录去重得到894个设计，无冲突；任务失败共46正/848负。
其822条原始训练记录为46正/776负，最近72条校准全部可行，因此主要校准头
均为 `fallback_insufficient_classes`，其余四类为 `fallback_base_unfitted`。
全量模型没有可用的经验校准头，不能将原始评分当作已校准输出。
实际启用状态分别记录在各自 `release.json` 中，不用实验后模型改写原实验预测。

## 只读加载

在 `D:\LSBO\RA-LSBO` 下，使用已经验证的
`D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe`：

```python
import json
from pathlib import Path
from tools.release_system_training_data import load_model_bundle

root = Path(r"D:\LSBO\RA-LSBO\runs")
bundle = root / "system_shared_reference_v5_20261021"
raw, responses, calibrated = load_model_bundle(bundle)
candidate = json.loads(
    (root / "system_best_design_v5_20261021" / "best_design.json")
    .read_text(encoding="utf-8")
)["candidate"]

print(raw.predict([candidate]))
print(responses.predict([candidate]))
print(calibrated.predict([candidate]))
```

上述操作只恢复模型并预测，不执行真实MDA。尚未拟合的校准头返回
不可用状态，不能把原始评分直接改名为校准概率。组合评分采用可用
分量的最小值及经验校准，不是联合概率或安全置信下界。

每个权重包包含 `encoder.pt`、`risk_forests.joblib`、`gp_states.pt`、
`model_metadata.json` 和 `bundle_spec.json`。加载时核验Python、依赖版本、
原始来源和权重哈希，并恢复RF、GP及校准参数。发布时另做16个候选的
完整恢复预测一致性检查。另已分别加载两套模型，对导出的最优构型执行
RF和7个目标/约束GP头预测，数值均有限；共同参考模型最终校准值可用，
实验后模型返回 `null`，与其降级状态一致。记录见
`runs/system_model_usage_check_v5_20261021.log`。当前加载器依赖保存的本地来源路径；移动
权重文件不等于已经迁移到另一物理模型或运行环境。

## 使用范围

真实任务仅有任务求解不收敛正例，其余四类保持训练不足。受控串行案例
中的五类正例另存，不进入飞机训练。定向邻域中的多个正例不能当作
多个独立失效区域；共同参考模型的校准拟合段也不是独立的校准效果测试集。
真实worker对前三类构造错误尚缺独立机制观测映射，详见
`PHYSICAL_MECHANISM_OBSERVABILITY_20261021.md`；缺正例不能简单归因于采样数量。

本轮每个真实优化运行最多12次，短于72条滚动校准窗口。新增优化标签
会影响最优值及候选池，但尚未进入该运行原始编码器、RF和GP的训练前缀。
实验后全量模型属于另一次拟合，不能据此宣称短程运行已完成长期在线学习。
