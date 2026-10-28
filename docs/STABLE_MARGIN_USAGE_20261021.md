# 归一化约束桥接入口

日期：2026-10-21。综合报告生成时234项测试通过；随后补齐本入口的
在线模型工厂和完整恢复测试，最新全套235项通过。
研究结果见METHOD_REVISION_RESULTS_20261021.md。

## 用途

`ra_lsbo.stable_margin.StableMarginLearning`沿用部分标签终止网络、
固定参考训练隔离、约束GP和温度校准，并在概率输出前用float64逐成员归一化。
它处理原桥接在近乎确定满足约束时发生的float32求和越界。
参考数据上的最大数值修订小于3e-7。

原两轮138组开发证据和被冻结源码保持不变。此入口完成拟合、校准和保存恢复测试，
没有重新运行一轮优化效果试验；不能将其视为一个已证明优越的新优化器。

## 模型接口

已规范化的`rows`、变量定义`variables`和原LSBO路径`base_project`由调用方提供。
初始参考ID预先固定，其标签只用于温度校准，后续其他观测进入下一次拟合。

```python
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.stable_margin import StableMarginLearning, StableMarginRisk

settings = SystemSettings()
reference_ids = [row["evaluation"] for row in rows[-settings.calibration_window:]]
learner = StableMarginLearning("adaptive_full", settings, reference_ids, seed=0)
(risk, responses, calibration), metadata = learner.fit(rows, variables, base_project)

forecast = risk.predict(candidates)
event_probabilities = calibration.transform(forecast["probabilities"], risk.model_id)
```

拟合后向同一个`learner`传入严格追加的历史即可增量更新，参考行不能修改。
各版本模型、温度和训练历史身份独立；更新后旧温度不能用于新风险模型。

## 保存恢复

在新输出目录保存`learner.features.save_checkpoint(encoder_path)`和
`risk.save_checkpoint(risk_path)`。后者包含：

- 桥接checkpoint：目标及约束GP的全部权重、归一化和来源。
- 同名`.terminal.pt`：bootstrap终止网络及输入缩放。
- 同名`.normalization.json`：归一化适配、工厂源码及模型身份。

先用`StructuredFeatures.load_checkpoint`恢复相同训练历史的编码器，
再用`StableMarginRisk.load_checkpoint(risk_path, features, training_rows)`恢复。
恢复后的模型需要对同一固定参考重新构建`ReferenceTemperature`，
或独立核对保存的温度标识；不能借用其他版本的校准。

该模块只提供模型工厂，没有物理`evaluate`入口。正式优化仍须使用冻结协议和统一
调用账本；预期风险评分不是实际失败次数上界，也不是安全保证。
