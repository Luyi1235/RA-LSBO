"""Collect audited physical and controlled evidence without new evaluations."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.mechanism_system import METHODS
from ra_lsbo.system_experiment import save_new


NAMES = {
    "geometry_generation_failure": "几何生成失败",
    "network_configuration_failure": "网络配置失败",
    "motor_design_failure": "电机设计失败",
    "mission_solver_nonconvergence": "任务求解不收敛",
    "energy_audit_failure": "能量审计失败",
}


def summarize(analysis, reference, poststudy, transfer, components, replay, sparse, output):
    paths = dict(
        analysis=analysis/"analysis.json", reconstruction=analysis/"forecast_reconstruction.json",
        reference=reference/"release.json", poststudy=poststudy/"release.json",
        transfer=transfer/"transfer.json", components=components/"components.json",
        native_replay=replay/"native_replay.json", sparse=sparse/"analysis.json")
    data = {k: load_json(p.read_bytes()) for k, p in paths.items()}
    result = data["analysis"]
    verify_sources(result["input_sha256"])
    verify_sources(data["sparse"]["input_sha256"])
    audit_path = Path(result["source_folder"])/"independent_audit.json"
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if any(data[name]["input_audit_sha256"] != audit_sha for name in
           ("reconstruction", "reference", "poststudy", "transfer", "native_replay")):
        raise ValueError("Evidence artifacts refer to different study audits")
    if data["components"]["bundle_spec_sha256"] != hashlib.sha256(
            (reference/"bundle_spec.json").read_bytes()).hexdigest():
        raise ValueError("Component diagnostic used a different reference model")
    for name, directory in (("reference", reference), ("poststudy", poststudy)):
        verify_sources({str(directory/filename): sha
                        for filename, sha in data[name]["output_sha256"].items()})
    if data["poststudy"]["complete_new_mda_cost"] != result["accounting"]["complete_new_mda_calls"]:
        raise ValueError("Training release and comparison accounting differ")
    if (data["reconstruction"]["status"] != "passed" or data["native_replay"]["status"] != "passed"
            or data["reference"]["full_model_bundle_roundtrip"] != "passed"
            or data["poststudy"]["full_model_bundle_roundtrip"] != "passed"):
        raise ValueError("Evidence bundle did not pass required reconstruction")
    physical = result["physical"]
    if len(physical["trials"]) != 24 or {r["method"] for r in physical["trials"]} != set(METHODS):
        raise ValueError("Missing physical comparison runs")
    if result["accounting"]["complete_new_mda_calls"] > 500:
        raise ValueError("Cumulative actual cost exceeds standing authorization")
    comparisons = []
    for other in METHODS:
        if other == "ra_full":
            continue
        rows = [r for r in physical["paired_common_actual_calls"] if r["other"] == other]
        valid = [r for r in rows if r["common_actual_calls"] > 0 and r["other_minus_full"] is not None]
        comparisons.append(dict(
            other=other, pairs=len(rows), nonzero_common_call_pairs=len(valid),
            full_better=sum(r["other_minus_full"] > 1e-8 for r in valid),
            other_better=sum(r["other_minus_full"] < -1e-8 for r in valid),
            ties=sum(abs(r["other_minus_full"]) <= 1e-8 for r in valid),
            rows=rows))
    training = data["reference"]["training_support"]
    all_support = data["poststudy"]["all_available_mechanism_support"]
    completed_comparison = all(
        r["evaluations"] > 0 and r["status"] in ("complete", "stopped_by_policy")
        for r in physical["trials"])
    optimizer_hidden_failures = sum(r["states"].get("hidden_failure", 0)
                                    for r in physical["aggregate"])
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), additional_mda_calls=0,
        scope="Completed finite validation; evidence acquisition does not require a positive conclusion",
        accounting=result["accounting"], historical_mda_cost=420,
        physical_runs=len(physical["trials"]),
        every_physical_repetition_has_new_call_evidence=completed_comparison,
        physical_optimizer_actual_calls=sum(r["calls"] for r in physical["aggregate"]),
        physical_optimizer_observed_hidden_failures=optimizer_hidden_failures,
        common_reference_training_support=training,
        common_reference_calibration_support=data["reference"]["calibration_support"],
        actual_active_mechanisms=data["reference"]["active_mechanisms"],
        poststudy_active_mechanisms=data["poststudy"]["active_mechanisms"],
        poststudy_training_support=data["poststudy"]["training_mechanism_support"],
        all_available_mechanism_support=all_support,
        zero_positive_real_mechanisms=[n for n, c in all_support.items() if c["positive"] == 0],
        all_five_mechanisms_have_20_positive_20_negative=all(
            c["positive"] >= 20 and c["negative"] >= 20 for c in training.values()),
        broad_failure_generalization_proven=False,
        stable_combination_superiority_proven=False,
        physical_aggregate=physical["aggregate"], physical_common_call_comparisons=comparisons,
        controlled_original_aggregate=result["controlled"]["aggregate"],
        controlled_sparse_aggregate=data["sparse"]["controlled"]["aggregate"],
        component_diagnostic=data["components"],
        mechanism_transfer=data["transfer"],
        verification=dict(
            independently_audited=True, forecast_reconstruction="passed", native_baseline_replay="passed",
            shared_reference_bundle_roundtrip="passed", poststudy_bundle_roundtrip="passed"),
        limitations=[
            "Single physical task, three seeds, at most twelve optimization calls per run",
            "Directed local training/calibration acquisition is correlated, not population sampling",
            "Current physical domain may never expose some supported failure classes",
            "Native worker does not directly emit the first three mechanism tags; "
            "generic construction errors lack a separately audited mechanism mapping",
            "Empirical calibration and expected-risk debits are not safety confidence bounds",
            "Physical engineering branch has 14 continuous and 5 discrete variables; not the paper's 24-variable task",
            "RA and CBO score a finite legal candidate pool; native LSBO retains its original search",
            "Training-grammar reconstruction does not prove latent prior matching or unseen-configuration generalization",
            "No-mechanisms also removes the aggregate hidden-success score",
            "No-calibration uses raw-risk weighted EI, full risk debits and baseline mode; "
            "it is an operational policy ablation, not only a probability-transform replacement",
            "The 72-record rolling calibration window exceeds the 12-call run; new optimization "
            "labels affect incumbents and candidate pools but do not yet enter RA raw encoder/GP/RF training",
            "Random uses the first admissible common-pool candidate, not independent uniform domain sampling"],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"acceptance.json", summary)
    lines = [
        "# 机制数据与完整系统验证结果", "",
        "本报告区分已经完成的有限实验与尚未成立的研究结论。"
        "无改善、失败与提前停止运行均保留，没有按结果筛选种子。", "",
        f"本轮所有版本累计真实MDA：{summary['accounting']['complete_new_mda_calls']}/500；"
        f"其中真实优化{summary['physical_optimizer_actual_calls']}次。"
        "历史热启动另计420次，共享训练补采和校准的收益不归给任一优化器。", "",
        "## 真实机制训练数据", "",
        f"共同参考模型：{data['reference']['training_observations']}训练、"
        f"{data['reference']['calibration_observations']}全新留出校准；"
        "完整RF、GP和编码器权重恢复检查通过。优化后另存全量去重模型，不覆盖共同参考模型。", "",
        "| 机制 | 训练正例 | 训练负例 | 训练未知 | 新校准正例 | 新校准负例 | 全部真实正例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, counts in training.items():
        cal = data["reference"]["calibration_support"][name]
        lines.append(f"| {NAMES[name]} | {counts['positive']} | {counts['negative']} | "
                     f"{counts['unknown']} | {cal['positive']} | {cal['negative']} | "
                     f"{all_support[name]['positive']} |")
    missing = "、".join(NAMES[n] for n in summary["zero_positive_real_mechanisms"])
    lines += ["", f"没有真实正例的机制：{missing or '无'}。未知标签保留未知。"
              "达到分类头拟合门槛不代表泛化充分；定向邻域不能充当独立失效区域。", "",
              "经验校准器的实际启用状态如下；拟合状态不是可靠性或置信覆盖证明。", "",
              "| 头 | 共同参考校准正/负例 | 参考状态 | 优化后校准正/负例 | 优化后状态 |",
              "| --- | ---: | --- | ---: | --- |"]
    for name in ("final_feasibility", "hidden_success", "mission_solver_nonconvergence", "any_hidden_failure"):
        first = data["reference"]["calibration_diagnostics"][name]
        last = data["poststudy"]["calibration_diagnostics"][name]
        lines.append(f"| {name} | {first['positive']}/{first['negative']} | {first['status']} | "
                     f"{last['positive']}/{last['negative']} | {last['status']} |")
    lines += ["",
              "## 真实优化比较", "",
              "全部方法使用共同物理契约和历史，8方法×3种子；每运行最多12次新增MDA。", "",
              "| 方法 | 实际调用 | 新可行 | 平均目标改进% | 改善运行 | 终止状态 |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in physical["aggregate"]:
        mean = row["mean_improvement_percent"]
        text = "NA" if mean is None else f"{mean:.6f}"
        lines.append(f"| {row['method']} | {row['calls']} | {row['feasible_calls']} | "
                     f"{text} | {row['improved_runs']}/3 | {row['statuses']} |")
    lines += ["", f"全部新增真实优化调用中，已观察隐藏失败{optimizer_hidden_failures}次。"]
    if optimizer_hidden_failures == 0:
        lines.append("各基线也没有在这些优化轨迹中遇到隐藏失败，"
                     "因此不能通过新增优化的隐藏失败率识别机制头的防失败效应；"
                     "训练和新校准中的真实失败另行统计。")
    lines += ["", "按每对方法较小的实际调用数比较，正差表示完整RA较好；"
              "不存在的调用不填补为已完成实验。", "",
              "| 对照 | 有新增证据的配对 | 完整RA较好 | 对照较好 | 持平 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['other']} | {row['nonzero_common_call_pairs']} | "
                     f"{row['full_better']} | {row['other_better']} | {row['ties']} |")
    lines += ["", "## 区域迁移与机制额外贡献", "",
              "以下是实验后探索性诊断，不用于重调冻结优化器。"
              "留出某失效区域时同时删除锚点及其全部邻居，且不使用全新校准"
              "或任何优化结果训练；0.5阈值只用于描述原始分类头的识别能力。", "",
              "| 留出区域 | 风险表示 | 剩余训练正例 | 留出正例 | 失败召回率@0.5 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for fold in data["transfer"]["folds"]:
        recall = (fold["scores"] or {}).get("recall_at_half")
        text = "NA" if recall is None else f"{recall:.6f}"
        lines.append(f"| {fold['held_out_anchor']} | {fold['view']} | "
                     f"{fold['training_support']['positive']} | "
                     f"{fold['test_support']['positive']} | {text} |")
    component = data["components"]
    lines += ["", f"在共同参考模型的{component['observations']}条留出记录中，"
              f"加入总体隐藏成功项后有{component['aggregate_changes_base_count']}条"
              "原始最终评分发生变化；在此基础上加入分解机制头后有"
              f"{component['separate_mechanisms_change_aggregate_count']}条进一步变化。"
              "这些是原始评分贡献，不是闭环优化收益或校准效果。"]
    for name, diagnostic in component["mechanism_aggregate_redundancy"].items():
        proof = diagnostic["forest_structure_complement"]
        lines.append(
            f"{NAMES[name]}与总体隐藏成功头的RF树级互补检查："
            f"{proof.get('complement_for_all_inputs_within_tolerance', False)}，"
            f"核验树数{proof.get('verified_trees', 0)}，"
            f"叶节点概率容差{proof.get('leaf_fraction_absolute_tolerance', 'NA')}。"
            "该检查只比较原始RF评分；不能据此断言含边界不确定性选点的整套算法相同。")
    comparison_text = (
        "已完成真实数据补采、分层留出校准及有限的完整系统对照，"
        if completed_comparison else
        "当前真实对照仍有未执行新增调用或异常终止的重复，不能判为完整效果验证；"
        "实际已经运行的补采、校准和对照均保留，")
    lines += ["", "## 能够支持的结论", "", comparison_text+
              "并可从原始请求、响应、账本、训练前缀和实际权重独立重建。"
              "当前证据不能把五类真实失效全部判为训练充分，"
              "也不能证明“机制分解＋潜空间＋校准＋边界选点”稳定优于各基线。", "",
              "真实worker对几何、网络及电机构造异常主要返回通用错误，"
              "尚缺各阶段的独立、可审计机制映射；其他四类没有正例也包含"
              "这一观测限制。进一步采集前须补充阶段诊断及设计相关性证据，"
              "详见 `docs/PHYSICAL_MECHANISM_OBSERVABILITY_20261021.md`。", "",
              "两个受控案例分开报告，标签不进入飞机训练。"
              "稀疏受控案例廉价筛选后可行率5.02%；完整RA与去机制的优化终点相同，"
              "one-hot平均改进更高。受控与真实结果不能拼成跨物理任务的优越性证明。", "",
              "本轮RA的72条滚动留出窗大于12次调用上限。原始编码器、GP和RF的"
              "训练前缀在短程中吸收较早校准记录，新增优化标签尚在留出窗内；"
              "它们仍通过最优值和候选池影响选点。CBO能使用全部已知数据，"
              "因此这不是相同原始训练样本数下的比较，也不能当作长期在线学习验证。", "",
              "RA/CBO使用有限合法候选池编码选点，原LSBO保留原生搜索。"
              "编码器的训练语法恢复诊断不能替代潜变量先验匹配或未见构型泛化；"
              "潜变量逐维方差、熵及恢复指标保存在预测重建结果中。"
              "本轮是19变量工程分支的有限比较，不是论文24变量任务的精确复现。", "",
              "去校准版本还采用原始风险加权EI、完整风险扣减和基线选点模式；"
              "它反映禁用校准后的操作策略差异，不能单独识别sigmoid变换的因果收益。"
              "去机制版本同时移除了总体隐藏成功项，因此也不能单独识别"
              "五个分解头相对于单一隐藏事件分类器的收益。", "",
              "仍需的证据是合法物理域中的其他真实失效类别，以及独立物理任务和更多"
              "独立失效区域的外部验证。不能通过放宽预检、注入程序错误、伪造标签"
              "或合并不同物理版本来补齐这些证据。", "",
              "## 产物", ""]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"summary.source.py").write_bytes(Path(__file__).read_bytes())
    print("Acceptance evidence written", output, "real optimization calls",
          summary["physical_optimizer_actual_calls"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "reference", "poststudy", "transfer", "components", "replay", "sparse", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    summarize(**{k: v.resolve() for k, v in vars(args).items()})
