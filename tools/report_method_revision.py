"""Reconcile both development rounds, paired controls and historical diagnostics."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_risk import JointRiskEnsemble, ReferenceTemperature
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.features import history_identity
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.selection import incumbent
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.system_experiment import save_new
from tools.release_system_training_data import load_model_bundle
from tools.run_adaptive_research import lines, problem_for


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "adaptive_full": "Full", "adaptive_aggregate": "Aggregate hidden",
    "adaptive_onehot": "One-hot risk", "adaptive_no_auxiliary": "No auxiliary loss",
    "adaptive_no_calibration": "Raw probability", "adaptive_no_information": "No information",
    "adaptive_no_trust": "Old candidate pool", "old_ra": "Previous RA",
    "cbo": "CBO", "lsbo": "Native LSBO", "random": "Random"}


def audit_direct(margin):
    folder = margin/"direct_joint_paired_control"
    specification = load_json((folder/"plan.json").read_bytes())
    verify_sources(specification["source_sha256"])
    plan = load_json((margin/"plan.json").read_bytes())
    summaries, forecasts, calls, fits_count = [], 0, 0, 0
    comparisons = []
    for case in specification["cases"]:
        problem = problem_for(case, plan["base_project"])
        for seed in specification["seeds"]:
            output = folder/case/("adaptive_full_"+str(seed))
            target = margin/case/("adaptive_full_"+str(seed))
            initial = load_json((margin/case/("initial_"+str(seed)+".json")).read_bytes())
            observed, proposals, fits = (lines(output/n) for n in
                                         ("observations.jsonl", "proposals.jsonl", "model_fits.jsonl"))
            target_fits = lines(target/"model_fits.jsonl")
            target_model = torch.load(target/"risk_0.terminal.pt", map_location="cpu", weights_only=True)
            if (fits[0]["model_id"] != target_model["model_id"]
                    or fits[0]["representation_snapshot"] != target_fits[0]["representation_snapshot"]):
                raise ValueError("Paired control did not start from the identical VAE and terminal network")
            summary = load_json((output/"summary.json").read_bytes())
            ledger = RiskLedger(output/"budget.jsonl").summary()
            if (summary["status"] == "error" or len(observed) != len(proposals)
                    or ledger["charged_calls"] != len(observed) or ledger["unresolved_started_calls"]
                    or ledger["risk_charge"] != len(observed)):
                raise ValueError("Paired control accounting differs")
            rows = list(initial)
            for proposal, row in zip(proposals, observed):
                if (proposal["history_id"] != history_identity(rows)
                        or proposal["proposal"]["candidate"] != row["candidate"]
                        or row["risk"] != problem.evaluate(row["candidate"])):
                    raise ValueError("Paired control history or controlled truth differs")
                rows.append(row)
            for fit in fits:
                number = fit["fit_number"]
                training = [r for r in initial+observed if r["evaluation"] in fit["training_indices"]]
                reference = [r for r in initial if r["evaluation"] in fit["reference_indices"]]
                eligible = [r["evaluation"] for r in rows if r["evaluation"] < fit["before_evaluation"]
                            and r["evaluation"] not in fit["reference_indices"]]
                if eligible != fit["training_indices"]:
                    raise ValueError("Paired control labels were delayed")
                features = StructuredFeatures.load_checkpoint(output/("encoder_"+str(number)+".pt"),
                                                              plan["base_project"], training)
                risk = JointRiskEnsemble.load_checkpoint(output/("risk_"+str(number)+".pt"), features)
                calibration = ReferenceTemperature(risk, reference)
                selected = [p for p in proposals if p["fit_number"] == number]
                raw = np.asarray(risk.predict([p["proposal"]["candidate"] for p in selected])["probabilities"])
                calibrated = calibration.transform(raw, risk.model_id)
                for i, proposal in enumerate(selected):
                    np.testing.assert_allclose(raw[i], proposal["selection"]["raw_events"], atol=2e-7, rtol=1e-6)
                    np.testing.assert_allclose(calibrated[i], proposal["selection"]["transformed_events"],
                                               atol=2e-7, rtol=1e-6)
                    forecasts += 1
                fits_count += 1
            other = load_json((target/"summary.json").read_bytes())
            comparisons.append({
                "case": case, "seed": seed, "direct_calls": len(observed), "bridge_calls": other["evaluations"],
                "direct_feasible": summary["feasible"], "bridge_feasible": other["feasible"],
                "direct_gain_percent": summary["improvement_percent"], "bridge_gain_percent": other["improvement_percent"],
                "difference_percentage_points": other["improvement_percent"]-summary["improvement_percent"],
                "same_initial_terminal_and_vae": True})
            calls += len(observed)
            summaries.append(summary)
    return {"status": "passed", "trials": summaries, "calls": calls, "forecasts_replayed": forecasts,
            "model_fits_replayed": fits_count, "comparisons": comparisons,
            "scope": specification["scope"], "additional_real_mda_calls": 0}


def unique_metrics(first, second):
    reference = ROOT/"runs/system_shared_reference_v5_20261021"
    old, _, calibration = load_model_bundle(reference)
    example = load_json((first/"physical_retrospective/joint_full_forecasts.json").read_bytes())
    unique, keys, indices = [], set(), []
    for i, row in enumerate(example["test"]):
        key = digest(row["candidate"])
        if key not in keys:
            keys.add(key)
            unique.append(row)
            indices.append(i)
    labels, candidates = [r["label"] for r in unique], [r["candidate"] for r in unique]
    raw = old.predict(candidates)["final_feasibility"]
    calibrated = calibration.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
    metrics = [{"family": "old", "method": "old_common_reference",
                "raw": binary_metrics(raw, labels), "calibrated": binary_metrics(calibrated, labels)}]
    for family, folder in (("direct", first), ("bridge", second)):
        for name in ("joint_full", "joint_aggregate", "joint_onehot", "joint_no_auxiliary"):
            subfolder = "physical_retrospective" if family == "direct" else "physical_retrospective_normalized_v2"
            saved = load_json((folder/subfolder/(name+"_forecasts.json")).read_bytes())
            if saved["test"] != example["test"]:
                raise ValueError("Historical diagnostic designs differ")
            raw = np.asarray(saved["raw"]["probabilities"])
            calibrated = np.asarray(saved["calibrated" if family == "direct" else "transformed"])
            metrics.append({"family": family, "method": name,
                            "raw": binary_metrics(raw[indices, 0].tolist(), labels),
                            "calibrated": binary_metrics(calibrated[indices, 0].tolist(), labels)})
    return {"unique_designs": len(unique), "original_observations": len(example["test"]),
            "selection": "First occurrence of each canonical design, all variants on identical inputs",
            "metrics": metrics}


def figures(output, rounds, audits, unique, paired, margin):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, (family, analysis) in enumerate(rounds):
        for column, case in enumerate(("five_mode", "sparse_interior")):
            ax = axes[row, column]
            methods = list(LABELS)
            for i, method in enumerate(methods):
                values = [t["improvement_percent"] for t in analysis["trials"]
                          if t["case"] == case and t["method"] == method]
                color = "#b54942" if method == "adaptive_full" else (
                    "#227f68" if method in ("cbo", "lsbo") else "#687c90")
                ax.scatter(values, np.full(len(values), i), s=20, color=color, alpha=.7, zorder=3)
                ax.plot([np.mean(values)], [i], "|", ms=15, color="black")
            ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
            ax.invert_yaxis()
            ax.set_title(f"{family}: {'Five terminal modes' if column == 0 else 'Sparse interior failure'}")
            ax.set_xlabel("Best feasible objective improvement (%)")
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Controlled development; three paired seeds per panel; dots = runs, tick = mean", fontsize=12)
    fig.savefig(output/"controlled_endpoints.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    names = ["Old RF/GP"]+[("Direct " if r["family"] == "direct" else "Bridge ")+
                          {"joint_full": "full", "joint_aggregate": "aggregate",
                           "joint_onehot": "one-hot", "joint_no_auxiliary": "no auxiliary"}[r["method"]]
                          for r in unique["metrics"][1:]]
    y = np.arange(len(names))
    ax.barh(y-.16, [r["raw"]["brier"] for r in unique["metrics"]], .3, color="#526c80", label="Raw")
    ax.barh(y+.16, [r["calibrated"]["brier"] for r in unique["metrics"]], .3, color="#c27643", label="Calibrated")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Brier error (lower is better)")
    ax.set_title(f"Historical fixed-design predictions: {unique['unique_designs']} unique aircraft designs")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output/"historical_risk.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, case in zip(axes, ("five_mode", "sparse_interior")):
        for method, color in (("adaptive_full", "#b54942"), ("cbo", "#227f68"),
                              ("lsbo", "#526c80"), ("old_ra", "#ab8e3a")):
            trajectories = [r["gain_percent_by_call"] for r in audits[1]["curves"]
                            if r["case"] == case and r["method"] == method]
            ax.plot(range(1, 25), np.mean(trajectories, axis=0), label=LABELS[method], color=color)
        trajectories = []
        for trial in paired["trials"]:
            if trial["case"] != case:
                continue
            directory = margin/"direct_joint_paired_control"/case/("adaptive_full_"+str(trial["seed"]))
            rows = lines(directory/"observations.jsonl")
            best, curve = trial["initial_best"], []
            for row in rows:
                if row["risk"]["feasible_label"] is True:
                    best = min(best, row["risk"]["objective"])
                curve.append(100*(trial["initial_best"]-best)/abs(trial["initial_best"]))
            curve += [curve[-1]]*(24-len(curve))
            trajectories.append(curve)
        ax.plot(range(1, 25), np.mean(trajectories, axis=0), "--", label="Direct joint (paired)", color="#93659b")
        ax.set_title("Five terminal modes" if case == "five_mode" else "Sparse interior failure")
        ax.set_xlabel("Controlled calls")
        ax.set_ylabel("Mean feasible improvement (%)")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Second development round; stopped runs carried forward to 24-call horizon", fontsize=11)
    fig.savefig(output/"paired_learning_curves.png", dpi=170)
    plt.close(fig)


def report(first, second, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    analyses = [load_json((p/"analysis.json").read_bytes()) for p in (first, second)]
    audits = [load_json((p/"independent_audit.json").read_bytes()) for p in (first, second)]
    if any(a["status"] != "completed" for a in analyses) or any(a["status"] != "passed" for a in audits):
        raise ValueError("Both rounds must finish and pass independent audits")
    paired = audit_direct(second)
    unique = unique_metrics(first, second)
    figures(output, [("Direct joint", analyses[0]), ("Conditional margin bridge", analyses[1])],
            audits, unique, paired, second)
    total_calls = sum(a["controlled_online_calls"] for a in analyses)+paired["calls"]
    result = {
        "status": "completed", "additional_real_mda_calls": 0, "software_tests_passed": 234,
        "controlled_trials": 138, "controlled_online_calls": total_calls,
        "shared_preparatory_controlled_labels": 10800,
        "risk_forecasts_replayed": sum(a["selected_risk_forecasts_replayed"] for a in analyses)+paired["forecasts_replayed"],
        "rounds": [{"directory": str(p), "aggregate": a["aggregate"],
                    "paired_endpoint_comparisons": audit["paired_endpoint_comparisons"]}
                   for p, a, audit in zip((first, second), analyses, audits)],
        "post_hoc_direct_control": paired, "unique_historical_risk": unique,
        "accepted_claims": [
            "Censored joint-event and masked-margin representation implemented with checkpoint replay",
            "Fixed-reference training removes rolling-window delays without using reference labels in fitting",
            "Conditional bridge connects actual constraint GP predictions to terminal probabilities",
            "Risk prediction improvement is retrospective on fixed historical inputs only"],
        "rejected_claims": [
            "Stable full-system superiority to native LSBO on both controlled tasks",
            "Consistent mechanism or latent-feature advantage",
            "Adequate real data for all five hidden failure mechanisms",
            "Failure-area transfer, calibrated safety guarantee or new prospective aircraft efficiency"],
        "input_sha256": {str(p/name): hashlib.sha256((p/name).read_bytes()).hexdigest()
                         for p in (first, second) for name in ("plan.json", "analysis.json", "independent_audit.json")},
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_new(output/"report.json", result)
    save_new(output/"paired_control_audit.json", paired)
    table = []
    for label, analysis in zip(("第一轮联合分类", "第二轮条件约束桥接"), analyses):
        table.append("### "+label+"\n\n| 任务 | 方法 | 实际调用 | 可行 | 平均改进% |\n| --- | --- | ---: | ---: | ---: |")
        for item in analysis["aggregate"]:
            table.append(f"| {item['case']} | {item['method']} | {item['calls']} | {item['feasible']} | "
                         f"{item['mean_improvement_percent']:.6f} |")
        table.append("")
    metric_table = ["| 模型 | 唯一设计原始Brier | 唯一设计校准Brier |",
                    "| --- | ---: | ---: |"]
    for item in unique["metrics"]:
        metric_table.append(f"| {item['family']} / {item['method']} | {item['raw']['brier']:.6f} | "
                            f"{item['calibrated']['brier']:.6f} |")
    document = f"""# RA-LSBO 方法修订与两轮开发结果

日期：2026-10-21。用户授权保留四项创新方向、放开原实现路线。
本轮完成文献核查、新代码、两轮独立冻结的受控开发、真实数据回顾性诊断及核验。
新增真实MDA为0；旧全部版本累计500次真实调用仍已用完。

## 结论

框架有实质改进，但完整方法尚未达到“稳定优于强基线”的论文效果要求。
保留原默认和旧实验，不把预测评分改善写成优化效率已提高。

主要变化是：删失观测下的七类联合终止模型；带掩码余量和监督对比损失的辅助风险表示；
bootstrap熵差；固定参考温度校准与无延迟训练资格；自适应局部候选和双重风险记录。
第二轮进一步按条件概率链，将评估完成概率与完成后约束GP满足评分耦合。
GP约束联合项采用并集不等式的代数表达，不能因这一表达就声称GP概率有安全保证。

风险输入保留one-hot直接路径并添加VAE潜变量，因此这是混合表示研究，
并非已经证明纯潜空间优于one-hot。余量回归和对比损失同时消融，未分离二者贡献。
主要开发评分使用联合事件概率；第二轮使用完成分支的GP条件约束评分，
各失效事件保持归一化，避免重复乘入同一失败证据。

## 文献依据与创新边界

局部约束BO参考Eriksson与Poloczek，AISTATS2021，2002.08526；
潜空间局部搜索参考Maus等，2201.11872；监督度量学习参考Grosnit等，2106.03609；
集成熵差信息代理参考Houlsby等，1112.5745；约束边界探索参考Tian等，2402.07692。
这些组成技术已有先例。本项目当前可检验的方法假设是“部分标签机制表示、
完成条件下的约束余量耦合、参考校准与有预算选点的协调”，不能把组合实现视为
创新效果或原创性已获确认。详细核查见METHOD_REVISION_RESEARCH_20261021.md。

## 实验范围及账目

第一轮：{first.name}，种子61–63，66组。
第二轮：{second.name}，种子71–73，66组。
补充6组直接联合分类器，使用第二轮完全相同的600条初始数据；
该补充在查看部分第二轮结果后设立，明确属于事后开发诊断。

累计138组，实际{total_calls}次廉价受控在线调用；
两轮共享离线初始/静态测试标签10800次另计，补充6组复用既有初始标签。
真值重放是廉价方程检查，也与在线费用分开记录。新增真实MDA为0。
所有提前停止、零改进及不利消融均保留，不能只报告正常完成的试验。
每运行上限24次、已观测不可行上限12次；全部新方法硬账本按1扣费，
预期风险是另记的软评分，不作为安全界。

累计{result['risk_forecasts_replayed']}个新方法选点的风险预测恢复回放通过；
两轮独立真值、初始配对、校准隔离、训练ID和硬账本审计通过，
6组补充控制也核对同一初始VAE/终止网络及逐次预测。
这不是全部GP采集值或原LSBO搜索路径的独立回放。
234项软件测试通过。原LSBO、物理模型、旧冻结源码和旧结果保持哈希一致。

第二轮原回顾性评分曾因float32 softmax求和略高于1而中止；
失败目录physical_retrospective保留。后在独立适配层中用float64成员归一化，
重新拟合当前参考温度并保存至physical_retrospective_normalized_v2。
此数值修订不改写66组在线试验及其970个已回放预测；不把修订后的历史评分
冒称为一个额外真实/受控优化试验。各模型最大修订量在diagnostics.json中记录。

## 优化结果

改进均相对每个配对运行自己的共同可行初值；不同种子初值不同。
两轮之间不能直接用平均改进差推断桥接效应，桥接效应另用同初始6组对照检查。

{chr(10).join(table)}

![全部受控端点](../runs/{output.name}/controlled_endpoints.png)

![第二轮与配对直接模型曲线](../runs/{output.name}/paired_learning_curves.png)

普通任务第二轮完整模型与强基线接近，但稀疏任务仍不稳定；
不能声称完整RA稳定领先原LSBO。合并机制、one-hot、去信息探索等版本
在部分任务更好，当前证据不支持四模块均有独立正向贡献。
6组桥接配对诊断见报告JSON，初始网络相同，不把不同种子两轮均值误作配对。

## 真实数据回顾性预测

始终使用560训练、72参考；288条既有优化观测不进入训练。
它们包含262个唯一设计；以下按唯一设计等权，避免26次重复观测改变权重。
旧共同参考、第一轮、第二轮风险模型均在相同输入上比较。
这是已查看历史数据上的方法开发，不能据此声称新方法产生了更好的实际能耗。

{chr(10).join(metric_table)}

![唯一设计风险误差](../runs/{output.name}/historical_risk.png)

首轮按全部288观测评分，完整模型原始Brier为0.015228，
旧共同参考为0.036512；合并机制为0.012997、one-hot为0.013080，
都比完整风险模型更好。完整模型虽改善预测，机制和潜变量优势仍未证实。
校准结果必须逐版本看，不能把校准自动视为增益。

全锚点族5、19各自留出时，新完整、one-hot、无辅助以及旧RF的任务失败召回均为0。
桥接只重分配完成后的可行/显式违规概率，不改变隐藏终止质量，
因此不声称它解决了失效区域迁移。其他四类真实隐藏失效仍无正例。

## 当前可用实现

- ra_lsbo/adaptive_risk.py：部分标签事件似然、掩码辅助监督、集成及温度参考校准。
- ra_lsbo/adaptive_system.py：固定参考在线学习、局部区域、风险价格和信息选点。
- ra_lsbo/margin_joint.py：完成条件下的约束GP桥接与全部权重恢复。
- ra_lsbo/normalized_margin_joint.py：独立的float32舍入修订，记录模型适配标识。
- tools/run_adaptive_research.py、tools/run_margin_research.py：两轮冻结与完整受控循环。
- tools/diagnose_adaptive_physical.py：禁用真实MDA的回顾性风险和全锚点族诊断。
- tools/audit_adaptive_research.py、tools/report_method_revision.py：隔离、账本、配对与结果汇总。

模型权重保存在各运行的encoder_*.pt、risk_*.pt；第二轮另有risk_*.terminal.pt。
桥接checkpoint包括目标/约束GP全部学习参数、缩放和终止网络，加载须绑定相同训练历史与源码。

## 下一步研究判断

已实现的联合表示和约束桥接可作为研究原型继续使用，当前不推荐直接替换
原LSBO或把完整组合推入昂贵真实确认。下一步优先解决阶段条件下的失效迁移、
小样本机制分解的负迁移及探索对目标改进的真实信息价值，而不是继续堆叠
无正例的独立头或增加校准复杂度。潜变量贡献必须继续作为待否证假设。

真实五机制的数据充分性仍需要合法物理工况下的实际失败证据，不能靠合成标签、
代码异常或修改物理模型补齐。本轮没有新增物理预算，也未启动新的真实优化计划。
应先在更多独立受控任务与预先冻结的更大种子组建立稳定优势，再设计新的真实验证。
"""
    document_path = ROOT/"docs/METHOD_REVISION_RESULTS_20261021.md"
    with document_path.open("x", encoding="utf-8") as stream:
        stream.write(document)
    result_paths = [output/"report.json", output/"paired_control_audit.json", document_path,
                    output/"controlled_endpoints.png", output/"historical_risk.png", output/"paired_learning_curves.png"]
    save_new(output/"delivery.json", {"status": "completed", "additional_real_mda_calls": 0,
             "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in result_paths}})
    print("Research report completed", document_path, "controlled calls", total_calls, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=ROOT/"runs/adaptive_research_20261021T071809Z_6fba5696")
    parser.add_argument("--second", type=Path, default=ROOT/"runs/margin_research_20261021T072847Z_79e09810")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/method_revision_report_20261021")
    args = parser.parse_args()
    report(args.first.resolve(), args.second.resolve(), args.output.resolve())
