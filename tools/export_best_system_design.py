"""Export an actual audited feasible design and its physical evidence; zero MDA."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.partial_observation import normalize_with_partial_lift
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import read_lines


def export(folder, output):
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal independently audited study")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                            output/"unused", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    shared = load_json((folder/"shared_history.json").read_bytes())
    entries = [(r, {"kind": "shared", "evaluation": r["evaluation"]}) for r in shared]
    for run in report["physical"]:
        entries.extend((r, {"kind": "optimizer", "method": run["method"], "seed": run["seed"],
                            "evaluation": r["evaluation"], "directory": run["directory"]})
                       for r in read_lines(Path(run["directory"])/"observations.jsonl"))
    valid = [(r, source) for r, source in entries
             if r["risk"]["feasible_label"] is True and r["risk"]["objective_observed"]]
    row, source = min(valid, key=lambda entry: entry[0]["risk"]["objective"])
    risk = row["risk"]
    worker = Path(risk["evidence"]["worker_dir"])
    request_path, response_path = worker/"request.json", worker/"response.json"
    request, response = load_json(request_path.read_bytes()), load_json(response_path.read_bytes())
    normalized = normalize_with_partial_lift(
        response, problem.required_constraints, problem.analytical_constraints).to_dict()
    if (request["candidate"] != row["candidate"] or request["config"] != problem.config
            or normalized["feasible_label"] is not True or any(
                normalized[k] != risk[k] for k in
                ("objective", "constraints", "constraint_observed", "hidden_labels"))):
        raise ValueError("Best design differs from actual physical request/response")
    candidate_key = problem.space.key(problem.candidate_class(**row["candidate"]))
    replicas = [(r, s) for r, s in entries if problem.space.key(
        problem.candidate_class(**r["candidate"])) == candidate_key]
    metadata = response.get("metrics", {})
    result = dict(
        schema="ra_audited_best_observed_design_v1", additional_mda_calls=0,
        objective_Wh=risk["objective"], candidate=row["candidate"], candidate_key=candidate_key,
        candidate_identity="Original DesignSpace.key: canonical branch and active unit coordinates rounded to 10 decimals",
        constraints=risk["constraints"], constraint_observed=risk["constraint_observed"],
        origin=source, physical_contract_id=problem.contract_id,
        label_contract_id=plan.get("label_contract_id"),
        worker_dir=str(worker), raw_status=normalized["raw_status"],
        energy_audit_status=metadata.get("energy_audit_status"),
        mission_solver_segments=metadata.get("mission_solver_segments"),
        available_same_design_evaluations=len(replicas),
        replicas=[dict(origin=s, objective=r["risk"]["objective"],
                       feasible_label=r["risk"]["feasible_label"], state=r["risk"]["state"])
                  for r, s in replicas],
        all_observed_replicas_feasible=all(r["risk"]["feasible_label"] is True for r, s in replicas),
        scope="Best actually observed feasible objective under the frozen conceptual physical model",
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (request_path, response_path, folder/"independent_audit.json")},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"best_design.json", result)
    lines = [
        "# 最优已观察可行构型", "", f"任务能耗：{risk['objective']:.8f} Wh。",
        "数值来自已完成且独立审计通过的原始物理响应；导出未增加MDA。",
        f"来源：{source}。按原设计空间规范化身份匹配的已观察评估数：{len(replicas)}。", "",
        "| 变量 | 数值 |", "| --- | --- |"]
    for name, value in row["candidate"]["discrete"].items():
        lines.append(f"| {name} | {value} |")
    for name, value in row["candidate"]["continuous"].items():
        active = next(v for v in problem.space.continuous if v.name == name)
        text = f"{value:.12g}" if problem.space.active(active, row["candidate"]["discrete"]) else f"{value:.12g} (inactive)"
        lines.append(f"| {name} | {text} |")
    lines += ["", "约束使用原物理定义和单位，满足条件为 `g(x) <= 0`。", "",
              "| 约束 | 原始余量 |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value:.12g} |" for name, value in risk["constraints"].items())
    lines += ["", "本结果依赖冻结的概念设计物理近似，不等于飞行试验验证。"
              "完整请求、响应、重复评估状态与来源哈希见 `best_design.json`。", ""]
    (output/"DESIGN.md").write_text("\n".join(lines), encoding="utf-8")
    (output/"export.source.py").write_bytes(Path(__file__).read_bytes())
    print("Best audited design exported", risk["objective"], "Wh", source, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.folder.resolve(), args.output.resolve())
