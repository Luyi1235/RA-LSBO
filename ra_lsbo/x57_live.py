"""Opt-in X57 boundary using the existing isolated worker without physical edits."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .audit import load_json
from .budget import digest
from .legacy import normalize_evaluation


def evaluator_files(base):
    base = Path(base).resolve()
    folder = base / "problems" / "x57"
    paths = [path for path in sorted(folder.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts
             and path.suffix.lower() in (".py", ".txt", ".dat", ".json")]
    paths.extend(base / name for name in ("lsbo/__init__.py", "lsbo/space.py", "lsbo/records.py",
                                         "problems/__init__.py"))
    return {str(path.relative_to(base)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


class X57LiveProblem:
    def __init__(self, base_project, config_path, work_dir, allow_mda=False):
        if type(allow_mda) is not bool:
            raise ValueError("Explicit boolean MDA permission required")
        self.base = Path(base_project).resolve()
        self.config_path = Path(config_path).resolve()
        self.config_bytes = self.config_path.read_bytes()
        self.config = load_json(self.config_bytes)
        self.source_hashes = evaluator_files(self.base)
        if not self.source_hashes:
            raise ValueError("No evaluator files found")
        # The ordinary worker packages live alongside an independently named
        # math-only LSBO package. Reject collisions with any other checkout.
        for name in ("lsbo", "problems"):
            module = sys.modules.get(name)
            if module is not None and self.base not in Path(module.__file__).resolve().parents:
                raise RuntimeError("Conflicting imported package: " + name)
        cached = sys.modules.get("problems.x57.problem")
        if cached is not None and getattr(cached, "_ra_import_snapshot", None) != self.source_hashes:
            raise RuntimeError("Physical modules already loaded without this frozen identity; use a fresh process")
        previous = list(sys.path)
        try:
            sys.path.insert(0, str(self.base))
            module = importlib.import_module("problems.x57.problem")
            importlib.import_module("problems.x57.adapter")
            module._ra_import_snapshot = dict(self.source_hashes)
            self.problem = module.X57Problem(self.config_path, Path(work_dir))
            self.candidate_class = importlib.import_module("lsbo.space").Candidate
        finally:
            sys.path[:] = previous
        self.space = self.problem.space
        self.required_constraints = tuple(self.problem.required_constraints)
        self.analytical_constraints = tuple(self.problem.analytical_constraints)
        self.allow_mda = allow_mda
        self.dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
        self.contract_id = digest({
            "config": self.config, "evaluator_files": self.source_hashes,
            "variables": [variable.__dict__ for variable in self.space.variables],
            "required_constraints": self.required_constraints,
            "python_executable": sys.executable, "python_version": sys.version,
            "dependencies": self.dependencies,
        })
        self.verify_frozen()

    def verify_frozen(self):
        if self.config_path.read_bytes() != self.config_bytes or evaluator_files(self.base) != self.source_hashes:
            raise RuntimeError("Physical evaluator/config changed; prepare a new request")

    def canonical(self, raw):
        if (set(raw.get("continuous", {})) != {v.name for v in self.space.continuous}
                or set(raw.get("discrete", {})) != {v.name for v in self.space.discrete}):
            raise ValueError("Candidate must explicitly match the physical schema")
        canonical = self.space.canonical(self.candidate_class(**raw))
        for variable in self.space.continuous:
            if self.space.active(variable, canonical.discrete):
                value = raw["continuous"][variable.name]
                if isinstance(value, bool) or not variable.bounds[0] <= value <= variable.bounds[1]:
                    raise ValueError("Refusing active-coordinate clipping")
        return canonical.to_dict()

    def precheck(self, raw):
        self.verify_frozen()
        return self.problem.precheck(self.candidate_class(**self.canonical(raw)))

    def evaluate(self, raw):
        if not self.allow_mda:
            raise RuntimeError("Real MDA was not explicitly enabled")
        self.verify_frozen()
        result = self.problem.evaluate(self.candidate_class(**self.canonical(raw)))
        # The original adapter launches a fresh worker and enforces its timeout.
        risk = normalize_evaluation(result.to_dict(), self.required_constraints, self.analytical_constraints).to_dict()
        self.verify_frozen()
        return risk


def prepare_smoke(base_project, config_path, output, active_run=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem = X57LiveProblem(base_project, config_path, output / "workers", allow_mda=False)
    candidate = problem.space.default().to_dict()
    checks = problem.precheck(candidate)
    request = {
        "schema": "ra_x57_single_call_request_v1", "operator_approved": False,
        "purpose": "One current-version adapter smoke evaluation, not optimization or calibration evidence.",
        "base_project": str(problem.base), "config_path": str(problem.config_path),
        "contract_id": problem.contract_id, "candidate": candidate, "precheck": checks,
        "config": problem.config, "evaluator_sha256": problem.source_hashes,
        "dependencies": problem.dependencies,
        "active_run_to_wait_for": None if active_run is None else str(Path(active_run).resolve()),
        "max_mda_calls": 1, "max_risk_charge": 1.0,
        "worker_timeout_seconds": problem.config["runtime"]["timeout_seconds"],
        "notes": ["No historical objective or labels imported into this physical call.",
                  "Approval, unchanged contract and a terminal existing experiment are required.",
                  "No automatic retry after a started call or worker crash."],
    }
    path = output / "request.json"
    path.write_text(json.dumps(request, indent=2, allow_nan=False), encoding="utf-8")
    return path


def execute_smoke(request_path):
    from .budget import BudgetPolicy, RiskLedger
    from .optimizer import dispatch
    request_path = Path(request_path).resolve()
    request_bytes = request_path.read_bytes()
    request = load_json(request_bytes)
    if (request.get("schema") != "ra_x57_single_call_request_v1"
            or request.get("operator_approved") is not True
            or request.get("max_mda_calls") != 1 or request.get("max_risk_charge") != 1.):
        raise ValueError("An explicitly approved single-call request is required")
    if request["active_run_to_wait_for"] is not None:
        summary = load_json((Path(request["active_run_to_wait_for"]) / "summary.json").read_bytes())
        if summary.get("status") not in ("complete", "stopped_with_error"):
            raise RuntimeError("Existing experiment is not terminal; do not compete with or modify it")
    problem = X57LiveProblem(request["base_project"], request["config_path"],
                             request_path.parent / "workers", allow_mda=True)
    if problem.contract_id != request["contract_id"]:
        raise RuntimeError("Approved evaluator identity changed")
    proposal = {"candidate": request["candidate"], "mode": "bootstrap",
                "source": "approved_single_call", "evidence": {"basis": "unavailable"}}
    # Creation with x-mode also blocks repeat executions of the same request.
    with RiskLedger(request_path.parent / "budget.jsonl", BudgetPolicy(1, 1., 1, 0, 0)) as ledger:
        risk = dispatch(problem, ledger, proposal)
        report = {"status": "precheck_rejected" if risk is None else "evaluated",
                  "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                  "contract_id": problem.contract_id, "risk": risk, "ledger": ledger.summary()}
    (request_path.parent / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
