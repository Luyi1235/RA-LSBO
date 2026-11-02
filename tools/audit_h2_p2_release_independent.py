"""Independent P2 Release auditor: bytes, sealed ledger, H2 chronology, paired results.

Uses numpy/scipy only for Gaussian probability arithmetic, not models/evaluators.
No GP/VAE training, optimizer fitting or aircraft simulations.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr

TAG = "h2-p2-windows-pilot-20261102"
ZIP_NAME = "RA-LSBO-H2-P2-pilot-evidence-20261102_02.zip"
URL = f"https://github.com/Luyi1235/RA-LSBO/releases/download/{TAG}/{ZIP_NAME}"
ZIP_SHA256 = "5af27b7fcec3a1d6aaf8a9d139f13c2d61f976d4067ae30f062eceb7c0d5f2e0"
ZIP_SIZE = 58694910
CODE_SHA = "7ca420613ccb50e138dc3c9d4292ac8e863c4ecf"
RAW = "RA-LSBO-H2-P2-pilot-20261102_02/"
PACKAGE = "RA-LSBO-H2-P2-package-20261102_02/"
LOGS = "RA-LSBO-H2-P2-logs-20261102_02/"
ARMS = ("random", "onehot_stable", "corpus_stable",
        "hybrid_onehot", "hybrid_guard", "h2_causal")
CASES = ("aircraft_ordinary", "aircraft_sparse")
SEEDS = (11062, 11063, 11064, 11065)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def stamp(value):
    return sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def parse_sha256_sums(raw):
    entries = {}
    for row in raw.decode("utf-8").splitlines():
        if not row.strip():
            continue
        split = row.split("  ", 1)
        require(len(split) == 2 and len(split[0]) == 64, "invalid SHA256SUMS format")
        hexsha, path = split
        require(path not in entries and all(c in "0123456789abcdef" for c in hexsha.lower()),
                "duplicate or invalid SHA256SUMS record")
        entries[path] = hexsha.lower()
    return entries


def download_with_bounded_retry(path):
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent":"RA-LSBO-H2-independent-pilot-audit"})
            with urllib.request.urlopen(req, timeout=150) as response, path.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            require(path.stat().st_size == ZIP_SIZE, "download size differs")
            require(sha_file(path) == ZIP_SHA256, "download checksum differs")
            print(f"Independent Release ZIP download validated on attempt {attempt}", flush=True)
            return
        except Exception as error:
            print(f"Download attempt {attempt} incomplete: {type(error).__name__}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * attempt)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_members(archive, manifest):
    actual = {}
    sizes = {}
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename
        parts = PurePosixPath(name)
        require(name not in actual and not parts.is_absolute() and ".." not in parts.parts
                and "\\" not in name and not name.startswith("/"), "unsafe/duplicate ZIP path")
        require(member.file_size < 160_000_000, "unexpectedly large archive member")
        hasher = hashlib.sha256()
        total = 0
        with archive.open(member) as stream:
            while True:
                chunk = stream.read(1024*1024)
                if not chunk:
                    break
                total += len(chunk)
                hasher.update(chunk)
        require(total == member.file_size, "bad ZIP member size: " + name)
        digest = hasher.hexdigest()
        if name in manifest:
            require(manifest[name] == digest, "original manifest hash mismatch: " + name)
        actual[name] = digest
        sizes[name] = total
    require(len(manifest) == 789 and len(actual) == 790, "unexpected member/manifest count")
    extra = set(actual).difference(manifest)
    require(extra == {PACKAGE + "SHA256SUMS.txt"},
            "unexpected unlisted ZIP file(s): " + repr(sorted(extra)))
    require(set(manifest).issubset(actual), "manifest file absent from ZIP")
    return actual, sizes


def raw_json(archive, name):
    return json.loads(archive.read(RAW + name))


def raw_jsonl(archive, name):
    return [json.loads(v) for v in archive.read(RAW + name).splitlines() if v.strip()]


def nll(pred, obs):
    mean, var = float(pred["mean"]), float(pred["variance"])
    require(math.isfinite(mean) and math.isfinite(var) and var > 0,
            "H2 invalid Gaussian variance")
    sd = math.sqrt(var)
    if obs["kind"] == "missing":
        return None
    if obs["kind"] == "exact":
        z = (obs["value"]-mean)/sd
        return ("exact_density", .5*math.log(2*math.pi*var) + .5*z*z)
    require(obs["kind"] == "interval", "unsupported observation type")
    lo = -math.inf if obs["lower"] is None else (obs["lower"]-mean)/sd
    hi = math.inf if obs["upper"] is None else (obs["upper"]-mean)/sd
    require(lo < hi, "bad interval")
    if lo == -math.inf:
        ln = float(log_ndtr(hi))
    elif hi == math.inf:
        ln = float(log_ndtr(-lo))
    elif lo >= 0:
        large, small = float(log_ndtr(-lo)), float(log_ndtr(-hi))
        ln = large+float(np.log(-np.expm1(small-large)))
    else:
        large, small = float(log_ndtr(hi)), float(log_ndtr(lo))
        ln = large+float(np.log(-np.expm1(small-large)))
    require(math.isfinite(ln), "nonfinite censored likelihood")
    return ("interval_mass", -ln)


def close(a, b, tolerance=2e-8):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def weight_for(n, clipped):
    if n < 4:
        return .5
    z = .35 * clipped/math.sqrt(n)
    return min(.9, max(.1, 1/(1+math.exp(-z))))


def check_ledger(archive, plan, result):
    ledger = raw_jsonl(archive, "calls.jsonl")
    require(len(ledger) == 2014, "expected 1007 started/settled pairs")
    by_tag, roles, seen_ids = {}, Counter(), set()
    for i, (before, after) in enumerate(zip(ledger[::2],ledger[1::2]), 1):
        require(before["event"] == "started" and after["event"] == "settled", "unpaired ledger")
        require(before["number"] == i, "nonsequential ledger numbers")
        ident = {k: before[k] for k in
                 ("number","tag","role","candidate","contract","new_rcaide_mda")}
        ticket = stamp(ident)
        row = after["record"]
        require(before["call_id"] == after["call_id"] == ticket, "bad ledger call identity")
        require(before["tag"] == after["tag"] and before["tag"] not in by_tag
                and ticket not in seen_ids, "repeated or mismatched call ID/tag")
        require(before["new_rcaide_mda"] is False, "RCAIDE charge detected")
        require(row["candidate"] == before["candidate"], "physical candidate mismatch")
        require(row["observation"]["design_id"] == stamp(row["candidate"]), "observation ID mismatch")
        require(row["observation"]["contract_id"] == before["contract"], "contract mismatch")
        require(after["record_digest"] == stamp(row), "settled record not authenticated by digest")
        require(math.isfinite(after["seconds"]) and after["seconds"] >= 0, "invalid duration")
        by_tag[before["tag"]] = (before, row)
        seen_ids.add(ticket)
        roles[before["role"]] += 1
    expected = {"shared_initial":96,"online":864,"incumbent_step_verification":47}
    require(dict(roles) == expected and result["charged_by_role"] == expected, "cost roles differ")
    require(result["charged_model_calls"] == len(seen_ids) == 1007
            and result["unused_calls_not_reassigned"] == 1
            and result["maximum_model_calls"] == plan["budget"]["maximum_model_calls"] == 1008
            and result["new_rcaide_mda_calls"] == 0, "budget counters differ")
    return by_tag, dict(roles)


def check_trials_and_h2(archive, plan, result, ledger, analysis):
    summary_by_key = {(x["case"], x["seed"], x["method"]): x
                      for x in result["summaries"]}
    require(len(summary_by_key) == len(result["summaries"]) == 48, "duplicate/missing summary")
    require(set(summary_by_key) ==
            {(c,s,a) for c in CASES for s in SEEDS for a in ARMS},
            "trial roster mismatch")
    require(all(r["model_failure_coverage_calls"] == 0 for r in result["summaries"]),
            "unexpected surrogate-failure fallback")

    expected_h2, scores_exact, scores_interval, changes = {},0,0,0
    verifications, cold, records_matched = 0, 0, 0
    for case in CASES:
        for seed in SEEDS:
            initial = [ledger[f"{case}/{seed}/initial/{i}"][1] for i in range(12)]
            initial_ids = {stamp(x["candidate"]) for x in initial}
            require(len(initial_ids) == 12, "repeated shared initial candidate")
            for arm in ARMS:
                folder = f"optimization/{case}/{seed}_{arm}/"
                saved = raw_json(archive, folder+"summary.json")
                summary = summary_by_key[(case,seed,arm)]
                require(saved == summary, "trial summary differs from raw file")
                manifest = raw_json(archive, folder+"manifest.json")
                require(manifest["initial"] == initial and manifest["initial_digest"] == stamp(initial),
                        "shared initial no longer common")
                require((manifest["case"],manifest["seed"],manifest["method"]) == (case,seed,arm),
                        "trial manifest misrouted")
                online = raw_jsonl(archive, folder+"observations.jsonl")
                proposals = raw_jsonl(archive, folder+"proposals.jsonl")
                require(len(proposals) == len(online) == summary["online_calls"] == 18
                        and summary["stop_reason"] is None and not summary["stopped"],
                        "incomplete 18-step trial")
                history = list(initial)
                seen = set(initial_ids)
                for t, (p, row) in enumerate(zip(proposals,online)):
                    require(p["iteration"] == t and p["prefix_digest"] == stamp(history),
                            "history prefix or proposal sequence mutated")
                    require(p["candidate"] == row["candidate"], "proposal-observation mismatch")
                    require(p["metadata"].get("raw_history_digest") in (None,p["prefix_digest"]),
                            "model history differs from raw prefix")
                    ticket,row_ledger = ledger[f"{case}/{seed}/{arm}/online/{t}"]
                    require(ticket["role"] == "online" and row == row_ledger,
                            "online record differs from sealed ledger")
                    identity = stamp(row["candidate"])
                    require(identity not in seen, "duplicate sampled candidate")
                    seen.add(identity)
                    history.append(row)
                    records_matched += 1
                if summary["step_verification"]["status"] == "evaluated":
                    verifications += 1
                    tag = f"{case}/{seed}/{arm}/verification/5"
                    require(tag in ledger and
                            ledger[tag][0]["role"] == "incumbent_step_verification",
                            "missing registered five-second check")
                    refinement = raw_json(archive,folder+"verification.json")
                    require(refinement == ledger[tag][1], "verification ledger mismatch")
                    require(summary["step_verification"]["verified_usable"] is True
                            and summary["step_verification"]["screen"]["status"] == "passed_screen",
                            "invalid numerical verification")
                else:
                    require(summary["step_verification"]["status"] == "no_nominal_feasible_incumbent"
                            and not summary["has_feasible"],
                            "skipped verification without a reason")
                    cold += 1
                if arm != "h2_causal":
                    continue
                require(summary["H2_controller_replay"]["settled_events"] == 18,
                        "missing H2 causal replay")
                n_scored, clipped_sum = 0, 0.
                weights, current_exact, current_interval = [], [], []
                for t, (p,row) in enumerate(zip(proposals,online)):
                    meta = p["metadata"]
                    before = meta["H2_controller_before"]
                    commit = meta["H2_prior_forecast_commitment"]
                    require(before["settled"] == t and before["scored"] == n_scored
                            and before["pending"] is False, "causal predecessor state mismatch")
                    require(close(before["clipped_sum"], clipped_sum)
                            and close(before["onehot_weight"], weight_for(n_scored,clipped_sum)),
                            "H2 pre-proposal weight future-leak or score mismatch")
                    require(commit["step"] == t and commit["design_id"] == stamp(p["candidate"])
                            and commit["prefix"] == p["prefix_digest"]
                            and commit["prior_chain"] == before["event_chain"]
                            and close(commit["weight_at_proposal"], before["onehot_weight"]),
                            "H2 proposal commitment doesn't match predecessor")
                    require(commit["commitment"] == stamp({k:v for k,v in commit.items()
                              if k != "commitment"}), "H2 committed forecast has been altered")
                    weights.append(before["onehot_weight"])
                    observation = dict(row["observation"]["constraints"])["soc"]
                    if commit["skip"] is not None:
                        require(commit["skip"] in ("coverage","model_failure","no_forecast")
                                and meta.get("posterior_fit_skipped") is True
                                and commit["onehot"] is None and commit["latent"] is None,
                                "H2 skip was not explicit/registered")
                    elif observation["kind"] != "missing":
                        parts = meta["forecasts"]["constraints"]["soc"]["components"]
                        idx = meta["selected_index"]
                        require(len(parts["weights"]) == 2
                                and close(parts["weights"][0],before["onehot_weight"])
                                and close(parts["weights"][1],1-before["onehot_weight"]),
                                "H2 mixture doesn't use prior weight")
                        for k,view in enumerate(("onehot","latent")):
                            require(close(commit[view]["mean"],parts["mean"][k][idx])
                                    and close(commit[view]["variance"],parts["variance"][k][idx]),
                                    "H2 selected Gaussian differs from committed")
                        first,second = nll(commit["onehot"],observation),nll(commit["latent"],observation)
                        require(first[0] == second[0], "different H2 censor types")
                        delta = second[1]-first[1]
                        clipped_sum += min(4,max(-4,delta))
                        n_scored += 1
                        mixture_nll = float(-logsumexp([
                            math.log(before["onehot_weight"])-first[1],
                            math.log(1-before["onehot_weight"])-second[1]]))
                        if first[0] == "exact_density":
                            current_exact.append(mixture_nll)
                            scores_exact += 1
                        else:
                            current_interval.append(mixture_nll)
                            scores_interval += 1
                    elif observation["kind"] == "missing":
                        require(commit["skip"] is None, "missing observations are not forecasts")
                weights.append(summary["H2_controller_replay"]["final_state"]["onehot_weight"])
                saved_final = summary["H2_controller_replay"]["final_state"]
                require(n_scored == summary["H2_controller_replay"]["scored_events"]
                        and n_scored == saved_final["scored"]
                        and close(clipped_sum,saved_final["clipped_sum"])
                        and close(weight_for(n_scored,clipped_sum),saved_final["onehot_weight"]),
                        "H2 settled controller state diverged from raw evidence")
                require(len(current_exact) == summary["H2_controller_replay"]["exact_scores"]
                        and len(current_interval) == summary["H2_controller_replay"]["interval_scores"],
                        "H2 observation type count differs")
                if current_exact:
                    require(close(np.mean(current_exact),
                                  summary["soc_prequential"]["exact_density_nll_mean"]),
                            "H2 saved exact-mixture proper score does not match raw forecast")
                if current_interval:
                    require(close(np.mean(current_interval),
                                  summary["soc_prequential"]["interval_mass_nll_mean"]),
                            "H2 saved interval-mixture score does not match raw forecast")
                local_changes = sum(abs(b-a)>1e-14 for a,b in zip(weights,weights[1:]))
                changes += local_changes
                expected_h2[(case,seed)] = {"scored":n_scored,"final_weight":saved_final["onehot_weight"],
                                            "weight_changes":local_changes}
    require(records_matched == 864 and verifications == 47 and cold == 1,
            "trial-to-ledger association or verification counts differ")
    require((scores_exact,scores_interval,changes) == (128,14,118),
            "H2 scored/weight changes disagree with original independent summary")
    require(analysis["h2_scored_events"] == scores_exact + scores_interval
            and analysis["h2_effective_weight_changes"] == changes,
            "checked-in analysis counts disagree with raw evidence")
    return expected_h2


def independent_comparison(result):
    rows = {(r["case"],r["seed"],r["method"]):r for r in result["summaries"]}
    summary = {}
    for case in CASES:
        summary[case] = {}
        for baseline in ("hybrid_onehot","hybrid_guard","onehot_stable","corpus_stable","random"):
            deltas, unavailable = [], []
            for seed in SEEDS:
                a=rows[(case,seed,"h2_causal")]["best_feasible"]
                b=rows[(case,seed,baseline)]["best_feasible"]
                if a is None or b is None:
                    unavailable.append(seed)
                else:
                    deltas.append({"seed":seed, "h2_minus_baseline_Wh":a-b,
                                   "initial_kind":rows[(case,seed,"h2_causal")]["initial_kind"]})
            ds=sorted(x["h2_minus_baseline_Wh"] for x in deltas)
            n=len(ds)
            summary[case][baseline]={"n":n,"mean_Wh":sum(ds)/n if n else None,
                "median_Wh":(ds[n//2] if n%2 else (ds[n//2-1]+ds[n//2])/2) if n else None,
                "h2_wins":sum(x<0 for x in ds),"h2_losses":sum(x>0 for x in ds),
                "h2_ties":sum(x==0 for x in ds),"excluded_unpaired":unavailable,"paired":deltas}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zip-file", type=Path)
    args=parser.parse_args()

    manifest_bytes=(args.results_dir/"SHA256SUMS.txt").read_bytes()
    manifest=parse_sha256_sums(manifest_bytes)
    if args.zip_file is None:
        temp=tempfile.TemporaryDirectory(prefix="h2p2independent_")
        zip_path=Path(temp.name)/ZIP_NAME
        download_with_bounded_retry(zip_path)
    else:
        temp=None
        zip_path=args.zip_file
    require(zip_path.stat().st_size == ZIP_SIZE and sha_file(zip_path) == ZIP_SHA256,
            "ZIP SHA256/size did not match independent Release expectation")
    with zipfile.ZipFile(zip_path) as archive:
        file_hashes, file_sizes = inspect_members(archive, manifest)
        inside=archive.read(PACKAGE+"SHA256SUMS.txt")
        require(parse_sha256_sums(inside) == manifest,
                "Internal checksum list does not match published frozen manifest")

        files_to_compare = {
            "plan.json":"plan.json", "registration.json":"registration.json",
            "qualification_diagnostics.json":"qualification_diagnostics.json",
            "audit.json":"audit.json", "results.json":"results.json"}
        for path,check_in in files_to_compare.items():
            require(raw_json(archive,path) == json.loads(
                (args.results_dir/check_in).read_text(encoding="utf-8")),
                "GitHub PR metadata JSON differs from the sealed ZIP: "+path)
        analysis=json.loads(archive.read(PACKAGE+"analysis.json"))
        require(analysis == json.loads((args.results_dir/"analysis.json").read_text(encoding="utf-8")),
                "PR analysis disagrees with sealed read-only analysis")
        plan=raw_json(archive,"plan.json")
        results=raw_json(archive,"results.json")
        audit=raw_json(archive,"audit.json")
        reg=raw_json(archive,"registration.json")
        require(plan["source_commit"] == CODE_SHA
                and plan["profile"]["name"] == "pilot"
                and tuple(plan["profile"]["seeds"]) == SEEDS
                and plan["profile"]["horizon"] == 18
                and tuple(plan["cases"]) == CASES and tuple(plan["arms"]) == ARMS,
                "pilot identity/roster changed")
        require(plan["budget"]["maximum_model_calls"] == 1008
                and plan["budget"]["maximum_rcaide_mda_calls"] == 0
                and plan["budget"]["online"] == 864
                and plan["budget"]["shared_initial"] == 96
                and plan["budget"]["incumbent_step_verification"] == 48,
                "pilot budget specification changed")
        require(reg["plan_sha256"] == sha_bytes(archive.read(RAW+"plan.json"))
                and reg["maximum_model_calls"] == 1008, "plan registration changed")
        require(audit["passed"] is True and audit["settled_calls"] == 1007
                and audit["trial_count"] == 48 and audit["checked_proposals"] == 864
                and audit["H2_controller_replayed"] is True
                and audit["physical_validation"] is False
                and audit["algorithm_replay"] is False
                and len(audit["evidence_hashes"]) == 580, "original audit seal changed")
        require(len(plan["sources"]) == 182
                and len(set(plan["sources"].values())) == 181,
                "source path/blob count changed")
        for sha in set(plan["sources"].values()):
            name=RAW+"source_snapshot/"+sha+".source"
            require(file_hashes.get(name) == sha, "missing or altered source snapshot: "+sha)
        for rel,sha in audit["evidence_hashes"].items():
            require(file_hashes.get(RAW+rel) == sha,
                    "one or more audit-covered original files changed: "+rel)
        require(plan["smoke_prerequisite"]["source_commit"] == CODE_SHA
                and plan["reviewed_qualification"]["settled_calls"] == 316,
                "P1 smoke or 316-call prerequisite missing")
        require(archive.read(LOGS+"run-exit-code.txt").strip() == b"0",
                "original model-run exit code not successful")
        require(file_sizes[LOGS+"run-stderr.txt"] == 0,
                "stderr log unexpectedly differs")
        ledger, roles=check_ledger(archive,plan,results)
        h2_state=check_trials_and_h2(archive,plan,results,ledger,analysis)
        comparison=independent_comparison(results)

    answer={"schema":"ra_lsbo_h2_p2_independent_sealed_audit_v1",
        "passed":True,"source_commit":CODE_SHA,"download_sha256":ZIP_SHA256,
        "download_bytes":ZIP_SIZE,"zip_members":len(file_hashes),
        "manifest_entries":len(manifest),"audit_evidence_hashes_checked":580,
        "source_path_bindings":182,"distinct_source_snapshots":181,
        "started_and_settled":1007,"charged_roles":roles,"online_proposals":864,
        "trials":48,"five_second_verifications":47,
        "h2_replayed_prediction_prefixes":144,"h2_scored_events":142,
        "h2_weight_changes":118,"h2_final_weights":{f"{case}/{seed}":v
            for (case,seed),v in h2_state.items()},
        "comparison":comparison,
        "new_model_calls":0,"new_rcaide_calls":0,
        "GP_or_VAE_refit":False,"algorithm_decisions_reexecuted":False,
        "physical_validation":False,"confirmatory_evidence":False,
        "important_limit":"Adaptive H2 vs fixed hybrid_onehot conflates dual-view mixture and weighting: no fixed-weight mixture control."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(answer,sort_keys=True,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({k:v for k,v in answer.items() if k not in
                      ("h2_final_weights","comparison")},sort_keys=True))
    print("INDEPENDENT RAW RELEASE AUDIT PASSED")
    if temp is not None:
        temp.cleanup()


if __name__ == "__main__":
    main()
