"""Read-only GP numerical-geometry audit of fixed Windows D29 SOC evidence.

No VAE, EP fitting, GP fitting, BO, simulator, or evaluator calls. This script
checks the Release's original bytes, then independently examines frozen inputs,
Gaussian kernel matrices, approximate evidence weights and EP trace metadata.
It does NOT establish predictive calibration or out-of-sample performance.

python tools/audit_ep_d29_static.py ORIGINAL_RELEASE_ZIP NEW_OUTPUT_DIR
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
from scipy.linalg import cho_factor, cho_solve

EXPECTED_ZIP = "0787fed8ce9ce59fa568c761e1bfe01337c672ebd74965ec970235cf30e1279f"
OLD_ZIP_SHA = "aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9"
EXPECTED_D29_FEATURE_ID = "e458ae755ff5abbdbba4fa5069ad54f4cb1a7d004377831e69d409b48b88f687"
EXPECTED_D28_FEATURE_ID = "75aa389992a576bacc3af2c0abe77526e7338bac01c1c12ea2b1525cc6c9ebfa"
GRID = ((c, d) for c in (.05, .15, .4, 1.) for d in (.5, 1.5))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _as_json(data):
    return json.loads(data.decode("utf-8-sig"))


def load_release(path):
    if sha(path) != EXPECTED_ZIP:
        raise ValueError("Fixed result ZIP sha256 does not match release")
    with zipfile.ZipFile(path) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        if len(files) != 14:
            raise ValueError("Unexpected number of evidence files")
        names = [PurePosixPath(x.filename).name for x in files]
        if len(set(names)) != len(names):
            raise ValueError("Repeated evidence basename")
        for x in files:
            posix = PurePosixPath(x.filename)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError("Unsafe evidence member path")
        raw = {PurePosixPath(i.filename).name: z.read(i) for i in files}
        manifest = _as_json(raw["manifest.json"])
        for item in manifest["files"]:
            by = raw[item["name"]]
            if len(by) != item["bytes"] or hashlib.sha256(by).hexdigest() != item["sha256"]:
                raise ValueError("Manifest member failed byte integrity")
        for line in raw["SHA256SUMS.txt"].decode("utf8").splitlines():
            digest, name = line.split(None, 1)
            if hashlib.sha256(raw[name.strip()]).hexdigest() != digest:
                raise ValueError("SHA256SUMS member failed integrity")
        if manifest["input_archive_sha256"] != OLD_ZIP_SHA:
            raise ValueError("Wrong original interrupted pilot identity")
        if manifest["diagnostic_commit"] != "b5744736c942eadf0a56f79d811f7e7b8ca3d52b":
            raise ValueError("Wrong Windows numerical candidate")
        import io
        with np.load(io.BytesIO(raw["D29_soc_gp_features.npz"]), allow_pickle=False) as data:
            if data.files != ["inputs"]:
                raise ValueError("Unexpected saved feature names")
            X = np.array(data["inputs"], dtype=float)
        observations = _as_json(raw["D29_soc_normalized_evidence.json"])
        attempts = [_as_json(x) for x in raw["candidate_attempts.jsonl"].splitlines() if x]
        plan = _as_json(raw["plan.json"])
        result = _as_json(raw["result.json"])
        features = _as_json(raw["features.json"])
    return X, observations, attempts, plan, result, features, {
        "archive_sha256": EXPECTED_ZIP,
        "files_checked": len(raw),
        "manifest_checked": len(manifest["files"]),
        "source_archive_sha256": OLD_ZIP_SHA,
    }


def independent_matern52(a, b):
    # a/b are already scaled to the kernel's actual length scale 1.
    differences = a[:, None] - b[None, :]
    r = np.sqrt(5) * np.linalg.norm(differences, axis=2)
    return (1 + r + r*r/3) * np.exp(-r)


def independent_group_divisors(dim, ncont, c, d):
    if dim <= 2*ncont:
        raise ValueError("Feature schema is not a continuous/mask/latent layout")
    return np.r_[
        np.full(ncont, c*np.sqrt(ncont)),
        np.full(ncont, np.sqrt(ncont)),
        np.full(dim-2*ncont, d*np.sqrt(dim-2*ncont))
    ]


def candidate_geometry(X, obs, c, d, ncont):
    # Equivalent to original two-block grouping, but independently coded.
    divisor = independent_group_divisors(X.shape[1], ncont, c, d)
    z = X/divisor
    exact = np.asarray([i for i, o in enumerate(obs) if o["kind"] == "exact"], dtype=int)
    interval = np.asarray([i for i, o in enumerate(obs) if o["kind"] == "interval"], dtype=int)
    Zx, Zi = z[exact], z[interval]
    Kxx = independent_matern52(Zx, Zx)
    Kxx.flat[::len(Kxx)+1] += 1e-6 + 1e-8
    f = cho_factor(Kxx, lower=True, check_finite=True)
    Kix = independent_matern52(Zi, Zx)
    Kii = independent_matern52(Zi, Zi)
    Kcond = Kii - Kix @ cho_solve(f, Kix.T)
    Kcond = .5*(Kcond+Kcond.T)
    Kcond.flat[::len(Kcond)+1] += 1e-8
    eigen = np.linalg.eigvalsh(Kcond)
    # An independent *pre-EP* standardized lower-bound check: the SOC model
    # conditions analytically on the 17 exact values and observes 24 positive
    # one-sided lower inequalities. Large tail distances can cause EP numeric
    # instability without requiring any GP/EP fitting here.
    y_exact = np.asarray([obs[i]["value"] for i in exact], dtype=float)
    base_mean = Kix @ cho_solve(f, y_exact)
    lower = np.asarray([obs[i]["lower"] for i in interval], dtype=float)
    if not np.isfinite(lower).all():
        raise ValueError("Expected finite one-sided lower SOC constraints")
    sd = np.sqrt(np.diag(Kcond))
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("Invalid pre-EP conditional variance")
    standardized_lower = (lower-base_mean)/sd
    if not np.isfinite(standardized_lower).all():
        raise ValueError("Nonfinite pre-EP interval tail distance")
    # Observational geometry only, not EP posterior or prediction quality.
    dist = np.sqrt(np.sum((z[:,None]-z[None,:])**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return {
        "continuous_length": c,
        "discrete_length": d,
        "pre_EP_exact_kernel_cond": float(np.linalg.cond(Kxx)),
        "pre_EP_interval_cov_min_eigen": float(eigen[0]),
        "pre_EP_interval_cov_max_eigen": float(eigen[-1]),
        "pre_EP_interval_cov_cond": float(eigen[-1]/eigen[0]) if eigen[0] > 0 else None,
        "minimum_distinct_row_distance": float(np.min(dist)),
        "zero_distance_nonidentical_rows": int(np.count_nonzero(np.triu(dist == 0, k=1))),
        "pre_EP_interval_lower_standardized_min": float(np.min(standardized_lower)),
        "pre_EP_interval_lower_standardized_median": float(np.median(standardized_lower)),
        "pre_EP_interval_lower_standardized_max": float(np.max(standardized_lower)),
        "pre_EP_interval_lower_above_8sigma_count": int(np.sum(standardized_lower > 8.0)),
        "pre_EP_interval_lower_above_20sigma_count": int(np.sum(standardized_lower > 20.0)),
    }


def audit(path):
    X, evidence, attempts, plan, result, features, integrity = load_release(path)
    obs = evidence["observations"]
    if (X.shape[0] != 41 or not np.isfinite(X).all() or len(obs) != 41
            or X.ndim != 2):
        raise ValueError("D29 features/observations inconsistent")
    if (evidence["D29_feature_identity"] != EXPECTED_D29_FEATURE_ID
            or features[0]["feature_identity"] != EXPECTED_D28_FEATURE_ID
            or features[1]["feature_identity"] != EXPECTED_D29_FEATURE_ID):
        raise ValueError("Feature identities differ")
    kinds = Counter(x["kind"] for x in obs)
    if kinds != {"exact": 17, "interval": 24}:
        raise ValueError("Unexpected D29 SOC evidence type distribution")
    if not math.isclose(evidence["scale"], .09529698726991598, rel_tol=1e-12):
        raise ValueError("Unexpected D29 output normalization")
    if result["encoder_fit_count"] != 2 or result["GP_attempt_count"] != 8:
        raise ValueError("Unexpected frozen diagnostic cost")
    grid = list(GRID)
    if (len(attempts) != 8 or len(plan["attempts"]) != 8
            or not all((a["continuous_length"], a["discrete_length"]) == g
                       and a["candidate_index"] == i and a["success"]
                       and a["raw_D29_feature_identity"] == EXPECTED_D29_FEATURE_ID
                       for i, (a, g) in enumerate(zip(attempts, grid)))):
        raise ValueError("Saved candidate identity/result inconsistent")
    rec = []
    for a, (c,d) in zip(attempts, grid):
        diag = a["ep_diagnostics"]
        changes = [s["last_site_change"] for s in diag["ep_trace"]]
        if (not diag["converged"] or not 1 <= diag["sweeps"] <= 60
                or len(changes) != diag["sweeps"] or changes[-1] > 1e-8
                or a["inner_errors"] or not math.isfinite(a["evidence"])
                or any(not math.isfinite(x) for x in
                       a["prediction_mean_minmax"]+a["prediction_variance_minmax"])
                or min(a["prediction_variance_minmax"]) < 0):
            raise ValueError("Not all sealed EP candidates have valid convergence records")
        geometry = candidate_geometry(X, obs, c, d, ncont=6)  # 6 active-schema continuous variables
        rec.append({**geometry,
            "ep_sweeps": diag["sweeps"],
            "ep_terminal_relative_site_change": changes[-1],
            "ep_max_site_precision": diag["ep_trace"][-1]["site_precision_max"],
            "ep_max_linear_term_abs": diag["ep_trace"][-1]["linear_term_max_abs"],
            "log_evidence": a["evidence"],
            "wall_fit_seconds": a["elapsed_seconds"],
            "min_predictive_variance_on_training_X": min(a["prediction_variance_minmax"]),
        })
    # Concentration of approximate evidence weights, not validated probabilities.
    ev = np.array([a["evidence"] for a in attempts])
    raw_weight = np.exp(ev-ev.max())
    w = raw_weight/raw_weight.sum()
    return {
        "schema": "ep_D29_readonly_geometry_and_trace_audit_v1",
        "fixed_source_commit": "b5744736c942eadf0a56f79d811f7e7b8ca3d52b",
        "integrity": integrity,
        "input_shape": list(X.shape),
        "observation_counts": dict(kinds),
        "output_normalization": {"center": evidence["center"], "scale": evidence["scale"]},
        "candidate_attempts": rec,
        "approximate_evidence_weights": w.tolist(),
        "effective_weighted_components": float(1/np.sum(w*w)),
        "evidence_spread": float(ev.max()-ev.min()),
        "statistical_inference": "EP approximate, not independent likelihood validity, calibration, or prequential evidence",
        "new_encoder_fits": 0, "new_EP_fits": 0, "new_aircraft_evaluations": 0, "new_RCAIDE_calls": 0,
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_ep_d29_static.py RELEASE_ZIP NEW_OUTPUT_DIRECTORY")
    archive, output = Path(sys.argv[1]), Path(sys.argv[2])
    if output.exists():
        raise FileExistsError("Use an unused directory; no overwrite or resume")
    output.mkdir(parents=True)
    report = audit(archive)
    with (output/"audit.json").open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    if sha(archive) != EXPECTED_ZIP:
        raise ValueError("Original evidence mutated")
    print(json.dumps({"status":"read_only_complete",
                      "candidates":len(report["candidate_attempts"]),
                      "weight_top":max(report["approximate_evidence_weights"]),
                      "evidence_spread":report["evidence_spread"]}, indent=2))


if __name__ == "__main__":
    main()
