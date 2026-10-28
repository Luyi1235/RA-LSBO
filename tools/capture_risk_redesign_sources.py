"""Archive primary papers and pinned, read-only implementation excerpts."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


PAPERS = {
    "cbob_aaai_2024": "https://ojs.aaai.org/index.php/AAAI/article/view/29488",
    "cbob_aaai_2024_full": "https://ojs.aaai.org/index.php/AAAI/article/view/29488/30805",
    "becbo_icml_2024": "https://proceedings.mlr.press/v235/tian24g.html",
    "becbo_full_v1": "https://arxiv.org/html/2402.07692v1",
    "architecture_hidden_constraints_v1": "https://arxiv.org/html/2504.08721v1",
    "anubis_rsc_2025": "https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00018a",
    "anubis_publisher_mirror_2025": (
        "https://www.sciencedirect.com/org/science/article/pii/S2635098X25001329"),
    "safe_bocp_full_v3": "https://arxiv.org/html/2306.17815v3",
    "locbo_full_v2": "https://arxiv.org/html/2411.17387v2",
    "online_calibration_aistats_2024": "https://proceedings.mlr.press/v238/deshpande24a.html",
}

# These are retrieval commits, not claims about the versions used in the papers.
REPOSITORIES = {
    "cbob": {
        "repository": "COLA-Laboratory/CBOB",
        "commit": "a3339f167765ab8678575c5981c5e6238d7be33f",
        "files": [
            "README.md", "single_run.py", "utils/sequential.py",
            "models/gp_multiple_constraints.py", "models/gp_heteroscedastic_regression.py",
            "constraint_acquisition/eicb.py", "constraint_acquisition/eicb_cdf.py",
            "utils/ep_processing.py", "benchmark/data_observation.py", "cbob_gpflow.zip",
        ],
    },
    "becbo": {
        "repository": "yunshengtian/BE-CBO",
        "commit": "76c12b049c8748d0616039608323d78ab0c51f43",
        "files": [
            "README.md", "LICENSE", "algorithms/becbo/becbo.py",
            "model/classifier/de.py", "model/classifier/gp.py", "model/fit.py",
            "model/predict.py", "algorithms/base.py", "exp/run_exp.py",
        ],
    },
    "atlas_anubis": {
        "repository": "the-matter-lab/atlas",
        "commit": "ebb1f4f728107f759c48289c2faecfa3af19254f",
        "files": [
            "README.md", "LICENSE", "requirements.txt",
            "src/atlas/acquisition_functions/acqfs.py",
            "src/atlas/acquisition_functions/acqf_utils.py",
            "src/atlas/unknown_constraints/unknown_constraints.py",
            "examples/unknown_constraints/run.py",
        ],
    },
    "sbarchopt": {
        "repository": "jbussemaker/SBArchOpt",
        "commit": "b78705044ad45653e5e77d2f0e1dda67581d6096",
        "files": [
            "README.md", "LICENSE", "sb_arch_opt/sampling.py",
            "sb_arch_opt/algo/arch_sbo/hc_strategy.py",
            "sb_arch_opt/algo/arch_sbo/infill.py",
            "sb_arch_opt/problems/hidden_constraints.py",
        ],
    },
}


def retrieve(url):
    result = subprocess.run([
        "curl.exe", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "5", "--max-time", "15",
        "--user-agent", "RA-LSBO research source audit",
        "--write-out", "\n%{http_code}\n%{url_effective}\n%{content_type}", url,
    ], capture_output=True, check=True, timeout=20)
    data, status, resolved_url, content_type = result.stdout.rsplit(b"\n", 3)
    return data, {
        "http_status": int(status), "resolved_url": resolved_url.decode(),
        "content_type": content_type.decode(),
    }


def archive(url, target, compressed=False):
    result = {"url": url, "path": str(target)}
    try:
        if target.exists():
            payload = target.read_bytes()
            data = gzip.decompress(payload) if compressed else payload
            metadata = {"resumed_existing_archive": True}
        else:
            data, metadata = retrieve(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = gzip.compress(data, mtime=0) if compressed else data
            with target.open("xb") as stream:
                stream.write(payload)
        result.update(metadata, status="retrieved", bytes=len(data),
                      original_sha256=hashlib.sha256(data).hexdigest(),
                      archived_sha256=hashlib.sha256(payload).hexdigest())
    except Exception as error:
        result.update(status="unavailable", error=f"{type(error).__name__}: {error}")
    return result


def capture(output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    if (output/"manifest.json").exists():
        raise FileExistsError("A completed capture must not be overwritten")
    papers, repositories = [], []
    for name, url in PAPERS.items():
        entry = archive(url, output/"papers"/(name+".source.gz"), compressed=True)
        papers.append(dict(name=name, **entry))
        print(name, entry["status"], flush=True)
    for name, specification in REPOSITORIES.items():
        repository, commit = specification["repository"], specification["commit"]
        entries = []
        for relative in specification["files"]:
            url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
            entries.append(archive(url, output/"repositories"/name/relative))
            print(name, relative, entries[-1]["status"], flush=True)
        repositories.append(dict(name=name, **specification, captures=entries))
        print(name, sum(e["status"] == "retrieved" for e in entries),
              "/", len(entries), flush=True)
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only excerpts; no upstream code executed or installed.",
        "version_note": (
            "Pinned retrieval commits may differ from the publication experiment versions. "
            "Absence of a license is not permission to copy an implementation."),
        "papers": papers, "repositories": repositories,
    }
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    capture(arguments.output.resolve(), resume=arguments.resume)
