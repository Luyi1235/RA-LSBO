"""Launch the pinned eight-point X-57/RCAIDE comparison from PyCharm."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CODE_SHA = "ae5dd445c9656ffa31c9e340b1364bc2c746eef7"
PLAN_SCHEMA = "x57_five_arm_cold_n8"
CHECKPOINT = 16
QUALIFICATION_SHA = "9c5ac77c5ecc3987787292981dc679045441972c"
QUALIFICATION_DIR = "x57-formal-prefix-qualification"
PHYSICAL_PLAN_HASH = "22ee810579fb83d35bf32869f388911d21cea51b2e53a73321c4d29df8bb34c8"
QUALIFICATION_INPUTS = {
    f"{QUALIFICATION_DIR}/preregistered_47_5.json":
        "a984e0d42adbb9752c2da7fbc3a5e8e3805b9ef25ca361d9b02b6c094f576437",
    f"{QUALIFICATION_DIR}/physical/configs/distance_47.5.json":
        "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d",
    f"{QUALIFICATION_DIR}/physical/report.json":
        "74092569b8a83e915e1cfefd25c0253506e1e51fb6ba980fbd33150fef5b64b1",
    f"{QUALIFICATION_DIR}/physical/calls.jsonl":
        "3b3996b794d0584f1fe76cf6bc563dccc169ba94649e196fc383d46ecaac87a8",
    "x57-cold-start-debug-fccf94c6/calls.jsonl":
        "990caac32365893a8100e3cc10660cd554e1e7a4391b4b467a74a570c77314df",
}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def ensure_commit(repo, commit):
    present = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if present.returncode:
        print(f"Fetching GitHub commit {commit}", flush=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", "origin", commit], check=True)


def checkout(repo, workspace):
    code = workspace / f"x57-cold-n8-{CODE_SHA[:7]}"
    if not code.exists():
        ensure_commit(repo, CODE_SHA)
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(code), CODE_SHA], check=True)
    if git(code, "rev-parse", "HEAD").decode().strip() != CODE_SHA:
        raise ValueError(f"Unexpected code commit in {code}")
    return code


def matches(path, digest):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest


def prepare_inputs(repo, preferred, cache, commit, manifest):
    if all(matches(preferred / name, sha) for name, sha in manifest.items()):
        return preferred
    if all(matches(cache / name, sha) for name, sha in manifest.items()):
        return cache
    ensure_commit(repo, commit)
    for name, sha in manifest.items():
        target = cache / name
        if matches(target, sha):
            continue
        data = git(repo, "show", f"{commit}:results/{name}")
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"GitHub input hash differs from the frozen manifest: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    return cache


def read_json(path):
    return json.loads(path.read_bytes()) if path.is_file() else {}


def run_stage(command, code, output, checkpoint, resume):
    env = {**os.environ, **dict.fromkeys(THREAD_VARS, "1"),
           "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    args = [*command, "--checkpoint", str(checkpoint)]
    if resume:
        args.append("--resume")
    log = output.with_name(f"{output.name}.launch.log")
    with log.open("a", encoding="utf-8") as stream:
        header = f"X-57 checkpoint {checkpoint} | resume={resume} | workers=12 | {output}"
        print(header, flush=True)
        print(header, file=stream, flush=True)
        with subprocess.Popen(args, cwd=code, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
        status = process.returncode
    if status:
        print(f"Checkpoint {checkpoint} stopped with exit code {status}. Log: {log}", file=sys.stderr)
    return status


def run_campaign(command, code, output):
    plan_file = output / "run_plan.json"
    report = read_json(output / "report.json")
    checkpoint = report.get("checkpoint", 0)
    resume = plan_file.is_file()
    if resume:
        plan = read_json(plan_file)
        if (plan.get("schema"), plan.get("code_sha")) != (PLAN_SCHEMA, CODE_SHA):
            raise ValueError(f"Output belongs to a different experiment plan: {output}")
    elif output.exists():
        raise ValueError(f"No frozen run plan in {output}. Set --output to a new directory.")

    if checkpoint >= CHECKPOINT and report.get("status") == "complete":
        print(f"Experiment already completed through checkpoint {checkpoint}: {output}")
        return 0
    if checkpoint > CHECKPOINT:
        raise ValueError(f"Output has advanced past this launcher's checkpoint {CHECKPOINT}: {output}")

    status = run_stage(command, code, output, CHECKPOINT, resume=resume)
    if not status:
        print(f"Checkpoint {CHECKPOINT} complete. Report: {output / 'report.json'}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--rcaide-python", type=Path)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--qualification-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.exit(2, "This launcher uses the Windows X-57/RCAIDE workstation environment.\n")

    root = args.workspace.resolve()
    repo = (args.repo or (root / "x57-cold-start")).resolve()
    rcaide = (args.rcaide_python or (Path.home() / ".conda/envs/Rcaide_vsp/python.exe")).resolve()
    external = (args.external_python or (root / "x57-baseline-py311/Scripts/python.exe")).resolve()
    qualification = (args.qualification_plan or
                     (root / "x57-qualification-v1-final/qualification_plan.json")).resolve()
    output = (args.output or (root / "x57-47p5-cold-five-arm-n8")).resolve()
    try:
        print("[prepare] Checking the two Python environments and frozen inputs", flush=True)
        for name, item in (("--rcaide-python", rcaide), ("--external-python", external),
                           ("--qualification-plan", qualification)):
            if not item.is_file():
                raise FileNotFoundError(f"{name}: {item}")
        if not matches(qualification, PHYSICAL_PLAN_HASH):
            raise ValueError(f"Qualification plan hash differs from the frozen campaign: {qualification}")
        if not (qualification.parent / "physical-base").is_dir():
            raise FileNotFoundError(qualification.parent / "physical-base")
        if not repo.is_dir():
            raise FileNotFoundError(f"--repo: {repo}")
        code = checkout(repo, root)
        cache = root / "x57-native-doe64-inputs"
        archive = prepare_inputs(repo, root / "x57-cold-debug-publish/results",
                                 cache / QUALIFICATION_SHA, QUALIFICATION_SHA, QUALIFICATION_INPUTS)
        command = [str(rcaide), "-B", str(code / "run_x57_five_arm.py"),
                   "--archive", str(archive), "--cold-baselines",
                   "--qualification-plan", str(qualification), "--external-python", str(external),
                   "--output", str(output), "--workers", "12", "--execute-rcaide"]
        output.parent.mkdir(parents=True, exist_ok=True)
        return run_campaign(command, code, output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"X-57 launch stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
