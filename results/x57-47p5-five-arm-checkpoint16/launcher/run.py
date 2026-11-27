"""Launch the pinned eight-point X-57/RCAIDE comparison from PyCharm."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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
EXTERNAL_SOURCE_VAR = "RA_LSBO_X57_EXTERNAL_SOURCE"
# The embedded Python distribution ignores PYTHONPATH. This hook is local to
# the external venv and only activates for the optimizer service.
EXTERNAL_PATH_HOOK = (
    f"import os, sys; source = os.environ.get({EXTERNAL_SOURCE_VAR!r}); "
    "source and sys.argv[1:2] == ['--external-service'] and sys.path.insert(0, source)\n"
)


_GIT = None


def git_exe():
    """Find Git even when PyCharm's PATH omits the DSH/portable Git bin."""
    global _GIT
    if _GIT:
        return _GIT
    candidates = [
        os.environ.get("GIT_EXE"),
        shutil.which("git"),
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        str(Path.home() / ".dsh/profiles/desktop/node_modules/@dsh-ops/"
            "bash-win32-x64/ucrt64/bin/git.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            _GIT = candidate
            return _GIT
    raise FileNotFoundError(
        "Git executable not found. Install Git for Windows, add it to PATH, "
        "or set GIT_EXE to git.exe."
    )


def git(repo, *args):
    return subprocess.check_output([git_exe(), "-C", str(repo), *args])


def ensure_commit(repo, commit):
    present = subprocess.run(
        [git_exe(), "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if present.returncode:
        print(f"Fetching GitHub commit {commit}", flush=True)
        subprocess.run([git_exe(), "-C", str(repo), "fetch", "--no-tags", "origin", commit], check=True)


def checkout(repo, workspace):
    code = workspace / f"x57-cold-n8-{CODE_SHA[:7]}"
    if not code.exists():
        ensure_commit(repo, CODE_SHA)
        subprocess.run([git_exe(), "-C", str(repo), "worktree", "add", "--detach",
                        str(code), CODE_SHA], check=True)
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


def launch_environment(external, code):
    env = {**os.environ, **dict.fromkeys(THREAD_VARS, "1"),
           "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    env["PATH"] = str(Path(git_exe()).parent) + os.pathsep + env.get("PATH", "")
    env.pop(EXTERNAL_SOURCE_VAR, None)
    probe = (
        "import json, sys, sysconfig; "
        "print(json.dumps([sys.flags.isolated, sys.prefix, sys.base_prefix, "
        "sysconfig.get_path('purelib')]))"
    )
    isolated, prefix, base_prefix, site_packages = json.loads(
        subprocess.check_output([str(external), "-B", "-c", probe], env=env)
    )
    if isolated:
        packages = Path(site_packages).resolve()
        if prefix == base_prefix or not packages.is_relative_to(Path(prefix).resolve()):
            raise ValueError(f"An isolated external Python must run from a venv: {external}")
        hook = packages / "ra_lsbo_x57_external.pth"
        expected = EXTERNAL_PATH_HOOK.encode("ascii")
        try:
            with hook.open("xb") as stream:
                stream.write(expected)
        except FileExistsError:
            if hook.read_bytes() != expected:
                raise ValueError(f"Unexpected content in external Python path hook: {hook}")
        env[EXTERNAL_SOURCE_VAR] = str(code)
    return env


def default_output(root):
    original = root / "x57-47p5-cold-five-arm-n8"
    recovery = root / "x57-47p5-cold-five-arm-n8-recovery"
    manifest_file = recovery / "recovery_manifest.json"
    if not manifest_file.is_file():
        return original
    manifest = read_json(manifest_file)
    hashes = manifest.get("source_sha256", {})
    if (manifest.get("original_output") == str(original)
            and all(matches(original / name, hashes.get(name, "").lower())
                    for name in ("calls.jsonl", "run_plan.json"))
            and matches(recovery / "run_plan.json", manifest.get("recovery_plan_sha256", ""))):
        return recovery
    raise ValueError(f"Recovery files do not match the manifest: {manifest_file}")


def run_stage(command, code, output, checkpoint, resume, workers, env):
    args = [*command, "--checkpoint", str(checkpoint)]
    if resume:
        args.append("--resume")
    log = output.with_name(f"{output.name}.launch.log")
    with log.open("a", encoding="utf-8") as stream:
        header = f"X-57 checkpoint {checkpoint} | resume={resume} | workers={workers} | {output}"
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


def run_campaign(command, code, output, env, target=CHECKPOINT):
    plan_file = output / "run_plan.json"
    report = read_json(output / "report.json")
    checkpoint = report.get("checkpoint", 0)
    resume = plan_file.is_file()
    workers = 12
    if resume:
        plan = read_json(plan_file)
        if (plan.get("schema"), plan.get("code_sha")) != (PLAN_SCHEMA, CODE_SHA):
            raise ValueError(f"Output belongs to a different experiment plan: {output}")
        workers = plan["max_workers"]
    elif output.exists():
        raise ValueError(f"No frozen run plan in {output}. Set --output to a new directory.")

    if checkpoint >= target and report.get("status") == "complete":
        print(f"Experiment already completed through checkpoint {checkpoint}: {output}")
        return 0
    if checkpoint > target:
        raise ValueError(f"Output has advanced past the requested checkpoint {target}: {output}")

    status = run_stage([*command, "--workers", str(workers)], code, output,
                       target, resume, workers, env)
    if not status:
        print(f"Checkpoint {target} complete. Report: {output / 'report.json'}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--rcaide-python", type=Path)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--qualification-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=int, choices=range(8, CHECKPOINT + 1), default=CHECKPOINT)
    args = parser.parse_args()
    if os.name != "nt":
        parser.exit(2, "This launcher uses the Windows X-57/RCAIDE workstation environment.\n")

    root = args.workspace.resolve()
    repo = (args.repo or (root / "x57-cold-start")).resolve()
    rcaide = (args.rcaide_python or (Path.home() / ".conda/envs/Rcaide_vsp/python.exe")).resolve()
    external = (args.external_python or (root / "x57-baseline-py311/Scripts/python.exe")).resolve()
    qualification = (args.qualification_plan or
                     (root / "x57-qualification-v1-final/qualification_plan.json")).resolve()
    try:
        print("[prepare] Checking the two Python environments and frozen inputs", flush=True)
        output = (args.output or default_output(root)).resolve()
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
        entrypoint = [str(code / "run_x57_five_arm.py")]
        if (output / "recovery_manifest.json").is_file():
            entrypoint = [str(Path(__file__).with_name("resume_recovery.py").resolve()), str(code)]
        command = [str(rcaide), "-B", *entrypoint,
                   "--archive", str(archive), "--cold-baselines",
                   "--qualification-plan", str(qualification), "--external-python", str(external),
                   "--output", str(output), "--execute-rcaide"]
        env = launch_environment(external, code)
        output.parent.mkdir(parents=True, exist_ok=True)
        return run_campaign(command, code, output, env, target=args.checkpoint)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"X-57 launch stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
