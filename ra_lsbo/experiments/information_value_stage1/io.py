"""Write-once evidence, source pinning, integrity checking and case ZIPs."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

PARENT = "18b48e25dda6ff7d17296087981d0c4dc3ce99b2"
CASES = ("branin_ordinary", "branin_sparse", "rosenbrock_ordinary", "rosenbrock_sparse")


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def tree_hashes(root, exclude=()):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence must not contain symlinks")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in exclude:
                files[name] = file_sha(path)
    return files


def source_receipt(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Use a clean fixed-SHA worktree; store logs/results outside it")
    sha = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", PARENT, sha)
    def entries(ref):
        output = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", ref])
        return dict(row.decode("utf-8").split("\t", 1)[::-1]
                    for row in output.split(b"\0") if row)
    old, current = entries(PARENT), entries(sha)
    changed = [p for p, identity in old.items() if current.get(p) != identity]
    if changed:
        raise RuntimeError("Frozen parent files changed: " + repr(changed[:10]))
    sources = {p: file_sha(root/p) for p in current
               if p.endswith(".py") and (p.startswith("ra_lsbo/") or "/" not in p)}
    return {"head_sha": sha, "parent_sha": PARENT, "all_parent_blobs_unchanged": True,
            "python_sources_sha256": sources, "python": platform.python_version(),
            "platform": platform.platform()}


def manifest(root):
    write_new(Path(root)/"manifest.json", tree_hashes(root, ("manifest.json",)))


def verify(root):
    root = Path(root)
    expected = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    actual = tree_hashes(root, ("manifest.json",))
    if expected != actual:
        raise ValueError("Evidence manifest mismatch (missing, changed or extra files)")
    return {"verified_files": len(actual), "manifest_sha256": file_sha(root/"manifest.json")}


def pack(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    status = verify(root)
    if root == destination or root in destination.parents:
        raise ValueError("ZIP destination must be outside sealed results")
    destination.mkdir(parents=True, exist_ok=False)
    groups = {case: [] for case in CASES}
    groups["summary"] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            parts = path.relative_to(root).parts
            case = next((c for c in CASES if c in parts), "summary")
            groups[case].append(path)
    for case, paths in groups.items():
        if not paths:
            continue
        # Split by uncompressed input size, so no ZIP exceeds the upload limit.
        part, size, bucket = 1, 0, []
        def emit(items, number):
            target = destination/("stage1_%s_%02d.zip" % (case, number))
            with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
                for item in items:
                    z.write(item, item.relative_to(root).as_posix())
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None:
                    raise ValueError("ZIP integrity failure")
        for path in paths:
            n = path.stat().st_size
            if n > 48 * 1024**2:
                raise ValueError("Single evidence file exceeds 48 MiB; report, do not omit it")
            if bucket and size + n > 48 * 1024**2:
                emit(bucket, part)
                part, size, bucket = part+1, 0, []
            bucket.append(path)
            size += n
        if bucket:
            emit(bucket, part)
    write_new(destination/"bundle_manifest.json", {"input_verification": status,
              "archives": tree_hashes(destination)})
    return status
