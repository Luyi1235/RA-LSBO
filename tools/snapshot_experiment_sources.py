"""Preserve content-addressed source copies only when current hashes still match."""

import argparse
import hashlib
import json
from pathlib import Path


def snapshot(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    blobs = {}
    for filename, expected in report["source_sha256"].items():
        content = Path(filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Cannot snapshot changed source: " + filename)
        blobs[expected] = content
    folder = report_path.parent / "source_snapshot"
    folder.mkdir(exist_ok=False)
    for sha, content in blobs.items():
        (folder / (sha + ".source")).write_bytes(content)
    with (folder / "index.json").open("x", encoding="utf-8") as stream:
        json.dump(report["source_sha256"], stream, indent=2)
    return {"unique_blobs": len(blobs), "files": len(report["source_sha256"]), "path": str(folder)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.report), indent=2))
