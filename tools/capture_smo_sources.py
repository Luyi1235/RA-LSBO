"""Archive primary-source HTML and retrieval hashes for the SMO research audit."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.system_experiment import save_new

SOURCES = {
    "smo_submission": "https://link.springer.com/journal/158/submission-guidelines",
    "smo_algorithms_2016": "https://link.springer.com/article/10.1007/s00158-016-1491-5",
    "smo_replication_2019": "https://link.springer.com/article/10.1007/s00158-019-02298-4",
    "architecture_hidden_constraints_2025": "https://arxiv.org/html/2504.08721v1",
    "scbo_2021": "https://proceedings.mlr.press/v130/eriksson21a.html",
    "boundary_exploration_2024": "https://arxiv.org/html/2402.07692v1",
}


class Title(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attributes):
        if tag == "title":
            self.inside = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, url in SOURCES.items():
        entry = {"name": name, "url": url}
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                data = response.read()
                entry.update(http_status=response.status, resolved_url=response.url)
            path = output/(name+".html.gz")
            with path.open("xb") as stream:
                stream.write(gzip.compress(data, mtime=0))
            parser = Title()
            parser.feed(data.decode("utf-8", errors="replace"))
            entry.update(status="retrieved", title="".join(parser.parts).strip(),
                         original_html_sha256=hashlib.sha256(data).hexdigest(),
                         archived_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         archived_path=str(path))
        except Exception as error:
            entry.update(status="unavailable", error=f"{type(error).__name__}: {error}")
        entries.append(entry)
        print(name, entry["status"], flush=True)
    save_new(output/"sources.json", {
        "audit_date": "2026-10-25", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "note": "A retrieved paywalled landing page does not establish access to the full article.",
        "sources": entries})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output.resolve())
