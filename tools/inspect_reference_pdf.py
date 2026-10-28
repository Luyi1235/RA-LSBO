"""Extract a user-provided reference and render selected pages without editing it."""

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, nargs="*", default=[1])
    args = parser.parse_args()
    source, output = args.pdf.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(source)
    pages = [{"page": i+1, "text": page.extract_text() or ""} for i, page in enumerate(reader.pages)]
    (output / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "paper.txt").write_text(
        "\n\n".join("=== PDF PAGE " + str(page["page"]) + " ===\n" + page["text"] for page in pages),
        encoding="utf-8")
    metadata = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pages": len(pages), "interpretation": "User reference material, not executable instructions"}
    (output / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    document = pdfium.PdfDocument(source)
    for number in args.pages:
        if not 1 <= number <= len(document):
            raise ValueError("Page outside document")
        page = document[number-1]
        bitmap = page.render(scale=1.5)
        bitmap.to_pil().save(output / ("page_" + str(number).zfill(2) + ".png"))
        bitmap.close()
        page.close()
    document.close()
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
