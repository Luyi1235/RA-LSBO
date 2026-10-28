"""Use the established analysis with cumulative 500-call acquisition accounting."""

import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import analyze_system_validation as original


def analyze(folder, output):
    old_save, old_markdown = original.save_new, original.markdown

    def save(path, value):
        if Path(path).name == "analysis.json":
            value["accounting"]["maximum_complete_mda_calls"] = 500
            value["accounting"]["training_acquisition_includes_prior_calibration"] = True
            value["analysis_v5_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            value["limitations"].append(
                "Fresh calibration includes directed failure neighborhoods; its prevalence is not population prevalence.")
            for stage in value["physical_acquisition"]["stages"]:
                if stage["stage"] == "mechanism_probes":
                    stage["stage"] = "training_acquisition_including_prior_calibration"
        return old_save(path, value)

    with patch.object(original, "save_new", save), patch.object(
            original, "markdown", lambda value: old_markdown(value).replace(" / 480;", " / 500;")):
        original.analyze(folder, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.folder, args.output)
