"""Recompute the paired and cross-source RSA estimates with SciPy's direct APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .acquire import save_json, sha


def recheck(root: Path) -> dict:
    source = root / "features.npz"
    result = root / "geometry.json"
    measured = json.loads(result.read_text(encoding="utf8"))
    if sha(source.read_bytes()) != measured["features_sha256"]:
        raise ValueError("geometry references a different feature snapshot")
    output = []
    with np.load(source, allow_pickle=False) as data:
        lookup = {value: i for i, value in enumerate(data["source_ids"].tolist())}
        for pair in measured["pairs"]:
            if pair["status"] != "measured":
                continue
            views = []
            for representation in (pair["left"], pair["right"]):
                values = data[representation].astype(float)
                views.append([
                    pdist([np.mean([values[lookup[source_id]] / np.linalg.norm(values[lookup[source_id]])
                                   for source_id in concept], axis=0) for concept in fold], metric="cosine")
                    for fold in pair["source_ids_by_fold"]
                ])
            left, right = views
            for name, order in [("paired", [0, 1]), ("cross_source", [1, 0])]:
                estimate = float(np.mean([spearmanr(left[i], right[j]).statistic for i, j in enumerate(order)]))
                expected = pair["outcomes"][name]["global_rsa"]["estimate"]
                output.append(dict(pair=pair["pair"], condition=name, reanalysis=estimate,
                                   recorded=expected, absolute_error=abs(estimate - expected)))
    if not output:
        raise ValueError("no measured ordinal outcomes to reanalyse")
    report = dict(source_sha256=sha(source.read_bytes()), geometry_sha256=sha(result.read_bytes()),
                  scope="same cached features and roster; no independent data, API repeat or reviewer",
                  outcomes=output, max_absolute_error=max(row["absolute_error"] for row in output))
    save_json(root / "reanalysis.json", report)
    if report["max_absolute_error"] > 1e-8:
        raise ValueError("same-data RSA disagreed; inspect ties and rounding")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    args = parser.parse_args()
    result = recheck(args.root)
    print(f"Checked {len(result['outcomes'])} outcomes; max error {result['max_absolute_error']:.3g}")


if __name__ == "__main__":
    main()
