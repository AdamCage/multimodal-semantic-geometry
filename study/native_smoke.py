"""Exercise native statistics on synthetic vectors and a real candidate layout."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .native_cohort import exposed_recordings, freeze_layout
from .native_measure import measure
from .native_routes import PROTOCOL, read_json


def run(manifest: Path, output: Path) -> dict:
    registration = read_json(PROTOCOL)
    analysis = read_json(PROTOCOL.with_name(registration["analysis_protocol"]))
    analysis["creator_bootstraps"] = 20
    datasets, rows, _ = freeze_layout(read_json(manifest), registration, exposed_recordings())
    models = {row["id"]: row for row in registration["candidates"]}
    rng = np.random.default_rng(analysis["seed"])
    concepts = {name: i for i, name in enumerate(sorted({row["concept"] for row in rows}))}
    latent = rng.normal(size=(len(concepts), 9))[[concepts[row["concept"]] for row in rows]]
    vectors = {}
    for i, model in enumerate(models):
        dimension = 16 + i
        vectors[model] = latent @ rng.normal(size=(9, dimension)) + 0.25 * rng.normal(size=(len(rows), dimension))
    measured = measure(dict(sources=rows, datasets=datasets, models=models), registration, analysis, vectors)
    patterns = Counter((row["dataset"], row["concept_split"], row.get("distinct_taxonomy_permutations"),
                        row.get("taxonomy_null_sampling")) for row in measured["native_statistical_panels"])
    report = dict(
        synthetic_only=True, manifest_sha256=sha(manifest.read_bytes()), sources=len(rows),
        holm_family_slots=measured["holm_family_slots"], statuses=measured["panel_status_counts"],
        groups=[dict(dataset=dataset, concept_split=split, exact_null_support=support, sampling=sampling, panels=count)
                for (dataset, split, support, sampling), count in patterns.items()],
        protocol_sha256=sha(PROTOCOL.read_bytes()), creator_bootstraps=analysis["creator_bootstraps"],
        scope="Synthetic numerical vectors share a simulated concept effect. Output verifies only the analysis code.",
    )
    save_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.manifest, args.output)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
