"""Build a larger Commons candidate with jointly blocked concepts and creators."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .acquire import save_json, sha
from .native_discovery import materialize
from .native_routes import read_json
from .roster import source_components

GROUPS = ("instruments", "animal_vocalizations", "powered_transport", "mechanical_or_natural_events")
SHAPES = ((4, 3, 3, 4), (4, 2, 4, 4), (5, 2, 3, 4), (4, 2, 3, 5), (4, 3, 2, 5))
PARTITIONS = (("development", "A"), ("development", "B"), ("heldout", "A"), ("heldout", "B"))


def allocate(data: dict) -> tuple[list[dict], dict]:
    sources = []
    for components in source_components(data["sources"], data["grouped_perceptual_matches"]):
        cluster = sha("|".join(sorted(item["source_id"] for item in components)).encode())
        sources.extend(dict(item, creator_component=cluster) for item in components)
    sources.sort(key=lambda row: row["source_id"])
    memberships = defaultdict(set)
    groups = {}
    for row in sources:
        memberships[row["concept"]].add(row["creator_component"])
        groups[row["concept"]] = row["group"]
    admissible = {key for key, units in memberships.items() if len(units) >= 4}
    coverage = dict(Counter(groups[key] for key in admissible))
    rows = [dict(row, concept_split=split, fold=fold) for row in sources if row["concept"] in admissible
            for split, fold in PARTITIONS]

    terms, definitions = defaultdict(list), {}
    weights = [1 + int(sha(f"{row['source_id']}:{row['concept_split']}:{row['fold']}".encode())[:6], 16) / 2**32
               for row in rows]

    def indicator(key: tuple) -> int:
        if key not in definitions:
            definitions[key] = len(weights)
            weights.append(0.0)
        return definitions[key]

    for i, row in enumerate(rows):
        source, concept, fold, split, component = (row[k] for k in
                                                   ("source_id", "concept", "fold", "concept_split", "creator_component"))
        for target in (("source", source), ("dependency-concept", component, concept), ("dependency-block", component, split, fold),
                       ("fold-lower", concept, split, fold), ("fold-upper", concept, split, fold)):
            terms[target].append((i, 1))
        terms["balanced", concept, split].append((i, 1 if fold == "A" else -1))
        indicator(("concept", concept, split))
        indicator(("dependency", component, split, fold))

    for key, i in definitions.items():
        if key[0] == "concept":
            _, concept, split = key
            terms["unique-concept", concept].append((i, 1))
            terms["taxonomy", groups[concept], split].append((i, 1))
            for fold in ("A", "B"):
                terms["fold-lower", concept, split, fold].append((i, -2))
                terms["fold-upper", concept, split, fold].append((i, -5))
        else:
            _, creator, split, fold = key
            same_panel = terms["dependency-block", creator, split, fold]
            same_panel.append((i, -len(same_panel)))
            terms["unique-dependency", creator].append((i, 1))

    indexes, columns, values = [], [], []
    for i, members in enumerate(terms.values()):
        indexes.extend([i] * len(members))
        columns.extend(k for k, _ in members)
        values.extend(w for _, w in members)
    matrix = coo_matrix((values, (indexes, columns)), shape=(len(terms), len(weights))).tocsc()
    results = []
    for shape in SHAPES:
        expected = dict(zip(GROUPS, shape, strict=True))
        if any(coverage.get(group, 0) < 2 * count for group, count in expected.items()):
            continue
        lower = [expected[k[1]] if k[0] == "taxonomy" else (-np.inf if k[0] in {"dependency-block", "fold-upper"} else 0)
                 for k in terms]
        upper = [expected[k[1]] if k[0] == "taxonomy" else
                 {"dependency-block": 0, "fold-upper": 0, "balanced": 0, "fold-lower": np.inf}.get(k[0], 1)
                 for k in terms]
        fitted = milp(-np.array(weights), integrality=np.ones(len(weights)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 50, "mip_rel_gap": 0.002})
        results.append(dict(taxonomy=expected, solver_status=fitted.status))
        if fitted.x is not None:
            pick = (fitted.x > 0.5).astype(int)
            observed = matrix @ pick
            if np.all(observed >= lower) and np.all(observed <= upper):
                selected = [row for row, keep in zip(rows, pick[:len(rows)], strict=True) if keep]
                return selected, dict(taxonomy=expected, permutation_support_per_panel=math.prod(math.factorial(n) for n in shape),
                                      eligible_concepts_by_group=coverage, attempted_designs=results,
                                      available_creator_components=len({row["creator_component"] for row in sources}))
    return [], dict(eligible_concepts_by_group=coverage, attempted_designs=results,
                    available_creator_components=len({row["creator_component"] for row in sources}),
                    status="no_feasible_creator_blocked_layout_at_registered_resolution")


def run(source: Path, output: Path, *, augmented: bool = False) -> dict:
    input_path = source / ("screened-augmented.json" if augmented else "screened.json")
    selected, report = allocate(read_json(input_path))
    report.update(screened_manifest_sha256=sha(input_path.read_bytes()), assigned_sources=len(selected))
    if selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            jobs = [pool.submit(materialize, source, output, row, row["concept_split"]) for row in selected]
            data = [job.result() for job in jobs]
        dataset = dict(role="discovery", origin="https://commons.wikimedia.org/", license="per-recording licenses",
                       training_overlap_assessment="Unknown. Some sources were exposed in the contextual discovery pilot.")
        manifest = dict(version=1, datasets={"commons-discovery": dataset}, sources=data)
        file = output / "discovery-candidate.json"
        save_json(file, manifest)
        report.update(manifest_sha256=sha(file.read_bytes()),
                      sources_by_panel=dict(Counter(f"{r['concept_split']}/{r['fold']}" for r in data)),
                      status="candidate_inputs_materialized")
    save_json(output / "discovery-summary.json", report)
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--augmented", action="store_true")
    args = cli.parse_args()
    print(json.dumps(run(args.source, args.output, augmented=args.augmented)))


if __name__ == "__main__":
    main()
