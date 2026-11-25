"""Run the frozen native cohorts without substituting a contextual representation."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .geometry import ORIENTATIONS, adjust, compare, matrices, memberships, normalize, permute_indices, rng_for
from .native_cohort import SUFFIXES, checked_asset, freeze
from .native_routes import MODALITIES, PROTOCOL, read_json
from .router import RequestError, embedding, media_bytes


def load_frozen(root: Path) -> tuple[dict, dict, dict]:
    frozen = read_json(root / "native-freeze.json")
    if frozen["native_protocol_sha256"] != sha(PROTOCOL.read_bytes()):
        raise ValueError("native protocol changed after cohort freeze")
    protocol = read_json(PROTOCOL)
    analysis = PROTOCOL.with_name(protocol["analysis_protocol"])
    if frozen["analysis_protocol_sha256"] != sha(analysis.read_bytes()):
        raise ValueError("metric protocol changed after cohort freeze")
    return frozen, protocol, read_json(analysis)


def vector(root: Path, data_root: Path, row: dict, model: dict) -> np.ndarray:
    channel = model["modality"]
    _, data = checked_asset(data_root, channel, row["assets"][channel])
    content = data.decode("utf8").strip() if channel == "text" else {
        "content": [media_bytes(data, SUFFIXES[channel])],
    }
    try:
        native, alias = embedding(root, model["model"], [content],
                                  context=f"native:{model['id']}:{row['source_id']}", provider=model.get("provider"))
        value = np.asarray(native[0], dtype=float)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RequestError("native response did not contain one embedding") from exc
    if alias != model["response_model"]:
        raise RequestError("native observation returned a different model alias")
    norm = np.linalg.norm(value)
    if value.shape != (model["dimension"],) or not np.isfinite(norm) or norm <= 0:
        raise RequestError("native observation has invalid shape or values")
    return value


def extract(root: Path, manifest: Path) -> dict:
    frozen, _, _ = load_frozen(root)
    if frozen["manifest_sha256"] != sha(manifest.read_bytes()):
        raise ValueError("source manifest changed after cohort freeze")
    destination = root / "native-features.npz"
    freeze_digest = sha((root / "native-freeze.json").read_bytes())
    extraction_record = root / "native-extraction.json"
    if extraction_record.exists():
        prior = read_json(extraction_record)
        if prior["freeze_sha256"] != freeze_digest or prior["features_sha256"] != sha(destination.read_bytes()):
            raise ValueError("recorded native features do not match the frozen cohort")
        return prior
    rows, models = frozen["sources"], frozen["models"]
    arrays = {key: np.full((len(rows), model["dimension"]), np.nan, dtype=float) for key, model in models.items()}
    failures = []
    for key, model in models.items():
        # Complete one encoder bank at a time and retain transport failures.
        def try_measure(row):
            try:
                return vector(root, manifest.resolve().parent, row, model), None
            except RequestError as exc:
                return None, str(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for i, (values, failure) in enumerate(pool.map(try_measure, rows)):
                if failure is None:
                    arrays[key][i] = values
                else:
                    failures.append(dict(source_id=rows[i]["source_id"], model=key, detail=failure))

    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=root, delete=False) as temp:
        np.savez_compressed(temp, source_ids=[row["source_id"] for row in rows], **arrays)
    os.replace(temp.name, destination)
    result = dict(freeze_sha256=freeze_digest,
                  features_sha256=sha(destination.read_bytes()),
                  measured={key: int(np.isfinite(values).all(axis=1).sum()) for key, values in arrays.items()},
                  failures=failures)
    save_json(extraction_record, result)
    return result


def paired_matrices(rows: list[dict], left: np.ndarray, right: np.ndarray, minimum: int) -> tuple:
    norms = [np.linalg.norm(v, axis=1, keepdims=True) for v in (left, right)]
    valid = [np.isfinite(norm) & (norm > 0) for norm in norms]
    names, groups, folds = memberships(rows, [x[:, 0] for x in valid], minimum)
    if len(names) < 3:
        return names, np.array(groups), None
    unit = [np.divide(values, norm, out=np.zeros_like(values), where=active)
            for values, norm, active in zip((left, right), norms, valid, strict=True)]
    return names, np.array(groups), tuple(matrices(values, folds) for values in unit)


def transfer_metric(dev: tuple, test: tuple, orientation: tuple, draws: int, seed: int, label: str) -> dict:
    dev_names, dev_groups, dev_data = dev
    test_names, test_groups, test_data = test
    if dev_data is None or test_data is None:
        return {"status": "coverage_insufficient"}
    generator = rng_for(seed, label, "prospective_metric")
    ids = []
    for names, groups in ((dev_names, dev_groups), (test_names, test_groups)):
        n = len(names)
        permutations = np.vstack([np.arange(n), permute_indices(n, groups, draws, generator, True)])
        ids.append((np.triu_indices(n, 1), permutations))
    fits, errors, constants = [], [], []
    for a, b in orientation:
        fitted = []
        for data, ((i, j), p) in zip((dev_data, test_data), ids, strict=True):
            fitted.append((data[0][a][i, j], data[1][b][p[:, i], p[:, j]]))
        (x, y), (tx, ty) = fitted
        norm = np.linalg.norm(ty, axis=1)
        denominator = x @ x
        if denominator <= 0 or np.any(norm <= 0):
            return {"status": "metric_degenerate"}
        scale = np.maximum(0, (y @ x) / denominator)
        fits.append(float(scale[0]))
        errors.append(np.linalg.norm(scale[:, None] * tx - ty, axis=1) / norm)
        constants.append(float(np.linalg.norm(y[0].mean() - ty[0]) / norm[0]))
    measured = np.mean(errors, axis=0)
    return dict(status="measured", calibration_concepts=dev_names, evaluation_concepts=test_names,
                nonnegative_scales=fits, relative_rmse=float(measured[0]), constant_rmse=float(np.mean(constants)),
                blocked_null_median=float(np.median(measured[1:])),
                blocked_p_lower=float((1 + np.sum(measured[1:] <= measured[0])) / (1 + draws)))


def measure(frozen: dict, protocol: dict, plan: dict, arrays: dict) -> dict:
    rows, models = frozen["sources"], frozen["models"]
    datasets = list(sorted(frozen["datasets"]))
    discovery = next(d for d in datasets if frozen["datasets"][d]["role"] == "discovery")
    plan = dict(plan, fit_metric_within_panel=False,
                exact_taxonomy_limit=50_000,
                minimum_concepts=protocol["minimum_concepts_per_panel"],
                minimum_sources_per_concept_fold=protocol["minimum_sources_per_concept_fold"])
    pairs = []
    for family_split in protocol["family_split"]:
        model_ids = {row["modality"]: key for key, row in models.items() if row["family_split"] == family_split}
        pairs.extend((model_ids[left], model_ids[right], left + "_" + right, family_split)
                     for left, right in itertools.combinations(MODALITIES, 2))
    for channel in MODALITIES:
        left, right = [key for key, row in models.items() if row["modality"] == channel]
        pairs.append((left, right, "within_" + channel, "between_families"))
    total = len(datasets) * len(protocol["concept_split"]) * len(pairs) * 6
    plan["permutations"] = max(plan["permutations"], math.floor(total / .05))
    panels = {}
    for dataset, concept_split in itertools.product(datasets, protocol["concept_split"]):
        indexes = [i for i, row in enumerate(rows) if row["dataset"] == dataset and row["concept_split"] == concept_split]
        panels[(dataset, concept_split)] = ([rows[i] for i in indexes], {key: a[indexes] for key, a in arrays.items()})
    outcomes = []
    reference_rows, reference = panels[(discovery, "development")]
    for pair in pairs:
        left, right, _, family_split = pair
        dev = paired_matrices(reference_rows, reference[left], reference[right], plan["minimum_sources_per_concept_fold"])
        for (dataset, concept_split), (subset, panel_arrays) in panels.items():
            value = compare(list(pair), panel_arrays, subset, plan)
            value.update(dataset=dataset, dataset_role=frozen["datasets"][dataset]["role"],
                         concept_split=concept_split, family_split=family_split)
            if concept_split == "heldout" and value["status"] == "measured":
                sample = paired_matrices(subset, panel_arrays[left], panel_arrays[right],
                                         plan["minimum_sources_per_concept_fold"])
                value["heldout_metric"] = {
                    name: transfer_metric(dev, sample, orientation, plan["permutations"], plan["seed"],
                                          f"{dataset}:{left}:{right}:{name}")
                    for name, orientation in ORIENTATIONS.items()
                }
            outcomes.append(value)
    adjust(outcomes)
    return dict(native_statistical_panels=outcomes, permutations=plan["permutations"], holm_family_slots=total,
                panel_status_counts=dict(Counter(row["status"] for row in outcomes)),
                scale_calibration_dataset=discovery,
                inference="Separate creator-disjoint, concept-held-out, dataset-held-out and family-held-out strata.",
                native_replication_status="awaiting_scientific_review",
                alignment_empirics="requires_native_replication")


def analyze(root: Path) -> dict:
    frozen, protocol, geometry = load_frozen(root)
    meta = read_json(root / "native-extraction.json")
    archive = root / "native-features.npz"
    if (meta["features_sha256"] != sha(archive.read_bytes())
            or meta["freeze_sha256"] != sha((root / "native-freeze.json").read_bytes())):
        raise ValueError("native features lack the frozen source and route lineage")
    with np.load(archive, allow_pickle=False) as vectors:
        if vectors["source_ids"].tolist() != [record["source_id"] for record in frozen["sources"]]:
            raise ValueError("native features lack exact frozen source order")
        result = measure(frozen, protocol, geometry, {key: vectors[key] for key in frozen["models"]})
    result.update(freeze_sha256=meta["freeze_sha256"], features_sha256=meta["features_sha256"])
    save_json(root / "native-geometry.json", result)
    return result


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["freeze", "extract", "analyze"])
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--routes", type=Path)
    args = cli.parse_args()
    if args.stage == "freeze":
        if args.manifest is None or args.routes is None:
            cli.error("freeze requires --manifest and --routes")
        value = freeze(args.root, args.manifest, args.routes)["panels"]
    elif args.stage == "extract":
        if args.manifest is None:
            cli.error("extract requires the frozen --manifest")
        value = extract(args.root, args.manifest)["measured"]
    else:
        value = analyze(args.root)["panel_status_counts"]
    print(json.dumps(value))


if __name__ == "__main__":
    main()
