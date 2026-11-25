"""Post-pilot partial RSA controlling image-caption geometry and category-pair means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .geometry import ORIENTATIONS, empirical, matrices, memberships, normalize, permute_indices, ranks, rng_for

TESTS = (
    ("text_bge", "image_titan", "image_caption_bge"),
    ("text_qwen", "image_voyage", "image_caption_qwen"),
    ("text_gemini", "image_gemini", "image_caption_bge"),
)


def center_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = values.copy()
    for group in np.unique(groups):
        keep = groups == group
        result[..., keep] -= result[..., keep].mean(axis=-1, keepdims=True)
    return result


def correlations(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    xx, yy, xy = x @ x, np.einsum("ij,ij->i", y, y), y @ x
    if z is not None:
        zz = np.einsum("ij,ij->i", z, z)
        xz, yz = z @ x, np.einsum("ij,ij->i", y, z)
        inverse = np.divide(1., zz, out=np.zeros_like(zz), where=zz > 0.)
        xx = xx - xz * xz * inverse
        yy = yy - yz * yz * inverse
        xy = xy - xz * yz * inverse
    denominator = np.sqrt(np.maximum(xx, 0.) * np.maximum(yy, 0.))
    return np.divide(xy, denominator, out=np.zeros_like(xy), where=denominator > 0.)


def compare(arrays, roster: list[dict], spec: tuple[str, str, str]) -> dict:
    left, right, caption = spec
    data = [arrays[key] for key in spec]
    finite = [np.isfinite(value).all(axis=1) & (np.linalg.norm(value, axis=1) > 0) for value in data]
    concepts, population, folds = memberships(roster, finite, 2)
    scope = dict(left=left, right=right, nuisance_caption=caption, concepts=concepts)
    if len(concepts) < 8:
        return dict(scope, status="insufficient_concepts")
    used = np.concatenate([cohort for fold in folds for cohort in fold])
    views = []
    for values in data:
        norm = np.zeros_like(values, dtype=float)
        norm[used] = normalize(values[used])
        views.append([ranks(d) for d in matrices(norm, folds)])
    a, b, c = views
    i, j = np.triu_indices(len(concepts), 1)
    taxonomy = np.asarray(["|".join(sorted((population[u], population[v]))) for u, v in zip(i, j, strict=True)])
    ph = permute_indices(len(concepts), np.asarray(population), 4999,
                         rng_for(20261125, left, right, "caption_partial"), True)
    perms = np.vstack([np.arange(len(concepts)), ph])
    results = {name: [] for name in ("unconditioned", "taxonomy_only", "caption_only", "taxonomy_and_caption")}
    for l, r in ORIENTATIONS["cross_source"]:
        x, y, z = a[l][i, j], b[r][perms[:, i], perms[:, j]], c[r][perms[:, i], perms[:, j]]
        gx, gy, gz = [center_within(value, taxonomy) for value in (x, y, z)]
        results["unconditioned"].append(correlations(x, y, None))
        results["taxonomy_only"].append(correlations(gx, gy, None))
        results["caption_only"].append(correlations(x, y, z))
        results["taxonomy_and_caption"].append(correlations(gx, gy, gz))
    outcomes = {}
    for label, values in results.items():
        result = np.mean(values, axis=0)
        outcomes[label] = empirical(float(result[0]), result[1:])
    return dict(scope, status="measured", sources=len(used), cross_source=outcomes)


def evaluate(root: Path) -> dict:
    features = root / "features.npz"
    roster = root / "roster.json"
    rows = json.loads(roster.read_text(encoding="utf8"))["sources"]
    with np.load(features, allow_pickle=False) as arrays:
        if arrays["source_ids"].tolist() != [row["source_id"] for row in rows]:
            raise ValueError("roster and features disagree")
        observations = [compare(arrays, rows, pair) for pair in TESTS]
    result = dict(features_sha256=sha(features.read_bytes()), roster_sha256=sha(roster.read_bytes()),
                  scope="adaptive descriptive sensitivity; no causal mediation, no additional inferential family",
                  label_null="blocked on broad taxonomy, permuting image and image-caption axes together",
                  outcomes=observations)
    save_json(root / "caption-partial.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    result = evaluate(parser.parse_args().root)
    print(json.dumps([dict(pair=[p["left"], p["right"]], **p.get("cross_source", {})) for p in result["outcomes"]]))
