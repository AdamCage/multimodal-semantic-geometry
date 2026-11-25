"""Exploratory concept geometry with label nulls and creator-aware sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .acquire import save_json, sha

PROTOCOL = Path(__file__).with_name("protocol.json")
ORIENTATIONS = {"paired": ((0, 0), (1, 1)), "cross_source": ((0, 1), (1, 0))}
NULLS = ("global", "blocked", "within")


def rng_for(seed: int, *parts: str) -> np.random.Generator:
    material = "|".join([str(seed), *parts]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def normalize(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("cosine geometry requires finite, nonzero vectors")
    return values / norms


def rdm(centroids: np.ndarray) -> np.ndarray:
    center = normalize(centroids)
    matrix = np.clip(1.0 - center @ center.T, 0.0, 2.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def memberships(rows: list[dict], masks: list[np.ndarray], minimum: int) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    retained = np.logical_and.reduce(masks)
    labels = sorted({(row["group"], row["concept"]) for row in rows})
    concepts, groups = [], []
    folds = [[], []]
    for group, concept in labels:
        indices = [np.asarray(sorted((i for i, row in enumerate(rows)
                                     if retained[i] and row["concept"] == concept and row["fold"] == fold),
                                     key=lambda i: rows[i]["source_id"]), dtype=int) for fold in ("A", "B")]
        count = min(map(len, indices))
        if count >= minimum:
            concepts.append(concept)
            groups.append(group)
            for target, index in zip(folds, indices, strict=True):
                target.append(index[:count])
    return concepts, groups, folds


def matrices(values: np.ndarray, folds: list[list[np.ndarray]], weights: np.ndarray | None = None) -> list[np.ndarray]:
    result = []
    for fold in folds:
        if weights is None:
            means = [values[selection].mean(axis=0) for selection in fold]
        else:
            means = [np.average(values[selection], weights=weights[selection], axis=0) for selection in fold]
        result.append(rdm(np.stack(means)))
    return result


def ranks(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(len(matrix), 1)
    ordered = rankdata(matrix[upper]).astype(float)
    ordered -= ordered.mean()
    norm = np.linalg.norm(ordered)
    output = np.zeros_like(matrix)
    if norm > 0:
        output[upper] = ordered / norm
    return output + output.T


def permute_indices(n: int, groups: np.ndarray, draws: int,
                    rng: np.random.Generator, blocked: bool) -> np.ndarray:
    pool = np.tile(np.arange(n), (draws, 1))
    if not blocked:
        return rng.permuted(pool, axis=1)
    for group in np.unique(groups):
        indexes = np.flatnonzero(groups == group)
        pool[:, indexes] = rng.permuted(pool[:, indexes], axis=1)
    return pool


def exact_taxonomy_permutations(groups: np.ndarray) -> np.ndarray:
    blocks = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    pool = np.empty((math.prod(math.factorial(len(block)) for block in blocks), len(groups)), dtype=int)
    for i, assignment in enumerate(itertools.product(*(itertools.permutations(block) for block in blocks))):
        for indexes, places in zip(blocks, assignment, strict=True):
            pool[i, indexes] = places
    return pool


def pairs_average(x: list, y: list, orientations: tuple, evaluator) -> np.ndarray:
    return (evaluator(x[orientations[0][0]], y[orientations[0][1]]) +
            evaluator(x[orientations[1][0]], y[orientations[1][1]])) / 2


def score_rank(left: list, right: list, orientations: tuple, permutations: np.ndarray) -> np.ndarray:
    n = left[0].shape[0]
    i, j = np.triu_indices(n, 1)
    pi, pj = permutations[:, i], permutations[:, j]
    return pairs_average(left, right, orientations,
                         lambda x, y: np.sum(x[i, j] * y[pi, pj], axis=1))


def score_within(left: list, right: list, orientations: tuple, permutations: np.ndarray,
                 groups: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    scores, names = [], []
    for group in np.unique(groups):
        loc = np.flatnonzero(groups == group)
        if len(loc) < 3:
            continue
        sliced_x = [ranks(view[np.ix_(loc, loc)]) for view in left]
        sliced_y = [ranks(view[np.ix_(loc, loc)]) for view in right]
        # Convert each global index into the corresponding contiguous within-group index.
        lookup = np.full(len(groups), -1, dtype=int)
        lookup[loc] = np.arange(len(loc))
        local_perm = lookup[permutations[:, loc]]
        scores.append(score_rank(sliced_x, sliced_y, orientations, local_perm))
        names.append(str(group))
    return (np.mean(scores, axis=0) if scores else None), names


def empirical(observed: float, null: np.ndarray) -> dict:
    return dict(estimate=observed, null_median=float(np.median(null)),
                null_q95=float(np.quantile(null, .95)),
                p=float((1 + (null >= observed - 1e-12).sum()) / (len(null) + 1)))


def centered(matrix: np.ndarray) -> np.ndarray:
    kernel = 1 - matrix
    corrected = kernel - kernel.mean(axis=0) - kernel.mean(axis=1)[:, None] + kernel.mean()
    norm = np.linalg.norm(corrected)
    return corrected / norm if norm > 0 else np.zeros_like(corrected)


def cka_scores(left: list, right: list, orientation: tuple, perm: np.ndarray) -> np.ndarray:
    px, py = [centered(a) for a in left], [centered(a) for a in right]
    return pairs_average(px, py, orientation, lambda x, y: np.sum(
        x * y[perm[:, :, None], perm[:, None, :]], axis=(1, 2)))


def neighbors(distances: np.ndarray, k: int) -> np.ndarray:
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    kth = np.partition(masked, k - 1, axis=1)[:, k - 1, None]
    closer = masked < kth
    ties = masked == kth
    return closer + ties * ((k - closer.sum(axis=1)) / ties.sum(axis=1))[:, None]


def triplet_agreement(x: np.ndarray, y: np.ndarray) -> float:
    values = []
    for anchor in range(len(x)):
        i, j = np.triu_indices(len(x), 1)
        keep = (i != anchor) & (j != anchor)
        px = np.sign(x[anchor, i[keep]] - x[anchor, j[keep]])
        py = np.sign(y[anchor, i[keep]] - y[anchor, j[keep]])
        values.append(np.mean((1 + px * py) / 2))
    return float(np.mean(values))


def heldout_scale(left: list, right: list, orient: tuple, concepts: list[str],
                  seed: int, permutations: np.ndarray | None = None) -> dict:
    order = rng_for(seed, "metric", *concepts).permutation(len(concepts))
    dev, test = np.sort(order[::2]), np.sort(order[1::2])
    def upper(matrix: np.ndarray, indexes: np.ndarray) -> np.ndarray:
        i, j = np.triu_indices(len(indexes), 1)
        return matrix[indexes[i], indexes[j]]
    folds, null = [], []
    for a, b in orient:
        xa, ya = upper(left[a], dev), upper(right[b], dev)
        xb, yb = upper(left[a], test), upper(right[b], test)
        scale_denom, norm = xa @ xa, np.linalg.norm(yb)
        if scale_denom <= 0 or norm <= 0:
            continue
        scale = max(0.0, float((xa @ ya) / scale_denom))
        folds.append(dict(scale=scale, relative_rmse=float(np.linalg.norm(scale * xb - yb) / norm),
                          constant_rmse=float(np.linalg.norm(ya.mean() - yb) / norm)))
        if permutations is not None:
            di, dj = np.triu_indices(len(dev), 1)
            ti, tj = np.triu_indices(len(test), 1)
            dev_null = right[b][permutations[:, dev[di]], permutations[:, dev[dj]]]
            test_null = right[b][permutations[:, test[ti]], permutations[:, test[tj]]]
            scale_null = np.maximum(0, (dev_null @ xa) / scale_denom)
            test_denom = np.linalg.norm(test_null, axis=1)
            rmse_null = np.full(len(test_null), np.nan)
            np.divide(np.linalg.norm(scale_null[:, None] * xb - test_null, axis=1),
                      test_denom, out=rmse_null, where=test_denom > 0)
            null.append(rmse_null)
    estimate = float(np.mean([row["relative_rmse"] for row in folds])) if folds else None
    report = dict(development=[concepts[i] for i in dev], heldout=[concepts[i] for i in test],
                  orientations=folds, relative_rmse=estimate,
                  constant_rmse=float(np.mean([row["constant_rmse"] for row in folds])) if folds else None)
    if estimate is not None and null:
        draws = np.mean(null, axis=0)
        draws = draws[np.isfinite(draws)]
        if len(draws):
            report["permutation"] = dict(n=len(draws), null_median=float(np.median(draws)),
                                         null_q05=float(np.quantile(draws, .05)),
                                         p_lower_unadjusted=float((1 + (draws <= estimate + 1e-12).sum()) /
                                                                  (len(draws) + 1)))
    return report


def rank_point(x: list, y: list, orientation: tuple) -> float:
    i, j = np.triu_indices(len(x[0]), 1)
    return float(pairs_average(x, y, orientation,
                               lambda a, b: np.dot(a[i, j], b[i, j])))


def creator_intervals(left: np.ndarray, right: np.ndarray, folds: list, component: np.ndarray,
                      draws: int, rng: np.random.Generator) -> dict:
    estimates = []
    for _ in range(draws):
        weights = rng.exponential(size=int(component.max()) + 1)[component]
        lx = [ranks(a) for a in matrices(left, folds, weights)]
        ry = [ranks(a) for a in matrices(right, folds, weights)]
        estimates.append([rank_point(lx, ry, ORIENTATIONS[key]) for key in ORIENTATIONS])
    result = np.asarray(estimates)
    all_metrics = dict(zip(ORIENTATIONS, result.T, strict=True))
    all_metrics["paired_minus_cross_source"] = result[:, 0] - result[:, 1]
    return {key: np.quantile(values, [.025, .975]).tolist() for key, values in all_metrics.items()}


def compare(pair: list[str], arrays: dict, rows: list[dict], plan: dict) -> dict:
    lkey, rkey, dyad, interpretation = pair
    tag = f"{lkey}__{rkey}"
    result = dict(pair=tag, left=lkey, right=rkey, dyad=dyad, interpretation=interpretation)
    if lkey not in arrays or rkey not in arrays:
        return dict(result, status="representation_unavailable")
    left, right = arrays[lkey], arrays[rkey]
    valid = [np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1) > 0) for v in (left, right)]
    concepts, groups, folds = memberships(rows, valid, plan["minimum_sources_per_concept_fold"])
    result.update(concepts=concepts, groups=dict(Counter(groups)),
                  source_count=sum(len(selection) for fold in folds for selection in fold))
    if len(concepts) < plan["minimum_concepts"]:
        return dict(result, status="insufficient_complete_concepts")
    selected = np.concatenate([indices for side in folds for indices in side])
    norm_left, norm_right = np.zeros_like(left, dtype=float), np.zeros_like(right, dtype=float)
    norm_left[selected], norm_right[selected] = normalize(left[selected]), normalize(right[selected])
    left_rdm, right_rdm = [matrices(view, folds) for view in (norm_left, norm_right)]
    lx, ry = [[ranks(item) for item in matrices_] for matrices_ in (left_rdm, right_rdm)]
    if any(not np.any(r) for r in lx + ry):
        return dict(result, status="degenerate_geometry")
    _, component = np.unique([row["creator_component"] for row in rows], return_inverse=True)
    categories = np.asarray(groups)
    seed = plan["seed"]
    generator = rng_for(seed, tag, "perm")
    group_size = Counter(groups)
    support = math.prod(math.factorial(num) for num in group_size.values())
    perms = {null: np.vstack((np.arange(len(concepts)),
                             permute_indices(len(concepts), categories, plan["permutations"], generator, null != "global")))
             for null in ("global", "blocked")}
    exact_blocked = support <= plan.get("exact_taxonomy_limit", 0)
    if exact_blocked:
        perms["blocked"] = exact_taxonomy_permutations(categories)
    random = rng_for(seed, tag, "dimension-matched-features")
    rx, rr = [[ranks(d) for d in matrices(normalize(random.normal(size=v.shape)), folds)]
              for v in (left, right)]
    result.update(
        status="measured",
        fold_source_counts=[sum(map(len, fold)) for fold in folds],
        creator_components=int(len(set(component[selected]))),
        source_ids_by_fold=[[[rows[i]["source_id"] for i in cohort] for cohort in fold] for fold in folds],
        reliability={"left": rank_point(lx, lx, ((0, 1), (1, 0))),
                     "right": rank_point(ry, ry, ((0, 1), (1, 0)))},
        distinct_taxonomy_permutations=support,
        intervals=creator_intervals(norm_left, norm_right, folds, component, plan["creator_bootstraps"],
                                    rng_for(seed, tag, "creator")),
        outcomes={},
    )
    if plan.get("exact_taxonomy_limit"):
        result.update(taxonomy_null_sampling="exact" if exact_blocked else "monte_carlo",
                      taxonomy_null_minimum_exact_p=1 / support)
    identity = perms["global"][:1]
    for name, orientation in ORIENTATIONS.items():
        full = score_rank(lx, ry, orientation, perms["global"])
        blocked = score_rank(lx, ry, orientation, perms["blocked"])
        within, within_groups = score_within(left_rdm, right_rdm, orientation, perms["blocked"], categories)
        cka_global = cka_scores(left_rdm, right_rdm, orientation, perms["global"])
        left_out = {}
        for group in sorted(group_size):
            keep = np.flatnonzero(categories != group)
            if len(keep) >= 4:
                xa, ya = [[ranks(d[np.ix_(keep, keep)]) for d in view] for view in (left_rdm, right_rdm)]
                left_out[group] = rank_point(xa, ya, orientation)
        within_points = {}
        for group in within_groups:
            subset = np.flatnonzero(categories == group)
            xa, ya = [[ranks(d[np.ix_(subset, subset)]) for d in view] for view in (left_rdm, right_rdm)]
            within_points[group] = rank_point(xa, ya, orientation)
        random_value = float(score_rank(rx, rr, orientation, identity)[0])
        outcomes = dict(global_rsa=empirical(float(full[0]), full[1:]),
                        blocked_rsa=empirical(float(blocked[0]), blocked[1:]),
                        within_rsa=empirical(float(within[0]), within[1:]) if within is not None else None,
                        within_groups=within_groups, within_group_estimates=within_points,
                        leave_one_group_out=left_out,
                        dimension_matched_random_rsa=random_value, cka=empirical(float(cka_global[0]), cka_global[1:]),
                        triplet_agreement=float(pairs_average(left_rdm, right_rdm, orientation, triplet_agreement)),
                        metric=(heldout_scale(left_rdm, right_rdm, orientation, concepts, seed, perms["global"][1:])
                                if plan.get("fit_metric_within_panel", True) else None),
                        neighborhood={})
        for k in plan["neighbors"]:
            if k >= len(concepts):
                continue
            nx, ny = [[neighbors(d, k) for d in view] for view in (left_rdm, right_rdm)]
            def overlap(x, y):
                perm = perms["global"]
                return np.sum(x * y[perm[:, :, None], perm[:, None, :]], axis=(1, 2)) / (len(x) * k)
            nhood = pairs_average(nx, ny, orientation, overlap)
            outcomes["neighborhood"][str(k)] = empirical(float(nhood[0]), nhood[1:])
            if k == 3:
                # Relation is present only if it belongs unambiguously to all four lists.
                shared = np.logical_and.reduce([item == 1 for item in nx + ny])
                outcomes["shared_top3_both_source_folds"] = {
                    concept: [concepts[j] for j in np.flatnonzero(shared[i])]
                    for i, concept in enumerate(concepts) if shared[i].any()
                }
        result["outcomes"][name] = outcomes
    return result


def adjust(pairs: list[dict]) -> None:
    tests = []
    for row in pairs:
        for condition in ORIENTATIONS:
            for null in NULLS:
                outcome = row.get("outcomes", {}).get(condition, {}).get(f"{null}_rsa")
                tests.append((outcome["p"] if outcome else 1.0, outcome))
    running_max = 0.0
    for idx, (p, outcome) in enumerate(sorted(tests, key=lambda item: item[0])):
        running_max = max(running_max, min(1.0, p * (len(tests) - idx)))
        if outcome is not None:
            outcome["p_holm"] = running_max


def analyze(root: Path, protocol: Path = PROTOCOL) -> dict:
    plan = json.loads(protocol.read_text(encoding="utf8"))
    roster, feature_path = root / "roster.json", root / "features.npz"
    rows = json.loads(roster.read_text(encoding="utf8"))["sources"]
    with np.load(feature_path, allow_pickle=False) as archive:
        if archive["source_ids"].tolist() != [row["source_id"] for row in rows]:
            raise ValueError("feature source order does not match frozen roster")
        arrays = {key: archive[key] for key in archive.files if key != "source_ids"}
    pairs = []
    for spec in plan["pairs"]:
        item = compare(spec, arrays, rows, plan)
        pairs.append(item)
        print(item["pair"], item["status"], len(item.get("concepts", [])), flush=True)
    adjust(pairs)
    report = dict(protocol_sha256=sha(protocol.read_bytes()), roster_sha256=sha(roster.read_bytes()),
                  features_sha256=sha(feature_path.read_bytes()), pair_count=len(pairs),
                  holm_family_size=len(pairs) * len(ORIENTATIONS) * len(NULLS), pairs=pairs)
    save_json(root / "geometry.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    analyze(args.root, args.protocol)


if __name__ == "__main__":
    main()
