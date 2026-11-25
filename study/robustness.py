"""Post-pilot sensitivity to metadata quality, speech and literal vocabulary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .geometry import ORIENTATIONS, PROTOCOL, empirical, matrices, memberships, normalize, permute_indices, ranks, rng_for, score_rank

STOP = set("a an and or but the for in of to at It with from two is are was appears this that then can be on by has there around as its their some sounds someone while".lower().split())
SELECTED = [
    ("text_bge", "image_titan"),
    ("text_qwen", "image_voyage"),
    ("text_bge", "audio_caption_bge"),
    ("text_bge", "video_caption_bge"),
    ("image_titan", "video_caption_bge"),
    ("audio_caption_bge", "video_caption_bge"),
    ("image_caption_bge", "video_caption_bge"),
    ("audio_signal", "video_motion"),
]
SUBGROUPS = {
    "acoustic_guitar": "bowed_or_plucked_string", "violin": "bowed_or_plucked_string", "cello": "bowed_or_plucked_string",
    "piano": "keyboard", "accordion": "keyboard", "trumpet": "wind",
    "dog_bark": "mammal", "cow_low": "mammal", "horse_neigh": "mammal", "donkey_bray": "mammal", "lion_roar": "mammal",
    "rooster_crow": "bird", "steam_train": "rail", "diesel_train": "rail", "electric_train": "rail",
    "propeller_aircraft": "fixed_wing", "jet_aircraft": "fixed_wing", "helicopter": "rotary_wing",
    "church_bell": "human_made", "fireworks": "human_made", "chainsaw": "human_made",
    "waterfall": "water", "ocean_waves": "water", "pouring_water": "water",
}


def evaluate(x: np.ndarray, y: np.ndarray, roster: list[dict], inclusion: np.ndarray, seed: int,
             label: str, use_subgroups: bool = False) -> dict:
    masks = [np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1) > 0) for v in (x, y)]
    concepts, groups, folds = memberships(roster, masks + [inclusion], minimum=2)
    report = dict(concepts=concepts, source_count=sum(len(partition) for side in folds for partition in side))
    if len(concepts) < 8:
        return dict(report, status="insufficient")
    selected = np.concatenate([indices for side in folds for indices in side])
    axes = []
    for a in (x, y):
        normalized = np.zeros_like(a, dtype=float)
        normalized[selected] = normalize(a[selected])
        axes.append([ranks(d) for d in matrices(normalized, folds)])
    coarse = ranks((np.asarray(groups)[:, None] != np.asarray(groups)[None, :]).astype(float))
    pair_counts = np.triu_indices(len(concepts), 1)
    taxonomy = [float(np.mean([np.dot(coarse[pair_counts], fold[pair_counts]) for fold in axis]))
                for axis in axes]
    if use_subgroups:
        groups = [f"{group}:{SUBGROUPS[concept]}" for group, concept in zip(groups, concepts, strict=True)]
    global_draws = permute_indices(len(concepts), np.asarray(groups), 4999, rng_for(seed, label, "global"), False)
    blocked_draws = permute_indices(len(concepts), np.asarray(groups), 4999, rng_for(seed, label, "blocked"), True)
    scores = {}
    for key, draws in [("global", global_draws), ("blocked", blocked_draws)]:
        ordered = np.vstack((np.arange(len(concepts)), draws))
        null = score_rank(*axes, ORIENTATIONS["cross_source"], ordered)
        scores[key] = empirical(float(null[0]), null[1:])
    return dict(report, status="measured", cross_source=scores, broad_taxonomy_rsa=taxonomy)


def bag_of_words(values: list[list[str]]) -> list[np.ndarray]:
    tokens = [[set(re.findall("[a-z]{2,}", caption.casefold())) - STOP for caption in channel] for channel in values]
    words = sorted(set().union(*(terms for channel in tokens for terms in channel)))
    lookup = {word: index for index, word in enumerate(words)}
    output = []
    for channel in tokens:
        data = np.zeros((len(channel), len(words)), dtype=np.uint8)
        for i, terms in enumerate(channel):
            data[i, [lookup[word] for word in terms]] = 1
        output.append(data)
    return output


def run(root: Path) -> dict:
    data_path = root / "features.npz"
    roster = json.loads((root / "roster.json").read_text(encoding="utf8"))["sources"]
    captions = json.loads((root / "captions.json").read_text(encoding="utf8"))
    seed = json.loads(PROTOCOL.read_text(encoding="utf8"))["seed"]
    inclusion = dict(
        all=np.ones(len(roster), dtype=bool),
        strong_metadata=np.asarray([row["metadata_assessment"] == "strong" for row in roster]),
        description_text=np.asarray([row["source_text_origin"] == "commons_description" for row in roster]),
        caption_observer_no_speech=np.asarray([
            captions["audio"].get(row["source_id"], {}).get("speech") is False for row in roster]),
    )
    report = []
    geometry = json.loads((root / "geometry.json").read_text(encoding="utf8"))
    passed_blocked = [pair for pair in geometry["pairs"] if pair["status"] == "measured"
                      and pair["outcomes"]["cross_source"]["blocked_rsa"]["p_holm"] < .05]
    with np.load(data_path, allow_pickle=False) as arrays:
        if arrays["source_ids"].tolist() != [row["source_id"] for row in roster]:
            raise ValueError("sensitivity source order differs from feature input")
        for lkey, rkey in SELECTED:
            if lkey not in arrays.files or rkey not in arrays.files:
                continue
            for condition, mask in inclusion.items():
                tag = f"{lkey}:{rkey}:{condition}"
                report.append(dict(left=lkey, right=rkey, restriction=condition,
                                   **evaluate(arrays[lkey], arrays[rkey], roster, mask, seed, tag)))
        detail = []
        for pair in passed_blocked:
            lkey, rkey = pair["left"], pair["right"]
            detail.append(dict(pair=pair["pair"],
                               **evaluate(arrays[lkey], arrays[rkey], roster, inclusion["all"], seed,
                                          f"nested:{pair['pair']}", use_subgroups=True)))
    names = ["text", "audio", "image", "video"]
    texts = [[row["source_text"] if name == "text" else
              captions[name].get(row["source_id"], {}).get("description", "") for row in roster] for name in names]
    lexical_arrays = dict(zip(names, bag_of_words(texts), strict=True))
    lexical = []
    for lkey, rkey in (("text", "audio"), ("text", "video"), ("audio", "video"), ("image", "video")):
        lexical.append(dict(left=lkey, right=rkey,
                            **evaluate(lexical_arrays[lkey], lexical_arrays[rkey], roster, inclusion["all"],
                                       seed, f"lexical:{lkey}:{rkey}")))
    result = dict(features_sha256=sha(data_path.read_bytes()), scope="same-cohort sensitivity, unadjusted descriptive p-values",
                  filter_counts={key: int(mask.sum()) for key, mask in inclusion.items()},
                  filtered=report, lexical=lexical,
                  adaptive_taxonomy=dict(
                      rule="If cross-source blocked RSA survives Holm, test the same pairs with a nested taxonomy. Otherwise use the broad-taxonomy RDM correlations already recorded for the fixed sensitivity comparisons.",
                      selected="nested_blocking" if detail else "broad_taxonomy_reference",
                      nested_results=detail))
    save_json(root / "robustness.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    options = parser.parse_args()
    result = run(options.root)
    print(json.dumps(dict(restrictions=len(result["filtered"]), lexical=len(result["lexical"]))))
