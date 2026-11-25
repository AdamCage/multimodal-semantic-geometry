"""Verify replication inputs and audit the resolution of the proposed native tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .epic_replication import csv_rows, stamp
from .native_cohort import exposed_recordings, freeze_layout, validate_assets
from .geometry import NULLS, ORIENTATIONS
from .native_routes import MODALITIES, PROTOCOL, read_json


def merge(root: Path, paths: tuple[Path, Path]) -> dict:
    combined = dict(version=1, datasets={}, sources=[])
    for path in paths:
        data = read_json(path)
        if data["version"] != combined["version"] or combined["datasets"].keys() & data["datasets"].keys():
            raise ValueError("inconsistent manifest version or duplicate dataset")
        combined["datasets"].update(data["datasets"])
        prefix = path.resolve().parent.relative_to(root.resolve())
        for source in data["sources"]:
            assets = {channel: dict(asset, path=(prefix / asset["path"]).as_posix())
                      for channel, asset in source["assets"].items()}
            combined["sources"].append(dict(source, assets=assets))
    return combined


def unconditioned_narration(root: Path, rows: list[dict]) -> dict:
    videos = defaultdict(list)
    needed = {row["selection"]["source"] for row in rows}
    for partition in ("train", "validation"):
        for value in csv_rows(root / f"epic_visual_{partition}.csv"):
            if value["video_id"] in needed:
                videos[value["video_id"]].append((stamp(value["start_timestamp"]), stamp(value["stop_timestamp"]),
                                                 value["narration_id"], " ".join(value["narration"].split())))
    different = Counter()
    for row in rows:
        choice = row["selection"]
        a, b = choice["audio_start"], choice["audio_end"]
        options = [(min(b, end) - max(a, start), narration_id, text)
                   for start, end, narration_id, text in videos[choice["source"]] if text]
        best = max(options)
        different["tested"] += 1
        different["different_narration_id"] += best[1] != choice["visual_action"]["narration_id"]
        different["different_narration_text"] += best[2] != choice["visual_action"]["narration"]
    return dict(different)


def midpoint_dbfs(base: Path, row: dict) -> float:
    offset = row["selection"]["image_time"]
    with wave.open(str(base / row["assets"]["audio"]["path"]), "rb") as audio:
        rate = audio.getframerate()
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("expected mono, 16-bit replication audio")
        audio.setpos(max(0, round((offset - 0.05) * rate)))
        raw = np.frombuffer(audio.readframes(round(0.10 * rate)), dtype="<i2").astype(float)
    if len(raw) == 0:
        raise ValueError("missing audio near the still frame")
    return float(20 * np.log10(max(10 ** -7, np.sqrt(np.mean(np.square(raw / 32768))))))


def resolution(rows: list[dict], dataset_count: int) -> dict:
    concepts = defaultdict(dict)
    for row in rows:
        concepts[row["dataset"], row["concept_split"]][row["concept"]] = row["group"]
    protocol = read_json(PROTOCOL)
    families = len(protocol["family_split"])
    split_count = len(protocol["concept_split"])
    comparisons = families * math.comb(len(MODALITIES), 2) + len(MODALITIES)
    tests = dataset_count * split_count * comparisons * len(ORIENTATIONS) * len(NULLS)
    first_threshold = 0.05 / tests
    panels = {}
    for (dataset, split), labels in sorted(concepts.items()):
        group_sizes = dict(Counter(labels.values()))
        permutations = math.prod(math.factorial(size) for size in group_sizes.values())
        panels[f"{dataset}/{split}"] = dict(
            taxonomy_group_sizes=group_sizes, exact_taxonomy_label_permutations=permutations,
            minimum_exact_p=1 / permutations,
            reaches_first_holm_threshold=(1 / permutations <= first_threshold),
            groups_with_at_least_three_concepts=sum(size >= 3 for size in group_sizes.values()),
        )
    return dict(planned_family_tests=tests, first_holm_threshold=first_threshold, panels=panels,
                interpretation="The support of the exact randomization, including identity, sets a resolution ceiling. "
                               "Additional Monte Carlo draws do not create independent assignments.")


def run(root: Path, discovery: Path, replication: Path,
        output: Path | None = None, report_output: Path | None = None) -> dict:
    destination = output or root / "native-combined-candidate.json"
    receipt = report_output or replication.parent / "replication-check.json"
    if destination.parent.resolve() != root.resolve():
        raise ValueError("the combined manifest must be at the root of all candidate paths")
    if destination.resolve() == receipt.resolve():
        raise ValueError("the manifest and verification must have separate destinations")
    combined = merge(root, (discovery, replication))
    datasets, rows, panels = freeze_layout(combined, read_json(PROTOCOL), exposed_recordings())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda row: validate_assets(root, row), rows))
    subset = [row for row in rows if row["dataset"] == "epic-sounds"]
    dbfs = [midpoint_dbfs(root, row) for row in subset]
    narrations = [row["selection"]["visual_action"]["narration"] for row in subset]
    labels = [row["selection"]["label"] for row in subset]
    exact_matches = sum(bool(re.search(rf"\b{re.escape(label.casefold())}\b", text.casefold()))
                        for label, text in zip(labels, narrations, strict=True))
    available = sum(row["dataset"] == "commons-discovery" and row["concept_split"] == "development"
                    and row["fold"] == "A" for row in rows)
    save_json(destination, combined)
    report = dict(
        combined_manifest_sha256=sha(destination.read_bytes()), datasets=sorted(datasets), design_panels=panels,
        raw_assets_verified=sum(len(row["assets"]) for row in rows),
        unique_epic_participants=len({row["selection"]["participant"] for row in subset}),
        unique_epic_original_recordings=len({row["selection"]["source"] for row in subset}),
        unique_epic_narrations=len(set(narrations)),
        exact_audio_class_name_in_human_narration=exact_matches,
        audio_100ms_near_frame_dbfs=dict(minimum=min(dbfs), median=float(np.median(dbfs)), below_minus_60=sum(x < -60 for x in dbfs)),
        narration_selection_sensitivity=unconditioned_narration(replication.parent, subset),
        paired_training_capacity_discovery_development_A=available,
        taxonomy_test_resolution=resolution(rows, len(datasets)),
        status="local_inputs_verified_awaiting_native_routes_and_scientific_review",
        limitations=["Temporal and verb compatibility do not independently certify visibility of the named event or material in a still frame.",
                     "EPIC-SOUNDS material labels were checked visually by the original dataset authors.",
                     "Both EPIC annotations cover the same source video, intentionally for the paired condition.",
                     "The normalized window digest is an observed local digest. The complete original was not downloaded or checked against its published MD5."],
    )
    save_json(receipt, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research"))
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.discovery, args.replication, args.manifest_output, args.report_output)
    print(json.dumps({k: result[k] for k in ("raw_assets_verified", "design_panels", "status")}))


if __name__ == "__main__":
    main()
