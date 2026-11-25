"""Prepare the existing curated Commons recordings as a native discovery candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .acquire import save_json, sha
from .metadata_control import strip
from .native_cohort import validate_assets
from .native_routes import read_json

MEDIA = {"image": "image.jpg", "audio": "audio.wav", "video": "video.mp4"}


def concept_partition(rows: list[dict]) -> dict[str, str]:
    group = {row["concept"]: row["group"] for row in rows}
    components = {key: {key} for key in group}
    links = {}
    for row in rows:
        concept = row["concept"]
        previous = links.setdefault(row["creator_component"], concept)
        merged = components[concept] | components[previous]
        for name in merged:
            components[name] = merged
    blocks = sorted({tuple(sorted(value)) for value in components.values()})
    if len(group) > 20:
        raise ValueError("this component enumeration is scoped to the curated Commons pilot")
    complete = set(group)
    totals = Counter(group.values())
    preferred = None
    for keep in itertools.product((False, True), repeat=len(blocks)):
        training = {concept for include, block in zip(keep, blocks, strict=True) if include for concept in block}
        testing = complete - training
        if min(len(training), len(testing)) < 8:
            continue
        a = Counter(group[c] for c in training)
        # Prioritize coverage of the four broad groups, then balance their counts.
        score = (sum(0 < a[k] < n for k, n in totals.items()),
                 -sum(abs(2 * a[k] - n) for k, n in totals.items()),
                 -abs(len(training) - len(testing)), tuple(sorted(training)))
        if preferred is None or score > preferred[0]:
            preferred = score, training
    if preferred is None:
        raise ValueError("creator dependencies prevent an eight-concept split on both sides")
    return {concept: "development" if concept in preferred[1] else "heldout" for concept in group}


def materialize(source: Path, output: Path, original: dict, split: str) -> dict:
    unit = original["source_id"]
    if re.fullmatch(r"commons-[0-9]+", unit) is None:
        raise ValueError("invalid Commons source ID")
    folder = output / "assets" / unit
    folder.mkdir(parents=True, exist_ok=True)
    assets = {}
    for channel, filename in MEDIA.items():
        src, dst = source / "media" / unit / filename, folder / filename
        if sha(src.read_bytes()) != original["media_sha256"][filename]:
            raise ValueError(f"original media digest mismatch for {unit}/{channel}")
        if channel == "image":
            shutil.copyfile(src, dst)
        else:
            strip(src, dst, channel)
        assets[channel] = {"path": dst.relative_to(output).as_posix(), "sha256": sha(dst.read_bytes())}
    text = folder / "description.txt"
    text.write_text(original["source_text"].strip() + "\n", encoding="utf8")
    assets["text"] = {"path": text.relative_to(output).as_posix(), "sha256": sha(text.read_bytes())}
    row = dict(source_id=unit, dataset="commons-discovery", concept=original["concept"],
               group=original["group"], concept_split=split, fold=original["fold"],
               recording_sha256=original["source_sha256"], license=original["license"],
               source_url=original["source_page"], attribution=original["author"],
               text_authorship="Existing Commons contributor description, already used in contextual pilot.",
               dependency_ids=[original["creator_component"]], assets=assets)
    validate_assets(output, row)
    return row


def prepare(source: Path, output: Path) -> dict:
    original = read_json(source / "roster.json")["sources"]
    split = concept_partition(original)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(materialize, source, output, row, split[row["concept"]]) for row in original]
        rows = [job.result() for job in jobs]
    loci = defaultdict(set)
    for row in rows:
        locus = (row["concept_split"], row["fold"])
        for key in row["dependency_ids"]:
            loci[("creator", key)].add(locus)
        for channel in MEDIA:
            loci[(channel, row["assets"][channel]["sha256"])].add(locus)
    if any(len(partitions) > 1 for partitions in loci.values()):
        raise ValueError("a creator or a media digest crosses the proposed discovery splits")
    manifest = dict(version=1, datasets={"commons-discovery": dict(role="discovery",
                    origin="https://commons.wikimedia.org/", license="per-recording licenses",
                    training_overlap_assessment="Unknown. Existing public recordings and contextual pilot exposures are known.")},
                    sources=rows)
    save_json(output / "discovery-candidate.json", manifest)
    taxonomy = {row["concept"]: row["group"] for row in rows}
    counts = {role: dict(
        concepts=sorted(c for c, s in split.items() if s == role),
        source_counts=dict(Counter(r["fold"] for r in rows if r["concept_split"] == role)),
        concept_counts_by_taxonomy=dict(Counter(taxonomy[c] for c, s in split.items() if s == role)),
    ) for role in ("development", "heldout")}
    result = dict(sources=len(rows), concept_partitions=counts,
                  readiness="discovery_candidate_only; requires independently assembled replication cohort and valid routes",
                  manifest_sha256=sha((output / "discovery-candidate.json").read_bytes()))
    save_json(output / "discovery-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))


if __name__ == "__main__":
    main()
