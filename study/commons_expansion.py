"""Estimate metadata coverage for expanding the Commons discovery concepts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from .acquire import CONCEPTS, inventory_rows, prepare, save_json, sha
from .roster import near_duplicates, screen, source_components
from .router import RequestError

EXTENSION = Path(__file__).with_name("commons_extension.json")


def inventory(root: Path, base: Path) -> dict:
    original = json.loads(CONCEPTS.read_text(encoding="utf8"))
    extension = json.loads(EXTENSION.read_text(encoding="utf8"))
    concepts = [(group, concept, phrase, is_new) for plan, is_new in ((original, False), (extension, True))
                for group, queries in plan.items() for concept, phrase in queries.items()]
    indexed = {}

    def load(group: str, concept: str, phrase: str, is_new: bool) -> tuple[str, dict]:
        target = (root if is_new else base) / "inventory" / f"{concept}.json"
        try:
            if target.is_file():
                sources = json.loads(target.read_text(encoding="utf8"))
            else:
                sources = inventory_rows(phrase)
                save_json(root / "inventory" / f"{concept}.json", sources)
        except (OSError, RuntimeError, ValueError) as exc:
            return concept, dict(group=group, query=phrase, added=is_new, error=type(exc).__name__)
        return concept, dict(group=group, query=phrase, added=is_new, page_ids=len({x["pageid"] for x in sources}),
                             authors=len({x["author_key"] for x in sources}), uploaders=len({x["uploader_key"] for x in sources}),
                             metadata_sha256=sha(json.dumps(sources, ensure_ascii=False).encode()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for key, result in pool.map(lambda row: load(*row), concepts):
            indexed[key] = result
    thresholds = {}
    for count in (4, 8, 30):
        thresholds[str(count)] = dict(Counter(value["group"] for value in indexed.values()
                                              if min(value.get(key, 0) for key in ("page_ids", "authors", "uploaders")) >= count))
    report = dict(concepts=indexed, metadata_upper_ceiling_by_minimum_count=thresholds,
                  new_queries=sum(row["added"] for row in indexed.values()),
                  unavailable_queries=sum("error" in row for row in indexed.values()),
                  limitations="Metadata is an upper ceiling. Sources can be mislabeled, silent, duplicated, inaccessible, or share author components.",
                  design_sha256=sha(EXTENSION.read_bytes()))
    save_json(root / "commons-inventory-extension.json", report)
    return report


def supplement_animals(root: Path, base: Path) -> dict:
    data = json.loads((root / "screened.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    former = json.loads((base / "roster.json").read_text(encoding="utf8"))["sources"]
    collected = {row["pageid"]: row for row in data["sources"]}
    allocated = {row["pageid"]: row["concept"] for row in manifest}
    previous = [row for row in former if row["group"] == "animal_vocalizations" and row["pageid"] not in collected
                and allocated.get(row["pageid"], row["concept"]) == row["concept"]]
    for row in previous:
        origin = base / "media" / row["source_id"]
        destination = root / "media" / row["source_id"]
        for name, digest in row["media_sha256"].items():
            if sha((origin / name).read_bytes()) != digest:
                raise ValueError("prior accepted animal media failed the recorded digest")
        if not (destination / "media.json").is_file():
            shutil.copytree(origin, destination, dirs_exist_ok=True)
        collected[row["pageid"]] = dict(row, screening_origin="prior discovery metadata decision")
    clusters = Counter()
    for dependency in source_components(list(collected.values()), data["grouped_perceptual_matches"]):
        clusters.update({row["concept"] for row in dependency})
    concepts = {}
    reserved = set(allocated) | set(collected)
    for folder in (base, root):
        design = CONCEPTS if folder == base else EXTENSION
        for concept in json.loads(design.read_text(encoding="utf8"))["animal_vocalizations"]:
            file = folder / "inventory" / f"{concept}.json"
            if file.is_file() and clusters[concept] < 5:
                rows = [row for row in json.loads(file.read_text(encoding="utf8")) if row["pageid"] not in reserved][:48]
                if rows:
                    concepts[concept] = rows
                    reserved.update(row["pageid"] for row in rows)

    def qualify(concept: str, candidates: list[dict]) -> tuple[str, list[dict], list[dict]]:
        added, receipts = [], []
        # Metadata is screened before fetching uncaptured media. The definitions
        # and acceptance rules are identical to the main screening pass.
        for offset in range(0, len(candidates), 16):
            batch = candidates[offset:offset + 16]
            try:
                review = screen(root, concept, batch)
                receipts.append(review)
            except (RequestError, KeyError, TypeError, ValueError) as exc:
                receipts.append(dict(failure=type(exc).__name__, batch=offset))
                continue
            for row in batch:
                status = review["labels"][str(row["pageid"])]
                if status not in {"strong", "probable"}:
                    continue
                try:
                    media = prepare(row, root)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    receipts.append(dict(pageid=row["pageid"], acquisition_failure=type(exc).__name__))
                    continue
                added.append(dict(row, **media, concept=concept, group="animal_vocalizations", metadata_assessment=status,
                                  screening_origin="prospective metadata replenishment"))
                if len(added) + clusters[concept] >= 6:
                    return concept, added, receipts
        return concept, added, receipts

    supplementation = {}
    recordings = {row["source_sha256"] for row in collected.values()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for concept, added, receipts in pool.map(lambda item: qualify(*item), concepts.items()):
            count = 0
            for row in added:
                if row["source_sha256"] not in recordings:
                    recordings.add(row["source_sha256"])
                    collected[row["pageid"]] = row
                    count += 1
            supplementation[concept] = dict(added=count, metadata_and_transfer=receipts)
    rows = list(collected.values())
    result = dict(data, sources=rows, prior_metadata_records_retained=len(previous), supplementation=supplementation,
                  grouped_perceptual_matches=near_duplicates(root, rows),
                  screened_concept_counts=dict(Counter(row["concept"] for row in rows)))
    save_json(root / "screened-augmented.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--supplement-animals", action="store_true")
    args = parser.parse_args()
    if args.supplement_animals:
        result = supplement_animals(args.root, args.base)
        print(json.dumps({key: value["added"] for key, value in result["supplementation"].items()}))
        return
    result = inventory(args.root, args.base)
    print(json.dumps(result["metadata_upper_ceiling_by_minimum_count"]))


if __name__ == "__main__":
    main()
