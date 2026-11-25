"""Prepare a source-blocked EPIC-SOUNDS / EPIC-KITCHENS replication cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .acquire import fetch, save_json, sha
from .native_cohort import validate_assets

BASE = "https://data.bris.ac.uk/datasets/"
REPOSITORIES = {"55": "3h91syskeag572hl6tvuovwv4d", "100": "2g1n6qdydwa9u22shpxqzp0t8m"}
SOUNDS = "epic-kitchens/epic-sounds-annotations/57a922f0d352e9429f1ef8a37eee21758dd3a33c"
VISUAL = "epic-kitchens/epic-kitchens-100-annotations/ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
DOWNLOAD = "epic-kitchens/epic-kitchens-download-scripts/4f11fb2b579833f360c3c7bb917bf1e24a9787b5"
INPUTS = {
    **{f"epic_sounds_{p}.csv": f"{SOUNDS}/EPIC_Sounds_{p}.csv" for p in ("train", "validation")},
    **{f"epic_visual_{p}.csv": f"{VISUAL}/EPIC_100_{p}.csv" for p in ("train", "validation")},
    "epic_md5.csv": f"{DOWNLOAD}/data/md5.csv",
    "epic_licence.txt": f"{VISUAL}/license.txt",
}
INPUT_SHA256 = {
    "epic_sounds_train.csv": "225427ab288cc5c72c1680abb29f5ab8ccd3608c3f4cf409541d2c5cf720dd32",
    "epic_sounds_validation.csv": "fff66485d8478762fd9cde1a829e6d91713a5f2f7f917e3af746cf891dd62e5a",
    "epic_visual_train.csv": "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87",
    "epic_visual_validation.csv": "35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d",
    "epic_md5.csv": "3f9c31899ff5807c8f0fed5d26a8fe62300f3f4c40dc3eea767345212c2e83c7",
    "epic_licence.txt": "acb7384c3f73cc96367ac132bc0bd57efa38c4495c17781032a2cf1444b65ebe",
}
SEED = "epic-independent-cohort-v1"
ORIGIN = "https://epic-kitchens.github.io/epic-sounds/site"
LICENSE = "Bristol Non-Commercial Government Licence (footage) and CC BY-NC 4.0 (visual narrations)"
DATASET = dict(role="replication", origin=ORIGIN, license=LICENSE,
               training_overlap_assessment="Potential overlap with encoder training sets is unknown. "
                                           "No provider weights, pretraining corpus, or exact compound overlap were audited.")

TAXONOMY = {
    "metal impacts": ((0, 12), (10, 17)),
    "nonmetal impacts": ((6, 20), (14, 21)),
    "object manipulation": ((1, 4), (3, 9)),
    "food and liquids": ((2, 5), (8, 19)),
}
EXPANDED_TAXONOMY = {
    "metal impacts": ((0, 12, 11, 23), (10, 17, 25, 28)),
    "nonmetal impacts": ((6, 20, 26, 30, 32), (14, 21, 22, 27, 34)),
    "object manipulation": ((1, 4, 18), (3, 9, 36)),
    "food and liquids": ((2, 5), (8, 19)),
}
IMPACTS = {"adjust", "close", "cut", "move", "open", "place", "put", "remove", "set", "take"}
HANDLE = IMPACTS | {"fold", "pull", "tear", "unwrap"}
VERBS = {
    **{code: IMPACTS for code in (0, 6, 10, 11, 12, 14, 17, 20, 21, 22, 23, 25, 26, 27, 28, 30, 32, 34)},
    1: {"clean", "dry", "scrape", "scrub", "wash", "wipe"},
    2: {"chop", "cut", "dice", "peel", "slice"},
    3: {"close", "open", "uncover"},
    4: HANDLE,
    5: {"fill", "pour", "rinse", "run", "turn-off", "turn-on", "wash"},
    8: {"beat", "mix", "stir", "whisk"},
    9: {"drag", "move", "pull", "push", "slide"},
    18: HANDLE,
    19: {"empty", "fill", "pour"},
    36: HANDLE | {"dry", "grab", "rub", "wipe"},
}


def stamp(text: str) -> float:
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def inputs(root: Path) -> dict:
    result = {}
    for name, path in INPUTS.items():
        url = f"https://raw.githubusercontent.com/{path}"
        target = root / name
        if not target.exists():
            raw = fetch(url, 16_000_000)
            if sha(raw) != INPUT_SHA256[name]:
                raise ValueError(f"upstream annotation hash changed for {name}")
            target.write_bytes(raw)
        digest = sha(target.read_bytes())
        if digest != INPUT_SHA256[name]:
            raise ValueError(f"local annotation hash changed for {name}")
        result[name] = dict(url=url, sha256=digest)
    return result


def annotations(root: Path, classes: dict) -> tuple[list[dict], dict]:
    reference = {}
    for row in csv_rows(root / "epic_md5.csv"):
        path = row["file_remote_path"]
        if row["version"] in REPOSITORIES and "/videos/" in f"/{path}" and path.lower().endswith(".mp4"):
            source = Path(path).stem
            reference[source] = dict(url=f"{BASE}{REPOSITORIES[row['version']]}/{path}",
                                     index_md5=row["md5"], publication=row["version"])

    visual = defaultdict(list)
    for partition in ("train", "validation"):
        for row in csv_rows(root / f"epic_visual_{partition}.csv"):
            visual[row["video_id"]].append(dict(
                narration=" ".join(row["narration"].split()), verb=row["verb"], noun=row["noun"],
                narration_id=row["narration_id"], participant=row["participant_id"], partition=partition,
                start=stamp(row["start_timestamp"]), end=stamp(row["stop_timestamp"]),
            ))

    eligibility, counts = {}, Counter()
    for partition in ("train", "validation"):
        for row in csv_rows(root / f"epic_sounds_{partition}.csv"):
            counts["human_audio_annotations"] += 1
            cls, source = int(row["class_id"]), row["video_id"]
            if cls not in classes or source not in reference:
                continue
            if row["description"].strip().casefold() in {"", "unlabelled", "unlabeled"}:
                continue
            counts["classified_indexed_audio_annotations"] += 1
            a, b = stamp(row["start_timestamp"]), stamp(row["stop_timestamp"])
            if not 0.25 <= b - a <= 6:
                continue
            evidence = []
            for action in visual[source]:
                if action["participant"] != row["participant_id"] or action["verb"] not in VERBS[cls] or not action["narration"]:
                    continue
                left, right = max(a, action["start"]), min(b, action["end"])
                overlap = right - left
                if overlap < max(0.20, 0.35 * (b - a)):
                    continue
                evidence.append((overlap, action["narration_id"], left, right, action))
            if not evidence:
                continue
            overlap, _, left, right, action = max(evidence, key=lambda x: (x[0], x[1]))
            counts["overlapping_compatible_audio_annotations"] += 1
            # The image is sampled from the intersection. A centered three-second
            # clip preserves nearby motion and sound without using model scores.
            midpoint = (left + right) / 2
            start = max(0.0, midpoint - 1.5)
            score = min(overlap, 1.5) / 1.5 + min(overlap / (b - a), 1)
            tie = int(sha(f"{SEED}:{row['annotation_id']}".encode())[:8], 16) / 2**32
            candidate = dict(cls=cls, label=row["class"], participant=row["participant_id"], source=source,
                             annotation_id=row["annotation_id"], audio_partition=partition, audio_description=row["description"],
                             audio_start=a, audio_end=b, visual_action=action, overlap=round(overlap, 6),
                             start=round(start, 6), image_time=round(midpoint - start, 6), score=score + tie * 0.01,
                             **reference[source])
            key = cls, source
            if key not in eligibility or candidate["score"] > eligibility[key]["score"]:
                eligibility[key] = candidate
    candidates = sorted(eligibility.values(), key=lambda c: (c["cls"], c["source"]))
    return candidates, dict(counts, unique_class_source_candidates=len(candidates),
                            initial_participants=len({c["participant"] for c in candidates}),
                            initial_recordings=len({c["source"] for c in candidates}))


def assign(candidates: list[dict], classes: dict, unavailable: set[str], verified: set[str]) -> tuple[list[dict], int]:
    participants = sorted({r["participant"] for r in candidates})
    panels = [(split, fold) for split in ("development", "heldout") for fold in ("A", "B")]
    pool = [c for c in candidates if c["source"] not in unavailable]
    for attempt in range(48):
        ordered = sorted(participants, key=lambda p: sha(f"{SEED}:{attempt}:{p}".encode()))
        mapping = {p: panels[i % len(panels)] for i, p in enumerate(ordered)}
        rows = [dict(c, split=classes[c["cls"]][0], group=classes[c["cls"]][1], fold=mapping[c["participant"]][1])
                for c in pool if classes[c["cls"]][0] == mapping[c["participant"]][0]]
        buckets = defaultdict(list)
        for i, row in enumerate(rows):
            cls, fold, source, participant = (row[k] for k in ("cls", "fold", "source", "participant"))
            for key in (("class", cls, fold), ("recording", source), ("class-participant", cls, participant),
                        ("participant", participant)):
                buckets[key].append(i)
        if any(len({rows[i]["participant"] for i in buckets[("class", cls, fold)]}) < 2
               for cls in classes for fold in ("A", "B")):
            continue
        indices, columns, lower, upper = [], [], [], []
        for i, (key, members) in enumerate(buckets.items()):
            indices.extend([i] * len(members))
            columns.extend(members)
            lower.append(2 if key[0] == "class" else 0)
            upper.append({"class": 2, "participant": 5}.get(key[0], 1))
        matrix = coo_matrix((np.ones(len(indices)), (indices, columns)), shape=(len(buckets), len(rows))).tocsc()
        # After transport failures, keep materialized records first and prefer
        # shorter seeks among replacements. The annotation gate stays fixed.
        weight = [r["score"] + 200 * (r["annotation_id"] in verified)
                  - (max(0, r["start"] - 60) / 1000 if unavailable else 0) for r in rows]
        result = milp(-np.array(weight), integrality=np.ones(len(rows)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 12})
        if result.x is not None:
            pick = (result.x > 0.5).astype(int)
            values = matrix @ pick
            if np.all(values >= lower) and np.all(values <= upper):
                return [r for r, selected in zip(rows, pick) if selected], attempt
    raise ValueError(f"no participant-blocked {len(classes)}-concept assignment satisfies the registered constraints")


def joint_assignment(candidates: list[dict], classes: dict, unavailable: set[str], verified: set[str]) -> list[dict]:
    rows = [dict(value, split=classes[value["cls"]][0], group=classes[value["cls"]][1], fold=fold)
            for value in candidates if value["source"] not in unavailable for fold in ("A", "B")]
    constraints = defaultdict(list)
    indicators = {}
    weights = [row["score"] + 200 * (row["annotation_id"] in verified)
               - (max(0, row["start"] - 60) / 1000 if unavailable else 0) for row in rows]
    for i, row in enumerate(rows):
        cls, fold, source, participant, split = (row[k] for k in ("cls", "fold", "source", "participant", "split"))
        panel = participant, split, fold
        if panel not in indicators:
            indicators[panel] = len(weights)
            weights.append(0)
        for limit in (("class", cls, fold), ("recording", source), ("class-participant", cls, participant), ("panel", *panel)):
            constraints[limit].append((i, 1))
    for panel, i in indicators.items():
        constraints[("panel", *panel)].append((i, -5))
        constraints[("participant", panel[0])].append((i, 1))
    for cls in classes:
        for fold in ("A", "B"):
            constraints[("class", cls, fold)]
    indexes, columns, lower, upper, values = [], [], [], [], []
    for i, (key, terms) in enumerate(constraints.items()):
        indexes.extend([i] * len(terms))
        columns.extend(position for position, _ in terms)
        values.extend(value for _, value in terms)
        lower.append(2 if key[0] == "class" else (-np.inf if key[0] == "panel" else 0))
        upper.append({"class": 2, "panel": 0}.get(key[0], 1))
    matrix = coo_matrix((values, (indexes, columns)), shape=(len(constraints), len(weights))).tocsc()
    result = milp(-np.array(weights), integrality=np.ones(len(weights)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 45, "mip_rel_gap": 0.001})
    if result.x is not None:
        pick = (result.x > 0.5).astype(int)
        totals = matrix @ pick
        if np.all(totals >= lower) and np.all(totals <= upper):
            return [row for row, selected in zip(rows, pick[:len(rows)], strict=True) if selected]
    raise ValueError(f"expanded labels do not have a verified joint participant assignment (solver status {result.status})")


def ffmpeg(args: list[str], timeout: int = 30) -> None:
    call = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=False,
                          capture_output=True, timeout=timeout)
    if call.returncode:
        raise ValueError(call.stderr.decode(errors="replace").strip()[-500:])


def prepared(root: Path, choice: dict, download_timeout: int) -> dict:
    source = f"epic-{choice['annotation_id']}"
    dest = root / "assets" / source
    dest.mkdir(parents=True, exist_ok=True)
    files = {k: dest / f"native{ext}" for k, ext in (("text", ".txt"), ("image", ".jpg"),
                                                   ("audio", ".wav"), ("video", ".mp4"))}
    lock = dest / "prepared.json"
    if lock.exists():
        manifest = json.loads(lock.read_text(encoding="utf8"))
        if any(manifest["selection"][key] != choice[key] for key in ("annotation_id", "start", "image_time", "url")):
            raise ValueError("cached preparation disagrees with current timestamps or source")
        validate_assets(root, manifest)
    else:
        with tempfile.TemporaryDirectory(dir=dest) as work:
            clip = Path(work) / "window.mkv"
            ffmpeg(["-rw_timeout", "11000000", "-tls_verify", "1", "-short_seek_size", "1048576", "-multiple_requests", "1",
                    "-ss", str(choice["start"]), "-i", choice["url"], "-t", "3", "-map", "0:v:0", "-map", "0:a:0",
                    "-map_metadata", "-1", "-vf", "scale=480:-2,fps=12", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-threads", "2", "-preset", "fast", "-c:a", "flac", "-ar", "16000", "-ac", "1", str(clip)],
                   timeout=download_timeout)
            ffmpeg(["-i", str(clip), "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart", str(files["video"])])
            ffmpeg(["-i", str(clip), "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", str(files["audio"])])
            ffmpeg(["-ss", str(choice["image_time"]), "-i", str(clip), "-map", "0:v:0", "-frames:v", "1",
                    "-q:v", "3", "-update", "1", str(files["image"])])
        files["text"].write_text(choice["visual_action"]["narration"] + "\n", encoding="utf8", newline="\n")
        with wave.open(str(files["audio"]), "rb") as audio:
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float)
            if audio.getframerate() != 16000 or not 46400 <= len(samples) <= 48400:
                raise ValueError("truncated or invalid three-second audio")
        rms = float(np.sqrt(np.mean(np.square(samples / 32768))))
        if rms < 0.0003:
            raise ValueError("audio energy below registered threshold")
        assets = {kind: dict(path=file.relative_to(root).as_posix(), sha256=sha(file.read_bytes())) for kind, file in files.items()}
        manifest = dict(source_id=source, dataset="epic-sounds", concept=f"epic-{choice['label']}", group=choice["group"],
                        concept_split=choice["split"], fold=choice["fold"], license=LICENSE, source_url=choice["url"],
                        text_authorship="Original human EPIC-KITCHENS visual-first narration from the overlapping action",
                        recording_sha256=sha((assets["video"]["sha256"] + assets["audio"]["sha256"]).encode()),
                        recording_sha256_scope="Ordered concatenation of normalized video and audio SHA-256, rehashed as ASCII",
                        dependency_ids=[f"epic-participant:{choice['participant']}", f"epic-recording:{choice['source']}",
                                        f"epic-original-index-md5:{choice['index_md5']}"],
                        license_note="Research windows sampled and re-encoded from Bristol footage. No footage redistributed.",
                        assets=assets, selection=choice, rms=rms)
        validate_assets(root, manifest)
        save_json(lock, manifest)
    return dict(manifest, concept_split=choice["split"], fold=choice["fold"], selection=choice)


def run(root: Path, acquire: bool, expanded: bool) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    input_manifest = inputs(root)
    taxonomy = EXPANDED_TAXONOMY if expanded else TAXONOMY
    prior_selection = root / "epic-selection.json"
    if prior_selection.exists():
        previous = json.loads(prior_selection.read_text(encoding="utf8")).get("taxonomy")
        chosen = json.loads(json.dumps(taxonomy))
        if (previous is not None and previous != chosen) or (previous is None and expanded):
            raise ValueError("use a separate output directory for a different EPIC taxonomy")
    classes = {code: (split, group) for group, subsets in taxonomy.items()
               for split, codes in zip(("development", "heldout"), subsets)
               for code in codes}
    candidates, counts = annotations(root, classes)
    failures = {}
    prior = root / "epic-acquisition-failures.json"
    if acquire and prior.exists():
        failures = json.loads(prior.read_text(encoding="utf8"))
    report = {}
    for _ in range(6):
        verified = {lock.parent.name.removeprefix("epic-") for lock in (root / "assets").glob("*/prepared.json")}
        available_cache = verified if failures or expanded else set()
        if expanded:
            selected = joint_assignment(candidates, classes, set(failures), available_cache)
            attempt = None
        else:
            selected, attempt = assign(candidates, classes, set(failures), available_cache)
        report = dict(counts=counts, input_manifest=input_manifest, assignment_attempt=attempt, taxonomy=taxonomy,
                      assignment_method="joint participant and event constraints" if expanded else "seeded participant blocks",
                      panels={f"{split}/{fold}": dict(sources=len(items), recordings=len({x['source'] for x in items}),
                                                       participants=len({x['participant'] for x in items}),
                                                       classes=len({x['cls'] for x in items}))
                              for split in ("development", "heldout") for fold in ("A", "B")
                              if (items := [x for x in selected if x["split"] == split and x["fold"] == fold])},
                      failures=failures, sources=selected, dataset=DATASET,
                      fallback_rule="After observed failures, prioritize verified windows and prefer shorter source seeks.")
        save_json(root / "epic-selection.json", report)
        if not acquire:
            return report
        ready = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            deadline = 32 if failures else 75
            jobs = {pool.submit(prepared, root, row, deadline): row for row in selected}
            for future in concurrent.futures.as_completed(jobs):
                row = jobs[future]
                try:
                    ready.append(future.result())
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    failures[row["source"]] = dict(annotation=row["annotation_id"], type=type(exc).__name__, error=str(exc)[-500:])
        save_json(prior, failures)
        if len(ready) == len(selected):
            save_json(root / "epic-candidate.json", dict(version=1, datasets={"epic-sounds": DATASET},
                                                        sources=sorted(ready, key=lambda row: row["source_id"])))
            return dict(report, prepared=len(ready))
    return dict(report, prepared=len(ready), status="incomplete_window_acquisition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--expanded", action="store_true", help="Use the separate 28-concept design in a new root")
    args = parser.parse_args()
    report = run(args.root, args.acquire, args.expanded)
    print(json.dumps({key: value for key, value in report.items() if key in ("counts", "panels", "prepared", "status")}))
    if report.get("status"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
