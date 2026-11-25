"""Run a dataset-held-out, source-level audio–text check on licensed Clotho audio."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import urllib.error
from pathlib import Path
from urllib.parse import urlencode, urlparse

import numpy as np

from .acquire import command, fetch, save_json, sha
from .features import TEXT_MODELS, audio_features, caption, text_vectors
from .geometry import empirical, normalize, permute_indices, ranks, rdm, rng_for, score_rank
from .router import RequestError

DATASET = "gijs/clotho"
REVISION = "93be4f14a2d662389bb30c324781d2b489b583fd"
ORIGINAL = "https://zenodo.org/records/4783391/files/"
SEED = 20261125
TARGET = 64


def table(name: str) -> tuple[list[dict], str]:
    raw = fetch(ORIGINAL + name, 1_000_000)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    return list(csv.DictReader(io.StringIO(text))), sha(raw)


def roster_for_clotho(root: Path) -> dict:
    cached = root / "clotho-roster.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf8"))
    metadata, meta_digest = table("clotho_metadata_evaluation.csv")
    captions, caption_digest = table("clotho_captions_evaluation.csv")
    by_name = {row["file_name"]: row for row in metadata}
    options = []
    for i, row in enumerate(captions):
        meta = by_name[row["file_name"]]
        license_link = urlparse(meta["license"].strip())
        rights = license_link.path.strip("/").split("/")
        if license_link.hostname != "creativecommons.org" or rights[:2] not in (
            ["publicdomain", "zero"], ["licenses", "by"], ["licenses", "by-sa"],
        ):
            continue
        author, sound_id = meta["manufacturer"].strip(), meta["sound_id"].strip()
        if not author or not sound_id:
            continue
        options.append(dict(index=i, source_id=f"clotho-{i:04}", file_name=row["file_name"],
                            source_url=meta["sound_link"], author=author, sound_id=sound_id,
                            license=meta["license"], captions=[row[f"caption_{j}"] for j in range(1, 6)]))
    options.sort(key=lambda row: sha(f"{SEED}|clotho|{row['sound_id']}".encode()))
    seen_author, seen_sound, cohort = set(), set(), []
    for row in options:
        author = row["author"].casefold()
        if author not in seen_author and row["sound_id"] not in seen_sound:
            cohort.append(row)
            seen_author.add(author)
            seen_sound.add(row["sound_id"])
        if len(cohort) >= TARGET:
            break
    result = dict(dataset=DATASET, media_commit=REVISION, source_table_sha256=meta_digest,
                  captions_table_sha256=caption_digest, seed=SEED, rows=cohort)
    save_json(cached, result)
    return result


def download_clip(root: Path, row: dict) -> dict:
    folder = root / "media" / row["source_id"]
    clip = folder / "audio.wav"
    sheet = folder / "clotho-source.json"
    if sheet.exists() and clip.exists():
        saved = json.loads(sheet.read_text(encoding="utf8"))
        if sha(clip.read_bytes()) != saved["input_sha256"]:
            raise ValueError("cached Clotho audio hash drifted")
        return saved
    params = dict(dataset=DATASET, config="default", split="test", offset=row["index"], length=1)
    response = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + urlencode(params), 500_000))
    media = response["rows"][0]["row"]
    mirrored = [media[f"caption_{i}"] for i in range(1, 6)]
    if media["file_name"] != row["file_name"] or mirrored != row["captions"]:
        raise ValueError("mirror row does not match official captions")
    url = media["audio"][0]["src"]
    parsed = urlparse(url)
    if parsed.hostname != "datasets-server.huggingface.co" or f"/{REVISION}/" not in parsed.path:
        raise ValueError("unexpected mirror host or revision")
    payload = fetch(url, 4_000_000)
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "mirror.wav"
    raw.write_bytes(payload)
    command(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(raw),
             "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clip)])
    evidence = dict(source_id=row["source_id"], mirror_path=parsed.path, mirror_sha256=sha(payload),
                    input_sha256=sha(clip.read_bytes()))
    save_json(sheet, evidence)
    return evidence


def gather(root: Path) -> dict:
    cohort = roster_for_clotho(root)
    sources, captions, failures = {}, {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(download_clip, root, row): row for row in cohort["rows"]}
        for future in concurrent.futures.as_completed(tasks):
            row = tasks[future]
            try:
                sources[row["source_id"]] = future.result()
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                failures.append(dict(source_id=row["source_id"], stage="download", reason=type(exc).__name__,
                                     http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None))
            if (len(sources) + len(failures)) % 16 == 0:
                print(f"Clotho sources {len(sources)}/{len(cohort['rows'])}", flush=True)
        tasks = {pool.submit(caption, root, row, "audio"): row for row in cohort["rows"] if row["source_id"] in sources}
        for future in concurrent.futures.as_completed(tasks):
            row = tasks[future]
            try:
                captions[row["source_id"]] = future.result()
            except (KeyError, RequestError, ValueError) as exc:
                failures.append(dict(source_id=row["source_id"], stage="caption", reason=str(exc)))
    save_json(root / "clotho-inputs.json", dict(sources=sources, machine_captions=captions, failures=failures))
    return cohort


def vectors(root: Path, cohort: dict) -> dict:
    inputs = json.loads((root / "clotho-inputs.json").read_text(encoding="utf8"))
    available = sorted(inputs["sources"])
    text = {f"{row['source_id']}::{i}": cap for row in cohort["rows"]
            if row["source_id"] in available for i, cap in enumerate(row["captions"])}
    machine = {source: inputs["machine_captions"][source]["description"] for source in sorted(inputs["machine_captions"])}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        human_task = pool.submit(text_vectors, root, text, "clotho_human_bge", TEXT_MODELS["text_bge"])
        machine_task = pool.submit(text_vectors, root, machine, "clotho_machine_bge", TEXT_MODELS["text_bge"])
        qwen_task = pool.submit(text_vectors, root, machine, "clotho_machine_qwen", TEXT_MODELS["text_qwen"])
        qwen_human_task = pool.submit(text_vectors, root, text, "clotho_human_qwen", TEXT_MODELS["text_qwen"])
        human, human_model, human_errors = human_task.result()
        bge, bge_model, bge_errors = machine_task.result()
        qwen, qwen_model, qwen_errors = qwen_task.result()
        qwen_human, qwen_human_model, qwen_human_errors = qwen_human_task.result()
    signal, human_all, human_early, human_late, human_qwen_all = {}, {}, {}, {}, {}
    for source in available:
        signal[source] = audio_features(root / "media" / source / "audio.wav")
        keys = [f"{source}::{i}" for i in range(5)]
        if all(key in human for key in keys):
            unit = normalize(np.asarray([human[key] for key in keys]))
            human_all[source], human_early[source], human_late[source] = (
                unit.mean(axis=0), unit[:3].mean(axis=0), unit[3:].mean(axis=0))
        if all(key in qwen_human for key in keys):
            human_qwen_all[source] = normalize(np.asarray([qwen_human[key] for key in keys])).mean(axis=0)
    maps = dict(audio_signal=signal, human_bge=human_all, human_bge_early=human_early,
                human_bge_late=human_late, human_qwen=human_qwen_all, machine_bge=bge, machine_qwen=qwen)
    saved = {}
    for name, mapping in maps.items():
        dims = {len(vector) for vector in mapping.values()}
        if len(dims) == 1:
            saved[name] = np.asarray([mapping.get(source, [np.nan] * next(iter(dims))) for source in available],
                                     dtype="float32")
    feature_file = root / "clotho-features.npz"
    np.savez_compressed(feature_file, source_ids=np.asarray(available), **saved)
    return dict(errors=human_errors + bge_errors + qwen_errors + qwen_human_errors,
                observed_models={key: sorted(set(values.values())) for key, values in
                                 [("human_bge", human_model), ("machine_bge", bge_model),
                                  ("machine_qwen", qwen_model), ("human_qwen", qwen_human_model)]})


def evaluate(root: Path) -> dict:
    plans = [("human_bge", "machine_bge"), ("human_bge", "machine_qwen"),
             ("human_bge", "audio_signal"), ("human_bge_early", "human_bge_late"),
             ("human_bge", "human_qwen")]
    data_path = root / "clotho-features.npz"
    results, generator = [], rng_for(SEED, "clotho-permutation")
    with np.load(data_path, allow_pickle=False) as archive:
        for left, right in plans:
            outcome = dict(left=left, right=right, status="insufficient")
            results.append(outcome)
            if left not in archive.files or right not in archive.files:
                continue
            a, b = archive[left], archive[right]
            keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            n = int(keep.sum())
            if n < 24:
                continue
            d1, d2 = rdm(a[keep]), rdm(b[keep])
            draws = permute_indices(n, np.zeros(n), 1999, generator, False)
            permutations = np.vstack((np.arange(n), draws))
            scores = score_rank([ranks(d1)] * 2, [ranks(d2)] * 2, ((0, 0), (1, 1)), permutations)
            sensitivity = []
            for _ in range(399):
                ix = generator.choice(n, size=int(n * .8), replace=False)
                x, y = [ranks(d[np.ix_(ix, ix)]) for d in (d1, d2)]
                sensitivity.append(float(np.sum(x * y) / 2))
            outcome.update(status="measured", sources=n, source_ids=archive["source_ids"][keep].tolist(),
                           **empirical(float(scores[0]), scores[1:]),
                           source_subsample_percentile=np.quantile(sensitivity, [.025, .975]).tolist())
    running = 0.
    for i, row in enumerate(sorted(results[:4], key=lambda value: value.get("p", 1.0))):
        running = max(running, min(1., row.get("p", 1.0) * (4 - i)))
        if row["status"] == "measured":
            row["p_holm"] = running
    report = dict(scope="matched recordings and human captions, dataset-held-out audio–text triangulation at the source level",
                  features_sha256=sha(data_path.read_bytes()), results=results[:4],
                  followup_text_encoder_control=dict(results[-1], inference="post-pilot descriptive text–text control"))
    save_json(root / "clotho-geometry.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/clotho"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        cohort = gather(args.root)
        save_json(args.root / "clotho-feature-meta.json", vectors(args.root, cohort))
    report = evaluate(args.root)
    print(json.dumps([{k: v for k, v in item.items() if k != "source_ids"} for item in report["results"]], indent=2))


if __name__ == "__main__":
    main()
