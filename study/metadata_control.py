"""Repeat affected caption inputs after removing auxiliary container metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .acquire import command, save_json, sha
from .features import CAPTION_MODEL, CAPTION_PROMPTS, CAPTION_SUFFIX, TEXT_MODELS, text_vectors
from .geometry import ORIENTATIONS, matrices, normalize, rank_point, ranks
from .router import RequestError, chat, media_part, parse_json

FORMATS = {"audio": ".wav", "video": ".mp4"}
STRUCTURAL_VIDEO = {"encoder", "major_brand", "minor_version", "compatible_brands"}


def decode(path: Path, channel: str) -> bytes:
    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path)]
    if channel == "audio":
        return command(args + ["-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"])
    return command(args + ["-map", "0:v:0", "-an", "-c:v", "rawvideo", "-threads", "1",
                           "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"])


def strip(source: Path, target: Path, channel: str) -> None:
    args = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(source)]
    args += ["-map", "0:a:0" if channel == "audio" else "0:v:0",
             "-map_metadata", "-1", "-map_chapters", "-1", "-c", "copy"]
    if channel == "video":
        args += ["-an", "-write_tmcd", "0", "-movflags", "+faststart"]
    command(args + [str(target)])
    if decode(source, channel) != decode(target, channel):
        raise ValueError("metadata stripping changed the decoded input")


def media_tags(source: Path) -> dict:
    return json.loads(command(["ffprobe", "-v", "error", "-show_entries",
                               "stream=codec_type:format_tags", "-of", "json", str(source)]))


def repeat(root: Path, source_id: str, channel: str) -> dict | None:
    extension = FORMATS[channel]
    source = root / "media" / source_id / (channel + extension)
    info = media_tags(source)
    structural_tags = STRUCTURAL_VIDEO if channel == "video" else {"encoder"}
    tags = sorted(set(info.get("format", {}).get("tags", {})) - structural_tags)
    streams = [stream["codec_type"] for stream in info["streams"]]
    if not tags and streams == [channel]:
        return None
    folder = root / "interventions" / "metadata"
    folder.mkdir(parents=True, exist_ok=True)
    cleaned = folder / f"{source_id}-{channel}{extension}"
    strip(source, cleaned, channel)
    clean_info = media_tags(cleaned)
    if (set(clean_info.get("format", {}).get("tags", {})) - structural_tags or
            [stream["codec_type"] for stream in clean_info["streams"]] != [channel]):
        raise ValueError("container metadata stripping was incomplete")
    text, identity = chat(root, CAPTION_MODEL, [
        {"type": "text", "text": CAPTION_PROMPTS[channel] + CAPTION_SUFFIX}, media_part(cleaned),
    ], context=f"metadata_control:{channel}:{source_id}", max_tokens=200)
    value = parse_json(text)
    if not isinstance(value, dict) or not isinstance(value.get("description"), str):
        raise ValueError("invalid metadata sensitivity caption")
    return dict(source_id=source_id, channel=channel, tag_names_removed=tags,
                prior_streams=streams, cleaned_sha256=sha(cleaned.read_bytes()),
                identical_decoded_signal=True, new_caption=value["description"].strip(), provider=identity)


def run(root: Path) -> dict:
    rows = json.loads((root / "roster.json").read_text(encoding="utf8"))["sources"]
    descriptions = json.loads((root / "captions.json").read_text(encoding="utf8"))
    main = json.loads((root / "geometry.json").read_text(encoding="utf8"))
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        ids = archive["source_ids"].tolist()
        values = {key: archive[key].copy() for key in archive.files if key != "source_ids"}
    index = {source: i for i, source in enumerate(ids)}
    changes, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {(row["source_id"], channel): pool.submit(repeat, root, row["source_id"], channel)
                 for row in rows for channel in FORMATS}
        for (source_id, channel), job in tasks.items():
            try:
                record = job.result()
                if record is not None:
                    original = descriptions[channel].get(source_id, {}).get("description")
                    record.update(old_caption=original, caption_changed=original != record["new_caption"])
                    changes.append(record)
            except (RequestError, OSError, TypeError, KeyError, ValueError, subprocess.SubprocessError) as exc:
                failures.append(dict(source_id=source_id, channel=channel, error=str(exc)))
    changed_keys = set()
    for channel in FORMATS:
        for head in ("bge", "qwen"):
            key = f"{channel}_caption_{head}"
            if key not in values:
                continue
            replacements = {item["source_id"]: item["new_caption"] for item in changes
                            if item["channel"] == channel and item["caption_changed"] and item["old_caption"]
                            and np.isfinite(values[key][index[item["source_id"]]]).all()}
            if replacements:
                vectors, _, errors = text_vectors(root, replacements, f"metadata_control_{key}",
                                                  TEXT_MODELS[f"text_{head}"])
                failures.extend(errors)
                for source_id, vector in vectors.items():
                    values[key][index[source_id]] = vector
                    changed_keys.add(key)
    comparisons = []
    for pair in main["pairs"]:
        if pair["status"] != "measured" or not changed_keys.intersection((pair["left"], pair["right"])):
            continue
        folds = [[np.asarray([index[source_id] for source_id in cohort]) for cohort in fold]
                 for fold in pair["source_ids_by_fold"]]
        used = np.concatenate([cohort for fold in folds for cohort in fold])
        axes = []
        for key in (pair["left"], pair["right"]):
            data = np.zeros_like(values[key], dtype=float)
            data[used] = normalize(values[key][used])
            axes.append([ranks(matrix) for matrix in matrices(data, folds)])
        estimate = rank_point(*axes, ORIENTATIONS["cross_source"])
        original = pair["outcomes"]["cross_source"]["global_rsa"]["estimate"]
        comparisons.append(dict(pair=pair["pair"], original=original, remux=estimate, difference=estimate - original))
    result = dict(scope="same decoded inputs, repeated RouterAI captions; service variability remains possible",
                  checked_sources=len(rows), repeated=changes, failures=failures, cross_source=comparisons)
    save_json(root / "metadata-control.json", result)
    print(f"Repeated {len(changes)} inputs and recomputed {len(comparisons)} affected comparisons.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    run(parser.parse_args().root)
