"""Check an independent audiovisual corpus without downloading its large media archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .acquire import fetch, save_json, sha

RECORD = "https://zenodo.org/records/14536325"
MIRROR = "https://huggingface.co/datasets/TUT-ARG/AVCaps"
PARENT = "https://xdshang.github.io/docs/vidor.html"
TAIL_BYTES = 65_557
VIDOR_ARCHIVES = {
    "training-annotation.zip": "85f39fdd81a780bbb9b975cca8f219a2",
    "validation-annotation.zip": "96c6870910e8d4fb3836878215432c1f",
}


def archive_members(url: str) -> tuple[dict, set[str]]:
    query = urllib.request.Request(url, headers={"Range": f"bytes=-{TAIL_BYTES}",
                                               "User-Agent": "multimodal-semantic-research"})
    try:
        with urllib.request.urlopen(query, timeout=25) as response:
            status = response.status
            range_header = response.headers.get("Content-Range")
            if status != 206:
                return dict(status=status, archive_index="range_not_available"), set()
            tail = response.read(TAIL_BYTES + 1)
        if len(tail) > TAIL_BYTES:
            raise ValueError("range exceeded requested bytes")
        with zipfile.ZipFile(io.BytesIO(tail)) as archive:
            ids = {PurePosixPath(row.filename).stem for row in archive.infolist()
                   if not row.is_dir() and PurePosixPath(row.filename).suffix.lower() == ".mp4"}
        return dict(status=status, range=range_header, archive_index="observed",
                    sha256=sha(tail), listed_mp4=len(ids)), ids
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return dict(status=getattr(exc, "code", None), archive_index="unverified",
                    error=type(exc).__name__), set()


def join_vidor(ids: set[str], archives: list[Path]) -> dict:
    objects, relations = Counter(), Counter()
    joined = set()
    checksums = {}
    for path in archives:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "md5").hexdigest()
        if checksum != VIDOR_ARCHIVES.get(path.name):
            raise ValueError(f"VidOR annotation checksum mismatch: {path.name}")
        checksums[path.name] = checksum
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = PurePosixPath(item.filename)
                if name.suffix != ".json" or name.stem not in ids:
                    continue
                if item.file_size > 16_000_000 or name.stem in joined:
                    raise ValueError("oversize or duplicate VidOR annotation in joined sample")
                row = json.loads(archive.read(item))
                if str(row["video_id"]) != name.stem:
                    raise ValueError("VidOR annotation filename and video ID differ")
                joined.add(name.stem)
                objects.update({x["category"] for x in row["subject/objects"]})
                relations.update({x["predicate"] for x in row["relation_instances"]})
    return dict(verified_annotation_md5=checksums, matched_clip_ids=len(joined),
                visual_objects_in_at_least_four_clips={k: v for k, v in sorted(objects.items()) if v >= 4},
                visual_relations_in_at_least_four_clips={k: v for k, v in sorted(relations.items()) if v >= 4},
                count_scope="Per-label video counts from parent annotations, with no audio or creator independence claim.")


def inspect_avcaps(archives: list[Path] | None = None) -> dict:
    captions = fetch(RECORD + "/files/test_captions.json?download=1", 1_000_000)
    mirror = fetch(MIRROR + "/resolve/main/test_captions.json", 1_000_000)
    catalog_data = fetch("https://zenodo.org/api/records/14536325", 1_000_000)
    rows = json.loads(captions)
    fields = ("audio_captions", "visual_captions", "audio_visual_captions")
    available = {key: sum(any(isinstance(value, str) and value.strip() for value in row.get(key, []))
                          for row in rows.values()) for key in fields}
    artifact = next(row for row in json.loads(catalog_data)["files"] if row["key"] == "test_videos.zip")
    archive, ids = archive_members(RECORD + "/files/test_videos.zip?download=1")
    result = dict(
        dataset="AVCaps test", record=RECORD, mirror=MIRROR, parent=PARENT,
        captions_sha256=sha(captions), mirror_sha256=sha(mirror), captions_byte_identical=captions == mirror,
        unique_caption_ids=len(rows), caption_fields=available,
        candidate_archive=dict(archive, advertised_bytes=artifact["size"],
                               advertised_checksum=artifact["checksum"],
                               matched_caption_ids=len(set(rows) & ids) if ids else None),
        dataset_card_license="CC BY-NC 4.0",
        unresolved=["per-item source rights and unique creator clusters",
                    "usable sound and visibility by concept",
                    "concept membership and independent-creator fold counts",
                    "training overlap with future encoder checkpoints"],
        disposition="candidate metadata only; no media inference and no admitted replication cohort",
    )
    if archives:
        result["parent_visual_annotation_join"] = join_vidor(set(rows), archives)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vidor", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    result = inspect_avcaps(args.vidor)
    save_json(args.output, result)
    print(json.dumps(result))
