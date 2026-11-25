"""Select source recordings using Commons metadata and deterministic media screening."""

from __future__ import annotations

import argparse
import array
import concurrent.futures
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

CONCEPTS = Path(__file__).with_name("concepts.json")
USER_AGENT = "SemanticGeometryResearch/0.1 (https://github.com/AdamCage/multimodal-semantic-geometry)"
API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSE = re.compile(r"(?:CC BY(?:-SA)? \d+\.\d+|CC0(?: 1\.0)?|Public domain)", re.I)
SOURCE_LIMIT = 32_000_000
SOURCE_SEED = "commons-2026-11-25-v3"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub("<[^>]*>", " ", value)).split())


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    Path(f.name).replace(path)


def fetch(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError("source exceeds byte limit")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            wait = exc.headers.get("Retry-After", "")
            time.sleep(min(30, int(wait)) if wait.isdecimal() else 2 ** (attempt + 1))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable retry state")


def inventory(query: str) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:video", "gsrnamespace": 6, "gsrlimit": 48,
        "prop": "videoinfo", "viprop": "url|mime|size|sha1|timestamp|user|extmetadata|derivatives",
        "viextmetadatalanguage": "en",
        "viextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        data = json.loads(fetch(url, 4_000_000))
        if data.get("error", {}).get("code") != "cirrussearch-too-busy-error" or attempt == 2:
            break
        time.sleep(3 * (attempt + 1))
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    rows = []
    for page in data.get("query", {}).get("pages", {}).values():
        info = page.get("videoinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        def value(key: str) -> str:
            return plain(meta.get(key, {}).get("value", ""))
        license_name = value("LicenseShortName")
        if not ALLOWED_LICENSE.fullmatch(license_name):
            continue
        if not info.get("mime", "").startswith("video/") and not page["title"].lower().endswith(".ogv"):
            continue
        duration = info.get("duration", 0)
        derivative = next((part for part in info.get("derivatives", [])
                           if part.get("transcodekey") == "240p.vp9.webm" and
                           0 < part.get("bandwidth", 0) * duration / 8 <= SOURCE_LIMIT), None)
        if duration < 4.1 or (not derivative and not 0 < info.get("size", 0) <= SOURCE_LIMIT):
            continue
        author, uploader = value("Artist"), info.get("user", "")
        author_key = re.sub(r"\W+", "", author.casefold())
        uploader_key = re.sub(r"\W+", "", uploader.casefold())
        if not author_key or not uploader_key:
            continue
        rows.append(dict(
            pageid=page["pageid"], search_rank=page.get("index"), title=page["title"],
            url=info["url"], source_page=info["descriptionurl"], bytes=info["size"],
            media_url=derivative["src"] if derivative else info["url"],
            derivative=derivative["transcodekey"] if derivative else None,
            commons_sha1=info.get("sha1"), source_revision=info.get("timestamp"),
            author=author, uploader=uploader, author_key=author_key, uploader_key=uploader_key,
            license=license_name, license_url=value("LicenseUrl"), credit=value("Credit"),
            source_text=(value("ImageDescription") or page["title"].split(":", 1)[1].rsplit(".", 1)[0])[:600],
            source_text_origin="commons_description" if value("ImageDescription") else "commons_filename",
        ))
    return sorted(rows, key=lambda row: (row["search_rank"] or 1000, row["pageid"]))


def command(argv: list[str], timeout: int = 45) -> bytes:
    return subprocess.run(argv, capture_output=True, check=True, timeout=timeout).stdout


def prepare(row: dict, root: Path) -> dict:
    source_id = f"commons-{row['pageid']}"
    folder = root / "media" / source_id
    metadata = folder / "media.json"
    if metadata.is_file():
        return json.loads(metadata.read_text(encoding="utf8"))
    folder.mkdir(parents=True, exist_ok=True)
    transfer = urllib.parse.urlparse(row["media_url"])
    if transfer.scheme != "https" or transfer.hostname != "upload.wikimedia.org":
        raise ValueError("unexpected Commons media host")
    original = fetch(row["media_url"], SOURCE_LIMIT)
    if row["derivative"] is None and (len(original) != row["bytes"] or
                                     hashlib.sha1(original).hexdigest() != row["commons_sha1"]):
        raise ValueError("source revision no longer matches Commons inventory")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        source = Path(tmp) / "original"
        source.write_bytes(original)
        info = json.loads(command(["ffprobe", "-v", "error", "-show_streams",
                                   "-show_format", "-of", "json", str(source)]))
        channels = {stream["codec_type"] for stream in info["streams"]}
        duration = float(info["format"].get("duration", 0))
        if not {"audio", "video"} <= channels or duration < 4.1:
            raise ValueError("missing audio or video or shorter than 4.1 seconds")
        start = round((duration - 4.0) / 2.0, 3)
        common = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-threads", "1",
                  "-filter_threads", "1", "-ss", str(start), "-i", str(source), "-t", "4"]
        command(common + ["-map", "0:a:0", "-vn", "-ar", "16000", "-ac", "1",
                          "-c:a", "pcm_s16le", str(folder / "audio.wav")])
        command(common + ["-map", "0:v:0", "-an", "-vf", "fps=6,scale=256:-2",
                          "-c:v", "libx264", "-threads", "1", "-crf", "25",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                          str(folder / "video.mp4")])
        command(["ffmpeg", "-y", "-nostdin", "-v", "error", "-threads", "1",
                 "-filter_threads", "1", "-ss", "2", "-i", str(folder / "video.mp4"),
                 "-an", "-frames:v", "1", "-q:v", "3", str(folder / "image.jpg")])
        with wave.open(str(folder / "audio.wav"), "rb") as wav:
            pcm = array.array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        rms = math.isqrt(sum(sample * sample for sample in pcm) // max(1, len(pcm)))
        if rms < 100:
            raise ValueError("center-window audio below frozen RMS floor")
    result = dict(source_id=source_id, source_sha256=sha(original), duration_s=duration,
                  start_s=start, clip_s=4, audio_rms=rms, media_sha256={
                      name: sha((folder / name).read_bytes())
                      for name in ("audio.wav", "video.mp4", "image.jpg")
                  })
    save_json(metadata, result)
    return result


def inventory_rows(query: str) -> list[dict]:
    unique = {}
    for alias in query.split(" OR "):
        for row in inventory(alias):
            prior = unique.get(row["pageid"])
            if prior is None or (row["search_rank"] or 1000) < (prior["search_rank"] or 1000):
                unique[row["pageid"]] = row
    return sorted(unique.values(), key=lambda row: (row["search_rank"] or 1000, row["pageid"]))


def select(root: Path, target: int, extension: Path | None = None, reuse: Path | None = None) -> None:
    original = json.loads(CONCEPTS.read_text(encoding="utf8"))
    additional = json.loads(extension.read_text(encoding="utf8")) if extension else {}
    queries = {
        concept: (group, query)
        for design in (original, additional)
        for group, concepts in design.items()
        for concept, query in concepts.items()
    }
    old_concepts = {concept for group in original.values() for concept in group}
    candidates, audit, selected = {}, [], []
    def load(item: tuple[str, tuple[str, str]]) -> tuple[str, list[dict]]:
        concept, (_, query) = item
        cached = root / "inventory" / f"{concept}.json"
        if not cached.is_file() and reuse is not None and concept in old_concepts:
            prior = reuse / "inventory" / cached.name
            if prior.is_file():
                cached = prior
        if cached.is_file():
            return concept, json.loads(cached.read_text(encoding="utf8"))
        try:
            rows = inventory_rows(query)
        except (OSError, ValueError, RuntimeError) as exc:
            if extension is None:
                raise
            return concept, [dict(inventory_error=str(exc)[:180])]
        save_json(cached, rows)
        return concept, rows

    def media(row: dict, allow_reuse: bool) -> dict:
        if reuse is not None and allow_reuse:
            key = f"commons-{row['pageid']}"
            previous, destination = reuse / "media" / key, root / "media" / key
            if (previous / "media.json").is_file() and not (destination / "media.json").is_file():
                metadata = json.loads((previous / "media.json").read_text(encoding="utf8"))
                for name, digest in metadata["media_sha256"].items():
                    if sha((previous / name).read_bytes()) != digest:
                        raise ValueError("prior local media digest mismatch")
                shutil.copytree(previous, destination, dirs_exist_ok=True)
        return prepare(row, root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for concept, rows in pool.map(load, queries.items()):
            candidates[concept] = rows
            print(f"inventory {concept}: {len(rows)}", flush=True)
    seen_pages, seen_hashes = set(), set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for concept, rows in candidates.items():
            if rows and "inventory_error" in rows[0]:
                audit.append(dict(concept=concept, status=rows[0]["inventory_error"]))
                continue
            creators, uploaders, result = set(), set(), []
            ranks = sorted(rows[:24], key=lambda row: sha(f"{SOURCE_SEED}|{row['pageid']}".encode()))
            for offset in range(0, len(ranks), 4):
                chunk = [row for row in ranks[offset:offset + 4] if row["pageid"] not in seen_pages
                         and row["author_key"] not in creators and row["uploader_key"] not in uploaders]
                jobs = [(row, pool.submit(media, row, concept in old_concepts)) for row in chunk]
                for row, future in jobs:
                    try:
                        captured = future.result()
                        status = "eligible"
                        if (row["author_key"] in creators or row["uploader_key"] in uploaders or
                                captured["source_sha256"] in seen_hashes):
                            status = "duplicate_or_creator"
                        elif len(result) >= target:
                            status = "eligible_surplus"
                        else:
                            seen_pages.add(row["pageid"])
                            seen_hashes.add(captured["source_sha256"])
                            creators.add(row["author_key"])
                            uploaders.add(row["uploader_key"])
                            result.append(dict(**row, **captured, concept=concept, group=queries[concept][0]))
                            status = "selected"
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        status = f"{type(exc).__name__}: {str(exc)[:200]}"
                    audit.append(dict(concept=concept, pageid=row["pageid"], status=status))
                if len(result) >= target:
                    break
            selected.extend(result)
            save_json(root / "manifest-draft.json", selected)
            save_json(root / "selection-audit.json", audit)
            print(f"selected {concept}: {len(result)}", flush=True)
    save_json(root / "manifest.json", selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--target", type=int, default=12)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    select(args.root, args.target, args.extension, args.reuse)


if __name__ == "__main__":
    main()
