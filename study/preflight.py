"""Check actual RouterAI media delivery with synthetic payloads before collecting data."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import os
import struct
import urllib.request
import wave
from pathlib import Path

from .acquire import command, save_json, sha
from .router import BASE, RequestError, media_part, request

GEMINI = "google/gemini-embedding-2"
NOTES = "Synthetic transport fixtures test the observed routes and encoding shapes."


def fixtures(root: Path) -> dict[str, dict]:
    folder = root / "transport-fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for i, (frequency, pattern) in enumerate(((400, "testsrc2"), (870, "color=red")), 1):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(b"".join(struct.pack("<h", int(18000 * math.sin(sample * frequency * 2 * math.pi / 16000)))
                                     for sample in range(16000)))
        audio, video, image = [folder / f"fixture-{i}.{ext}" for ext in ("wav", "mp4", "jpg")]
        audio.write_bytes(buffer.getvalue())
        if not video.exists():
            settings = "=s=128x128:r=6:d=2" if i == 1 else ":s=128x128:r=6:d=2"
            command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", pattern + settings,
                     "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(video)])
        if not image.exists():
            command(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "1",
                     "-q:v", "3", "-threads", "1", str(image)])
        payloads.update({f"audio{i}": media_part(audio), f"video{i}": media_part(video), f"image{i}": media_part(image)})
    return payloads


def catalog() -> dict:
    req = urllib.request.Request(BASE + "models", headers={
        "Authorization": "Bearer " + os.environ["ROUTERAI_API_KEY"],
    })
    with urllib.request.urlopen(req, timeout=30) as response:
        listed = json.load(response)["data"]
    items = sorted([dict(model=row["id"], input=row.get("architecture", {}).get("input_modalities", []),
                         output=row.get("architecture", {}).get("output_modalities", [])) for row in listed],
                   key=lambda row: row["model"])
    return dict(https_endpoint=BASE + "models", total=len(items), all_model_ids=[row["model"] for row in items],
                models_with_embeddings=[row for row in items if "embeddings" in (row["output"] or [])])


def audit(root: Path) -> dict:
    data = fixtures(root)
    prompt = {"type": "text", "text": "A test signal"}
    jobs = {
        "bge": ("baai/bge-m3", ["dog barking", "church bell ringing"]),
        "qwen8": ("qwen/qwen3-embedding-8b", ["dog barking", "church bell ringing"]),
        "only_text_control": (GEMINI, [dict(content=[prompt])]),
        "audio_plus_text": (GEMINI, [dict(content=[prompt, data["audio1"]])]),
        "audio2_plus_text": (GEMINI, [dict(content=[prompt, data["audio2"]])]),
        "video_plus_text": (GEMINI, [dict(content=[prompt, data["video1"]])]),
    }
    for model in (GEMINI, GEMINI + "-preview", "mongodb/voyage-multimodal-3.5",
                  "voyageai/voyage-multimodal-3.5", "amazon/titan-embed-image-v1"):
        for name, part in data.items():
            jobs[f"{model}/{name}"] = model, [dict(content=[part])]
    for model in ("mongodb/voyage-multimodal-3.5", "voyageai/voyage-multimodal-3.5"):
        url = data["video1"]["video_url"]["url"]
        jobs[f"{model}/video_base64"] = model, [dict(content=[dict(type="video_base64", video_base64=url)])]
    for model, asset in (
        ("facebook/dinov2-base", "image1"), ("facebook/vit-mae-base", "image1"),
        ("microsoft/beats", "audio1"), ("facebook/audiomae", "audio1"),
        ("MCG-NJU/videomae-base", "video1"), ("facebook/vjepa2-vitl-fpc64-256", "video1"),
    ):
        jobs[f"candidate-alias/{model}"] = model, [dict(content=[data[asset]])]

    def probe(item: tuple[str, tuple[str, list]]) -> list:
        name, (model, inputs) = item
        try:
            observed = request(root, "embeddings", {"model": model, "input": inputs, "encoding_format": "float"},
                               context=f"transport:{name}")
            embeddings = observed["data"]
            return [name, "OK", observed.get("model"),
                    [[len(item["embedding"]), sha(json.dumps(item["embedding"]).encode())] for item in embeddings],
                    observed.get("usage")]
        except (RequestError, KeyError, TypeError, ValueError) as exc:
            return [name, "failed", model, str(exc)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        observations = list(executor.map(probe, jobs.items()))
    result = dict(notes=NOTES, catalog=catalog(), opens_2=observations)
    save_json(root / "observed-transport.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/transport-audit"))
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps([{ "probe": row[0], "status": row[1]} for row in report["opens_2"]], indent=2))
