"""Acquire RouterAI embeddings and explicitly labeled semantic and signal controls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acquire import command, save_json, sha
from .router import RequestError, chat, embedding, media_part, parse_json

TEXT_MODELS = {
    "text_bge": "baai/bge-m3",
    "text_qwen": "qwen/qwen3-embedding-8b",
    "text_gemini": "google/gemini-embedding-2",
}
IMAGE_MODELS = {
    "image_titan": "amazon/titan-embed-image-v1",
    "image_voyage": "mongodb/voyage-multimodal-3.5",
    "image_gemini": "google/gemini-embedding-2",
}
CAPTION_MODEL = "google/gemini-2.5-flash"
CHANNELS = {"audio": "audio.wav", "image": "image.jpg", "video": "video.mp4"}
CAPTION_PROMPTS = {
    "audio": "Listen only to the supplied recording. Describe the main environmental sounds and the audible event in plain English. Omit speech transcription and qualify unclear sources. Report whether speech or singing was audible.",
    "image": "Look only at the supplied still. Describe what is directly visible in plain English. Mention objects and visible activity. Avoid assuming sound or changes over time. Set speech false.",
    "video": "Watch this silent clip. Describe the main objects and visible changes over time in plain English. Infer no sounds from the picture. Set speech false.",
}
CAPTION_SUFFIX = (' Return only a compact JSON object with an English "description" of 12 to 35 '
                  'words and boolean "speech". Describe this particular input without seeing other channels.')


def caption(root: Path, row: dict, channel: str) -> dict:
    path = root / "media" / row["source_id"] / CHANNELS[channel]
    result, provider = chat(root, CAPTION_MODEL, [
        {"type": "text", "text": CAPTION_PROMPTS[channel] + CAPTION_SUFFIX},
        media_part(path),
    ], context=f"caption:{channel}:{row['source_id']}", max_tokens=200)
    value = parse_json(result)
    if (not isinstance(value, dict) or not isinstance(value.get("description"), str) or
            not value["description"].strip() or type(value.get("speech")) is not bool):
        raise ValueError(f"invalid caption response for {row['source_id']} {channel}")
    return dict(description=value["description"].strip(), speech=value["speech"], provider=provider)


def audio_features(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError(f"unexpected waveform format: {path.name}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2") / 32768.0
    frames = np.lib.stride_tricks.sliding_window_view(samples, 400)[::160]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    freq = np.fft.rfftfreq(512, 1 / 16000)
    edges = np.geomspace(80, 8000, 37)
    band = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        bins = (lo <= freq) & (freq < hi)
        if not bins.any():
            bins[np.argmin(np.abs(freq - np.sqrt(lo * hi)))] = True
        band.append(spectrum[:, bins].mean(axis=1))
    bands = np.log10(np.maximum(np.stack(band, axis=1), 1e-8))
    bands -= bands.mean(axis=1, keepdims=True)
    return np.concatenate([bands.mean(axis=0), bands.std(axis=0), np.abs(np.diff(bands, axis=0)).mean(axis=0)])


def video_features(path: Path) -> dict[str, np.ndarray]:
    raw = command(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                   "-i", str(path), "-an", "-vf", "scale=64:64,format=gray",
                   "-f", "rawvideo", "-threads", "1", "pipe:1"])
    frames = np.frombuffer(raw, dtype="u1").astype(float).reshape((-1, 8, 8, 8, 8)) / 255
    if len(frames) < 2:
        raise ValueError("silent video yielded too few frames")
    cells = frames.mean(axis=(2, 4))
    mean_luminance = cells.mean(axis=0).ravel()
    motion_frames = np.abs(np.diff(frames, axis=0)).mean(axis=(2, 4))
    return {
        "video_static": mean_luminance,
        "video_motion": np.concatenate((motion_frames.mean(axis=0).ravel(), motion_frames.std(axis=0).ravel())),
    }


def image_vector(root: Path, row: dict, name: str, model: str) -> tuple[list, str]:
    path = root / "media" / row["source_id"] / CHANNELS["image"]
    vectors, provider = embedding(root, model, [{"content": [media_part(path)]}],
                                  context=f"{name}:{row['source_id']}")
    return vectors[0], provider


def text_vectors(root: Path, values: dict[str, str], name: str, model: str,
                 *, provider: str | None = None) -> tuple[dict, dict, list]:
    grouped = defaultdict(list)
    for item, text in values.items():
        grouped[text].append(item)
    unique = list(grouped)
    vectors, observed, failures = {}, {}, []
    for offset in range(0, len(unique), 12):
        batch = unique[offset:offset + 12]
        label = f"{name}:batch{offset // 12}"
        try:
            result, alias = embedding(root, model, batch, context=label, provider=provider)
        except (RequestError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(item=label, reason=str(exc)))
            continue
        for text, vector in zip(batch, result, strict=True):
            for item in grouped[text]:
                vectors[item] = vector
                observed[item] = alias
    return vectors, observed, failures


def extract(root: Path) -> dict:
    roster = root / "roster.json"
    rows = json.loads(roster.read_text(encoding="utf8"))["sources"]
    ids = [row["source_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source in frozen roster")
    data, providers, failures = defaultdict(dict), defaultdict(dict), []
    captions = {channel: {} for channel in CHANNELS}
    for row in rows:
        folder = root / "media" / row["source_id"]
        for name, expected in row["media_sha256"].items():
            if sha((folder / name).read_bytes()) != expected:
                raise ValueError(f"media hash mismatch: {row['source_id']} / {name}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for row in rows:
            source_id = row["source_id"]
            for name, model in IMAGE_MODELS.items():
                futures[executor.submit(image_vector, root, row, name, model)] = ("image", name, source_id)
            for channel in CHANNELS:
                futures[executor.submit(caption, root, row, channel)] = ("caption", channel, source_id)
            folder = root / "media" / source_id
            data["audio_signal"][source_id] = audio_features(folder / CHANNELS["audio"])
            for name, value in video_features(folder / CHANNELS["video"]).items():
                data[name][source_id] = value
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            kind, name, source = futures[future]
            try:
                response = future.result()
                if kind == "caption":
                    captions[name][source] = response
                else:
                    data[name][source], providers[name][source] = response
            except (RequestError, ValueError, KeyError, TypeError, OSError) as exc:
                failures.append(dict(item=f"{name}:{source}", reason=str(exc)))
            if n % 40 == 0:
                print(f"media calls {n}/{len(futures)}", flush=True)
    text_inputs = {name: (model, {row["source_id"]: row["source_text"] for row in rows})
                   for name, model in TEXT_MODELS.items()}
    for channel, descriptions in captions.items():
        for head in ("bge", "qwen"):
            text_inputs[f"{channel}_caption_{head}"] = (
                TEXT_MODELS[f"text_{head}"], {source: entry["description"] for source, entry in sorted(descriptions.items())})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {name: executor.submit(text_vectors, root, values, name, model)
                for name, (model, values) in text_inputs.items()}
        for name, future in jobs.items():
            data[name], providers[name], problems = future.result()
            failures.extend(problems)
    feature_arrays = {}
    for name, values in sorted(data.items()):
        dimensions = {len(vector) for vector in values.values()}
        if not dimensions:
            continue
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent dimensions within {name}")
        size = dimensions.pop()
        array = np.full((len(ids), size), np.nan, dtype="float32")
        for i, source in enumerate(ids):
            if source in values:
                vector = np.asarray(values[source], dtype="float32")
                # Motion can be exactly absent on static assets. Represent this as missing.
                if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                    array[i] = vector
                else:
                    failures.append(dict(item=f"{name}:{source}", reason="nonfinite_or_zero_vector"))
        feature_arrays[name] = array
    target = root / "features.npz"
    np.savez_compressed(target, source_ids=np.asarray(ids), **feature_arrays)
    save_json(root / "captions.json", captions)
    meta = dict(roster_sha256=sha(roster.read_bytes()), feature_sha256=sha(target.read_bytes()),
                models={"text": TEXT_MODELS, "image": IMAGE_MODELS, "caption": CAPTION_MODEL},
                observed_models={name: {provider: list(values.values()).count(provider)
                                       for provider in sorted(set(values.values()))}
                                 for name, values in providers.items()},
                coverage={name: int(np.isfinite(value).all(axis=1).sum()) for name, value in feature_arrays.items()},
                failures=sorted(failures, key=lambda row: (row["item"], row["reason"])))
    save_json(root / "feature-meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    args = parser.parse_args()
    print(json.dumps(extract(args.root)["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    main()
