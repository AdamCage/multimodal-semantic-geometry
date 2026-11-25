"""Test channel-conditioned recognition and media interventions with a separate observer."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from .acquire import command, save_json, sha
from .features import CHANNELS
from .geometry import memberships, matrices, normalize, rng_for
from .metadata_control import strip
from .roster import LABELS
from .router import RequestError, chat, media_part, parse_json

OBSERVER = "google/gemini-3.5-flash"
FACTORS = ("object", "action", "material", "color", "motion", "temporal_order", "sound_source")
CHANNEL_PROMPTS = {
    "audio": "Listen to the audio only. Ignore spoken labels and judgments from speech. Count color and visible motion only if present in your actual input.",
    "image": "Look only at this single still image. You cannot verify a sound by looking at its possible source.",
    "video": "Watch only this silent video. Report directly observable visual information. You cannot verify a sound by looking at its possible source.",
}


def observe(root: Path, path: Path, channel: str, options: list[str], source: str, condition: str) -> dict:
    enumerated = [{"id": label, "event": LABELS[label]} for label in options]
    enumerated.append({"id": "other_or_unclear", "event": "a different event or an unclear signal"})
    prompt = (f"{CHANNEL_PROMPTS[channel]} Choose the one best-supported event ID or other_or_unclear. "
              "Do not guess between equally plausible alternatives. For the factors listed, use true only when directly distinguishable "
              "from this channel. For temporal_order use true only if the ordering of at least two distinct events is distinguishable. "
              "Return compact JSON containing event and a factors object with booleans. "
              f"Factors: {', '.join(FACTORS)}. Event choices: {json.dumps(enumerated)}")
    output, info = chat(root, OBSERVER, [{"type": "text", "text": prompt}, media_part(path)],
                        context=f"observability:{source}:{channel}:{condition}", max_tokens=650, temperature=None)
    record = parse_json(output)
    if (not isinstance(record, dict) or record.get("event") not in options + ["other_or_unclear"] or
            not isinstance(record.get("factors"), dict) or
            set(record["factors"]) != set(FACTORS) or any(type(v) is not bool for v in record["factors"].values())):
        raise ValueError("observer response did not follow the registered schema")
    return dict(event=record["event"], factors=record["factors"], provider=info,
                input_sha256=sha(path.read_bytes()), condition=condition, channel=channel, source_id=source)


def write_audio(path: Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(np.rint(data).astype("<i2").tobytes())


def audio_pair(source: Path, folder: Path) -> tuple[Path, Path, dict]:
    with wave.open(str(source), "rb") as clip:
        original = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(float)
    rng = rng_for(20261125, source.parent.name, "phase")
    desired = np.fft.rfft(original)
    reordered = desired.copy()
    lo = 1
    hi = len(desired) - (len(original) % 2 == 0)
    reordered[lo:hi] = abs(desired[lo:hi]) * np.exp(1j * rng.uniform(0, 2 * np.pi, hi - lo))
    changed = np.fft.irfft(reordered, n=len(original))
    gain = min(1.0, 31000 / max(np.max(np.abs(original)), np.max(np.abs(changed))))
    original, changed = np.rint(gain * original), np.rint(gain * changed)
    p, q = np.abs(np.fft.rfft(original)), np.abs(np.fft.rfft(changed))
    error = float(np.linalg.norm(p - q) / np.linalg.norm(p))
    if error > .002 or np.array_equal(original, changed):
        raise ValueError("spectral control failed")
    folder.mkdir(parents=True, exist_ok=True)
    sham, scrambled = folder / "audio-original.wav", folder / "audio-phase.wav"
    write_audio(sham, original)
    write_audio(scrambled, changed)
    return sham, scrambled, dict(relative_spectral_magnitude_error=error, matched_gain=float(gain), samples=len(original))


def video_pair(source: Path, folder: Path) -> tuple[Path, Path, dict]:
    raw = command(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                   "-i", str(source), "-an", "-vf", "fps=1,scale=192:192,format=yuv420p",
                   "-frames:v", "4", "-c:v", "rawvideo", "-threads", "1", "-f", "rawvideo", "pipe:1"])
    stride = 192 * 192 * 3 // 2
    if len(raw) != 4 * stride:
        raise ValueError("expected four 1 Hz frames")
    frames = [raw[i * stride:(i + 1) * stride] for i in range(4)]
    if len(set(frames)) < 2:
        raise ValueError("temporal manipulation requires visibly nonidentical samples")
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / name for name in ("video-ordered.mp4", "video-permuted.mp4")]
    order = [2, 0, 3, 1]
    for path, indexes in zip(paths, [list(range(4)), order], strict=True):
        payload = b"".join(frames[i] * 4 for i in indexes)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                        "-s", "192x192", "-r", "4", "-i", "pipe:0", "-an",
                        "-c:v", "libx264", "-threads", "1", "-g", "1", "-qp", "0",
                        "-movflags", "+faststart", str(path)], input=payload, capture_output=True,
                       check=True, timeout=35)
        decoded = command(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                           "-i", str(path), "-an", "-pix_fmt", "yuv420p",
                           "-c:v", "rawvideo", "-threads", "1", "-f", "rawvideo", "pipe:1"])
        if decoded != payload:
            raise ValueError("video order control did not preserve decoded frames")
    return paths[0], paths[1], dict(order=order, source_frame_sha256=[sha(frame) for frame in frames])


def contrast(rows: list[dict], records: list[dict], channel: str,
             initial: str, altered: str, field: str = "event") -> dict:
    lookup = {(row["source_id"], row["condition"]): row for row in records if row["channel"] == channel}
    observed, components = [], Counter()
    for row in rows:
        source = row["source_id"]
        first, last = lookup.get((source, initial)), lookup.get((source, altered))
        if first is not None and last is not None:
            pair = ((first["event"] == row["concept"], last["event"] == row["concept"]) if field == "event"
                    else (first["factors"][field], last["factors"][field]))
            observed.append(pair)
            components[row["creator_component"]] += int(pair[0]) - int(pair[1])
    if not observed:
        return dict(n=0)
    table = Counter(observed)
    lost, gained = table[True, False], table[False, True]
    changes = np.asarray([v for _, v in sorted(components.items()) if v != 0])
    observed_change = abs(lost - gained)
    if len(changes) <= 15:
        indexes = np.arange(2 ** len(changes), dtype=np.uint32)[:, None]
        bits = ((indexes >> np.arange(len(changes))) & 1) * 2 - 1
        p = float(np.mean(np.abs(bits @ changes) >= observed_change))
    else:
        generator = rng_for(20261125, channel, field, "cluster-signs")
        signs = 2 * generator.integers(0, 2, (19999, len(changes))) - 1
        p = float((1 + (np.abs(signs @ changes) >= observed_change).sum()) / (len(signs) + 1))
    return dict(n=len(observed), initial_correct=table[True, True] + lost,
                altered_correct=table[True, True] + gained, lost=lost, gained=gained,
                creator_components=len(components), cluster_sign_p_unadjusted=p)


def conditioning(root: Path, roster: list[dict], audit: list[dict]) -> list[dict]:
    lookup = {(row["source_id"], row["channel"]): row["event"] for row in audit if row["condition"] == "native_input"}
    result = []
    checks = [("audio_caption_bge", "video_caption_bge", "audio", "video"),
              ("image_titan", "video_caption_bge", "image", "video")]
    from scipy.stats import spearmanr
    with np.load(root / "features.npz", allow_pickle=False) as arrays:
        for lkey, rkey, ch1, ch2 in checks:
            if lkey not in arrays.files or rkey not in arrays.files:
                continue
            data = [arrays[key] for key in (lkey, rkey)]
            masks = [np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1) > 0) for v in data]
            concepts, groups, folds = memberships(roster, masks, 2)
            if len(concepts) < 8:
                continue
            selected = np.concatenate([s for fold in folds for s in fold])
            norms = []
            for matrix in data:
                normalized = np.zeros_like(matrix, dtype=float)
                normalized[selected] = normalize(matrix[selected])
                norms.append(normalized)
            a, b = [matrices(v, folds) for v in norms]
            measurements = []
            for i, concept in enumerate(concepts):
                sources = sorted({roster[int(index)]["source_id"] for fold in folds for index in fold[i]})
                heard = [source for source in sources if (source, ch1) in lookup and (source, ch2) in lookup]
                if not heard:
                    continue
                target = sum(lookup[s, ch1] == lookup[s, ch2] == concept for s in heard)
                keep = np.arange(len(concepts)) != i
                agreement = (spearmanr(a[0][i, keep], b[1][i, keep]).statistic +
                             spearmanr(a[1][i, keep], b[0][i, keep]).statistic) / 2
                if np.isfinite(agreement):
                    measurements.append(dict(concept=concept, group=groups[i], measured=len(heard),
                                             jointly_recognized=target, fraction=target/len(heard), anchor_rsa=float(agreement)))
            x = [m["fraction"] for m in measurements]
            y = [m["anchor_rsa"] for m in measurements]
            estimate = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
            result.append(dict(pair=[lkey, rkey], channels=[ch1, ch2], descriptively_correlated=estimate,
                               observations=measurements))
    return result


def run(root: Path) -> dict:
    roster = json.loads((root / "roster.json").read_text(encoding="utf8"))["sources"]
    config = json.loads(Path(__file__).with_name("concepts.json").read_text(encoding="utf8"))
    groups = defaultdict(list)
    for row in roster:
        groups[row["concept"], row["fold"]].append(row)
    selected = [min(rows, key=lambda row: sha(f"observable|{row['source_id']}".encode()))
                for _, rows in sorted(groups.items())]
    audit, controls, failures = [], {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for row in selected:
            source = row["source_id"]
            path = root / "media" / source
            options = list(config[row["group"]])
            rng_for(20261125, "choice-order", source).shuffle(options)
            assets = [(path / CHANNELS["image"], "image", "native_input")]
            for channel, transform in [("audio", audio_pair), ("video", video_pair)]:
                try:
                    folder = root / "interventions" / source
                    folder.mkdir(parents=True, exist_ok=True)
                    clean = folder / CHANNELS[channel]
                    strip(path / CHANNELS[channel], clean, channel)
                    assets.append((clean, channel, "native_input"))
                    initial, altered, check = transform(clean, folder)
                    controls.setdefault(source, {})[channel] = check
                    assets.extend([(initial, channel, "control"), (altered, channel, "altered")])
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    failures.append(dict(source_id=source, channel=channel, condition="transform", error=type(exc).__name__))
            for clip, channel, condition in assets:
                futures[pool.submit(observe, root, clip, channel, options, source, condition)] = (
                    source, channel, condition)
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                audit.append(future.result())
            except (RequestError, ValueError, TypeError, KeyError, OSError) as exc:
                source, channel, condition = futures[future]
                failures.append(dict(source_id=source, channel=channel, condition=condition, error=str(exc)))
            if i % 30 == 0:
                print(f"observability calls {i}/{len(futures)}", flush=True)
    audit.sort(key=lambda row: (row["source_id"], row["channel"], row["condition"]))
    actual = {row["source_id"]: row["concept"] for row in selected}
    recognition = {}
    for channel in CHANNELS:
        items = [row for row in audit if row["condition"] == "native_input" and row["channel"] == channel]
        recognition[channel] = dict(n=len(items), correct=sum(row["event"] == actual[row["source_id"]] for row in items),
                                    unclear=sum(row["event"] == "other_or_unclear" for row in items),
                                    factor_counts={name: sum(row["factors"][name] for row in items) for name in FACTORS})
    report = dict(observer=OBSERVER, selected_source_ids=list(actual), recognition=recognition, record_count=len(audit),
                  phase_scrambling=contrast(selected, audit, "audio", "control", "altered"),
                  frame_permutation=contrast(selected, audit, "video", "control", "altered"),
                  frame_permutation_temporal_order=contrast(selected, audit, "video", "control", "altered", "temporal_order"),
                  conditioning=conditioning(root, roster, audit), controls=controls,
                  failures=sorted(failures, key=lambda row: (row["source_id"], row["channel"], row["condition"])),
                  audit=audit)
    save_json(root / "observability.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({key: result[key] for key in ("recognition", "phase_scrambling", "frame_permutation")}, indent=2))
