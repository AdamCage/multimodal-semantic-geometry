"""Freeze a source-disjoint and concept-disjoint cohort before native inference."""

from __future__ import annotations

import concurrent.futures
import io
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

from .acquire import command, save_json, sha
from .native_routes import MODALITIES, PROTOCOL, read_json, validate_registration

SUFFIXES = {"text": ".txt", "image": ".jpg", "audio": ".wav", "video": ".mp4"}
STUDY_ASSETS = Path(__file__).resolve().parent.parent / "evidence"


def required_string(mapping: dict, name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"missing or unnormalized {name}")
    return value


def digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def checked_asset(base: Path, channel: str, asset: dict) -> tuple[Path, bytes]:
    relative = Path(required_string(asset, "path"))
    path = (base / relative).resolve()
    if (relative.is_absolute() or not path.is_relative_to(base.resolve())
            or path.suffix.lower() != SUFFIXES[channel]):
        raise ValueError(f"invalid {channel} asset path")
    data = path.read_bytes()
    if not data or sha(data) != digest(asset.get("sha256")):
        raise ValueError(f"missing bytes or changed digest for {path.name}")
    return path, data


def validate_assets(base: Path, row: dict) -> None:
    for channel, asset in row["assets"].items():
        path, data = checked_asset(base, channel, asset)
        if channel == "text":
            text = data.decode("utf8").strip()
            if not text:
                raise ValueError("empty native text")
        elif channel == "audio":
            with wave.open(io.BytesIO(data), "rb") as audio:
                if audio.getnframes() == 0:
                    raise ValueError("empty audio")
        else:
            video = channel == "video"
            fields = "stream=codec_type,codec_name:format=format_name,duration"
            content = json.loads(command(["ffprobe", "-v", "error", "-show_entries", fields,
                                          "-of", "json", str(path)]))
            streams = [(item.get("codec_type"), item.get("codec_name")) for item in content["streams"]]
            if video:
                if len(streams) != 1 or streams[0][0] != "video" or "mp4" not in content["format"]["format_name"].split(","):
                    raise ValueError("source video must contain a single silent video stream")
            elif streams != [("video", "mjpeg")]:
                raise ValueError("isolated image must be a JPEG")


def exposed_recordings() -> set[str]:
    file = STUDY_ASSETS / "commons" / "roster.json"
    if not file.is_file():
        return set()
    return {item["source_sha256"] for item in read_json(file)["sources"]}


def freeze_layout(manifest: dict, protocol: dict, previously_exposed: set[str]) -> tuple[dict, list[dict], dict]:
    if manifest.get("version") != 1:
        raise ValueError("unsupported native manifest version")
    datasets = manifest["datasets"]
    if not isinstance(datasets, dict) or {v.get("role") for v in datasets.values()} != {"discovery", "replication"}:
        raise ValueError("at least one discovery dataset and one separate replication dataset are required")
    origins = set()
    for dataset, meta in datasets.items():
        origin = required_string(meta, "origin")
        if origin in origins:
            raise ValueError(f"duplicate dataset origin for {dataset}")
        origins.add(origin)
        required_string(meta, "license")
        required_string(meta, "training_overlap_assessment")

    concepts, data_splits, components, rows = {}, defaultdict(lambda: defaultdict(Counter)), {}, []
    source_ids, recordings = set(), set()
    ancestry = {}

    def find(source: str) -> str:
        while ancestry[source] != source:
            source = ancestry[source]
        return source

    for raw in manifest["sources"]:
        row = dict(raw)
        source = required_string(row, "source_id")
        origin = required_string(row, "dataset")
        concept = required_string(row, "concept")
        group = required_string(row, "group")
        split = required_string(row, "concept_split")
        fold = required_string(row, "fold")
        required_string(row, "license")
        required_string(row, "source_url")
        required_string(row, "text_authorship")
        if origin not in datasets or split not in protocol["concept_split"] or fold not in ("A", "B"):
            raise ValueError(f"invalid split for {source}")
        if concepts.setdefault(concept.casefold(), (concept, split, group)) != (concept, split, group):
            raise ValueError(f"concept crosses partitions or has inconsistent taxonomy: {concept}")
        recording = digest(row.get("recording_sha256"))
        if source in source_ids or recording in recordings:
            raise ValueError(f"recording repeated: {source}")
        if datasets[origin]["role"] == "replication" and recording in previously_exposed:
            raise ValueError(f"replication includes an exposed pilot recording: {source}")
        source_ids.add(source)
        recordings.add(recording)
        assets = row["assets"]
        if set(assets) != set(MODALITIES):
            raise ValueError(f"exactly the four raw-channel assets are required for {source}")
        asset_digests = {channel: digest(asset.get("sha256")) for channel, asset in assets.items()}
        dependencies = row.get("dependency_ids")
        if (not isinstance(dependencies, list) or not dependencies or any(
                not isinstance(value, str) or not value.strip() or value != value.strip() for value in dependencies)):
            raise ValueError(f"source grouping is missing for {source}")
        identity_keys = {("declared", value.casefold()) for value in dependencies}
        identity_keys.update((channel, asset_digests[channel]) for channel in MODALITIES if channel != "text")
        partition = (origin, split, fold)
        ancestry[source] = source
        for key in identity_keys:
            prior, previous_partition = components.setdefault(key, (source, partition))
            if previous_partition != partition:
                raise ValueError(f"shared source component crosses a dataset, concept split or recording fold: {source}")
            ancestry[find(source)] = find(prior)
        data_splits[(origin, split)][concept][fold] += 1
        rows.append(row)

    panels = {}
    min_count = protocol["minimum_sources_per_concept_fold"]
    for dataset in datasets:
        for role in protocol["concept_split"]:
            pool = data_splits[dataset, role]
            deficient = [concept for concept, counts in pool.items() if min(counts["A"], counts["B"]) < min_count]
            if deficient or len(pool) < protocol["minimum_concepts_per_panel"]:
                raise ValueError(f"insufficient paired- and cross-source concept coverage in {dataset}/{role}")
            panels[f"{dataset}/{role}"] = {"concepts": len(pool), "sources": sum(sum(c.values()) for c in pool.values())}

    grouped = defaultdict(list)
    for row in rows:
        grouped[find(row["source_id"])].append(row["source_id"])
    for row in rows:
        component = sorted(grouped[find(row["source_id"])])
        row["creator_component"] = sha(json.dumps(component, ensure_ascii=False).encode())
    return datasets, sorted(rows, key=lambda row: row["source_id"]), panels


def verified_routes(audit: dict, protocol: dict, protocol_digest: str) -> dict:
    if audit.get("protocol_sha256") != protocol_digest or not audit.get("native_acquisition_ready"):
        raise ValueError("direct native acquisition is blocked by the RouterAI route audit")
    validate_registration(protocol)
    observed = audit["candidate_routes"]
    rows = {value["id"]: value for value in observed}
    if len(rows) != len(observed) or set(rows) != {c["id"] for c in protocol["candidates"]}:
        raise ValueError("unexpected or duplicate route identities")
    registry = {}
    for candidate in protocol["candidates"]:
        entry = rows.get(candidate["id"], {})
        if any(entry.get(k) != candidate[k] for k in ("modality", "family_split", "model")):
            raise ValueError(f"route identity changed for {candidate['id']}")
        if entry.get("pinned_provider") != candidate.get("provider"):
            raise ValueError(f"pinned provider changed for {candidate['id']}")
        proof = entry.get("probe", {})
        alias = proof.get("response_model")
        dim = proof.get("dimension")
        if (entry.get("status") != "eligible" or not candidate["api_checkpoint_documentation"]
                or proof.get("transport") != "input_sensitive" or not isinstance(dim, int) or dim < 2
                or not isinstance(alias, str)
                or alias.casefold() not in {a.casefold() for a in candidate["response_models"]}):
            raise ValueError(f"route evidence incomplete for {candidate['id']}")
        registry[candidate["id"]] = dict(candidate, response_model=alias, dimension=dim)
    return registry


def freeze(root: Path, manifest: Path, audit: Path) -> dict:
    destination = root / "native-freeze.json"
    if destination.exists():
        raise ValueError("native cohort already frozen")
    config = read_json(PROTOCOL)
    protocol_digest = sha(PROTOCOL.read_bytes())
    analysis = PROTOCOL.with_name(config["analysis_protocol"])
    registry = verified_routes(read_json(audit), config, protocol_digest)
    dataset_info, sources, panels = freeze_layout(read_json(manifest), config, exposed_recordings())
    data_root = manifest.resolve().parent
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda row: validate_assets(data_root, row), sources))
    lock = dict(version=1, route_audit_sha256=sha(audit.read_bytes()), native_protocol_sha256=protocol_digest,
                analysis_protocol_sha256=sha(analysis.read_bytes()), manifest_sha256=sha(manifest.read_bytes()),
                models=registry, datasets=dataset_info, sources=sources, panels=panels)
    save_json(destination, lock)
    return lock
