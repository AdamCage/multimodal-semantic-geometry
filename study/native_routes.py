"""Authenticated evidence for the eligible, single-modality RouterAI routes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .acquire import command, save_json, sha
from .preflight import fixtures
from .router import BASE, RequestError, embedding, media_part

PROTOCOL = Path(__file__).with_name("native_protocol.json")
MODALITIES = ("text", "image", "audio", "video")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def validate_registration(plan: dict) -> None:
    candidates = plan["candidates"]
    expected = {(m, role) for m in MODALITIES for role in ("development", "heldout")}
    cells = {(row["modality"], row["family_split"]) for row in candidates}
    if (plan["family_split"] != ["development", "heldout"] or cells != expected
            or len(candidates) != len(expected)
            or len({row["id"] for row in candidates}) != len(expected)
            or len({row["model"].casefold() for row in candidates}) != len(expected)):
        raise ValueError("expected distinct development and held-out families for each modality")


def get(path: str) -> tuple[dict, dict]:
    req = urllib.request.Request(BASE + path, headers={
        "Authorization": "Bearer " + os.environ["ROUTERAI_API_KEY"], "Accept": "application/json",
    })
    status = None
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=25) as response:
            status, body = response.status, response.read(16_000_000)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return dict(status=status, sha256=sha(body)), value
    except urllib.error.HTTPError as exc:
        return dict(status=exc.code, sha256=sha(exc.read(1_000_000))), {}
    except (OSError, ValueError) as exc:
        return dict(status=status, error=type(exc).__name__), {}


def payloads(root: Path) -> dict:
    inputs = fixtures(root)
    # Both silent videos start with a second of the same black frames. The
    # later content changes, so a vector from just the first frame fails.
    folder = root / "transport-fixtures"
    videos = []
    for name, source in [("green", "color=green:s=128x128:r=6:d=2"),
                         ("test", "testsrc2=s=128x128:r=6:d=2")]:
        path = folder / f"first-frame-control-{name}.mp4"
        if not path.is_file():
            command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                     "color=black:s=128x128:r=6:d=1", "-f", "lavfi", "-i", source,
                     "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
                     "-pix_fmt", "yuv420p", "-an", "-threads", "1", "-c:v", "libx264", str(path)])
        videos.append({"content": [media_part(path)]})
    return {"text": ["A dog barks beside a fence.", "A freight train crosses a mountain pass."],
            "video": videos, **{k: [{"content": [inputs[k + str(i)]]} for i in (1, 2)]
                                for k in ("image", "audio")}}


def calibrated(result: list[tuple[list, str]], accepted_models: list[str]) -> dict:
    if len(result) != 4:
        raise ValueError("transport calibration requires two stimuli with two observations each")
    vectors = [np.asarray(value, dtype=float) for value, _ in result]
    aliases = [alias for _, alias in result]
    shapes = {v.shape for v in vectors}
    allowed = {value.casefold() for value in accepted_models}
    if any(alias.casefold() not in allowed for alias in aliases):
        return dict(transport="unexpected_response_model", aliases=aliases)
    if len(set(aliases)) != 1:
        return dict(transport="response_alias_changed", aliases=aliases)
    if (len(shapes) != 1 or any(v.ndim != 1 or v.size < 2 or
                                 not np.isfinite(v).all() or not (np.linalg.norm(v) > 0)
                                 for v in vectors)):
        return dict(transport="invalid_native_vectors", aliases=aliases)
    norm = [v / np.linalg.norm(v) for v in vectors]
    gap = float(np.linalg.norm(norm[0] - norm[1]))
    drift = max(float(np.linalg.norm(norm[i] - norm[i + 2])) for i in range(2))
    measured = gap > max(1e-5, drift * 5)
    return dict(transport="input_sensitive" if measured else "no_control_separation",
                aliases=aliases, response_model=aliases[0], dimension=vectors[0].size,
                control_gap=gap, max_repeat_drift=drift,
                vector_sha256=[sha(v.astype("<f8").tobytes()) for v in vectors])


def trial(root: Path, candidate: dict, catalog: dict, inputs: list) -> dict:
    model = candidate["model"]
    model_path = "/".join(quote(part, safe="") for part in model.split("/"))
    info, route = get(f"models/{model_path}/endpoints")
    advertised = catalog.get(model.casefold())
    detail = route.get("data", {})
    endpoints = detail.get("endpoints", []) if isinstance(detail, dict) else []
    record = dict(id=candidate["id"], model=model, modality=candidate["modality"],
                  family_split=candidate["family_split"], catalog_architecture=advertised,
                  endpoint_lookup=info, endpoint_names=[r.get("name") for r in endpoints],
                  pinned_provider=candidate.get("provider"))
    if advertised is None and info["status"] == 404:
        return dict(record, status="model_not_found")

    calls = []
    for repeat in range(2):
        for stimulus, asset in enumerate(inputs):
            try:
                vectors, observed = embedding(root / f"repeat-{repeat}", model, [asset],
                                              context=f"native-route:{candidate['id']}:{repeat}:{stimulus}",
                                              provider=candidate.get("provider"))
                calls.append((vectors[0], observed))
            except (RequestError, ValueError, KeyError, TypeError) as exc:
                return dict(record, status="embedding_request_failed", detail=str(exc))
    proof = calibrated(calls, candidate["response_models"])
    if proof["transport"] != "input_sensitive":
        return dict(record, status=proof["transport"], probe=proof)
    if not candidate["api_checkpoint_documentation"]:
        return dict(record, status="checkpoint_mapping_unverified", probe=proof)
    return dict(record, status="eligible", probe=proof)


def audit(root: Path, protocol: Path = PROTOCOL) -> dict:
    plan = read_json(protocol)
    validate_registration(plan)
    receipt, response = get("models")
    catalog = {row["id"].casefold(): row.get("architecture", {}) for row in response.get("data", [])}
    if not catalog:
        raise RuntimeError(f"Authenticated RouterAI catalog could not be verified: {receipt}")
    inputs = payloads(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        work = [pool.submit(trial, root, row, catalog, inputs[row["modality"]])
                for row in plan["candidates"]]
        rows = [task.result() for task in work]
    eligible = {m: [row["id"] for row in rows if row["modality"] == m and row["status"] == "eligible"]
                for m in MODALITIES}
    advertised_av = sorted(key for key, row in catalog.items()
                           if {"audio", "video"} & set(row.get("input_modalities") or [])
                           and "embeddings" in (row.get("output_modalities") or []))
    # The gate requires the named, frozen pair for every modality.
    ready = all(row["status"] == "eligible" for row in rows)
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(protocol.read_bytes()),
                  service=BASE, catalog=receipt, model_count=len(catalog), advertised_av=advertised_av,
                  native_acquisition_ready=ready, eligible=eligible, candidate_routes=rows,
                  alignment_empirics="requires_native_replication")
    save_json(root / "native-route-audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    outcome = audit(args.root)
    print(json.dumps({row["id"]: row["status"] for row in outcome["candidate_routes"]}))


if __name__ == "__main__":
    main()
