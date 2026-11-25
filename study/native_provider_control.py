"""Test provider pinning and a negative control at the RouterAI embeddings endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

from .acquire import save_json
from .native_routes import calibrated
from .router import RequestError, request

ROUTES = {
    "baai/bge-m3": (("deepinfra", "parasail"), ("BAAI/bge-m3", "parasail-bge-m3")),
    "qwen/qwen3-embedding-8b": (("deepinfra", "nebius", "siliconflow"), ("Qwen/Qwen3-Embedding-8B",)),
}
CONTROLS = ("A dog barks beside a fence.", "A freight train crosses a mountain pass.")
UNSUPPORTED_PROVIDER = "openai"


def probe(root: Path, model: str, tag: str, accepted: tuple[str, ...]) -> dict:
    values, observed, failures = [], [], []
    for repeat in range(2):
        for i, text in enumerate(CONTROLS):
            body = dict(model=model, input=[text], encoding_format="float",
                        provider=dict(only=[tag], allow_fallbacks=False))
            try:
                result = request(root / f"r{repeat}", "embeddings", body,
                                 context=f"pin:{model}:{tag}:{repeat}:{i}")
                values.append((result["data"][0]["embedding"], result["model"]))
                observed.append({key: result.get(key) for key in ("model", "provider")})
            except (RequestError, KeyError, IndexError, TypeError) as exc:
                failures.append(str(exc))
    proof = calibrated(values, list(accepted)) if len(values) == 4 else None
    verified = len(observed) == 4 and all(str(row["provider"]).casefold() == tag for row in observed)
    return dict(model=model, requested_only=tag, actual=observed, failures=failures,
                calibration=proof, pinned_provider_verified=verified,
                unsupported_provider_rejected=(tag == UNSUPPORTED_PROVIDER and len(failures) == 4
                                               and all(value.startswith("RouterAI HTTP 404:") for value in failures)))


def run(root: Path) -> dict:
    destination = root / "provider-pin.json"
    if destination.is_file():
        raise FileExistsError("use a new provider-control directory to preserve past observations")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(probe, root, model, tag, aliases)
                for model, (tags, aliases) in ROUTES.items() for tag in (*tags, UNSUPPORTED_PROVIDER)]
        rows = [future.result() for future in jobs]
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                  docs="https://routerai.ru/docs/guides/overview/provider-selection", rows=rows)
    save_json(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.root)
    print(json.dumps({f"{row['model']}/{row['requested_only']}": row["pinned_provider_verified"]
                      or row["unsupported_provider_rejected"] for row in value["rows"]}))


if __name__ == "__main__":
    main()
