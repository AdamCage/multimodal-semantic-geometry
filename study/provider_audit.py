"""Audit response model aliases and recompute sensitivity from cached vectors."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .acquire import save_json, sha
from .features import text_vectors
from .geometry import PROTOCOL, adjust, compare, normalize, ranks, rdm
from .robustness import evaluate

BGE = "baai/bge-m3"
ALIAS = "parasail-bge-m3"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def read_batches(root: Path, values: dict[str, str]) -> tuple[dict, dict]:
    groups = defaultdict(list)
    for item, text in values.items():
        groups[text].append(item)
    unique = list(groups)
    aliases, vectors = {}, {}
    for start in range(0, len(unique), 12):
        batch = unique[start:start + 12]
        body = {"model": BGE, "encoding_format": "float", "input": batch}
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        key = sha(b"embeddings\0" + raw)
        path = root / "api" / (key + ".json")
        if not path.is_file():
            continue
        receipt = load(path)
        if receipt["request_sha256"] != key:
            raise ValueError("request receipt has incorrect digest")
        response = receipt.get("response", {})
        for item in response.get("data", []):
            for unit in groups[batch[item["index"]]]:
                aliases[unit] = response["model"]
                vectors[unit] = np.asarray(item["embedding"], dtype="float32")
    return aliases, vectors


def commons_audit(root: Path) -> dict:
    rows = load(root / "roster.json")["sources"]
    original = load(root / "geometry.json")["pairs"]
    texts = load(root / "captions.json")
    spec = dict(text_bge={r["source_id"]: r["source_text"] for r in rows})
    for channel, captions in texts.items():
        spec[f"{channel}_caption_bge"] = {key: val["description"] for key, val in sorted(captions.items())}
    inputs = {name: read_batches(root, values) for name, values in spec.items()}
    output, aliases_by_rep = [], {}
    with np.load(root / "features.npz", allow_pickle=False) as arrays:
        sources = arrays["source_ids"].tolist()
        for name, (aliases, vectors) in inputs.items():
            for i, source in enumerate(sources):
                if np.isfinite(arrays[name][i]).all() and (source not in vectors or
                                                        not np.array_equal(vectors[source], arrays[name][i])):
                    raise ValueError(f"receipt does not match frozen {name} vector")
            aliases_by_rep[name] = aliases
        for pair in original:
            names = [key for key in (pair["left"], pair["right"]) if key in aliases_by_rep]
            if not names or pair["status"] != "measured":
                continue
            left, right = pair["left"], pair["right"]
            restricted = np.asarray([all(aliases_by_rep[name].get(source) == ALIAS for name in names)
                                     for source in sources])
            observed = evaluate(arrays[left], arrays[right], rows, restricted,
                                load(PROTOCOL)["seed"], f"alias:{left}:{right}")
            output.append(dict(pair=pair["pair"], original=pair["outcomes"]["cross_source"]["global_rsa"]["estimate"],
                               **observed))
        qwen_check = evaluate(arrays["image_caption_qwen"], arrays["video_caption_qwen"], rows,
                              np.ones(len(sources), dtype=bool), load(PROTOCOL)["seed"], "alias:qwen-caption-control")
    return dict(response_alias_by_rep=aliases_by_rep, parasail_only=output, qwen_caption_control=qwen_check)


def clotho_audit(root: Path) -> dict:
    cohort = load(root / "clotho-roster.json")["rows"]
    inputs = load(root / "clotho-inputs.json")
    available = set(inputs["sources"])
    human = {f"{row['source_id']}::{i}": cap for row in cohort if row["source_id"] in available
             for i, cap in enumerate(row["captions"])}
    machine = {key: value["description"] for key, value in sorted(inputs["machine_captions"].items())}
    human_alias, _ = read_batches(root, human)
    machine_alias, _ = read_batches(root, machine)
    results = defaultdict(list)
    with np.load(root / "clotho-features.npz", allow_pickle=False) as arrays:
        source_names = arrays["source_ids"].tolist()
        for alias in sorted(set(human_alias.values())):
            supported = {source for source in available
                         if all(human_alias.get(f"{source}::{i}") == alias for i in range(5))}
            for right in ("machine_bge", "machine_qwen", "audio_signal"):
                keep = np.asarray([name in supported and (right != "machine_bge" or machine_alias.get(name) == alias)
                                   for name in source_names])
                keep &= np.isfinite(arrays["human_bge"]).all(axis=1) & np.isfinite(arrays[right]).all(axis=1)
                n = int(keep.sum())
                estimate = None
                if n >= 24:
                    a, b = [ranks(rdm(normalize(arrays[key][keep]))) for key in ("human_bge", right)]
                    estimate = float(np.sum(a * b) / 2)
                results[alias].append(dict(right=right, sources=n, rsa=estimate))
    return dict(human_alias_counts=dict(Counter(human_alias.values())),
                machine_alias_counts=dict(Counter(machine_alias.values())), single_alias=results)


def pinned_sensitivity(root: Path, output: Path) -> dict:
    from .native_routes import PROTOCOL as native_protocol

    roster_path, feature_path, captions_path = (root / name for name in ("roster.json", "features.npz", "captions.json"))
    rows = load(roster_path)["sources"]
    ids = [row["source_id"] for row in rows]
    previous = load(root / "geometry.json")
    if previous["features_sha256"] != sha(feature_path.read_bytes()) or previous["roster_sha256"] != sha(roster_path.read_bytes()):
        raise ValueError("pilot features or roster changed after the original computation")
    inputs = {"text_bge": {row["source_id"]: row["source_text"] for row in rows}}
    inputs["text_qwen"] = inputs["text_bge"]
    for channel, captions in load(captions_path).items():
        inputs[f"{channel}_caption_bge"] = {key: value["description"] for key, value in sorted(captions.items())}
    routes = {"bge" if row["id"] == "bge_m3" else "qwen": row for row in load(native_protocol)["candidates"]
              if row["modality"] == "text"}
    with np.load(feature_path, allow_pickle=False) as frozen:
        if frozen["source_ids"].tolist() != ids:
            raise ValueError("pilot features lost their source order")
        arrays = {key: frozen[key] for key in frozen.files if key != "source_ids"}
    parts, outcomes = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for name, values in inputs.items():
            route = routes[name.rsplit("_", 1)[1]]
            jobs[name] = (route, pool.submit(text_vectors, output, values, name, route["model"], provider=route["provider"]))
        for name, (route, future) in jobs.items():
            vectors, aliases, failures = future.result()
            old = arrays[name]
            revised = np.full_like(old, np.nan)
            expected = {x.casefold() for x in route["response_models"]}
            for i, unit in enumerate(ids):
                if (unit in vectors and aliases[unit].casefold() in expected and np.isfinite(old[i]).all()
                        and np.asarray(vectors[unit]).shape == old[i].shape):
                    revised[i] = vectors[unit]
            arrays[name] = revised
            parts[name] = dict(model=route["model"], provider=route["provider"], aliases=dict(Counter(aliases.values())),
                               sources=int(np.isfinite(revised).all(axis=1).sum()), failures=failures,
                               feature_rows_sha256=sha(revised.astype("<f4").tobytes()))
    plan = load(PROTOCOL)
    pairs = [compare(pair, arrays, rows, plan) for pair in plan["pairs"]]
    adjust(pairs)
    for old, new in zip(previous["pairs"], pairs, strict=True):
        if old["pair"] != new["pair"]:
            raise ValueError("pilot comparison order changed")
        a, b = (value.get("outcomes", {}).get("cross_source", {}) for value in (old, new))
        outcomes.append(dict(pair=new["pair"], changed=bool({new["left"], new["right"]} & inputs.keys()),
                             concepts=len(new.get("concepts", [])), observed={key: b.get(key) for key in ("global_rsa", "blocked_rsa", "within_rsa")},
                             original={key: a.get(key) for key in ("global_rsa", "blocked_rsa", "within_rsa")}))
    report = dict(dataset="same exposed Commons pilot", inputs=parts, pairs=outcomes,
                  original_geometry_sha256=sha((root / "geometry.json").read_bytes()),
                  captions_sha256=sha(captions_path.read_bytes()),
                  note="Provider-fixed text reread. Unchanged channels retain their original embeddings. "
                       "This repeats exposed observations and corrects jointly across the original panel.")
    save_json(output / "pinned-text-sensitivity.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--clotho", type=Path, default=Path(".research/clotho"))
    parser.add_argument("--pinned-output", type=Path)
    args = parser.parse_args()
    if args.pinned_output is not None:
        result = pinned_sensitivity(args.root, args.pinned_output)
        print(json.dumps({key: value["sources"] for key, value in result["inputs"].items()}))
        raise SystemExit(0)
    result = dict(scope="Same cached vectors, restricted by literal returned model name; aliases do not identify checkpoint weights.",
                  commons=commons_audit(args.root), clotho=clotho_audit(args.clotho))
    save_json(args.root / "provider-audit.json", result)
    print(json.dumps(dict(
        commons=[(r["pair"], r["status"], len(r["concepts"]), r.get("cross_source", {}).get("global", {}).get("estimate"))
                 for r in result["commons"]["parasail_only"]],
        qwen_caption_control=result["commons"]["qwen_caption_control"],
        clotho=result["clotho"]), indent=2))
