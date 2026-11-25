"""Export the fixed reports, provenance, feature snapshots, and a checksum manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .acquire import save_json, sha

COMMONS = (
    "manifest.json", "selection-audit.json", "roster.json", "features.npz",
    "feature-meta.json", "captions.json", "geometry.json", "reanalysis.json",
    "observability.json", "robustness.json", "metadata-control.json", "caption-partial.json", "provider-audit.json",
)
CLOTHO = ("clotho-roster.json", "clotho-inputs.json", "clotho-features.npz",
          "clotho-feature-meta.json", "clotho-geometry.json")


def api_summary(roots: dict[str, Path]) -> dict:
    totals = defaultdict(Counter)
    for stage, root in roots.items():
        totals_for_root = totals[stage]
        for path in sorted((root / "api").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf8"))
            totals_for_root["unique_request_records"] += 1
            attempts = record.get("attempts", [])
            totals_for_root["attempts"] += len(attempts)
            for attempt in attempts:
                totals_for_root["http_" + str(attempt["status"]) if "status" in attempt else "transport_error"] += 1
            response = record.get("response") or {}
            if response:
                totals_for_root["responses"] += 1
            usage = response.get("usage") or {}
            for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]:
                if isinstance(usage.get(key), (int, float)):
                    totals_for_root[f"reported_{key}"] += usage[key]
                    totals_for_root[f"responses_reporting_{key}"] += 1
    return dict(scope="local request receipts only; provider token and cost fields when returned",
                by_stage=dict(totals))


def bundle(scratch: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    preflight = scratch / "preflight/combined.json"
    repeat = scratch / "transport-audit/observed-transport.json"
    first_pass = [(scratch / "clotho/first-pass" / name, "first-pass-" + name.removeprefix("clotho-"))
                  for name in ("clotho-geometry.json", "clotho-inputs.json")]
    episteme = scratch / "episteme-export"
    documented = {
        "commons": [(scratch / "commons-v3" / name, name) for name in COMMONS],
        "clotho": [(scratch / "clotho" / name, name) for name in CLOTHO]
        + [(source, target) for source, target in first_pass if source.is_file()],
        "routerai": [(preflight if preflight.is_file() else repeat, "preflight.json"),
                     (repeat, "repeat-transport.json")],
        "episteme": [(episteme / name, name) for name in ("summary.json", "events.json")]
        + ([(episteme / "graph.json", "graph.json")] if (episteme / "graph.json").is_file() else []),
        "roles": [(path, path.name) for path in sorted((scratch / "roles").glob("*.json"))],
    }
    native = scratch / "native-reopen"
    native_artifacts = {
        "route-discovery.json": "route-discovery.json",
        "method-critique.json": "method-critique.json",
        "live/native-route-audit.json": "native-route-audit.json",
        "ledger-reviewed/readiness.json": "readiness.json",
        "ledger-reviewed/graph.json": "graph.json",
        "mechanical-verification.json": "mechanical-verification.json",
        "discovery/discovery-candidate.json": "discovery-candidate.json",
        "discovery/discovery-summary.json": "discovery-summary.json",
        "data-audit/avcaps.json": "avcaps.json",
    }
    if (native / "ledger-reviewed/readiness.json").is_file():
        documented["native-reopen"] = [(native / source, target) for source, target in native_artifacts.items()]
    replication = scratch / "replication-lab"
    expanded = scratch / "replication-expanded"
    discovery_expansion = scratch / "commons-expansion"
    replication_artifacts = {
        replication / "avcaps-provenance.json": "avcaps-provenance.json",
        replication / "epic-selection.json": "epic-selection.json",
        replication / "epic-candidate.json": "epic-candidate.json",
        replication / "epic-acquisition-failures.json": "epic-acquisition-failures.json",
        replication / "replication-check.json": "replication-check.json",
        replication / "expanded-mechanical.json": "synthetic-layout-verification.json",
        replication / "roles/independent-replication.json": "independent-replication-review.json",
        replication / "pinned-pilot/pinned-text-sensitivity.json": "pinned-text-sensitivity.json",
        native / "provider-control-reviewed/provider-pin.json": "provider-pin.json",
        native / "live-pinned/native-route-audit.json": "native-route-audit-pinned.json",
        native / "live-expanded/native-route-audit.json": "native-route-audit-expanded.json",
        replication / "ledger/readiness.json": "readiness.json",
        replication / "ledger/graph.json": "graph.json",
        expanded / "epic-selection.json": "epic-expanded-selection.json",
        expanded / "epic-candidate.json": "epic-expanded-candidate.json",
        expanded / "replication-check.json": "epic-expanded-check.json",
        expanded / "synthetic-exact-verification.json": "epic-expanded-synthetic-verification.json",
        expanded / "roles/expanded-design.json": "expanded-design-review.json",
        discovery_expansion / "commons-inventory-extension.json": "commons-inventory-extension.json",
        scratch / "native-discovery-expanded/initial/discovery-summary.json": "commons-expanded-initial-summary.json",
        scratch / "native-discovery-expanded/supplemented/discovery-summary.json": "commons-expanded-summary.json",
        scratch / "native-discovery-expanded/supplemented/discovery-candidate.json": "commons-expanded-candidate.json",
        expanded / "full-expanded-check.json": "full-expanded-check.json",
        expanded / "full-expanded-synthetic.json": "full-expanded-synthetic.json",
        expanded / "full-expanded-gate-check.json": "full-expanded-gate-check.json",
        expanded / "ledger/readiness.json": "full-expanded-readiness.json",
        expanded / "ledger/graph.json": "full-expanded-graph.json",
    }
    if (replication / "replication-check.json").is_file():
        documented["replication"] = [(path, name) for path, name in replication_artifacts.items() if path.is_file()]
    for section, entries in documented.items():
        folder = output / section
        folder.mkdir(exist_ok=True)
        for original, target in entries:
            shutil.copyfile(original, folder / target)
    z = output / "commons"
    if json.loads((z / "geometry.json").read_text(encoding="utf8"))["features_sha256"] != sha(
            (z / "features.npz").read_bytes()):
        raise ValueError("bundle does not contain the measured Commons feature snapshot")
    z = output / "clotho"
    if json.loads((z / "clotho-geometry.json").read_text(encoding="utf8"))["features_sha256"] != sha(
            (z / "clotho-features.npz").read_bytes()):
        raise ValueError("bundle does not contain the measured Clotho feature snapshot")
    usage_roots = {name: scratch / name for name in ("commons-v3", "clotho", "roles", "transport-audit", "preflight")}
    if (native / "api").is_dir():
        usage_roots.update({"native-reopen": native, "repeat-0": native / "live/repeat-0", "repeat-1": native / "live/repeat-1"})
    usage_roots.update({"replication-roles": replication / "roles", "pinned-pilot": replication / "pinned-pilot",
                        "expanded-design-roles": expanded / "roles", "expanded-commons-screening": discovery_expansion,
                        "pinned-route-repeat-0": native / "live-pinned/repeat-0", "pinned-route-repeat-1": native / "live-pinned/repeat-1",
                        "expanded-route-repeat-0": native / "live-expanded/repeat-0", "expanded-route-repeat-1": native / "live-expanded/repeat-1",
                        "provider-control-repeat-0": native / "provider-control-reviewed/r0",
                        "provider-control-repeat-1": native / "provider-control-reviewed/r1"})
    save_json(output / "routerai/usage.json", api_summary(usage_roots))
    digest_path = output / "SHA256SUMS"
    rows = [f"{sha(path.read_bytes())}  {path.relative_to(output).as_posix()}"
            for path in sorted(output.rglob("*")) if path.is_file() and path != digest_path]
    digest_path.write_text("\n".join(rows) + "\n", encoding="utf8")


def verify(root: Path) -> None:
    for row in (root / "SHA256SUMS").read_text(encoding="utf8").splitlines():
        digest, path = row.split("  ", maxsplit=1)
        if sha((root / path).read_bytes()) != digest:
            raise ValueError(f"checksum mismatch: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path(".research"))
    parser.add_argument("--output", type=Path, default=Path("evidence"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        bundle(args.scratch, args.output)
    verify(args.output)
