"""Build a compact factual packet for the independent RouterAI review roles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .workflow import load


def compact(root: Path, clotho: Path, stage: str) -> dict:
    meta = load(root / "feature-meta.json")
    geometry = load(root / "geometry.json")
    cohorts = load(root / "roster.json")["sources"]
    results = []
    for pair in geometry["pairs"]:
        row = {k: pair[k] for k in ("pair", "interpretation", "status", "source_count", "groups") if k in pair}
        if row["status"] == "measured":
            outcome = pair["outcomes"]["cross_source"]
            row.update(
                paired=pair["outcomes"]["paired"]["global_rsa"]["estimate"],
                cross_source=outcome["global_rsa"]["estimate"],
                p_global_holm=outcome["global_rsa"]["p_holm"],
                p_blocked_holm=outcome["blocked_rsa"]["p_holm"],
                within_group=outcome["within_rsa"],
                reliability=pair["reliability"],
                source_sensitivity=pair["intervals"]["cross_source"],
                paired_excess_sensitivity=pair["intervals"]["paired_minus_cross_source"],
                neighbors_k3=outcome["neighborhood"]["3"]["estimate"],
                cka=outcome["cka"]["estimate"],
                metric_error=outcome["metric"]["relative_rmse"],
                constant_error=outcome["metric"]["constant_rmse"],
            )
        results.append(row)
    clotho_data = load(clotho / "clotho-geometry.json")
    result = dict(
        protocols="Ordinal Spearman in creator-disjoint folds, 4999 concept-label null draws with 126 Holm slots, plus separate descriptives.",
        measurements=results,
        roster_sources=len(cohorts),
        concept_source_counts=dict(Counter(row["concept"] for row in cohorts)),
        media_origin=dict(Counter(row["source_text_origin"] for row in cohorts)),
        feature_coverage=meta["coverage"],
        observed_models=meta["observed_models"],
        reanalysis=load(root / "reanalysis.json")["max_absolute_error"],
        clotho=[{k: v for k, v in row.items() if k != "source_ids"} for row in clotho_data["results"]],
        native_constraints="BGE and Qwen independently trained text checkpoints available. No eligible independent image, audio, or video checkpoints through the allowed provider. Raw image calls went to co-aligned image models. Audio and video geometric embedding inputs were caption proxies or fixed signals.",
    )
    integrity = root / "media-integrity.json"
    if integrity.exists():
        inputs = load(integrity)["sources"]
        result["container_audit"] = dict(
            samples=len(inputs), videos_with_auxiliary_streams=sum(len(row["video"]) > 1 for row in inputs),
            video_nonstructural_tag_keys=sorted(set().union(*(row["video_tags"] for row in inputs))
                                               - {"encoder", "major_brand", "minor_version", "compatible_brands"}),
            audio_nonstructural_tag_keys=sorted(set().union(*(row["audio_tags"] for row in inputs)) - {"encoder"}),
        )
    if stage == "final":
        obs = load(root / "observability.json")
        robustness = load(root / "robustness.json")
        result["observability"] = {k: obs[k] for k in
                                   ("recognition", "phase_scrambling", "frame_permutation",
                                    "frame_permutation_temporal_order", "conditioning")}
        result["robustness"] = robustness
        result["metadata_control"] = load(root / "metadata-control.json")
        if (root / "caption-partial.json").exists():
            result["caption_conditioning"] = load(root / "caption-partial.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["results", "final"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--clotho", type=Path, default=Path(".research/clotho"))
    args = parser.parse_args()
    packet = compact(args.root, args.clotho, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
