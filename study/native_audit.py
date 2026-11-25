"""Record the renewed readiness audit and paired-source capacity in EpistemeOS."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from episteme.kernel import Actor, Kernel
from episteme.planning import Planning
from episteme.reporting import export_store, inspect_store
from episteme.search import Search
from episteme.store import Store

from .acquire import save_json, sha
from .native_routes import MODALITIES, PROTOCOL, read_json
from .workflow import PIN, SEED, environment, implementation


def pilot_capacity(roster: dict, budgets: list[int]) -> dict:
    sources = roster["sources"]
    origin = {row["source_id"] for row in sources}
    if len(origin) != len(sources):
        raise ValueError("duplicate source in pilot roster")
    by_fold = Counter(row["fold"] for row in sources)
    by_digest = {fold: {row["source_sha256"] for row in sources if row["fold"] == fold} for fold in ("A", "B")}
    training_ceiling = len(by_digest["A"])
    return dict(total_sources=len(sources), concepts=len({row["concept"] for row in sources}),
                sources_by_fold=dict(by_fold), repeated_raw_source_across_folds=len(by_digest["A"] & by_digest["B"]),
                optimistic_unique_training_pairs_per_dyad=training_ceiling,
                counts_feasible_at_ceiling={str(count): count <= training_ceiling for count in budgets},
                interpretation="Count ceiling on the exploratory source roster with fold B reserved. No training, "
                               "concept holdout, model readiness or independent replication is inferred.")


def record(root: Path, routes: Path, roster: Path, critique: Path,
           replication_check: Path | None = None, provider_control: Path | None = None) -> dict:
    if (root / "episteme" / "state.sqlite3").exists():
        raise ValueError("use a new audit directory for another EpistemeOS run")
    route_data, config = read_json(routes), read_json(PROTOCOL)
    if route_data.get("protocol_sha256") != sha(PROTOCOL.read_bytes()):
        raise ValueError("route audit must match the registered native protocol")
    measured = {row["id"]: row["status"] for row in route_data["candidate_routes"]}
    candidates = config["candidates"]
    if len(measured) != len(candidates) or set(measured) != {row["id"] for row in candidates}:
        raise ValueError("native route evidence must cover exactly the registered candidate identities")
    availability = {k: [row["id"] for row in candidates
                        if row["modality"] == k and measured[row["id"]] == "eligible"] for k in MODALITIES}
    capacity = pilot_capacity(read_json(roster), config["alignment_pair_budgets"])
    native_open = all(len(items) == 2 for items in availability.values())
    if route_data["native_acquisition_ready"] is not native_open:
        raise ValueError("native route readiness summary contradicts the route evidence")
    metrics = dict(native_acquisition_ready=native_open, eligible_family_count=sum(map(len, availability.values())),
                   stable_verified_families=availability,
                   candidate_statuses=measured, alignment_empirics_ready=False, pilot_source_capacity=capacity)
    evidence_paths = {"routes": routes, "pilot_roster": roster, "critical_roles": critique}
    if replication_check is not None:
        verification = read_json(replication_check)
        resolution = verification["taxonomy_test_resolution"]
        metrics["independent_replication_materialized"] = {
            key: verification[key] for key in ("unique_epic_original_recordings", "unique_epic_participants",
                                               "raw_assets_verified", "design_panels", "status")
        }
        metrics["taxonomy_resolution_reaches_first_holm"] = {
            name: row["reaches_first_holm_threshold"] for name, row in resolution["panels"].items()
        }
        fit_pairs = verification["paired_training_capacity_discovery_development_A"]
        metrics["candidate_discovery_training_capacity"] = {
            "unique_recordings": fit_pairs,
            "available_budgets_by_count": {str(budget): budget <= fit_pairs for budget in config["alignment_pair_budgets"]},
        }
        evidence_paths["replication_cohort_verification"] = replication_check
    if provider_control is not None:
        evidence_paths["provider_selection_negative_control"] = provider_control

    with Store(root / "episteme") as store:
        planner = Kernel(store, Actor("native-planner", "planner"))
        planning = Planning(store, planner.actor)
        scope = dict(mode="native_readiness", population="authenticated RouterAI and existing pilot roster",
                     inference="experimental readiness only")
        question = planning.question(
            study_id="multimodal-semantic-geometry-native", statement="Is the original native experiment executable?",
            objective="Verify stable direct native routes and independent source capacity for the existing research question.",
            scope=scope, constraints=["RouterAI is the only model inference API.",
                                     "Empirical alignment requires reproducible native geometry."],
            stopping_criteria=["Verify the eight registered family routes.", "Count unique paired pilot sources."],
        )
        alternatives = [
            planner.hypothesis("The original native experiment is executable through the approved API.",
                               "All eight registered, separately trained family routes are eligible.",
                               "A required family has no verified native route.", scope),
            planner.hypothesis("Provider capacity blocks a subset of the native experiment.",
                               "One or more families fail catalog, modality transport or alias validation.",
                               "Every registered family passes the three route checks.", scope),
        ]
        explanations = planning.explanation_set(question=question, hypotheses=alternatives,
                                               comparison_plan="Audit authenticated routes and count capacity; distinguish both from empirical results.")
        evidence = {}
        for topic, path in evidence_paths.items():
            evidence[topic] = store.put(path.read_bytes())
            planner.expose_data(data=evidence[topic], purpose=f"Native reopen evidence: {topic}")
        inputs = store.put_json(evidence)
        sources = ["native_audit.py", "native_protocol.json", "native_routes.py", "native_cohort.py", "native_measure.py"]
        if replication_check is not None:
            sources.extend(["epic_replication.py", "replication_check.py", "geometry.py"])
        if provider_control is not None:
            sources.extend(["native_provider_control.py", "router.py"])
        impl = implementation(store, sources)
        env = store.put_json(environment())
        protocol = planner.preregister_for_set(
            explanation_set=explanations, design="Read authenticated model-route evidence and verified independent cohort inputs.",
            metric="eligible_family_count", analysis_plan="All eight rows must pass. Cohort counts and null resolution are readiness only.",
            stopping_rule="One route snapshot and one source-capacity audit.", seeds=[SEED], run_limit=2,
            implementation=impl, environment=env, data=inputs, replication_tolerance=0,
        )
        search = Search(store, planner.actor)
        tree = search.register_tree(weights=dict(discrimination=1., uncertainty=.4, coverage=.4, invalidity_risk=1.),
                                    cost_weight=.1, budget=1, cost_unit="readiness_audit",
                                    max_nodes=1, max_depth=1, max_width=1, max_selections=1, seed=SEED)
        node = search.add_node(tree, protocol=protocol, action="baseline",
                               components=dict(discrimination=1., uncertainty=.2, coverage=1., invalidity_risk=0.),
                               estimated_cost=1., rationale="Verify that new evidence can reopen the original experiment.")
        selected = search.select_next(tree)
        if selected["node"] != node:
            raise ValueError("EpistemeOS did not select the readiness audit")
        executor = Kernel(store, Actor("native-audit-executor", "executor"))
        run = executor.start_run(protocol, seed=SEED, implementation=impl, environment=env,
                                 command=["study.native_audit", "read authenticated route evidence and count pilot pairs"])
        executor.finish_run(run, status="completed", outputs={
            "raw_data": inputs, "metrics": store.put_json(metrics),
            "log": store.put_json(dict(scope="Readiness, without native outcome or alignment training.")),
        })
        search.finish_selection(selected["id"], status="completed", actual_cost=1., run=run,
                                reason="Readiness and source capacity recorded separately from scientific claims.")
        export_store(store)
        result = dict(episteme_commit=PIN, readiness=metrics, run=run, state=inspect_store(store))
        save_json(root / "readiness.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--critique", type=Path, required=True)
    parser.add_argument("--roster", type=Path, default=Path("evidence/commons/roster.json"))
    parser.add_argument("--replication-check", type=Path)
    parser.add_argument("--provider-control", type=Path)
    args = parser.parse_args()
    result = record(args.root, args.routes, args.roster, args.critique, args.replication_check, args.provider_control)
    print(json.dumps(result["readiness"]))


if __name__ == "__main__":
    main()
