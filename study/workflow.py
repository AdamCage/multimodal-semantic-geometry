"""Use EpistemeOS to freeze, select, execute, reanalyse, and audit the study."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from episteme.kernel import Actor, Kernel
from episteme.planning import Planning
from episteme.reporting import export_store, inspect_store
from episteme.search import Search
from episteme.store import Store

from .acquire import save_json, sha

PIN = "7f5a1a0e43d7c35196cc8b3c9c74a618aa2559c3"
REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "study/protocol.json"
SEED = 20261125
SCOPE = {
    "population": "bounded exploratory sample of licensed Commons concepts and observed RouterAI model routes",
    "inference": "concept relationship agreement across text, image, environmental audio, and silent video",
    "mode": "exploratory_multimodal_geometry",
}
HYPOTHESES = [
    ("Representations share concept geometry beyond broad taxonomy.",
     "Cross-source ordinal RSA exceeds taxonomy-preserving and within-group permutation nulls.",
     "For accessible, reliable representations, beyond-taxonomy agreement fails to exceed the registered nulls."),
    ("Broad category structure or recording context explains apparent agreement.",
     "Global ordinal RSA exceeds its null more often than within-group RSA, or same-source agreement is stronger.",
     "Within-group agreement survives creator-disjoint comparison after null calibration."),
    ("Text mediation or joint training explains apparent agreement in contextual representations.",
     "Caption-mediated and co-aligned comparisons can exhibit stronger agreement than fixed sensory controls.",
     "Agreement also appears in independently trained native checkpoint families using reliable disjoint inputs."),
]
RUN_FILES = ["acquire.py", "router.py", "features.py", "geometry.py", "roster.py", "concepts.json"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def environment() -> dict:
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                            text=True, check=True).stdout.splitlines()[0]
    return dict(episteme_version=importlib.metadata.version("epistemeos"), episteme_git_revision=PIN,
                python=sys.version.split()[0], platform=platform.system(), machine=platform.machine(),
                numpy=importlib.metadata.version("numpy"), scipy=importlib.metadata.version("scipy"),
                ffmpeg=ffmpeg, execution="trusted local subprocess, same OS user")


def implementation(store: Store, names: list[str]) -> str:
    return store.put_json({name: (REPO / "study" / name).read_text(encoding="utf8") for name in names})


def run_command(args: list[str], timeout: int = 3600) -> dict:
    process = subprocess.run(args, cwd=REPO, check=False, timeout=timeout,
                             encoding="utf8", errors="replace", capture_output=True)
    return dict(argv=args[1:], returncode=process.returncode,
                stdout=process.stdout[-25_000:], stderr=process.stderr[-25_000:])


def readiness(preflight: dict) -> dict:
    items = preflight["catalog"]
    available = set(items["all_model_ids"])
    text_results = {row[0]: row[1] for row in preflight["opens_2"]}
    audio_plus_text = next(row for row in preflight["opens_2"] if row[0] == "audio_plus_text")
    text_control = next(row for row in preflight["opens_2"] if row[0] == "only_text_control")
    expected = {"text": ["baai/bge-m3", "qwen/qwen3-embedding-8b"], "image": ["dino", "mae"],
                "audio": ["beats", "audiomae"], "video": ["videomae", "vjepa"]}
    native = dict(
        text=[model for model, label in zip(expected["text"], ("bge", "qwen8"), strict=True)
              if model in available and text_results[label] == "OK"],
        image=[], audio=[], video=[],
    )
    return dict(catalog_count=items["total"], candidate_families=expected, eligible=native,
                eligible_modality_count=sum(len(v) >= 2 for v in native.values()),
                modality_minimum=4, family_minimum=2,
                same_mixed_audio_and_text_control=(audio_plus_text[3:5] == text_control[3:5]),
                native_gate="open" if all(len(v) >= 2 for v in native.values()) else "blocked",
                applied_gate="blocked", interpretation="availability outcome; no empirical test of native cross-modal geometry")


def preflight_state(store: Store, preflight_path: Path, roles: Path) -> dict:
    planner = Kernel(store, Actor("geometry-planner", "planner"))
    planning = Planning(store, planner.actor)
    question = planning.question(
        study_id="multimodal-semantic-geometry", statement=load(PLAN)["question"],
        objective="Preserve the native research question and bound the reachable pilot by evidence.",
        scope=SCOPE, constraints=["RouterAI is the only model API", "no second-stage utility estimate without reproducible native geometry"],
        stopping_criteria=["Record unavailable modalities without treating them as observations.",
                           "Freeze the contextual portfolio before geometry evaluation."],
    )
    alternatives = [planner.hypothesis(*item, SCOPE) for item in HYPOTHESES]
    expl = planning.explanation_set(question=question, hypotheses=alternatives,
                                    comparison_plan="Separate native, jointly trained, mediated and fixed-signal claims.")
    primary = store.put(preflight_path.read_bytes())
    for path in sorted(roles.glob("*.json")):
        if path.name in {"initial.json", "feasibility.json"}:
            planner.expose_data(data=store.put(path.read_bytes()), purpose=f"RouterAI design roles: {path.stem}")
    env = store.put_json(environment())
    impl = implementation(store, ["workflow.py"])
    protocol = planner.preregister_for_set(
        explanation_set=expl, design="Replay the recorded authenticated RouterAI catalogue and input-format checks.",
        metric="eligible_modality_count", analysis_plan="Count modalities meeting the two-native-family requirement.",
        stopping_rule="One observed snapshot and its finite probe set.", seeds=[SEED], run_limit=2,
        implementation=impl, environment=env, data=primary, replication_tolerance=0,
    )
    search = Search(store, planner.actor)
    tree = search.register_tree(weights=dict(discrimination=1., uncertainty=.4, coverage=.4, invalidity_risk=1.),
                                cost_weight=.1, budget=4, cost_unit="bounded_study_stage",
                                max_nodes=4, max_depth=3, max_width=2, max_selections=4, seed=SEED)
    first = search.add_node(tree, protocol=protocol, action="baseline",
                            components=dict(discrimination=1., uncertainty=.1, coverage=1., invalidity_risk=0.),
                            estimated_cost=1., rationale="Checkpoint admissibility decides whether the original empirical stage can run.")
    decision = search.select_next(tree)
    if decision["node"] != first:
        raise ValueError("EpistemeOS failed to select the availability audit")
    executor = Kernel(store, Actor("availability-executor", "executor"))
    run = executor.start_run(protocol, seed=SEED, implementation=impl, environment=env,
                             command=["study.workflow", "readiness", "recorded RouterAI catalog and probes"])
    observed = readiness(load(preflight_path))
    metrics = store.put_json(observed)
    executor.finish_run(run, status="completed", outputs={
        "raw_data": primary, "metrics": metrics,
        "log": store.put_json(dict(execution="bounded audit of pre-existing probes", evidence_path=preflight_path.name)),
    })
    search.finish_selection(decision["id"], status="completed", actual_cost=1., run=run,
                            reason=f"Availability audit completed; native gate is {observed['native_gate']}.")
    return dict(question=question, explanation_set=expl, tree=tree, readiness_node=first,
                readiness_run=run, readiness=observed)


def register_geometry(store: Store, control: dict, root: Path) -> dict:
    if control["readiness"]["native_gate"] != "blocked":
        raise ValueError("the contextual fallback is gated on missing original native capacity")
    planner = Kernel(store, Actor("geometry-planner", "planner"))
    search = Search(store, planner.actor)
    config = store.put(PLAN.read_bytes())
    roster = store.put((root / "roster.json").read_bytes())
    plan = store.put_json(dict(config_sha256=config, roster_sha256=roster))
    planner.expose_data(data=config, purpose="Frozen ordinal estimands and registered model comparisons")
    planner.expose_data(data=roster, purpose="Frozen Commons licenses, source roster and creator-exclusive folds")
    impl = implementation(store, RUN_FILES)
    env = store.put_json(environment())
    protocol = planner.preregister_for_set(
        explanation_set=control["explanation_set"], design="Run the registered Commons contextual representation comparisons.",
        metric="rsa_estimates_checked", analysis_plan="Use the frozen protocol JSON and source roster. Recheck all overall RSA with SciPy.",
        stopping_rule="One fixed roster and one same-data computation check. Missing features retain error and coverage records.",
        seeds=[SEED], run_limit=2, implementation=impl, environment=env, data=plan, replication_tolerance=0,
    )
    node = search.add_node(control["tree"], protocol=protocol, action="discriminate",
                           parent=control["readiness_node"], estimated_cost=1.,
                           components=dict(discrimination=.8, uncertainty=.7, coverage=.75, invalidity_risk=.5),
                           rationale="Retain interpretable contextual and proxy evidence given the frozen native gate.")
    decision = search.select_next(control["tree"])
    if decision["node"] != node:
        raise ValueError("EpistemeOS failed to select the contextual study")
    return dict(protocol=protocol, node=node, selection=decision["id"],
                implementation=impl, environment=env, roster_sha256=roster, config_sha256=config)


def execute_geometry(store: Store, control: dict, root: Path) -> dict:
    stage = control["geometry"]
    if (implementation(store, RUN_FILES) != stage["implementation"] or sha(PLAN.read_bytes()) != stage["config_sha256"]
            or sha((root / "roster.json").read_bytes()) != stage["roster_sha256"]):
        raise ValueError("frozen study code, configuration, or roster drifted")
    executor = Kernel(store, Actor("geometry-executor", "executor"))
    commands = [[sys.executable, "-u", "-m", f"study.{module}", "--root", str(root)]
                for module in ("features", "geometry")]
    run = executor.start_run(stage["protocol"], seed=SEED, implementation=stage["implementation"],
                             environment=stage["environment"], command=commands[0] + ["then"] + commands[1][1:])
    search = Search(store, Actor("geometry-planner", "planner"))
    logs = []
    try:
        for args in commands:
            log = run_command(args)
            logs.append(log)
            if log["returncode"]:
                raise RuntimeError(f"subprocess returned {log['returncode']}: {' '.join(args[1:])}")
        measured = load(root / "geometry.json")
        estimates = sum(len(item.get("outcomes", {})) for item in measured["pairs"])
        if not estimates:
            raise RuntimeError("no measured comparisons on the frozen roster")
        raw = store.put((root / "features.npz").read_bytes())
        outputs = {
            "raw_data": raw, "metrics": store.put_json(dict(rsa_estimates_checked=estimates)),
            "log": store.put_json(logs), "geometry": store.put((root / "geometry.json").read_bytes()),
            "source_features_receipt": store.put((root / "feature-meta.json").read_bytes()),
            "descriptions": store.put((root / "captions.json").read_bytes()),
        }
        executor.finish_run(run, status="completed", outputs=outputs)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        executor.finish_run(run, status="failed", outputs={"log": store.put_json(logs)}, reason=str(exc))
        search.finish_selection(stage["selection"], status="failed", actual_cost=1.,
                                reason="A contextual-stage command failed; consult the recorded run.", run=run)
        raise
    actor = Kernel(store, Actor("geometry-scipy-check", "replicator"))
    args = [sys.executable, "-u", "-m", "study.check_geometry", "--root", str(root)]
    check_run = actor.start_run(stage["protocol"], seed=SEED, implementation=implementation(store, ["check_geometry.py"]),
                                environment=stage["environment"], command=args, replicate_of=run)
    log = run_command(args)
    if log["returncode"]:
        actor.finish_run(check_run, status="failed", outputs={"log": store.put_json(log)},
                         reason="Same-data reanalysis failed")
        raise RuntimeError(log["stderr"])
    checked = load(root / "reanalysis.json")
    actor.finish_run(check_run, status="completed", outputs={
        "raw_data": raw, "metrics": store.put_json(dict(rsa_estimates_checked=len(checked["outcomes"]))),
        "log": store.put_json(log), "reanalysis": store.put((root / "reanalysis.json").read_bytes()),
    })
    analyst = Kernel(store, Actor("geometry-analyst", "analyst"))
    claim = analyst.claim(protocol=stage["protocol"], statement=(
        f"The bounded contextual stage measured and cross-checked {estimates} overall RSA estimates. "
        "The independently trained four-modality claim remains inconclusive under the observed API route.")
        , scope=SCOPE, evidence=[run, check_run], outcome="inconclusive", inference_mode="exploratory",
        limitations=["Commons media availability and metadata-driven selection constrain generalization.",
                     "Image models include cross-modal training. Audio and video are represented by fixed signals or text-mediated features.",
                     "SciPy checks cached inputs and relies on the existing roster. Provider repeat, new data and independent scientific approval are distinct."])
    search.finish_selection(stage["selection"], status="completed", actual_cost=1., run=run, claim=claim,
                            reason="Frozen contextual computation and separate same-data RSA implementation finished.")
    return dict(run=run, check_run=check_run, claim=claim, gate=analyst.gate(claim))


def followup(store: Store, control: dict, root: Path, clotho: Path, opinion_path: Path) -> dict:
    claim = control["primary"]["claim"]
    planner = Kernel(store, Actor("geometry-planner", "planner"))
    review_info = load(opinion_path)
    sha_opinion = store.put(opinion_path.read_bytes())
    actions = ["Check selection and literal vocabulary on the frozen source cohort.",
               "Mask one information source at a time and record direct observer outcomes.",
               "Recompute the Clotho comparison on its own cached sources.",
               "Repeat affected caption inputs without auxiliary container metadata."]
    planner.expose_data(data=sha_opinion, purpose="RouterAI evidence critique, trusted-local context only")
    reviews = []
    for entry in review_info["roles"]:
        if entry.get("content"):
            actor = Kernel(store, Actor(f"routerai-{entry['role']}-reviewer", "reviewer"))
            basis = actor.gate(claim)["basis_hash"]
            reviews.append(actor.review(claim, verdict="request_changes", rationale=entry["content"],
                                        actions=actions, expected_basis=basis))
    if not reviews:
        raise ValueError("no complete review responses for the follow-up")
    sources = {
        "commons_features": store.put((root / "features.npz").read_bytes()),
        "commons_roster": store.put((root / "roster.json").read_bytes()),
        "clotho_features": store.put((clotho / "clotho-features.npz").read_bytes()),
        "clotho_roster": store.put((clotho / "clotho-roster.json").read_bytes()),
        "roles": sha_opinion,
    }
    inputs = store.put_json(sources)
    env = store.put_json(environment())
    code = implementation(store, RUN_FILES + ["clotho.py", "observability.py", "robustness.py", "metadata_control.py"])
    protocol = planner.preregister_for_set(
        explanation_set=control["explanation_set"], parent=control["geometry"]["protocol"],
        design="Observer interventions, container-metadata control, cohort sensitivity, and a cached separate-dataset audio–text check.",
        metric="followup_checks_reported", analysis_plan="Preserve paired intervention contrasts and the two source populations.",
        stopping_rule="One pass each for observer, metadata control, cohort sensitivity, and offline Clotho computations.",
        seeds=[SEED], run_limit=2, implementation=code, environment=env, data=inputs, replication_tolerance=0,
    )
    search = Search(store, planner.actor)
    node = search.add_node(control["tree"], protocol=protocol, action="robustness",
                           parent=control["geometry"]["node"], estimated_cost=1.,
                           components=dict(discrimination=.8, uncertainty=.85, coverage=.7, invalidity_risk=.5),
                           rationale="Evidence-critical checks can distinguish observational context, lexical mediation, and observer accessibility.")
    selected = search.select_next(control["tree"])
    if selected["node"] != node:
        raise ValueError("EpistemeOS failed to select the follow-up")
    executor = Kernel(store, Actor("geometry-followup-executor", "executor"))
    commands = {module: [sys.executable, "-u", "-m", f"study.{module}", "--root", str(where)]
                for module, where in [("observability", root), ("robustness", root),
                                      ("metadata_control", root), ("clotho", clotho)]}
    commands["clotho"].append("--offline")
    run = executor.start_run(protocol, seed=SEED, implementation=code, environment=env,
                             command=["study.workflow", "followup", *commands])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {module: pool.submit(run_command, args) for module, args in commands.items()}
        logs = {module: future.result() for module, future in jobs.items()}
    passed = [module for module, result in logs.items() if result["returncode"] == 0]
    reports = {"observability": root / "observability.json", "robustness": root / "robustness.json",
               "metadata_control": root / "metadata-control.json",
               "clotho": clotho / "clotho-geometry.json"}
    outputs = {module: store.put(reports[module].read_bytes()) for module in passed}
    outputs["log"] = store.put_json(logs)
    if len(passed) != len(commands):
        executor.finish_run(run, status="failed", outputs=outputs, reason="At least one registered follow-up did not complete.")
        search.finish_selection(selected["id"], status="failed", actual_cost=1.,
                                reason="Retained successful partial evidence and failed follow-up logs.", run=run)
    else:
        outputs["raw_data"] = inputs
        outputs["metrics"] = store.put_json(dict(followup_checks_reported=len(passed)))
        executor.finish_run(run, status="completed", outputs=outputs)
        search.finish_selection(selected["id"], status="completed", actual_cost=1., reason="Follow-up outputs verified and stored.", run=run)
    return dict(protocol=protocol, node=node, selection=selected["id"], run=run, reviews=reviews,
                completed=passed, outputs=outputs, next_action=planner.next_action(claim))


def adapt_caption(store: Store, control: dict, root: Path, opinion_path: Path) -> dict:
    opinions = load(opinion_path)
    proposals = [role["role"] for role in opinions["roles"] if isinstance(role.get("content"), str)
                 and "caption" in role["content"].casefold()
                 and any(word in role["content"].casefold() for word in ("partial", "residual", "condition"))]
    if not proposals:
        return dict(status="no_matching_reviewer_proposal")
    prior = control["followup"]
    planner = Kernel(store, Actor("geometry-planner", "planner"))
    data = store.put_json(dict(
        features=store.put((root / "features.npz").read_bytes()),
        roster=store.put((root / "roster.json").read_bytes()),
        motivating_review=store.put(opinion_path.read_bytes()),
    ))
    env = store.put_json(environment())
    code = implementation(store, ["caption_partial.py", "geometry.py", "acquire.py"])
    protocol = planner.preregister_for_set(
        explanation_set=control["explanation_set"], parent=prior["protocol"],
        design="Post-pilot descriptive stress test of the three direct text–image routes.",
        metric="controlled_pairs_reported",
        analysis_plan="Remove broad category-pair means and image-caption RDM associations. Jointly permute images and their caption axes.",
        stopping_rule="One pass on the same frozen cohort with no added confirmatory inference.",
        seeds=[SEED], run_limit=2, implementation=code, environment=env, data=data, replication_tolerance=0,
    )
    search = Search(store, planner.actor)
    node = search.add_node(control["tree"], protocol=protocol, action="robustness",
                           parent=prior["node"], estimated_cost=1.,
                           components=dict(discrimination=.65, uncertainty=.6, coverage=.45, invalidity_risk=.55),
                           rationale="A RouterAI role requested a caption-conditioned descriptive test after observing the pilot.")
    selected = search.select_next(control["tree"])
    if selected["node"] != node:
        raise ValueError("EpistemeOS failed to select the caption-conditioned sensitivity")
    executor = Kernel(store, Actor("geometry-adaptive-executor", "executor"))
    args = [sys.executable, "-u", "-m", "study.caption_partial", "--root", str(root)]
    run = executor.start_run(protocol, seed=SEED, implementation=code, environment=env, command=args)
    log = run_command(args)
    outputs = {"log": store.put_json(log)}
    if log["returncode"]:
        status = "failed"
        executor.finish_run(run, status=status, outputs=outputs, reason="Caption-conditioning returned a nonzero exit status.")
    else:
        status = "completed"
        report = root / "caption-partial.json"
        count = sum(item["status"] == "measured" for item in load(report)["outcomes"])
        outputs.update(raw_data=data, metrics=store.put_json(dict(controlled_pairs_reported=count)),
                       caption_partial=store.put(report.read_bytes()))
        executor.finish_run(run, status=status, outputs=outputs)
    search.finish_selection(selected["id"], status=status, actual_cost=1., run=run,
                            reason="Caption-conditioned sensitivity reached a terminal execution state.")
    end = search.select_next(control["tree"])
    return dict(status=status, reviews_matching=proposals, protocol=protocol, node=node, selection=selected["id"],
                run=run, outputs=outputs, search_stop=end["reason"])


def export(store: Store, control: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = inspect_store(store)
    summary["root"] = "local EpistemeOS Store"
    save_json(output / "summary.json", dict(episteme_git_revision=PIN, **summary, controller=control))
    save_json(output / "events.json", store.events())
    export_store(store)


def record_postflight(store: Store, root: Path, roles: Path) -> dict:
    planner = Kernel(store, Actor("geometry-planner", "planner"))
    artifacts = {
        "provider_alias_audit": root / "provider-audit.json",
        "closing_critique": roles / "adaptive-closure.json",
    }
    evidence = {}
    for topic, path in artifacts.items():
        digest = store.put(path.read_bytes())
        event = planner.expose_data(data=digest, purpose=f"Postflight provenance and critical reading: {topic}")
        evidence[topic] = dict(sha256=digest, event=event)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "freeze", "run", "followup", "adapt", "postflight", "export"])
    parser.add_argument("--root", type=Path, default=Path(".research/commons-v3"))
    parser.add_argument("--store", type=Path, default=Path(".research/episteme"))
    parser.add_argument("--preflight", type=Path, default=Path(".research/preflight/combined.json"))
    parser.add_argument("--roles", type=Path, default=Path(".research/roles"))
    parser.add_argument("--clotho", type=Path, default=Path(".research/clotho"))
    parser.add_argument("--opinion", type=Path, default=Path(".research/roles/results-review.json"))
    parser.add_argument("--export", type=Path, default=Path(".research/episteme-export"))
    args = parser.parse_args()
    state_path = args.store / "controller.json"
    with Store(args.store) as store:
        if args.action == "audit":
            if store.events():
                raise ValueError("audit would overwrite an existing study")
            state = preflight_state(store, args.preflight, args.roles)
        else:
            state = load(state_path)
            if args.action == "freeze":
                if "geometry" in state:
                    raise ValueError("contextual stage is already frozen")
                state["geometry"] = register_geometry(store, state, args.root)
            elif args.action == "run":
                if "primary" in state:
                    raise ValueError("contextual stage was already run")
                state["primary"] = execute_geometry(store, state, args.root)
            elif args.action == "followup":
                if "followup" in state:
                    raise ValueError("follow-up selection is already terminal")
                state["followup"] = followup(store, state, args.root, args.clotho, args.opinion)
            elif args.action == "adapt":
                if "adaptation" in state:
                    raise ValueError("caption-conditioned sensitivity was already scheduled")
                state["adaptation"] = adapt_caption(store, state, args.root, args.opinion)
            elif args.action == "postflight":
                if "postflight" in state:
                    raise ValueError("postflight artifacts were already recorded")
                state["postflight"] = record_postflight(store, args.root, args.roles)
        save_json(state_path, state)
        export(store, state, args.export)
    print(json.dumps(state, ensure_ascii=True))


if __name__ == "__main__":
    main()
