"""Validation-only selection, portable policy export and frozen Runner tests."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time
from types import SimpleNamespace

from boardbench.artifacts.store import (ArtifactStore, digest, read_json, save_checkpoint,
                                       source_id, write_json)
from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.runner.config import normalize
from boardbench.runner.context import empty_resources
from boardbench.runner.engine import evaluate
from boardbench.solvers.board import BoardSolver


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*q
    low = int(position)
    return values[low]+(values[min(low+1, len(values)-1)]-values[low])*(position-low)


def metrics(episodes, times, loading=()):
    scores = [e["score"] for e in episodes if e["status"] == "completed"]
    return {"episodes": len(episodes), "completed": len(scores),
            "completion_rate": len(scores)/len(episodes) if episodes else 0.,
            "score_mean": statistics.mean(scores) if scores else None,
            "score_std": statistics.pstdev(scores) if scores else None,
            "failed": sum(e["status"] == "failed" for e in episodes),
            "truncated": sum(e["status"] == "truncated" for e in episodes),
            "decision_p50_seconds": percentile(times, .5), "decision_p95_seconds": percentile(times, .95),
            "mean_game_decision_seconds": statistics.mean([e["resources"]["decision_seconds"] for e in episodes]) if episodes else None,
            "mean_simulation_steps": statistics.mean([e["resources"]["simulation_steps"] for e in episodes]) if episodes else None,
            "mean_model_load_seconds": statistics.mean(loading) if loading else None,
            "hardware": platform.node(), "device": "cpu", "threads": 2,
            "score_policy": "only completed scores; failures/truncations retained separately"}


def candidates(training):
    items = [{"id": "random", "method": "random", "params": {"method": "random"}},
             {"id": "heuristic", "method": "heuristic", "params": {"method": "heuristic"}}]
    for width, depth in ((2, 2), (4, 3)):
        items.append({"id": f"search_w{width}_d{depth}", "method": "search",
                      "params": {"method": "search", "width": width, "rollouts": 2,
                                 "depth": depth, "max_simulation_steps": width*2*depth}})
    for checkpoint in read_json(Path(training)/"checkpoints.json"):
        items.append({"id": "rl_"+Path(checkpoint["path"]).stem, "method": "rl", "params": {},
                      "weights": str((Path(training)/checkpoint["path"]).resolve()),
                      "weights_sha256": checkpoint["sha256"], "training": checkpoint})
    return items


def build_solver(task, candidate):
    if candidate["method"] == "rl":
        from boardbench.solvers.neural import NeuralSolver
        if digest(candidate["weights"]) != candidate["weights_sha256"]:
            raise ValueError("training weights changed since candidate definition")
        solver = NeuralSolver(90210, task, hidden=candidate.get("hidden", 128), device="cpu")
        solver.load_weights(candidate["weights"])
        return solver
    return BoardSolver(90210, **candidate["params"])


def validate(task, training, output, extra_training=None):
    import torch
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    from boardbench.artifacts.store import source_files, source_root
    for relative in source_files():
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root()/relative, destination)
    validation_code_version = source_id(out / "source")
    splits = read_json(Path(training)/"splits.json")
    write_json(out / "splits.json", splits)
    definitions = candidates(training)
    if extra_training:
        existing = {c["id"] for c in definitions}
        definitions += [c for c in candidates(extra_training) if c["id"] not in existing and c["method"] == "rl"]
    summaries = {}
    for candidate in definitions:
        times, episodes, loading = [], [], []
        records = out / (candidate["id"]+".jsonl")
        with records.open("x") as stream:
            for seed in splits["validation"]:
                started = time.perf_counter()
                solver = build_solver(task, candidate)
                loading.append(time.perf_counter()-started)
                env = TASKS[task].factory()
                obs, _ = env.reset(seed)
                simulation_steps, decision_seconds = [0], 0.
                def count():
                    simulation_steps[0] += 1
                ctx = SimpleNamespace(task_spec=TASKS[task].spec, reference_simulator=SearchAccess(env.fork, count))
                status, score, error = "truncated", None, None
                try:
                    for _ in range(256):
                        policy_obs = deepcopy(obs)
                        tick = time.perf_counter()
                        action = solver.decide(policy_obs, ctx)
                        elapsed = time.perf_counter()-tick
                        times.append(elapsed)
                        decision_seconds += elapsed
                        obs = env.step(action).observation
                        if obs["terminated"]:
                            status, score = "completed", obs["score"]
                            break
                except Exception as exc:
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
                record = {"seed": seed, "status": status, "score": score, "error": error,
                          "resources": {"simulation_steps": simulation_steps[0], "decision_seconds": decision_seconds}}
                episodes.append(record)
                stream.write(json.dumps(record)+"\n")
                stream.flush()
        summaries[candidate["id"]] = metrics(episodes, times, loading)
        write_json(out / "summaries.json", summaries)
        print(json.dumps({"candidate": candidate["id"], **summaries[candidate["id"]]}), flush=True)
    selected = []
    for method in ("random", "heuristic", "search", "rl"):
        pool = [c for c in definitions if c["method"] == method
                and c["id"] != "rl_untrained" and summaries[c["id"]]["completed"] == len(splits["validation"])]
        if not pool:
            raise RuntimeError(f"no fully evaluated candidate for {method}")
        chosen = max(pool, key=lambda c: (summaries[c["id"]]["score_mean"], -summaries[c["id"]]["decision_p95_seconds"]))
        selected.append(chosen)
    selected += [c for c in definitions if c["id"] == "rl_untrained"]
    selection = {"task_id": task, "rules_version": TASKS[task].spec["version"],
                 "source_id": validation_code_version, "selection_split": "validation", "test_used": False,
                 "selected": selected, "summaries": summaries, "splits": splits,
                 "training_config": read_json(Path(training)/"config.json")}
    write_json(out / "selection.json", selection)
    return selection


def export_policy(task, candidate, validation_summary, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    solver = build_solver(task, candidate)
    params = ({"task_id": task, "hidden": solver.hidden, "device": "cpu"} if candidate["method"] == "rl"
              else candidate["params"])
    config = normalize({"task": {"id": task, "version": TASKS[task].spec["version"]},
                        "solver": {"id": "ppo" if candidate["method"] == "rl" else "board", "params": params},
                        "episodes": 1, "seed": 90210, "model": {"max_calls": 0},
                        "limits": {"max_action_requests": 256, "max_actions_per_episode": 256, "max_wall_seconds": 120},
                        "capabilities": {"reference_simulator": candidate["method"] == "search"}})
    store = ArtifactStore(root / "export", config)
    try:
        checkpoint = save_checkpoint(store, solver, config, {"completed_episodes": 0, "started_episodes": 0},
                                     empty_resources(), store.root / "workspace")
    finally:
        store.close()
    artifact = checkpoint / "solver" / ("weights.pt" if candidate["method"] == "rl" else "board_solver.json")
    write_json(root / "config.json", config)
    write_json(root / "validation.json", validation_summary)
    training = candidate.get("training", {})
    manifest = {"policy_id": candidate["id"], "environment_id": task,
                "rules_version": TASKS[task].spec["version"], "method": candidate["method"],
                "label": candidate["id"], "code_version": source_id(),
                "training_code_version": training.get("code_version"),
                "config_path": "config.json", "config_sha256": digest(root / "config.json"),
                "artifact_path": str(artifact.relative_to(root.resolve())), "artifact_sha256": digest(artifact),
                "checkpoint_path": str(checkpoint.relative_to(root.resolve())), "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
                "training_steps": training.get("environment_steps"), "training_seed": training.get("training_seed"),
                "training": training or None, "selection_split": "validation",
                "validation_summary_path": "validation.json", "validation_mean": validation_summary["score_mean"],
                "inference_device": "cpu", "difficulty": None}
    write_json(root / "manifest.json", manifest)
    return checkpoint


def frozen_test(selection_path, output, policy_directory):
    import torch
    torch.set_num_threads(2)
    selection = read_json(selection_path)
    if selection["selection_split"] != "validation" or selection["test_used"]:
        raise ValueError("expected frozen validation-only selection")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_selection.json", selection)
    write_json(out / "selection_provenance.json", {"input_sha256": digest(selection_path), "runtime_source_id": source_id()})
    results = {}
    for candidate in selection["selected"]:
        cid = candidate["id"]
        checkpoint = export_policy(selection["task_id"], candidate, selection["summaries"][cid], Path(policy_directory)/cid)
        evaluation = {"episode_seeds": selection["splits"]["test"],
                      "per_episode_limits": {"max_action_requests": 256, "max_wall_seconds": 120, "max_model_calls": 0}}
        write_json(out / (cid+"_eval_config.json"), evaluation)
        run_dir = out / cid
        summary = evaluate(checkpoint, evaluation, run_dir)
        episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
        times, loads, illegal, pure_per_episode = [], [], 0, {}
        with (run_dir / "events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["kind"] == "decision_end":
                    pure = event["solver_seconds"]
                    times.append(pure)
                    ep = event["episode"]
                    pure_per_episode[ep] = pure_per_episode.get(ep, 0.)+pure
                if event["kind"] == "hook_end" and event.get("hook") == "load":
                    loads.append(event["elapsed_seconds"])
                illegal += event["kind"] == "action_error"
        for episode in episodes:
            episode["resources"]["decision_seconds"] = pure_per_episode.get(episode["episode"], 0.)
        results[cid] = {**metrics(episodes, times, loads), "illegal_actions": illegal,
                        "runner_status": summary["status"], "historical_training": candidate.get("training")}
        write_json(out / "results.json", results)
        print(json.dumps({"policy": cid, **results[cid]}), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--task", required=True, choices=("micro_tiles", "take_it_easy", "harmonies"))
    val.add_argument("--training", required=True)
    val.add_argument("--extra-training")
    test = sub.add_parser("test")
    test.add_argument("--selection", required=True)
    test.add_argument("--policies", required=True)
    for item in (val, test):
        item.add_argument("--out", required=True)
    args = p.parse_args()
    if args.phase == "validate":
        validate(args.task, args.training, args.out, args.extra_training)
    else:
        frozen_test(args.selection, args.out, args.policies)


if __name__ == "__main__":
    main()
