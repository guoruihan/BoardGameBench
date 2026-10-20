"""Agent advice -> paired PPO -> development-driven continuation -> fresh audit."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v121.registry import identity, validate_manifest
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import validate_allocation
from boardbench.v130.pipeline import checkpoint_spec, paired
from boardbench.v134 import campaign as orchestration
from .advice import METHODS


def new_manifest(old, old_sha, count=512):
    validate_manifest(old)
    forbidden = set().union(*(set(v) for v in old["sets"].values()))
    rng, fresh = random.Random(13520261021), []
    while len(fresh) < count * 3:
        s = rng.randrange(50000000, 60000000)
        if s not in forbidden:
            fresh.append(s)
            forbidden.add(s)
    sets = deepcopy(old["sets"])
    sets.update(development=fresh[:2 * count], development_health=fresh[:min(32, count)],
                confirmation=fresh[2 * count:])
    result = {"format_version": 1, "sets": sets, "generator_seed": 13520261021,
              "parent_split_sha256": old_sha, "test_state": "unchanged and sealed"}
    result["sha256"] = identity(result)
    validate_manifest(result)
    return result


def immutable(path, data):
    if path.exists() and read_json(path) != data:
        raise ValueError("immutable artifact changed: " + str(path))
    write_json(path, data)


def prepare(repo, root, config):
    parent = checkpoint_spec(repo / config["parent_checkpoint"])
    directory = Path(parent["checkpoint"]).parent
    if read_json(directory / "summary.json")["status"] != "completed":
        raise ValueError("predesignated reactive-a parent must finish normally")
    base = read_json(directory / "config.json")
    if digest(base["split_manifest"]) != base["split_sha256"]:
        raise ValueError("parent split changed")
    manifest = new_manifest(read_json(base["split_manifest"]), base["split_sha256"])
    split = root / "splits.json"
    immutable(split, manifest)
    immutable(root / "config.json", config)
    stage = root / "stage1"
    count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
    jobs = []
    for method in METHODS:
        for replica in range(2):
            name = method + "_" + "ab"[replica]
            trial = {**base, "advice_version": "1.3.5", "method": method, "advice_strength": .6,
                     "arm": "persistent" if method == "sticky" else "reactive",
                     "replicate": replica, "seed": 13501 + replica, "continuation": parent,
                     "split_manifest": str(split), "split_sha256": digest(split),
                     "curve_seeds": manifest["sets"]["development"][:512],
                     "final_seeds": manifest["sets"]["development"][512:],
                     "workers": config["training"]["workers"], "max_seconds": config["stage1_seconds"],
                     "target_actions": count + config["stage1_actions"],
                     "milestones": [count + 1000000, count + config["stage1_actions"]],
                     "trajectory_directory": str(stage / "trials" / name / "trajectories")}
            jobs.append([name, trial])
    immutable(stage / "jobs.json", jobs)
    immutable(stage / "config.json", config)
    (stage / "logs").mkdir(exist_ok=True)


def report(root, state):
    orchestration.report(root, state, title="V135 agent-authored action advice")


def promotion(root):
    jobs = read_json(root / "stage1/jobs.json")
    scores = {name: read_json(root / "stage1/trials" / name / "evaluations/final.json")["mean_score"]
              for name, _ in jobs}
    means = {method: float(np.mean([scores[method + "_" + suffix] for suffix in "ab"])) for method in METHODS}
    winner = max(METHODS[1:], key=lambda m: means[m])
    return {"means": means, "scores": scores, "selected": ["control", winner],
            "winner_minus_control": means[winner] - means["control"],
            "rule": "fixed endpoint development audit512 mean, not peak or confirmation",
            "interpretation": "exploratory continuation even if all variants lose; never implies a proven gain"}


def stage2(root):
    chosen = promotion(root)
    immutable(root / "promotion.json", chosen)
    config, stage, jobs = read_json(root / "config.json"), root / "stage2", []
    for name, trial in read_json(root / "stage1/jobs.json"):
        if trial["method"] not in chosen["selected"]:
            continue
        parent = checkpoint_spec(root / "stage1/trials" / name / "final.pt")
        count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
        jobs.append([name, {**trial, "continuation": parent, "target_actions": count + config["stage2_actions"],
                            "max_seconds": config["stage2_seconds"],
                            "milestones": [count + config["stage2_actions"] // 2, count + config["stage2_actions"]],
                            "trajectory_directory": str(stage / "trials" / name / "trajectories")}])
    immutable(stage / "jobs.json", jobs)
    immutable(stage / "config.json", {**config, "gpu_uuids": config["gpu_uuids"][:2],
                                     "preflight_cpu_core_seconds": 0., "preflight_gpu_seconds": 0.})
    (stage / "logs").mkdir(exist_ok=True)
    return stage


def finalize(root):
    import torch
    from .training import Trainer
    frozen = read_json(root / "finalization.json")
    seeds = read_json(root / "splits.json")["sets"]["confirmation"]
    results, interventions = {}, {}
    for name, spec in frozen["checkpoints"].items():
        c = (read_json(root / "stage1/jobs.json")[0][1] if name == "parent" else
             read_json(Path(spec["checkpoint"]).parent / "config.json"))
        t = Trainer(c, "cuda")
        try:
            if digest(spec["checkpoint"]) != spec["sha256"]:
                raise ValueError("frozen checkpoint hash changed")
            if name == "parent":
                t.import_sampling_continuation(spec)
            else:
                saved = torch.load(spec["checkpoint"], map_location="cuda", weights_only=False)
                if saved["version"] != 135 or saved["source_id"] != source_id() or saved["config"] != c:
                    raise ValueError("frozen source/config/version mismatch")
                t.model.load_state_dict(saved["model"])
                t.manager.load_state_dict(saved["manager"])
            results[name] = t.evaluate(seeds)
            write_json(root / "confirmation" / (name + ".json"), results[name])
            if name != "parent" and c["method"] != "control":
                intact = read_json(Path(spec["checkpoint"]).parent / "evaluations/final.json")
                interventions[name] = {}
                for intervention in ("zero", "shuffle"):
                    t.model.advice_intervention = intervention
                    changed = t.evaluate(c["final_seeds"])
                    comparison = paired(intact, changed)
                    interventions[name][intervention] = comparison
                    write_json(root / "ablations" / f"{name}_{intervention}.json",
                               {"normal": intact, "intervention": changed, "normal_minus_intervention": comparison,
                                "warning": "development-only off-distribution diagnostic; not independent causal proof"})
        finally:
            t.close()
    means, contrasts = {}, {}
    for method in read_json(root / "promotion.json")["selected"]:
        scores, gaps = [], []
        for i, seed in enumerate(seeds):
            if any(r["games"][i]["environment_seed"] != seed for r in results.values()):
                raise ValueError("environment pairing mismatch")
            score = float(np.mean([results[method + "_" + s]["games"][i]["score"] for s in "ab"]))
            baseline = np.mean([results["control_" + s]["games"][i]["score"] for s in "ab"])
            scores.append(score)
            gaps.append(score - baseline)
        means[method] = float(np.mean(scores))
        delta, half = float(np.mean(gaps)), 1.96 * float(np.std(gaps, ddof=1)) / np.sqrt(len(gaps))
        contrasts[method] = {"minus_control": delta, "environment_paired_descriptive95": [delta - half, delta + half],
                             "warning": "conditional on only two training seeds"}
    winner = read_json(root / "promotion.json")["selected"][1]
    useful = (contrasts[winner]["environment_paired_descriptive95"][0] > 0 and
              all(interventions[winner + "_" + s]["zero"]["mean"] > 0 for s in "ab"))
    # This is an analysis decision only: never auto-spend another campaign budget.
    write_json(root / "confirmation.json", {"n": len(seeds), "method_means": means, "contrasts": contrasts,
               "models": {name: r["mean_score"] for name, r in results.items()},
               "paired_vs_parent": {name: paired(r, results["parent"]) for name, r in results.items() if name != "parent"},
               "advice_interventions_development": interventions,
               "provisional_useful_advice_signal": useful,
               "next_direction": "expand strategic candidate quality" if useful else
                                 "inspect target execution and selective guidance before claiming agent benefit",
               "scope": "agent-authored public rules, not online LLM; no test tuning, no search"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "launch", "resume", "guard", "run", "finalize"))
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config")
    a = p.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    if not root.is_relative_to(repo / "outputs/v135"):
        raise ValueError("dedicated V135 directory required")
    if a.mode == "prepare":
        return prepare(repo, root, read_json(a.config))
    config = read_json(root / "config.json")
    if os.uname().nodename != config["host"]:
        raise ValueError("a226 only")
    if a.mode in ("launch", "resume"):
        if a.mode == "launch":
            if (root / "launch.json").exists():
                raise ValueError("already launched")
            gate = read_json(root / "preflight_cost.json")
            if not gate["passed"] or gate["source_id"] != source_id(repo):
                raise ValueError("source-matched CUDA gate required")
            all_cpus = physical_cpus()
            count = 8 * (config["training"]["workers"] + 1) + 1
            if count > min(128, len(all_cpus) // 2):
                raise ValueError("half-machine physical CPU cap")
            cpus = all_cpus[:count]
            validate_allocation(cpus, config["gpu_uuids"], config["training"]["workers"] + 1, 2, 8)
            local = read_json(root / "local_checks.json")
            if not local["passed"]:
                raise ValueError("local regression checks required")
            config.update(preflight_cpu_core_seconds=gate["cpu_core_seconds"] + local["cpu_core_seconds"],
                          preflight_gpu_seconds=gate["gpu_seconds"])
            write_json(root / "config.json", config)
            for relative in source_files(repo):
                dest = root / "source" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, dest)
            write_json(root / "launch.json", {"source_id": source_id(repo), "cpus": cpus,
                       "campaign_module": "boardbench.v135.campaign",
                       "started_unix": time.time(), "deadline_unix": time.time() + config["guard_seconds"]})
        launch = read_json(root / "launch.json")
        if time.time() >= launch["deadline_unix"] - 900:
            raise ValueError("original deadline expired; no renewal")
        env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTHONNOUSERSITE": "1",
               **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        os.execve(sys.executable, [sys.executable, "-u", "-m", "boardbench.v135.campaign", "guard",
                                  "--repo", str(repo), "--root", str(root)], env)
    if a.mode == "guard":
        orchestration.guard(root, repo)
    elif a.mode == "run":
        orchestration.run(root, repo, stage_builder=stage2, reporter=report,
                          training_module="boardbench.v135.training", campaign_module="boardbench.v135.campaign")
    else:
        finalize(root)


if __name__ == "__main__":
    main()
