"""Bounded two-stage exploration with development-only promotion and final audit."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v121.registry import identity, validate_manifest
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import execute, validate_allocation
from boardbench.v130.pipeline import checkpoint_spec, paired
from boardbench.v133.campaign import wait_resources
from .plans import ARMS


def new_manifest(old, old_sha, count=512):
    validate_manifest(old)
    forbidden = set().union(*(set(v) for v in old["sets"].values()))
    rng, fresh = random.Random(13420261021), []
    while len(fresh) < 3 * count:
        seed = rng.randrange(40000000, 50000000)  # Disjoint from all earlier V132/V133 fresh ranges.
        if seed not in forbidden:
            fresh.append(seed)
            forbidden.add(seed)
    sets = deepcopy(old["sets"])
    sets.update(development=fresh[:2 * count], development_health=fresh[:32], confirmation=fresh[2 * count:])
    result = {"format_version": 1, "sets": sets, "generator_seed": 13420261021,
              "parent_split_sha256": old_sha, "test_state": "unchanged and sealed"}
    result["sha256"] = identity(result)
    validate_manifest(result)
    return result


def prepare(repo, root, config):
    root.mkdir(parents=True, exist_ok=True)
    parent = checkpoint_spec(repo / config["parent_checkpoint"])
    if read_json(Path(parent["checkpoint"]).parent / "summary.json")["status"] != "completed":
        raise ValueError("predesignated PPO-a parent must finish normally")
    base = read_json(Path(parent["checkpoint"]).parent / "config.json")
    old = read_json(base["split_manifest"])
    manifest = new_manifest(old, base["split_sha256"])
    split = root / "splits.json"
    if split.exists() and read_json(split) != manifest:
        raise ValueError("immutable split")
    write_json(split, manifest)
    write_json(root / "config.json", config)
    stage = root / "stage1"
    stage.mkdir(exist_ok=True)
    (stage / "logs").mkdir(exist_ok=True)
    count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
    jobs = []
    for arm in ARMS:
        for replicate in range(2):
            name = arm + "_" + "ab"[replicate]
            trial = {**base, "planning_version": "1.3.4", "arm": arm, "replicate": replicate,
                     "seed": 13401 + replicate, "continuation": parent, "initial_checkpoint": None,
                     "initial_sha256": None, "split_manifest": str(split), "split_sha256": digest(split),
                     "curve_seeds": manifest["sets"]["development"][:512],
                     "final_seeds": manifest["sets"]["development"][512:],
                     "manager_lr": .00003, "plan_horizon": 2, "bc_weight": 0., "self_imitation_weight": 0.,
                     "workers": config["training"]["workers"], "max_seconds": config["stage1_seconds"],
                     "target_actions": count + config["stage1_actions"],
                     "milestones": [count + 1000000, count + config["stage1_actions"]],
                     "trajectory_directory": str(stage / "trials" / name / "trajectories")}
            jobs.append((name, trial))
    if (stage / "jobs.json").exists() and read_json(stage / "jobs.json") != [list(j) for j in jobs]:
        raise ValueError("immutable jobs")
    write_json(stage / "jobs.json", jobs)
    write_json(stage / "config.json", config)


def report(root, state, *, title="V134 public-plan RL"):
    rows, curves = [], {}
    jobs = read_json(root / "jobs.json")
    for name, config in jobs:
        trial = root / "trials" / name
        start = read_json(Path(config["continuation"]["checkpoint"]).with_suffix(".json"))["ppo_actions"]
        p = read_json(trial / "progress.json") if (trial / "progress.json").exists() else {}
        points = []
        for file in (trial / "evaluations").glob("*.json"):
            m = read_json(file)
            if m["label"] != "final":
                points.append({"actions": m["ppo_actions"] - start, "score": m["mean_score"],
                               "animal_score": m["animal_mean"]})
        points.sort(key=lambda v: v["actions"])
        curves[name] = points
        rows.append({"trial": name, "phase": p.get("phase", "pending"),
                     "additional_actions": max(0, p.get("ppo_actions", start) - start),
                     "latest_score": points[-1]["score"] if points else None,
                     "heartbeat_age": time.time() - p["updated_unix"] if p else None})
    write_json(root / "results.json", rows)
    write_json(root / "curves.json", curves)
    lines = ["# " + title, "", f"State: {state['phase']}",
             "Development512; fixed endpoints, no search or altered rewards.",
             "", "|trial|phase|additional actions|latest score|", "|---|---|---:|---:|"]
    lines += [f"|{r['trial']}|{r['phase']}|{r['additional_actions']}|{r['latest_score']}|" for r in rows]
    lines += ["", f"Resources: {state.get('cost', {})}"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    colors = ["#2563eb", "#60a5fa", "#dc2626", "#f87171", "#059669", "#34d399", "#7c3aed", "#a78bfa"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="450">',
           '<rect width="950" height="450" fill="white"/>',
           f'<text x="40" y="25">{title}: development512, additional actions per stage</text>']
    maximum = max(c["target_actions"] - read_json(Path(c["continuation"]["checkpoint"]).with_suffix(".json"))["ppo_actions"] for _, c in jobs)
    for score in range(80, 126, 5):
        y = 400 - (score - 75) * 6
        svg.append(f'<path d="M40 {y} H720" stroke="#ddd"/><text x="5" y="{y}">{score}</text>')
    for i, (name, points) in enumerate(curves.items()):
        coords = " ".join(f'{40+680*p["actions"]/maximum},{400-(p["score"]-75)*6}' for p in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i]}"/>')
        svg.append(f'<text x="735" y="{60+i*25}" fill="{colors[i]}">{name}</text>')
    svg.append("</svg>")
    (root / "curves.svg").write_text("\n".join(svg))


def promote(root):
    """Endpoint means on development audit only; preserve both training seeds."""
    stage = root / "stage1"
    jobs = dict(read_json(stage / "jobs.json"))
    scores = {name: read_json(stage / "trials" / name / "evaluations/final.json")["mean_score"] for name in jobs}
    means = {arm: float(np.mean([scores[arm + "_" + suffix] for suffix in "ab"])) for arm in ARMS}
    winner = max(ARMS[1:], key=lambda a: means[a])
    return {"means": means, "selected": ["capacity", winner], "scores": scores,
            "rule": "continue control and highest non-control fixed endpoint; no peak/confirmation selection",
            "warning": "promotion is exploratory, not evidence of statistical superiority"}


def stage2(root):
    choice = promote(root)
    if (root / "promotion.json").exists() and read_json(root / "promotion.json") != choice:
        raise ValueError("immutable promotion")
    write_json(root / "promotion.json", choice)
    config = read_json(root / "config.json")
    stage = root / "stage2"
    stage.mkdir(exist_ok=True)
    (stage / "logs").mkdir(exist_ok=True)
    jobs = []
    for name, trial in read_json(root / "stage1/jobs.json"):
        if trial["arm"] not in choice["selected"]:
            continue
        parent = checkpoint_spec(root / "stage1/trials" / name / "final.pt")
        count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
        jobs.append((name, {**trial, "continuation": parent, "target_actions": count + config["stage2_actions"],
                            "milestones": [count + config["stage2_actions"] // 2, count + config["stage2_actions"]],
                            "max_seconds": config["stage2_seconds"],
                            "trajectory_directory": str(stage / "trials" / name / "trajectories")}))
    if (stage / "jobs.json").exists() and read_json(stage / "jobs.json") != [list(j) for j in jobs]:
        raise ValueError("immutable second-stage jobs")
    write_json(stage / "jobs.json", jobs)
    write_json(stage / "config.json", {**config, "gpu_uuids": config["gpu_uuids"][:2],
                                      "preflight_cpu_core_seconds": 0., "preflight_gpu_seconds": 0.})
    return stage


def finalize(root):
    """Fixed endpoints on sealed-for-this-campaign confirmation; ablations separately."""
    import torch
    from .training import Trainer
    frozen = read_json(root / "finalization.json")
    seeds = read_json(root / "splits.json")["sets"]["confirmation"]
    results, ablations = {}, {}
    for name, spec in frozen["checkpoints"].items():
        c = (read_json(root / "stage1/jobs.json")[0][1] if name == "parent" else
             read_json(Path(spec["checkpoint"]).parent / "config.json"))
        trainer = Trainer(c, "cuda")
        try:
            if digest(spec["checkpoint"]) != spec["sha256"]:
                raise ValueError("changed frozen checkpoint")
            saved = torch.load(spec["checkpoint"], map_location="cuda", weights_only=False)
            if name == "parent":
                trainer.import_sampling_continuation(spec)
            else:
                if saved["source_id"] != source_id() or saved["config"] != c or saved["version"] != 134:
                    raise ValueError("frozen source/config mismatch")
                trainer.model.load_state_dict(saved["model"])
                trainer.manager.load_state_dict(saved["manager"])
            results[name] = trainer.evaluate(seeds)
            write_json(root / "confirmation" / (name + ".json"), results[name])
            if c["arm"] != "capacity" and name != "parent":
                # Development-only diagnostic; no choosing checkpoints based on interventions.
                trainer.model.intervention = "zero"
                removed = trainer.evaluate(c["final_seeds"])
                trainer.model.intervention = "normal"
                intact = read_json(Path(spec["checkpoint"]).parent / "evaluations/final.json")
                ablations[name] = paired(intact, removed)
                write_json(root / "ablations" / (name + ".json"), {"normal": intact, "zero_plan": removed,
                           "normal_minus_zero": ablations[name], "warning": "off-distribution intervention, not standalone proof"})
        finally:
            trainer.close()
    arms = sorted(set(name.rsplit("_", 1)[0] for name in results if name != "parent"))
    contrasts = {}
    for arm in arms:
        values, gaps = [], []
        for i, seed in enumerate(seeds):
            if any(result["games"][i]["environment_seed"] != seed for result in results.values()):
                raise ValueError("pairing mismatch")
            score = np.mean([results[arm + "_" + s]["games"][i]["score"] for s in "ab"])
            control = np.mean([results["capacity_" + s]["games"][i]["score"] for s in "ab"])
            values.append(score)
            gaps.append(score - control)
        half = 1.96 * float(np.std(gaps, ddof=1)) / np.sqrt(len(gaps))
        contrasts[arm] = {"mean_score": float(np.mean(values)), "minus_capacity": float(np.mean(gaps)),
                          "environment_paired_descriptive95": [float(np.mean(gaps) - half), float(np.mean(gaps) + half)],
                          "warning": "conditional on two training seeds"}
    write_json(root / "confirmation.json", {"n": len(seeds), "arms": contrasts,
               "models": {name: r["mean_score"] for name, r in results.items()}, "plan_ablation_development": ablations,
               "paired_vs_parent": {name: paired(r, results["parent"]) for name, r in results.items() if name != "parent"},
               "selection": "frozen matched-budget endpoints; no further tuning from these results"})


def run(root, repo, **hooks):
    def stop(*_):
        raise InterruptedError("campaign stopped at bounded deadline")
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _run(root, repo, **hooks)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run(root, repo, *, stage_builder=stage2, reporter=report,
         training_module="boardbench.v134.training", campaign_module="boardbench.v134.campaign"):
    config, launch = read_json(root / "config.json"), read_json(root / "launch.json")
    if source_id() != launch["source_id"] or os.uname().nodename != config["host"]:
        raise ValueError("authorized frozen source/host required")
    for index in (1, 2):
        stage = root / "stage1" if index == 1 else stage_builder(root)
        stage_config = read_json(stage / "config.json")
        jobs = read_json(stage / "jobs.json")
        cpus = launch["cpus"][:len(jobs) * (config["training"]["workers"] + 1) + 1]
        child_launch = {**launch, "cpus": cpus}
        write_json(stage / "launch.json", child_launch)
        if not (stage / "source").exists():
            (stage / "source").symlink_to(root / "source", target_is_directory=True)
        if index == 1:
            stage_config.update(preflight_cpu_core_seconds=config.get("preflight_cpu_core_seconds", 0),
                                preflight_gpu_seconds=config.get("preflight_gpu_seconds", 0))
            write_json(stage / "config.json", stage_config)
        tick = time.monotonic()
        wait_resources(root, repo, stage_config, launch["deadline_unix"] - 1800, f"stage{index}").close()
        costs = read_json(root / "waiting_cost.json") if (root / "waiting_cost.json").exists() else {}
        costs[f"stage{index}"] = costs.get(f"stage{index}", 0.) + (time.monotonic() - tick) / 3600
        write_json(root / "waiting_cost.json", costs)
        execute(stage, repo, training_module=training_module, reporter=reporter,
                expected_host=config["host"], absolute_deadline=launch["deadline_unix"] - 1200)
        state = read_json(stage / "state.json")
        if any(v["status"] != "completed" for v in state["jobs"].values()) or len(state["jobs"]) != len(jobs):
            write_json(root / "completion.json", {"status": "incomplete; checkpoints preserved", "stage": index, "state": state})
            return
    frozen = {"checkpoints": {name: checkpoint_spec(root / "stage2/trials" / name / "final.pt")
                             for name, _ in read_json(root / "stage2/jobs.json")},
              "selection": read_json(root / "promotion.json")}
    frozen["checkpoints"]["parent"] = read_json(root / "stage1/jobs.json")[0][1]["continuation"]
    if (root / "finalization.json").exists() and read_json(root / "finalization.json") != frozen:
        raise ValueError("frozen endpoints changed")
    write_json(root / "finalization.json", frozen)
    tick = time.monotonic()
    lock = wait_resources(root, repo, {**config, "gpu_uuids": config["gpu_uuids"][:1]},
                          launch["deadline_unix"] - 900, "confirmation")
    costs = read_json(root / "waiting_cost.json")
    costs["confirmation"] = costs.get("confirmation", 0.) + (time.monotonic() - tick) / 3600
    write_json(root / "waiting_cost.json", costs)
    child, code, tick = None, None, time.monotonic()
    try:
        with (root / "finalization.log").open("a") as stream:
            child = subprocess.Popen(["taskset", "-c", ",".join(map(str, launch["cpus"][:7])), sys.executable,
                                      "-u", "-m", campaign_module, "finalize", "--root", str(root),
                                      "--repo", str(repo)], env={**os.environ, "CUDA_VISIBLE_DEVICES": config["gpu_uuids"][0]},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait(timeout=min(900, max(1, launch["deadline_unix"] - time.time() - 60)))
    finally:
        if child is not None and child.poll() is None:
            kill_tree(child.pid)
            child.wait()
        elapsed = time.monotonic() - tick
        write_json(root / "finalization_cost.json", {"gpu_hours": elapsed / 3600,
                   "cpu_core_hours": 8 * elapsed / 3600, "returncode": code})
        lock.close()
    if code:
        raise RuntimeError("confirmation failed; endpoints preserved")
    write_json(root / "completion.json", {"status": "completed", "ended_unix": time.time(),
               "stages": {str(i): read_json(root / f"stage{i}/state.json")["cost"] for i in (1, 2)},
               "waiting_cpu_core_hours": sum(read_json(root / "waiting_cost.json").values()),
               "finalization": read_json(root / "finalization_cost.json")})


def guard(root, repo):
    launch = read_json(root / "launch.json")
    os.sched_setaffinity(0, {launch["cpus"][-1]})
    with (root / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-u", "-m", launch.get("campaign_module", "boardbench.v134.campaign"), "run",
                                  "--repo", str(repo), "--root", str(root)], start_new_session=True)
        write_json(root / "guard.json", {"pid": os.getpid(), "child_pid": child.pid,
                   "deadline_unix": launch["deadline_unix"]})
        signalled = False
        while child.poll() is None:
            if time.time() >= launch["deadline_unix"] - 120 and not signalled:
                child.terminate()
                signalled = True
            if time.time() >= launch["deadline_unix"]:
                kill_tree(child.pid)
                break
            time.sleep(5)
        code = child.wait()
        write_json(root / "guard_result.json", {"returncode": code, "ended_unix": time.time(),
                   "deadline_signalled": signalled})
        if code:
            raise SystemExit(code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "launch", "resume", "guard", "run", "finalize"))
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config")
    a = p.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    if not root.is_relative_to(repo / "outputs/v134"):
        raise ValueError("dedicated V134 output directory required")
    if a.mode == "prepare":
        prepare(repo, root, read_json(a.config))
        return
    config = read_json(root / "config.json")
    if os.uname().nodename != config["host"]:
        raise ValueError("a226 only")
    if a.mode in ("launch", "resume"):
        if a.mode == "launch":
            if (root / "launch.json").exists():
                raise ValueError("already launched")
            gate = read_json(root / "preflight_cost.json")
            if not gate["passed"] or gate["source_id"] != source_id(repo):
                raise ValueError("actual CUDA/source-matched gate required")
            all_cpus = physical_cpus()
            count = 8 * (config["training"]["workers"] + 1) + 1
            if count > min(128, len(all_cpus) // 2):
                raise ValueError("half-machine physical CPU cap")
            cpus = all_cpus[:count]
            validate_allocation(cpus, config["gpu_uuids"], config["training"]["workers"] + 1, 2, 8)
            local = read_json(root / "local_checks.json") if (root / "local_checks.json").exists() else {}
            config.update(preflight_cpu_core_seconds=gate["cpu_core_seconds"] + local.get("cpu_core_seconds", 0),
                          preflight_gpu_seconds=gate["gpu_seconds"])
            write_json(root / "config.json", config)
            for relative in source_files(repo):
                destination = root / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, destination)
            write_json(root / "launch.json", {"source_id": source_id(repo), "cpus": cpus,
                       "started_unix": time.time(), "deadline_unix": time.time() + config["guard_seconds"]})
        launch = read_json(root / "launch.json")
        if time.time() >= launch["deadline_unix"] - 900:
            raise ValueError("original deadline expired; no renewal")
        env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTHONNOUSERSITE": "1",
               **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        os.execve(sys.executable, [sys.executable, "-u", "-m", "boardbench.v134.campaign", "guard",
                                  "--repo", str(repo), "--root", str(root)], env)
    {"guard": guard, "run": run, "finalize": lambda r, _: finalize(r)}[a.mode](root, repo)


if __name__ == "__main__":
    main()
