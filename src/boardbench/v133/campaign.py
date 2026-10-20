"""Six continuously trained policies; immutable panels, bounded guard and final confirmation."""
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
from boardbench.v127.preflight import reserve
from boardbench.v130.pipeline import checkpoint_spec, paired
from .training import ARMS


def new_manifest(old, old_sha, extra_seeds, count=512):
    validate_manifest(old)
    forbidden = set(extra_seeds).union(*(set(v) for v in old["sets"].values()))
    rng, fresh = random.Random(13320261021), []
    while len(fresh) < 3 * count:
        candidate = rng.randrange(20000000, 30000000)
        if candidate not in forbidden:
            forbidden.add(candidate)
            fresh.append(candidate)
    sets = deepcopy(old["sets"])
    sets.update(development=fresh[:2 * count], development_health=fresh[:2 * count][:32],
                confirmation=fresh[2 * count:])
    value = {"format_version": 1, "generator_seed": 13320261021, "sets": sets,
             "parent_split_sha256": old_sha, "training_pool": "identical to parent",
             "test_state": "original formal test remains sealed"}
    value["sha256"] = identity(value)
    validate_manifest(value)
    return value


def prepare(repo, root, config):
    root.mkdir(parents=True, exist_ok=True)
    old_root = repo / config["parent_run"]
    frozen = read_json(old_root / "confirmation128_frozen/plan.json")
    parent = frozen["checkpoints"]["primary_round08_final"]
    if checkpoint_spec(parent["checkpoint"]) != parent:
        raise ValueError("predesignated parent changed")
    base = read_json(Path(parent["checkpoint"]).parent / "config.json")
    old = read_json(base["split_manifest"])
    if digest(base["split_manifest"]) != base["split_sha256"]:
        raise ValueError("old split changed")
    extra = read_json(old_root / "confirmation512_fresh_frozen/plan.json")["environment_seeds"]
    manifest = new_manifest(old, base["split_sha256"], extra)
    split = root / "splits.json"
    if split.exists() and read_json(split) != manifest:
        raise ValueError("new panels immutable")
    write_json(split, manifest)
    count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
    jobs = []
    for arm in ARMS:
        for replicate in range(2):
            name = arm + "_" + "ab"[replicate]
            trial = {**base, "continuous_version": "1.3.3", "arm": arm, "replicate": replicate,
                     "seed": 13301 + replicate, "split_manifest": str(split), "split_sha256": digest(split),
                     "curve_seeds": manifest["sets"]["development"][:512],
                     "final_seeds": manifest["sets"]["development"][512:],
                     "continuation": parent, "initial_checkpoint": None, "initial_sha256": None,
                     "learning_rate": .00003, "gae_lambda": .95, "entropy_coef": .005,
                     "epochs": 4, "minibatch": 1024, "batch_episodes": 128,
                     "workers": config["training"]["workers"], "critic_batches": 0,
                     "bc_weight": 0., "sequence_bc_weight": 0., "reset_mode": "none", "reset_fraction": 0.,
                     "target_actions": count + config["additional_actions"],
                     "milestones": [count + v for v in config["milestones"]],
                     "max_seconds": config["task_seconds"], "record_games_per_batch": 4,
                     "trajectory_directory": str(root / "trials" / name / "trajectories"),
                     "replay_directory": str(root / "trials" / name / "replay"),
                     "replay_batches": 8, "replay_minibatch": 256,
                     "self_imitation_weight": .02, "replay_advantage_cap": 1.}
            jobs.append((name, trial))
    if (root / "jobs.json").exists() and read_json(root / "jobs.json") != [list(j) for j in jobs]:
        raise ValueError("immutable job specification")
    write_json(root / "jobs.json", jobs)
    write_json(root / "config.json", config)
    (root / "logs").mkdir(exist_ok=True)
    return jobs


def report(root, state):
    root = Path(root)
    jobs = read_json(root / "jobs.json")
    curves, rows = {}, []
    for name, trial in jobs:
        directory = root / "trials" / name
        progress = read_json(directory / "progress.json") if (directory / "progress.json").exists() else {}
        starting = read_json(Path(trial["continuation"]["checkpoint"]).with_suffix(".json"))["ppo_actions"]
        points = []
        for path in (directory / "evaluations").glob("*.json"):
            result = read_json(path)
            if not result["label"].startswith("final"):
                points.append({"actions": result["ppo_actions"] - starting, "score": result["mean_score"],
                               "animal_score": result["animal_mean"]})
        points.sort(key=lambda r: r["actions"])
        curves[name] = points
        rows.append({"trial": name, "phase": progress.get("phase", "pending"),
                     "additional_actions": max(0, progress.get("ppo_actions", starting) - starting),
                     "latest_score": points[-1]["score"] if points else None,
                     "heartbeat_age": time.time() - progress["updated_unix"] if progress else None})
    write_json(root / "curves.json", curves)
    write_json(root / "results.json", rows)
    lines = ["# Harmonies V133 连续纯模型 RL", "", f"状态：{state['phase']}",
             "三组×两个种子；每条连续新增20M动作，不退回开发峰值。无搜索、专家损失或课程接手。",
             "曲线：新development512。另有512开发审计局；冻结终点后独立confirmation512。旧确认与正式test封存。",
             "", "|任务|状态|新增动作|最新开发均分|", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['trial']}|{row['phase']}|{row['additional_actions']}|{row['latest_score']}|")
    lines += ["", f"训练与预检资源：{state.get('cost', {})}",
              "finalization.json保留终点候选；confirmation.json与finalization_cost.json单独记录确认和费用。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    colors = ["#2563eb", "#60a5fa", "#dc2626", "#f87171", "#059669", "#34d399"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440">',
           '<rect width="960" height="440" fill="white"/>',
           '<text x="40" y="25">V133: pure model / fresh development512 / additional real actions</text>']
    for score in (80, 90, 100, 110, 120, 130):
        y = 390 - (score - 70) * 5
        svg.append(f'<path d="M50 {y} H750" stroke="#ddd"/><text x="10" y="{y}">{score}</text>')
    target = read_json(root / "config.json")["additional_actions"]
    for i, (name, points) in enumerate(curves.items()):
        coords = " ".join(f'{50+700*r["actions"]/target},{390-(r["score"]-70)*5}' for r in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        svg.append(f'<text x="765" y="{65+22*i}" fill="{colors[i]}">{name}</text>')
    svg += [f'<text x="50" y="425">0</text><text x="675" y="425">{target/1e6:g}M</text>', "</svg>"]
    (root / "curves.svg").write_text("\n".join(svg))


def finalize(root, repo):
    import torch
    from .training import Trainer
    frozen = read_json(root / "finalization.json")
    seeds = read_json(root / "splits.json")["sets"]["confirmation"]
    results = {}
    jobs = dict(read_json(root / "jobs.json"))
    for name, spec in frozen["checkpoints"].items():
        trial = jobs[name] if name != "parent" else next(iter(jobs.values()))
        trainer = Trainer(trial, "cuda")
        try:
            if name == "parent":
                trainer.import_sampling_continuation(spec)
            else:
                if digest(spec["checkpoint"]) != spec["sha256"]:
                    raise ValueError("frozen final checkpoint changed")
                saved = torch.load(spec["checkpoint"], map_location="cuda", weights_only=False)
                if saved["source_id"] != source_id() or saved["config"] != trial or saved["version"] != 133:
                    raise ValueError("final source/config mismatch")
                trainer.model.load_state_dict(saved["model"])
            results[name] = trainer.evaluate(seeds)
            results[name]["checkpoint"] = spec
            write_json(root / "confirmation" / (name + ".json"), results[name])
        finally:
            trainer.close()
    contrasts = {}
    for arm in ARMS:
        delta, mean = [], []
        for i, seed in enumerate(seeds):
            if any(row["games"][i]["environment_seed"] != seed for row in results.values()):
                raise ValueError("confirmation pairing mismatch")
            score = np.mean([results[arm + "_" + suffix]["games"][i]["score"] for suffix in "ab"])
            control = np.mean([results["ppo_" + suffix]["games"][i]["score"] for suffix in "ab"])
            mean.append(score)
            delta.append(score - control)
        half = 1.96 * np.std(delta, ddof=1) / np.sqrt(len(seeds))
        contrasts[arm] = {"mean_score": float(np.mean(mean)), "minus_ppo": float(np.mean(delta)),
                          "environment_paired_descriptive95": [float(np.mean(delta) - half), float(np.mean(delta) + half)],
                          "warning": "averaged over only two training seeds; not seed-general uncertainty"}
    write_json(root / "confirmation.json", {
        "n": len(seeds), "arms": contrasts,
        "paired_vs_parent": {name: paired(r, results["parent"]) for name, r in results.items() if name != "parent"},
        "models": {name: r["mean_score"] for name, r in results.items()},
        "selection": "all fixed20M endpoints, no checkpoint ranking or further tuning on confirmation"})


def wait_resources(root, repo, config, deadline, stage):
    """Bounded retry of resource gates only; never retry numerical training failures."""
    waited = time.monotonic()
    while True:
        try:
            lock = reserve(repo, config)
            return lock
        except (RuntimeError, BlockingIOError) as error:
            write_json(root / "waiting.json", {"unix": time.time(), "reason": str(error), "stage": stage})
            if time.monotonic() - waited >= config["resource_wait_seconds"] or time.time() >= deadline:
                raise RuntimeError("bounded resource wait exhausted") from error
            time.sleep(15)


def run(root, repo):
    # Turn external termination into an exception so finalization's finally
    # block also cleans its subprocess; the training runner has its own handler.
    def stop(*_):
        raise InterruptedError("bounded campaign requested shutdown")
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _run(root, repo)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run(root, repo):
    config, launch = read_json(root / "config.json"), read_json(root / "launch.json")
    if source_id() != launch["source_id"] or os.uname().nodename != config["host"]:
        raise ValueError("authorized host and frozen source required")
    before_training = time.monotonic()
    wait_resources(root, repo, config, launch["deadline_unix"] - 900, "training").close()
    prior_wait = read_json(root / "resource_wait_cost.json").get("cpu_core_hours", 0.) if (
        root / "resource_wait_cost.json").exists() else 0.
    write_json(root / "resource_wait_cost.json",
               {"cpu_core_hours": prior_wait + (time.monotonic() - before_training) / 3600})
    execute(root, repo, training_module="boardbench.v133.training", reporter=report,
            expected_host=config["host"], absolute_deadline=launch["deadline_unix"] - 600)
    state = read_json(root / "state.json")
    if any(v["status"] != "completed" for v in state["jobs"].values()) or len(state["jobs"]) != 6:
        write_json(root / "completion.json", {"status": "incomplete; checkpoints retained", "training": state["phase"]})
        return
    if (root / "confirmation.json").exists():
        return
    jobs = read_json(root / "jobs.json")
    checkpoints = {"parent": jobs[0][1]["continuation"]}
    checkpoints.update({name: checkpoint_spec(root / "trials" / name / "final.pt") for name, _ in jobs})
    frozen = {"checkpoints": checkpoints, "protocol": "all six fixed20M endpoints; no selection on confirmation"}
    if (root / "finalization.json").exists() and read_json(root / "finalization.json") != frozen:
        raise ValueError("frozen endpoints changed")
    write_json(root / "finalization.json", frozen)
    # One GPU and one worker core, with controller on its existing dedicated core.
    before_confirmation = time.monotonic()
    lock = wait_resources(root, repo, {**config, "gpu_uuids": [config["gpu_uuids"][0]]},
                          launch["deadline_unix"] - 330, "confirmation")
    started, child, code = time.monotonic(), None, None
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": config["gpu_uuids"][0]}
        with (root / "logs/finalization.log").open("a") as log:
            child = subprocess.Popen(["taskset", "-c", str(launch["cpus"][0]), sys.executable, "-u", "-m",
                                      "boardbench.v133.campaign", "finalize", "--repo", str(repo), "--root", str(root)],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait(timeout=min(300, max(1, launch["deadline_unix"] - time.time() - 30)))
    finally:
        if child is not None and child.poll() is None:
            kill_tree(child.pid)
            child.wait()
        elapsed = time.monotonic() - started
        write_json(root / "finalization_cost.json", {"returncode": code, "gpu_hours": elapsed / 3600,
                   "cpu_core_hours": (2 * elapsed + started - before_confirmation) / 3600,
                   "elapsed_seconds": elapsed})
        lock.close()
    if code:
        raise RuntimeError("final confirmation failed; endpoints and log preserved")
    write_json(root / "completion.json", {"status": "completed", "unix": time.time(),
               "training_cost": state["cost"], "confirmation_cost": read_json(root / "finalization_cost.json"),
               "waiting_cost": read_json(root / "resource_wait_cost.json")})


def guard(root, repo):
    launch = read_json(root / "launch.json")
    os.sched_setaffinity(0, {launch["cpus"][-1]})
    with (root / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-u", "-m", "boardbench.v133.campaign", "run",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "resume", "guard", "run", "finalize", "report"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
    if not root.is_relative_to(repo / "outputs/v133"):
        raise ValueError("dedicated V133 output directory required")
    if args.mode == "prepare":
        prepare(repo, root, read_json(args.config))
        return
    config = read_json(root / "config.json")
    if os.uname().nodename != config["host"]:
        raise ValueError("a226 only")
    if args.mode in ("launch", "resume"):
        if args.mode == "launch":
            if (root / "launch.json").exists():
                raise ValueError("already launched; resume preserves deadline")
            preflight = read_json(root / "preflight_cost.json")
            if not preflight.get("passed") or preflight["source_id"] != source_id(repo):
                raise ValueError("version-matched actual CUDA gate required")
            all_cpus = physical_cpus()
            count = 6 * (config["training"]["workers"] + 1) + 1
            if count > min(128, len(all_cpus) // 2):
                raise ValueError("half-machine CPU cap")
            cpus = all_cpus[:count]
            validate_allocation(cpus, config["gpu_uuids"], config["training"]["workers"] + 1, 2, 6)
            local_cost = read_json(root / "local_checks.json")["cpu_core_seconds"] if (
                root / "local_checks.json").exists() else 0.
            config.update(preflight_cpu_core_seconds=preflight["cpu_core_seconds"] + local_cost,
                          preflight_gpu_seconds=preflight["gpu_seconds"])
            write_json(root / "config.json", config)
            for relative in source_files(repo):
                destination = root / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, destination)
            write_json(root / "launch.json", {"source_id": source_id(repo), "cpus": cpus,
                       "started_unix": time.time(), "deadline_unix": time.time() + config["guard_seconds"],
                       "repo": str(repo)})
        launch = read_json(root / "launch.json")
        if time.time() >= launch["deadline_unix"] - 300:
            raise ValueError("original per-task launch deadline expired; no renewal")
        env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTHONNOUSERSITE": "1",
               **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        os.execve(sys.executable, [sys.executable, "-u", "-m", "boardbench.v133.campaign", "guard",
                                  "--repo", str(repo), "--root", str(root)], env)
    if args.mode == "report":
        report(root, read_json(root / "state.json"))
    else:
        {"guard": guard, "run": run, "finalize": finalize}[args.mode](root, repo)


if __name__ == "__main__":
    main()
