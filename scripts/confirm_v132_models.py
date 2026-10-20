"""Frozen 128 or fresh 512-environment comparison; no search, training, or promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from boardbench.v130.pipeline import checkpoint_spec, paired
from boardbench.v130.training import export_checkpoint
from boardbench.v132.training import Trainer


def fresh_panel(sets, count=512, seed=20261021):
    """Uniform seed sampling without replacement, excluding every prior split."""
    forbidden = set().union(*(set(values) for values in sets.values()))
    if not 1 <= count <= 10000000 - sum(20000000 <= s < 30000000 for s in forbidden):
        raise ValueError("invalid fresh panel size")
    rng = random.Random(seed)
    selected = []
    while len(selected) < count:
        candidate = rng.randrange(20000000, 30000000)
        if candidate not in forbidden:
            selected.append(candidate)
            forbidden.add(candidate)
    return selected


def validate_games(result, seeds):
    games = result["games"]
    if [g["environment_seed"] for g in games] != seeds or len(set(seeds)) != len(seeds):
        raise ValueError("paired seeds/order/uniqueness mismatch")
    if not all(g["natural_terminal"] and not g["curriculum_reset"]
               and g["initial_score"] == 0 and g["earned_score"] == g["score"] for g in games):
        raise ValueError("normal-start natural-terminal evaluation required")
    if abs(sum(g["score"] for g in games) / len(games) - result["mean_score"]) > 1e-10:
        raise ValueError("score aggregation mismatch")


def evaluate_panel(trainer, seeds, batch_size=128):
    """Bounded batches; aggregate by episode, never average unequal batch means."""
    if not seeds or batch_size < 1:
        raise ValueError("nonempty panel and positive batch required")
    batches = [trainer.evaluate(seeds[i:i + batch_size]) for i in range(0, len(seeds), batch_size)]
    for batch, start in zip(batches, range(0, len(seeds), batch_size)):
        validate_games(batch, seeds[start:start + batch_size])
    games = [game for batch in batches for game in batch["games"]]
    scores = [g["score"] for g in games]
    return {
        "games": games, "mean_score": statistics.mean(scores),
        "animal_mean": statistics.mean(g["animal_score"] for g in games),
        "completed_cards": statistics.mean(g["completed_cards"] for g in games),
        "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.,
        "score_min": min(scores), "score_max": max(scores),
        **{key: sum(batch[key] for batch in batches) for key in
           ("actions", "environment_seconds", "model_seconds")},
        "evaluation_batch_size": batch_size,
    }


def model_hash(trainer):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                  for v in trainer.model.state_dict().values())).hexdigest()


def export_primary(out, spec):
    # export_checkpoint also writes .json; keep it away from per-game results.
    return export_checkpoint(spec["checkpoint"], out / "model" / "primary_round08_final.npz")


def worker(out, repair=False):
    plan = read_json(out / "plan.json")
    script_sha = read_json(out / "repair.json")["script_sha256"] if repair else plan["script_sha256"]
    if source_id() != plan["source_id"] or digest(__file__) != script_sha:
        raise ValueError("frozen source/script mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one real CUDA device required")
    assert (torch.ones(2, device="cuda", dtype=torch.float64).sum().item() == 2)
    results = {}
    for name, spec in plan["checkpoints"].items():
        if repair and name != "primary_round08_final":
            results[name] = read_json(out / (name + ".json"))
            validate_games(results[name], plan["environment_seeds"])
            continue
        started = time.monotonic()
        trainer = Trainer(plan["trial_config"], "cuda")
        try:
            trainer.import_sampling_continuation(spec)
            trainer.model.eval()
            before = model_hash(trainer)
            counters = [trainer.actions, trainer.ppo_actions, trainer.updates]
            numpy_rng = json.dumps(trainer.rng.bit_generator.state, sort_keys=True)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            tick = time.monotonic()
            result = evaluate_panel(trainer, plan["environment_seeds"])
            result["evaluation_seconds"] = time.monotonic() - tick
            validate_games(result, plan["environment_seeds"])
            if (model_hash(trainer) != before
                    or counters != [trainer.actions, trainer.ppo_actions, trainer.updates]
                    or numpy_rng != json.dumps(trainer.rng.bit_generator.state, sort_keys=True)
                    or not torch.equal(cpu_rng, torch.get_rng_state())
                    or not torch.equal(cuda_rng, torch.cuda.get_rng_state())
                    or any(p.grad is not None for p in trainer.model.parameters())
                    or digest(spec["checkpoint"]) != spec["sha256"]):
                raise RuntimeError("inference mutated model, RNG, counters, gradients or checkpoint")
            result.update(checkpoint=spec, inference_only_verified=True,
                          total_seconds=time.monotonic() - started)
            if repair:
                previous = read_json(out / "summary.json")["models"][name]
                if any(result[k] != previous[k] for k in
                       ("mean_score", "animal_mean", "completed_cards")):
                    raise ValueError("deterministic artifact reconstruction changed scores")
            write_json(out / (name + ".json"), result)
            results[name] = result
            print(json.dumps({"model": name, **{k: result[k] for k in
                  ("mean_score", "animal_mean", "evaluation_seconds", "total_seconds",
                   "environment_seconds", "model_seconds")}}), flush=True)
        finally:
            trainer.close()
    comparisons = {}
    for name in ("primary_round08_final", "diagnostic_round04_best"):
        differences = [a["score"] - b["score"] for a, b in
                       zip(results[name]["games"], results["parent"]["games"])]
        comparisons[name] = {**paired(results[name], results["parent"]),
                             "wins": sum(d > 0 for d in differences),
                             "ties": sum(d == 0 for d in differences),
                             "losses": sum(d < 0 for d in differences)}
    export_primary(out, plan["checkpoints"]["primary_round08_final"])
    write_json(out / "summary.json", {
        "models": {name: {k: r[k] for k in ("mean_score", "animal_mean", "completed_cards",
                   "evaluation_seconds", "total_seconds")} for name, r in results.items()},
        "comparisons_vs_parent": comparisons, "primary": "primary_round08_final",
        "diagnostic_minus_primary": paired(results["diagnostic_round04_best"],
                                           results["primary_round08_final"]),
        "score_spread": {name: {k: r.get(k) for k in ("score_std", "score_min", "score_max")}
                         for name, r in results.items()},
        "environment_count": len(plan["environment_seeds"]),
        "split": plan.get("panel", "confirmation[:128]; not formal test"),
        "protocol": "greedy FP64 model only; no STOP, search, learning or confirmation-based promotion",
        "not_used_for_tuning": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh-512", action="store_true",
                       help="New 512-seed panel disjoint from all old splits; same three frozen candidates")
    modes.add_argument("--repair-primary-artifact", action="store_true",
                        help="Reconstruct only the known collided JSON, with the original frozen policy/seeds")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / ("confirmation512_fresh_frozen" if args.fresh_512 else "confirmation128_frozen")
    if args.worker:
        worker(out, repair=args.repair_primary_artifact)
        return
    repo = Path(__file__).resolve().parents[1]
    campaign = read_json(root / "config.json")
    if os.uname().nodename != campaign["host"]:
        raise ValueError("authorized a226 host required")
    if read_json(root / "state.json")["phase"] not in ("failed", "completed"):
        raise ValueError("training must be quiescent")
    repair = args.repair_primary_artifact
    if out.exists() and not repair:
        raise ValueError("refusing to overwrite or rerun confirmation panel")
    if repair:
        frozen = read_json(out / "plan.json")
        collided = read_json(out / "primary_round08_final.json")
        if ((out / "repair.json").exists() or "games" in collided
                or collided.get("parent_sha256") != frozen["checkpoints"]["primary_round08_final"]["sha256"]
                or source_id() != frozen["source_id"]):
            raise ValueError("only the known export-sidecar collision can be repaired")
    cpus = physical_cpus()
    if len(cpus) < 4:
        raise ValueError("two cores must fit within the half-machine cap")
    cpus = cpus[-2:]
    trial = read_json(root / "round_01/ppo/trials/control_a/config.json")
    if digest(trial["split_manifest"]) != trial["split_sha256"]:
        raise ValueError("split checksum mismatch")
    sets = read_json(trial["split_manifest"])["sets"]
    seeds = fresh_panel(sets) if args.fresh_512 else sets["confirmation"][:128]
    expected = 512 if args.fresh_512 else 128
    excluded = set().union(*(set(values) for values in sets.values())) if args.fresh_512 else (
        set(sets["train"]) | set(sets["development"]) | set(sets["test"]))
    if len(seeds) != expected or len(set(seeds)) != expected or set(seeds) & excluded:
        raise ValueError("confirmation panel size or disjointness violation")
    base = read_json(repo / campaign["base_config"])
    checkpoints = {
        "parent": checkpoint_spec(repo / base["parent"]),
        "primary_round08_final": checkpoint_spec(root / "round_08/ppo/trials/continue_winner_a/final.pt"),
        "diagnostic_round04_best": checkpoint_spec(root / "round_04/ppo/trials/small_step_b/best.pt"),
    }
    if args.fresh_512:
        previous = root / "confirmation128_frozen/plan.json"
        if read_json(previous)["checkpoints"] != checkpoints:
            raise ValueError("fresh panel must keep all three previously frozen candidates")
    gpu = campaign["gpu_uuids"][0]
    lock = reserve(repo, {"gpu_uuids": [gpu]})
    started, child, rc = time.monotonic(), None, None
    try:
        os.sched_setaffinity(0, {cpus[1]})
        out.mkdir(exist_ok=repair)
        if repair:
            write_json(out / "repair.json", {
                "unix": time.time(), "script_sha256": digest(__file__),
                "reason": "Export sidecar overwrote primary per-game JSON; replay same frozen primary and seeds only",
                "original_plan_sha256": digest(out / "plan.json"), "cpu_reservation": cpus,
                "original_summary": read_json(out / "summary.json"),
            })
            shutil.copyfile(out / "primary_round08_final.json", out / "collided_export_receipt.json")
        else:
            write_json(out / "plan.json", {
                "frozen_unix": time.time(), "source_id": source_id(), "script_sha256": digest(__file__),
                "checkpoints": checkpoints, "environment_seeds": seeds, "trial_config": trial,
                "split_sha256": trial["split_sha256"], "gpu": gpu, "cpu_reservation": cpus,
                "maximum_seconds": 180, "primary": "primary_round08_final",
                "panel": "fresh512; disjoint from all prior splits; not formal test" if args.fresh_512
                         else "confirmation[:128]; not formal test",
                "panel_generator": {"seed": 20261021, "range": [20000000, 30000000],
                                    "method": "uniform rejection without replacement; exclude every prior split"}
                                   if args.fresh_512 else None,
                "prior_candidate_plan_sha256": digest(previous) if args.fresh_512 else None,
                "selection": "round08 final selected on prior development64; round04 is diagnostic only",
                "authorization": "new bounded confirmation requested after original training window; no training renewal",
            })
        shutil.copyfile(__file__, out / (("repaired_" if repair else "") + Path(__file__).name))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1",
               **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "NUMEXPR_NUM_THREADS")}}
        with (out / ("repair_worker.log" if repair else "worker.log")).open("w") as log:
            command = ["taskset", "-c", str(cpus[0]), sys.executable, "-u", __file__,
                       "--worker", "--root", str(root)]
            if repair:
                command.append("--repair-primary-artifact")
            if args.fresh_512:
                command.append("--fresh-512")
            child = subprocess.Popen(
                command,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rc = child.wait(timeout=180)
    finally:
        if child is not None and child.poll() is None:
            kill_tree(child.pid)
            child.wait()
        elapsed = time.monotonic() - started
        if out.exists():
            write_json(out / ("repair_cost.json" if repair else "cost.json"), {
                "returncode": rc, "elapsed_seconds": elapsed,
                "cpu_core_hours": 2 * elapsed / 3600, "gpu_hours": elapsed / 3600,
                "basis": "separate bounded 2CPU+1GPU confirmation; original training ledger unchanged",
            })
        lock.close()
    if rc:
        raise RuntimeError("confirmation failed; inspect preserved worker.log")
    print(json.dumps(read_json(out / "summary.json")), flush=True)


if __name__ == "__main__":
    main()
