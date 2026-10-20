"""Deadline-bounded adaptive pure-model experiments, with frozen sources and data."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.artifacts.trajectories import archive
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import execute, report as base_report, validate_allocation
from boardbench.v130.pipeline import checkpoint_spec, paired
from .data import build as build_corpus


def excluded(sets):
    return (sets["train"][2176:2192] + sets["train"][2208:2224]
            + sets["train"][3028:3032] + sets["train"][3046:3048]
            + sets["train"][6112:6128])


def prepare_data(repo, root, config):
    base = read_json(repo / config["base_config"])
    manifest = repo / base["split_manifest"]
    if digest(manifest) != base["split_sha256"]:
        raise ValueError("split checksum mismatch")
    exclusions = excluded(read_json(manifest)["sets"])
    write_json(root / "excluded_train_seeds.json", exclusions)
    corpus = root / "data" / "original.json.gz"
    if not corpus.exists():
        build_corpus(repo / config["archive"], corpus, manifest, exclusions,
                     minimum_seconds=config["minimum_corpus_seconds"])
    meta = read_json(corpus.with_suffix(".json"))
    return {"path": str(corpus), "sha256": meta["sha256"]}


def refresh_corpus(repo, root, config, current):
    teacher = repo / config["fresh_teacher"]
    receipt = teacher / "watchdog_result.json"
    if not receipt.exists():
        return current, "new teacher still running; original corpus retained"
    target = root / "data" / "expanded.json.gz"
    if not target.exists():
        # Training admission still enforces split, complete replay and exclusions.
        archive(teacher, repo / config["archive"], workers=1)
        base = read_json(repo / config["base_config"])
        build_corpus(repo / config["archive"], target, repo / base["split_manifest"],
                     read_json(root / "excluded_train_seeds.json"),
                     minimum_seconds=config["minimum_corpus_seconds"])
    meta = read_json(target.with_suffix(".json"))
    return {"path": str(target), "sha256": meta["sha256"]}, "completed new train-only expert corpus admitted"


def choose_plan(number, previous, *, expanded, stagnation):
    """Evidence-directed branches, not a fixed parameter-grid queue.

    Rank methods on the common curve panel only. The 64-development comparison
    is reported separately; formal test outcomes never enter this function.
    """
    standard = {"reset_mode": "none", "reset_fraction": 0., "gae_lambda": .95,
                "sequence_bc_weight": 0., "learning_rate": .0001, "entropy_coef": .005}
    def arm(name, **changes):
        return name, {**standard, **changes}
    if number == 1:
        return {"reason": "matched ablation of expert state support and longer credit assignment",
                "arms": [arm("control"), arm("random", reset_mode="random", reset_fraction=.5),
                         arm("backward", reset_mode="backward", reset_fraction=.5),
                         arm("mc_backward", reset_mode="backward", reset_fraction=.5, gae_lambda=1.)]}
    groups = {}
    for row in previous:
        groups.setdefault(row["arm"], []).append(row)
    means = {name: float(np.mean([r["last_curve_score"] for r in rows])) for name, rows in groups.items()
             if len(rows) == 2}
    control = means.get("control", -float("inf"))
    useful = [name for name, mean in means.items() if name != "control" and mean > control + .75]
    if useful:
        winner = max(useful, key=lambda name: means[name])
        settings = groups[winner][0]["settings"]
        reset = settings["reset_mode"]
        base = {k: settings[k] for k in standard}
        return {"reason": f"{winner} exceeded paired-family control mean by {means[winner]-control:.3f}; test transfer and regularization",
                "evidence": means,
                "arms": [arm("control"),
                         ("continue_winner", base),
                         ("transfer", {**base, "reset_fraction": .2 if reset != "none" else 0.,
                                       "learning_rate": .00005}),
                         ("coherent_bc", {**base, "sequence_bc_weight": .01})]}
    if number == 2:
        return {"reason": "no reset family reliably dominates control; test coherent sequence supervision and normal-start long credit",
                "evidence": means,
                "arms": [arm("control"), arm("mc_normal", gae_lambda=1.),
                         arm("sequence", sequence_bc_weight=.01),
                         arm("sequence_reset", reset_mode="backward", reset_fraction=.25,
                             sequence_bc_weight=.01, gae_lambda=.99)]}
    if expanded and number == 3:
        return {"reason": "new independently generated expert environments available; ablate broader support and auxiliary strength",
                "evidence": means,
                "arms": [arm("control"), arm("sequence", sequence_bc_weight=.01),
                         arm("weak_sequence", sequence_bc_weight=.003, gae_lambda=.99),
                         arm("random_sequence", reset_mode="random", reset_fraction=.25,
                             sequence_bc_weight=.01)]}
    if stagnation >= 2:
        return {"reason": "multiple directions lack sustained curve gain; broaden on-policy exploration and long-horizon credit, retain parent",
                "evidence": means,
                "arms": [arm("control"),
                         arm("explore", entropy_coef=.02, gae_lambda=.99, learning_rate=.00005),
                         arm("long_credit", entropy_coef=.01, gae_lambda=1.),
                         arm("sequence_explore", sequence_bc_weight=.003, entropy_coef=.015,
                             reset_mode="random", reset_fraction=.15, gae_lambda=.99)]}
    return {"reason": "apparent gains are not stable against control; test lower update rate and reduced expert-state dependence",
            "evidence": means,
            "arms": [arm("control"), arm("small_step", learning_rate=.00003),
                     arm("long_credit", gae_lambda=.99, learning_rate=.00005),
                     arm("transfer", reset_mode="backward", reset_fraction=.15,
                         sequence_bc_weight=.003, learning_rate=.00005)]}


def build_jobs(repo, root, config, plan, parent, corpus, number, stage_seconds):
    base = read_json(repo / config["base_config"])
    sets = read_json(repo / base["split_manifest"])["sets"]
    count = read_json(Path(parent["checkpoint"]).with_suffix(".json"))["ppo_actions"]
    jobs = []
    # Match additional action counts within a round; no early score pruning.
    actions = config["actions_per_round"]
    for kind, changes in plan["arms"]:
        for replicate in range(2):
            name = kind + "_" + "ab"[replicate]
            trial = {**base["training"], **changes, "research_version": "1.2.7",
                     "curriculum_version": "1.2.12", "hidden": 256, "seed": 13201 + number * 100 + replicate,
                     "reward_scale": 150., "dataset": str(repo / base["dataset"]),
                     "dataset_sha256": base["dataset_sha256"], "split_manifest": str(repo / base["split_manifest"]),
                     "split_sha256": base["split_sha256"], "curve_seeds": sets["development"][:32],
                     "final_seeds": sets["development"][32:96],
                     "excluded_train_seeds": read_json(root / "excluded_train_seeds.json"),
                     "workers": config["workers"], "bc_weight": 0., "initial_checkpoint": None,
                     "initial_sha256": None, "continuation": parent, "corpus": corpus,
                     "arm": kind, "replicate": replicate, "round": number,
                     "curriculum_actions": actions * .75, "target_actions": count + actions,
                     "milestones": [count + int(actions * f) for f in (.125, .25, .5, .75, 1.)],
                     "max_seconds": stage_seconds,
                     "record_games_per_batch": config["record_games_per_batch"],
                     "trajectory_directory": str(root / f"round_{number:02d}" / "ppo" / "trials" / name / "trajectories")}
            jobs.append((name, trial))
    return jobs


def stage_report(root, state):
    base_report(root, state)
    path = Path(root) / "REPORT.md"
    path.write_text(path.read_text().replace("Harmonies V1.2.7 在线 PPO", "Harmonies V1.2.12 专家经验课程 RL")
        .replace("四组各两个种子：BC初始化、软标签初始化、BC初始化+衰减模仿约束、随机初始化。初始冻结策略作为逐环境对照。",
                 "四种方法各两个种子；具体机制和上轮证据见上级 decision.json。曲线均为普通开局，绝不包含专家前缀奖励。"))
    pipeline = Path(root).parents[1]
    current_path = pipeline / "state.json"
    if current_path.exists():
        current = read_json(current_path)
        current["live_stage"] = str(root)
        current["live_trials"] = state["jobs"]
        current["controller_seconds"] = max(current.get("controller_seconds", 0.),
                                           time.time() - read_json(pipeline / "launch.json")["started_unix"])
        publish(pipeline, current, read_json(pipeline / "config.json"),
                Path(read_json(pipeline / "launch.json")["repo"]))


def summarize(stage):
    rows = []
    for name, config in read_json(stage / "ppo" / "jobs.json"):
        trial = stage / "ppo" / "trials" / name
        summary = trial / "summary.json"
        if not summary.exists():
            continue
        curves = [read_json(p) for p in (trial / "evaluations").glob("*.json")
                  if not p.name.startswith("final")]
        if not curves:
            continue
        last = max(curves, key=lambda row: row["ppo_actions"])
        initial = read_json(trial / "evaluations" / "initial.json")
        row = {"name": name, "arm": config["arm"], "settings": config, **read_json(summary),
               "last_curve_score": last["mean_score"], "last_curve_animals": last["animal_mean"],
               "last_minus_initial": paired(last, initial), "checkpoint": str(trial / "best.pt")}
        final = trial / "final_comparison.json"
        if final.exists():
            comparison = read_json(final)
            row["comparison64"] = {key: comparison[key]["mean_score"]
                                   for key in ("initial", "latest", "best") if key in comparison}
            row["paired64"] = comparison.get("paired_best_minus_initial")
        rows.append(row)
    write_json(stage / "results.json", rows)
    return rows


def publish(root, state, config, repo):
    cpu, gpu, process_gpu = state.get("controller_seconds", 0.), 0., 0.
    launch = root / "launch.json"
    if launch.exists():
        cpu += read_json(launch).get("preparation_cpu_core_seconds", 0.)
    preflight = root / "preflight_cost.json"
    if preflight.exists():
        cost = read_json(preflight)
        cpu += cost["cpu_core_seconds"]
        gpu += cost["gpu_seconds"]
    for p in root.glob("round_*/ppo/state.json"):
        cost = read_json(p).get("cost", {})
        # Outer controller and stage controller are the same process/core.
        cpu += cost.get("cpu_core_hours", 0.) * 3600 - read_json(p).get("controller_seconds", 0.)
        gpu += cost.get("gpu_hours", 0.) * 3600
        process_gpu += cost.get("gpu_process_hours", 0.) * 3600
    teacher = repo / config["fresh_teacher"]
    if (teacher / "watchdog_result.json").exists():
        cpu += read_json(teacher / "watchdog_result.json")["allocated_cpu_core_seconds"]
    elif (teacher / "allocation.json").exists():
        allocation = read_json(teacher / "allocation.json")
        cpu += len(allocation["cpu_affinity"]) * (time.time() - allocation["started_unix"])
    if (root / "finalization_cost.json").exists():
        final = read_json(root / "finalization_cost.json")
        # CPU already included in outer controller reservation.
        gpu += final["gpu_seconds"]
    state.update(updated_unix=time.time(), remaining_seconds=max(0., config["deadline_unix"] - time.time()),
                 cost={"cpu_core_hours": cpu / 3600, "gpu_hours": gpu / 3600,
                       "gpu_process_hours": process_gpu / 3600,
                       "basis": "all attempts, preflight, expert generation and controller; physical GPU interval union per round"})
    write_json(root / "state.json", state)
    retained = read_json(root / "retained_best.json") if (root / "retained_best.json").exists() else {}
    rows = []
    for result in sorted(root.glob("round_*/results.json")):
        for row in read_json(result):
            rows.append(f"|{result.parent.name}|{row['name']}|{row['last_curve_score']:.2f}|"
                        f"{row['best_curve_score']:.2f}|{row.get('comparison64', {}).get('best', '—')}|")
    (root / "REPORT.md").write_text(
        "# Harmonies V1.2.12 自主模型迭代\n\n"
        f"状态：{state['phase']}；阶段：{state.get('stage')}；硬截止：2026-10-20 22:46:34 CST。\n\n"
        "目标为纯模型完整自然终局；32开发局用于选择，64开发局用于比较；重复使用的开发集不是最终泛化证明。正式测试未参与。\n\n"
        "|轮次|方法|latest/32|best/32|best/64|\n|---|---|---:|---:|---:|\n" + "\n".join(rows)
        + f"\n\n保留模型：{retained.get('checkpoint', 'parent pending')}。\n\n资源：{state['cost']}。\n"
        + "\n各轮 decision.json 记录读入的结果和自动选择理由；所有旧模型保留。latest 是固定训练终点，best 是开发曲线选择，二者不能混淆。\n")
    if (root / "confirmation.json").exists():
        final = read_json(root / "confirmation.json")
        with (root / "REPORT.md").open("a") as stream:
            stream.write(f"\n冻结后一次性 confirmation/128：旧模型 {final['parent']['mean_score']:.3f}，"
                         f"候选 {final['candidate']['mean_score']:.3f}，配对差 {final['paired']}。\n"
                         "此结果不再反馈训练或选择。候选未可靠胜出时应继续保留旧模型。\n")


def finalize(root, repo, config):
    """Freeze once; held-out confirmation never becomes another tuning signal."""
    from boardbench.v127.preflight import reserve
    from boardbench.v130.training import export_checkpoint
    from .training import Trainer
    retained = read_json(root / "retained_best.json")
    frozen = root / "finalization.json"
    if not frozen.exists():
        write_json(frozen, {"candidate": retained, "unix": time.time(),
                   "protocol": "freeze then one confirmation[:128] comparison; no subsequent training"})
    elif read_json(frozen)["candidate"] != retained:
        raise ValueError("cannot change candidate after confirmation freeze")
    model = root / "model" / "best.npz"
    if not model.exists():
        model.parent.mkdir(exist_ok=True)
        export_checkpoint(retained["checkpoint"], model)
    if (root / "confirmation.json").exists() or time.time() >= config["deadline_unix"] - 120:
        return
    lock = reserve(repo, config)
    started = time.monotonic()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuids"][0]
        base = read_json(repo / config["base_config"])
        manifest = read_json(repo / base["split_manifest"])
        seeds = manifest["sets"]["confirmation"][:128]
        assert not set(seeds) & (set(manifest["sets"]["train"]) | set(manifest["sets"]["development"]))
        plan = read_json(root / "round_01" / "decision.json")
        trial = build_jobs(repo, root, config, plan, plan["parent"], plan["corpus"], 1, 120)[0][1]
        results = {}
        for name, spec in (("parent", checkpoint_spec(repo / base["parent"])),
                           ("candidate", {k: retained[k] for k in ("checkpoint", "sha256", "source_id")})):
            trainer = Trainer(trial, "cuda")
            try:
                trainer.import_sampling_continuation(spec)
                results[name] = trainer.evaluate(seeds)
            finally:
                trainer.close()
        results["paired"] = paired(results["candidate"], results["parent"])
        results["candidate"] ["checkpoint"] = retained
        results["not_used_for_tuning"] = True
        write_json(root / "confirmation.json", results)
    finally:
        write_json(root / "finalization_cost.json", {"gpu_seconds": time.monotonic() - started,
                   "cpu_core_seconds": time.monotonic() - started, "includes_failed_attempt": True})
        lock.close()


def run(root, repo):
    config, launch = read_json(root / "config.json"), read_json(root / "launch.json")
    if os.uname().nodename != config["host"] or source_id() != launch["source_id"]:
        raise ValueError("authorized host and frozen source required")
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, {launch["cpus"][-1]})
    state = read_json(root / "state.json") if (root / "state.json").exists() else {}
    if state.get("phase") == "completed":
        return
    started = time.monotonic()
    previous_controller = state.get("controller_seconds", 0.)
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    def status(stage, phase="running"):
        state.update(stage=stage, phase=phase, pid=os.getpid(),
                     controller_seconds=previous_controller + time.monotonic() - started)
        publish(root, state, config, repo)
    try:
        if (root / "finalization.json").exists():
            finalize(root, repo, config)
            status("frozen final comparison", "completed")
            return
        corpus = prepare_data(repo, root, config)
        base = read_json(repo / config["base_config"])
        parent = checkpoint_spec(repo / base["parent"])
        previous, stagnation = [], 0
        best_score = None
        if (root / "retained_best.json").exists():
            # Do not use the latest checkpoint as parent when replaying a
            # completed decision; each stage has its immutable parent record.
            best_score = read_json(root / "retained_best.json")["curve_score"]
        for number in range(1, config["max_rounds"] + 1):
            stage = root / f"round_{number:02d}"
            if stopping[0] or time.time() >= config["deadline_unix"] - config["finish_buffer_seconds"]:
                break
            stage.mkdir(exist_ok=True)
            ready = stage / "decision.json"
            if not ready.exists():
                corpus, data_reason = refresh_corpus(repo, root, config, corpus)
                plan = choose_plan(number, previous, expanded="expanded" in corpus["path"], stagnation=stagnation)
                plan.update(parent=parent, corpus=corpus, data_reason=data_reason, created_unix=time.time())
                write_json(ready, plan)
            else:
                plan = read_json(ready)
            training = stage / "ppo"
            if not training.exists():
                seconds = min(config["max_round_seconds"],
                              config["deadline_unix"] - time.time() - config["finish_buffer_seconds"])
                if seconds < 600:
                    break
                jobs = build_jobs(repo, root, config, plan, plan["parent"], plan["corpus"], number, seconds)
                training.mkdir()
                (training / "logs").mkdir()
                (training / "source").symlink_to(root / "source", target_is_directory=True)
                write_json(training / "jobs.json", jobs)
                write_json(training / "config.json", {**config, "training": {
                    "workers": config["workers"], "target_actions": jobs[0][1]["target_actions"]}})
                write_json(training / "launch.json", launch)
            stage_state = training / "state.json"
            finished = stage_state.exists() and read_json(stage_state).get("phase") == "finished"
            status(f"round {number}: training")
            if not finished:
                execute(training, repo, training_module="boardbench.v132.training",
                        reporter=stage_report, expected_host=config["host"],
                        absolute_deadline=config["deadline_unix"])
            previous = summarize(stage)
            if time.time() >= config["deadline_unix"] - config["finish_buffer_seconds"]:
                break
            if len(previous) != 8 or any(row["status"] not in ("completed", "time_limit") for row in previous):
                raise RuntimeError("incomplete stage; saved checkpoints, do not silently replace missing experiments")
            if best_score is None:
                initial = read_json(training / "trials" / previous[0]["name"] / "evaluations" / "initial.json")
                best_score = initial["mean_score"]
                write_json(root / "retained_best.json", {**parent, "curve_score": best_score,
                           "selection": "frozen mature parent; 32 development panel"})
            winner = max(previous, key=lambda row: row["best_curve_score"])
            if winner["best_curve_score"] > best_score:
                best_score = winner["best_curve_score"]
                parent = checkpoint_spec(winner["checkpoint"])
                write_json(root / "retained_best.json", {**parent, "curve_score": best_score,
                           "round": number, "trial": winner["name"],
                           "selection": "32 development curve only; candidate, not proven generalization gain",
                           "comparison64": winner.get("comparison64"), "paired64": winner.get("paired64")})
                stagnation = 0
            else:
                parent = {k: read_json(root / "retained_best.json")[k]
                          for k in ("checkpoint", "sha256", "source_id")}
                stagnation += 1
            write_json(stage / "decision_outcome.json", {"retained": parent, "curve_score": best_score,
                       "stagnation_rounds": stagnation, "selection_panel": "development[:32]"})
            status(f"round {number}: analyzed")
        if (root / "retained_best.json").exists() and not stopping[0]:
            status("freeze and final confirmation")
            finalize(root, repo, config)
        status("saved; absolute window exhausted or round cap reached",
               "interrupted" if stopping[0] else "completed")
    except BaseException as exc:
        state["error"] = repr(exc)
        status(state.get("stage", "startup"), "failed")
        raise
    finally:
        lock.close()


def guard(root, repo):
    """Independent absolute deadline survives blocked training or reporting."""
    config = read_json(root / "config.json")
    launch = read_json(root / "launch.json")
    os.sched_setaffinity(0, {launch["cpus"][-1]})
    with (root / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-u", "-m", "boardbench.v132.pipeline",
                                  "run", "--repo", str(repo), "--root", str(root)], start_new_session=True)
        write_json(root / "guard.json", {"pid": os.getpid(), "child_pid": child.pid,
                   "started_unix": time.time(), "deadline_unix": config["deadline_unix"]})
        signalled = False
        while child.poll() is None:
            if time.time() >= config["deadline_unix"] - 120 and not signalled:
                child.terminate()
                signalled = True
            if time.time() >= config["deadline_unix"]:
                kill_tree(child.pid)
                break
            time.sleep(5)
        code = child.wait()
        write_json(root / "guard_result.json", {"returncode": code, "ended_unix": time.time(),
                   "deadline_unix": config["deadline_unix"], "deadline_signalled": signalled})
        if code:
            raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "resume", "run", "guard", "report"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
    if not root.is_relative_to(repo / "outputs" / "v132"):
        raise ValueError("dedicated V132 project output required")
    if args.mode == "prepare":
        config = read_json(args.config)
        root.mkdir(parents=True, exist_ok=True)
        print(prepare_data(repo, root, config), flush=True)
        return
    if args.mode in ("launch", "resume"):
        config = read_json(args.config) if args.mode == "launch" else read_json(root / "config.json")
        if os.uname().nodename != config["host"] or time.time() >= config["deadline_unix"] - 600:
            raise ValueError("authorized a226 host and unexpired original window required")
        if args.mode == "launch":
            if (root / "launch.json").exists():
                raise ValueError("launch exists; use resume without extending deadline")
            receipt = read_json(root / "preflight_cost.json")
            if not receipt.get("passed") or receipt["source_id"] != source_id(repo):
                raise ValueError("successful version-matched actual CUDA preflight required")
            all_cpus = physical_cpus()
            cpus = all_cpus[:min(128, len(all_cpus) // 2)]
            validate_allocation(cpus, config["gpu_uuids"], config["workers"] + 1, 2, 8)
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "config.json", config)
            for relative in source_files(repo):
                dest = root / "source" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, dest)
            write_json(root / "launch.json", {"host": config["host"], "started_unix": time.time(),
                       "repo": str(repo),
                       "preparation_cpu_core_seconds": max(0., time.time() - config["started_unix"]),
                       "source_id": source_id(root / "source"), "cpus": cpus,
                       "physical_cpu_count": len(all_cpus), "physical_cpu_cap": len(all_cpus) // 2,
                       "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()})
        os.environ.update(PYTHONPATH=str(root / "source" / "src"), PYTHONNOUSERSITE="1",
                          OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          NUMEXPR_NUM_THREADS="1")
        os.execv(sys.executable, [sys.executable, "-u", "-m", "boardbench.v132.pipeline",
                                "guard", "--repo", str(repo), "--root", str(root)])
    elif args.mode == "run":
        run(root, repo)
    elif args.mode == "guard":
        guard(root, repo)
    else:
        publish(root, read_json(root / "state.json"), read_json(root / "config.json"), repo)


if __name__ == "__main__":
    main()
