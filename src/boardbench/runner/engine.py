from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from boardbench.artifacts.store import (ArtifactStore, restore_solver_artifacts, save_checkpoint,
                                       source_id, validate_checkpoint, write_json)
from boardbench.contracts import EpisodeResult, InvalidAction, Transition, derive_seed
from boardbench.environments import get_task
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers import make_solver
from .config import evaluation_config, normalize
from .context import Context, empty_resources


class BudgetStop(Exception):
    pass


def invoke(context, name, function, *args, required=False):
    context.phase = name
    if context.expired() and not required:
        raise BudgetStop("wall_time")
    context.closing = context.expired()
    started = context.clock()
    context.event("hook_start", hook=name)
    callback_started = context.clock()
    try:
        return function(*args)
    finally:
        context.last_callback_seconds = context.clock() - callback_started
        context.event("hook_end", hook=name, elapsed_seconds=context.clock() - started,
                      callback_seconds=context.last_callback_seconds,
                      wall_overrun_seconds=max(0.0, context.clock() - context.deadline))
        context.closing = context.expired()


def resource_delta(before, after):
    delta = {k: after[k] - before[k] if before[k] is not None and after[k] is not None else None
             for k in after}
    if delta["model_calls"] == 0:
        delta["input_tokens"] = delta["output_tokens"] = 0
    return delta


def add_resources(total, values):
    for key, value in values.items():
        total[key] = total[key] + value if total[key] is not None and value is not None else None


def run_episode(env, solver, context, episode, seed):
    context.episode, context.decision_id, context.phase = episode, None, "reset"
    before = deepcopy(context.resources)
    started = context.clock()
    result = EpisodeResult(episode, seed, "truncated", "wall_time", None, 0, 0, started=False)
    try:
        if context.expired():
            raise BudgetStop("wall_time")
        observation, info = env.reset(seed)
        result.started = True
        if context.config["capabilities"]["reference_simulator"]:
            context.reference_simulator = SearchAccess(env.fork, context.count_simulation_step)
        context.event("episode_start", seed=seed, observation=deepcopy(observation), info=info,
                      solver_version=solver.version)
        invoke(context, "start_episode", solver.start_episode, deepcopy(observation), context)
        while True:
            if context.expired():
                result.reason = "wall_time"
                break
            if context.remaining_budget()["action_requests"] <= 0:
                result.reason = "run_action_limit"
                break
            if result.action_requests >= context.config["limits"]["max_actions_per_episode"]:
                result.reason = "episode_action_limit"
                break
            context.phase = "decide"
            context.decision_id = context.event("decision_start", step=result.environment_steps,
                                                 solver_version=solver.version)
            decision_started = context.clock()
            sim_before = context.resources["simulation_steps"]
            action, decision_ok = None, False
            context.last_callback_seconds = 0.
            try:
                action = invoke(context, "decide", solver.decide, deepcopy(observation), context)
                decision_ok = True
            finally:
                decision_seconds = context.clock() - decision_started
                context.resources["decision_seconds"] += decision_seconds
                context.event("decision_end", action=action, solver_version=solver.version,
                              status="returned" if decision_ok else "failed",
                              elapsed_seconds=decision_seconds,
                              solver_seconds=getattr(context, "last_callback_seconds", decision_seconds),
                              simulation_steps=context.resources["simulation_steps"] - sim_before,
                              simulation_count_source="reference_adapter")
            version = solver.version
            if context.expired():
                result.reason = "wall_time"
                context.event("action_not_executed", action=action, reason="wall_time")
                break
            result.action_requests += 1
            context.resources["action_requests"] += 1
            context.phase = "step"
            request_id = context.event("action_request", action=action, solver_version=version)
            try:
                step = env.step(deepcopy(action))
            except InvalidAction as exc:
                context.event("action_error", request_id=request_id, error=str(exc))
                continue
            result.environment_steps += 1
            context.resources["environment_steps"] += 1
            context.event("transition", step=result.environment_steps, request_id=request_id,
                          observation=deepcopy(observation), action=action, result=asdict(step),
                          solver_version=version, decision_seconds=decision_seconds)
            if step.terminated:
                result.status, result.reason = "completed", step.info.get("reason", "terminated")
                result.score = step.info["score"]
            elif step.truncated:
                result.reason = step.info.get("reason", "environment_truncated")
            # Even if the deadline was crossed in env.step, deliver exactly one feedback.
            invoke(context, "on_transition", solver.on_transition,
                   Transition(deepcopy(observation), deepcopy(action), deepcopy(step)), context, required=True)
            observation = step.observation
            if step.terminated or step.truncated:
                break
    except BudgetStop:
        result.reason = "wall_time"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        if result.status != "completed":
            result.status = "failed"
            result.reason = "exception" if result.started else "reset_error"
        context.event("error", error=result.error)
    finally:
        context.decision_id = None
        result.resources = resource_delta(before, context.resources)
        if result.started:
            try:
                feedback = deepcopy(result)
                if context.task_spec["id"] != "risk_collect":
                    feedback.seed = None
                invoke(context, "end_episode", solver.end_episode, feedback, context, required=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result.error = (result.error + "; " if result.error else "") + error
                context.event("error", error=error)
                if result.status != "completed":
                    result.status, result.reason = "failed", "exception"
        result.resources = resource_delta(before, context.resources)
        result.resources["wall_seconds"] = context.clock() - started
        result.resources["wall_overrun_seconds"] = max(0.0, context.clock() - context.deadline)
        record = asdict(result)
        context.phase = "episode_end" if result.started else "reset"
        context.event("episode_end" if result.started else "episode_initialization_failed", result=record)
        context.store.episode(record)
        context.reference_simulator = None
    return record


def initialize_solver(context, config, checkpoint=None, factory=None):
    solver = (factory or make_solver)(config["solver"], derive_seed(config["seed"], "solver"))
    invoke(context, "start_task", solver.start_task, context)
    if checkpoint is not None:
        directory = restore_solver_artifacts(checkpoint, context.workspace,
                                             context.workspace.parent / "restored_solver")
        invoke(context, "load", solver.load, directory)
    return solver


def summary_record(store, config, results, resources, started, checkpoints, error, reason,
                   historical=None):
    counts = Counter(r["status"] for r in results)
    status = ("failed" if error or any(r["error"] for r in results)
              else "completed" if counts["completed"] == config["episodes"] else "stopped")
    summary = {"status": status, "stop_reason": reason, "requested_episodes": config["episodes"],
               "attempted_episodes": len(results),
               "started_episodes": sum(r["started"] for r in results), "completed_episodes": counts["completed"],
               "truncated_episodes": counts["truncated"], "failed_episodes": counts["failed"],
               "episode_errors": sum(bool(r["error"]) for r in results),
               "unstarted_episodes": config["episodes"] - sum(r["started"] for r in results),
               "failure_reasons": dict(Counter(r["reason"] for r in results if r["status"] != "completed")),
               "resources": resources, "checkpoints": checkpoints, "error": error,
               "historical_adaptation": historical, "task": config["task"],
               "solver": config["solver"], "source_id": source_id(),
               "model_provider": config["model"]["provider"], "real_api_verified": False,
               "result_files": {"events": "events.jsonl", "episodes": "episodes.jsonl", "summary": "summary.json"},
               **get_task(config["task"]).aggregate(results, config["episodes"])}
    summary["resources"]["wall_seconds"] = time.monotonic() - started
    store.event("run_stop", status=status, reason=reason, error=error, resources=summary["resources"])
    write_json(store.root / "summary.json", summary)
    return summary


def run(config, output, *, checkpoint=None, solver_factory=None):
    started = time.monotonic()
    config = normalize(config)
    manifest = validate_checkpoint(checkpoint) if checkpoint else None
    if manifest and any(config[k] != manifest["config"][k] for k in ("task", "solver", "capabilities", "model")):
        raise ValueError("resume must retain checkpoint task, solver, capabilities and model config")
    task = get_task(config["task"])
    store = ArtifactStore(output, config)
    context = Context(store, config, deepcopy(task.spec), store.root / "workspace")
    context.started = started
    context.deadline = started + config["limits"]["max_wall_seconds"]
    results, checkpoints, error, reason = [], [], None, "requested_episodes_finished"
    store.event("run_start", mode=config["mode"], task=config["task"], source_checkpoint=str(checkpoint) if checkpoint else None)
    try:
        solver = initialize_solver(context, config, checkpoint, solver_factory)
        env = task.factory()
        for episode in range(1, config["episodes"] + 1):
            if context.expired() or context.remaining_budget()["action_requests"] <= 0:
                reason = "wall_time" if context.expired() else "run_action_limit"
                break
            result = run_episode(env, solver, context, episode, derive_seed(config["seed"], "environment", episode))
            results.append(result)
            completed = sum(r["status"] == "completed" for r in results)
            period = config["checkpoint"]["every_episodes"]
            if (result["status"] == "completed" and not result["error"] and period
                    and completed % period == 0 and not context.expired()):
                context.phase, context.decision_id = "save", None
                checkpoint_started = time.monotonic()
                resources = deepcopy(context.resources)
                resources["wall_seconds"] = checkpoint_started - started
                target = invoke(context, "save", save_checkpoint, store, solver, config,
                                {"started_episodes": len(results), "completed_episodes": completed,
                                 "environment_steps": context.resources["environment_steps"]},
                                resources, context.workspace,
                                manifest.get("cumulative_adaptation_resources", manifest["resources"]) if manifest else None)
                relative = str(target.relative_to(store.root))
                checkpoints.append(relative)
                context.event("checkpoint", path=relative, completed_episodes=completed,
                              elapsed_seconds=time.monotonic() - checkpoint_started)
            if result["error"]:
                reason = "solver_error" if result["started"] else result["reason"]
                break
            if context.expired():
                reason = "wall_time"
                break
    except BudgetStop:
        reason = "wall_time"
    except Exception as exc:
        error, reason = f"{type(exc).__name__}: {exc}", "exception"
        store.event("error", error=error)
    try:
        context.resources["wall_overrun_seconds"] = max(0.0, time.monotonic() - context.deadline)
        return summary_record(store, config, results, context.resources, started, checkpoints, error, reason,
                              {"resources": manifest.get("cumulative_adaptation_resources", manifest["resources"]),
                               "progress": manifest["progress"]} if manifest else None)
    finally:
        store.close()


def evaluate(checkpoint, evaluation, output):
    started = time.monotonic()
    manifest = validate_checkpoint(checkpoint)
    config, seeds = evaluation_config(evaluation, manifest)
    stored_config = {**config, "episode_seeds": seeds,
                     "budget_scope": "fresh per episode, including initialization and load"}
    store = ArtifactStore(output, stored_config)
    task = get_task(config["task"])
    resources, results, error = empty_resources(), [], None
    store.event("run_start", mode="evaluation", source_checkpoint=str(checkpoint),
                source_id=manifest["source_id"], task=config["task"])
    try:
        for episode, seed in enumerate(seeds, 1):
            with TemporaryDirectory(prefix=f"episode_{episode:06d}_", dir=store.root / "workspace") as replica:
                context = Context(store, config, deepcopy(task.spec), Path(replica) / "workspace")
                context.episode = episode
                try:
                    solver = initialize_solver(context, config, checkpoint)
                    result = run_episode(task.factory(), solver, context, episode, seed)
                except Exception as exc:
                    status = "truncated" if isinstance(exc, BudgetStop) else "failed"
                    result = asdict(EpisodeResult(episode, seed, status, "initialization", None, 0, 0,
                                                  error=None if status == "truncated" else str(exc), started=False))
                    result["resources"] = deepcopy(context.resources)
                    result["resources"]["wall_seconds"] = context.clock() - context.started
                    store.episode(result)
                    context.event("episode_initialization_failed", result=result)
                results.append(result)
                add_resources(resources, context.resources)
                context.reference_simulator = None
                # The working copy is discarded on exit, never reused or written back.
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.event("error", error=error)
    try:
        return summary_record(store, config, results, resources, started, [], error,
                              "evaluation_finished" if error is None else "exception",
                              {"resources": manifest.get("cumulative_adaptation_resources", manifest["resources"]),
                               "progress": manifest["progress"],
                               "source_id": manifest["source_id"]})
    finally:
        store.close()
