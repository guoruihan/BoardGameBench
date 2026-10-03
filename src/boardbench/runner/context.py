from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import os
import signal
import subprocess
import time

from boardbench.contracts import ConsultResult, JobResult


def empty_resources():
    return {"action_requests": 0, "environment_steps": 0, "simulation_steps": 0,
            "model_calls": 0, "model_rejections": 0, "model_failures": 0,
            "model_seconds": 0.0, "jobs": 0, "job_rejections": 0,
            "job_seconds": 0.0, "decision_seconds": 0.0,
            "input_tokens": 0, "output_tokens": 0}


class MockModelClient:
    provider = "mock"

    def consult(self, request, timeout):
        # Synthetic output and unknown token usage: never claim real model performance.
        observation = request.get("observation", {})
        return ConsultResult("success", "DRAW" if observation.get("pot", 0) < 3 else "BANK",
                             {"input_tokens": None, "output_tokens": None})


class Context:
    def __init__(self, store, config, task_spec, workspace, *, client=None, clock=time.monotonic):
        self.store, self.config, self.task_spec = store, deepcopy(config), deepcopy(task_spec)
        # Reproducibility seeds live in runner logs, not the policy API. This is
        # a cooperative information boundary, not an adversarial Python sandbox.
        if task_spec["id"] != "risk_collect":
            self.config.pop("seed", None)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode, self.allow_learning = config["mode"], config["allow_learning"]
        self.reference_simulator = None
        self.resources = empty_resources()
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + config["limits"]["max_wall_seconds"]
        self.client = client if client is not None else MockModelClient()
        self.episode = None
        self.phase = "start_task"
        self.decision_id = None
        self.closing = False

    def seconds_left(self):
        return max(0.0, self.deadline - self.clock())

    def expired(self):
        return self.clock() >= self.deadline

    def event(self, kind, **data):
        return self.store.event(kind, episode=self.episode, phase=self.phase,
                                decision_id=self.decision_id, **data)

    def remaining_budget(self):
        return {"wall_seconds": self.seconds_left(),
                "action_requests": max(0, self.config["limits"]["max_action_requests"] - self.resources["action_requests"]),
                "model_calls": max(0, self.config["model"]["max_calls"] - self.resources["model_calls"])}

    def emit(self, metrics):
        self.event("solver_report", source="solver", metrics=metrics)

    def count_simulation_step(self):
        self.resources["simulation_steps"] += 1

    def consult(self, request):
        remaining = self.remaining_budget()
        reason = ("closing" if self.closing else "wall_time" if remaining["wall_seconds"] <= 0
                  else "model_quota" if remaining["model_calls"] <= 0 else None)
        if reason:
            self.resources["model_rejections"] += 1
            self.event("model_rejected", provider=self.client.provider, reason=reason, request=request)
            return ConsultResult("budget_exhausted", error=reason)
        self.resources["model_calls"] += 1
        timeout = min(self.config["model"]["timeout_seconds"], remaining["wall_seconds"])
        call_id = self.event("model_start", provider=self.client.provider, request=request, timeout_seconds=timeout)
        started = self.clock()
        try:
            result = self.client.consult(request, timeout=timeout)
        except Exception as exc:
            result = ConsultResult("failed", error=f"{type(exc).__name__}: {exc}")
        result.elapsed_seconds = self.clock() - started
        if result.elapsed_seconds > timeout:
            result.status = "failed"
            result.error = "client exceeded timeout (cooperative client contract)"
        self.resources["model_seconds"] += result.elapsed_seconds
        if result.status != "success":
            self.resources["model_failures"] += 1
        for name in ("input_tokens", "output_tokens"):
            value = result.usage.get(name)
            old = self.resources[name]
            self.resources[name] = old + value if old is not None and value is not None else None
        self.event("model_end", provider=self.client.provider, call_id=call_id, **asdict(result))
        return result

    def run_job(self, argv, resources=None):
        options = resources or {}
        if set(options) - {"timeout_seconds"}:
            raise ValueError("V0 run_job supports timeout_seconds only, no CPU/GPU allocation")
        if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("run_job requires a nonempty argv list (no shell)")
        timeout = min(float(options.get("timeout_seconds", self.seconds_left())), self.seconds_left())
        if timeout <= 0 or self.closing:
            self.resources["job_rejections"] += 1
            self.event("job_rejected", reason="closing" if self.closing else "wall_time", argv=argv)
            return JobResult("budget_exhausted", None, None, 0.0)
        job_id = self.event("job_start", argv=argv, timeout_seconds=timeout)
        log = self.store.root / "jobs" / f"job_{job_id:06d}.log"
        self.resources["jobs"] += 1
        started = self.clock()
        returncode, error, status = None, None, "failed"
        try:
            with log.open("wb") as output:
                process = subprocess.Popen(argv, cwd=self.workspace, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=timeout)
                    status = "success" if returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    # Linux process group avoids orphaning local training descendants.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait()
                    status = "timeout"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = self.clock() - started
        self.resources["job_seconds"] += elapsed
        relative = str(log.relative_to(self.store.root))
        result = JobResult(status, returncode, relative, elapsed, error)
        self.event("job_end", job_id=job_id, **asdict(result))
        return result
