"""Execute the documented acceptance commands and retain their actual outputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from boardbench.artifacts.store import file_hashes, read_json, source_id, write_json


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def execute(name, command, extra_env=None):
        env = os.environ.copy()
        env.update(extra_env or {})
        started = time.monotonic()
        print(f"[{name}] {' '.join(command)}", flush=True)
        with (out / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        commands.append({"name": name, "argv": command, "cwd": str(root),
                         "environment_overrides": extra_env or {}, "returncode": returncode,
                         "elapsed_seconds": time.monotonic() - started, "log": f"{name}.log"})
        write_json(out / "commands.json", commands)
        if returncode:
            raise RuntimeError(f"{name} exited {returncode}; see {out / (name + '.log')}")

    execute("install", [sys.executable, "-m", "pip", "install", "-e", ".[dev]",
                         "--index-url", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"],
            {"PIP_CONFIG_FILE": "/dev/null", "PIP_EXTRA_INDEX_URL": "", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    execute("pytest", [sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'pytest.xml'}"])
    for solver in ("random", "search", "collaboration"):
        execute(solver, [sys.executable, "-m", "boardbench", "run", "--config",
                         f"configs/risk_{solver}.json", "--out", str(out / f"{solver}_demo")])
    checkpoint = out / "collaboration_demo/checkpoints/ep_000006"
    original = file_hashes(out / "collaboration_demo")
    for name in ("eval_demo", "eval_repeat"):
        execute(name, [sys.executable, "-m", "boardbench", "evaluate", "--checkpoint", str(checkpoint),
                       "--config", "configs/risk_eval.json", "--out", str(out / name)])
    assert original == file_hashes(out / "collaboration_demo")
    trajectories = []
    for name in ("eval_demo", "eval_repeat"):
        trajectories.append([(e["episode"], e["action"], e["result"]) for e in
                             records(out / name / "events.jsonl") if e["kind"] == "transition"])
    assert trajectories[0] == trajectories[1]
    audit = {}
    for name in ("random_demo", "search_demo", "collaboration_demo", "eval_demo", "eval_repeat"):
        summary = read_json(out / name / "summary.json")
        events = records(out / name / "events.jsonl")
        episodes = records(out / name / "episodes.jsonl")
        assert summary["status"] == "completed"
        for field, kind in (("action_requests", "action_request"), ("environment_steps", "transition"),
                            ("model_calls", "model_start"), ("model_rejections", "model_rejected"),
                            ("jobs", "job_start")):
            assert summary["resources"][field] == sum(e["kind"] == kind for e in events)
        assert summary["requested_episode_score_sum"] == sum(e["score"] or 0 for e in episodes)
        assert summary["resources"]["decision_seconds"] == sum(e["elapsed_seconds"] for e in events if e["kind"] == "decision_end")
        audit[name] = {"status": summary["status"], "score_sum": summary["requested_episode_score_sum"],
                       "episodes": summary["completed_episodes"], "resources": summary["resources"]}
    execute("rl_example", [sys.executable, "examples/rl_rollout.py"])
    if args.browser:
        execute("browser", [sys.executable, "scripts/browser_check.py", "--trajectory",
                            str(out / "collaboration_demo/events.jsonl"), "--out", str(out / "ui")])
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    write_json(out / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "execution": "CPU only", "source_id": source_id(),
        "packages": {name: importlib.metadata.version(name) for name in ("boardbench", "pytest", "setuptools")}})
    write_json(out / "audit.json", {"status": "passed", "runs": audit,
                                   "repeated_evaluation_identical": True,
                                   "adaptation_and_checkpoint_unchanged": True,
                                   "real_api_verified": False,
                                   "browser_verified": args.browser})
    print(f"Acceptance evidence: {out}")


if __name__ == "__main__":
    main()
