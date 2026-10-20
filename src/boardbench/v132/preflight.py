"""Bounded real-CUDA, native reset and mature-parent PPO gates on a226."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import read_json, write_json, source_id
from boardbench.v127.preflight import reserve
from boardbench.v130.pipeline import checkpoint_spec
from .pipeline import build_jobs, choose_plan, prepare_data
from .training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
    config = read_json(args.config)
    if os.uname().nodename != config["host"] or time.time() >= config["deadline_unix"] - 600:
        raise ValueError("authorized host and original deadline required")
    out = root / f"preflight_attempt_{len(list(root.glob('preflight_attempt_*'))) + 1:02d}"
    out.mkdir(exist_ok=False)
    lock = reserve(repo, config)
    started = time.monotonic()
    os.sched_setaffinity(0, {56, 57, 58, 59})
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuids"][0]
    receipt = {"source_id": source_id(), "gpu": config["gpu_uuids"][0], "checks": []}
    try:
        env = {**os.environ, "BOARDBENCH_V132_CUDA": "1"}
        with (out / "pytest.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                     str(repo / "tests/test_v132.py") + "::test_cuda_actual_resume",
                                     "--basetemp", str(out / "pytest")],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError("actual CUDA resume test failed; inspect preflight/pytest.log")
        base = read_json(repo / config["base_config"])
        corpus = prepare_data(repo, root, config)
        plan = choose_plan(1, [], expanded=False, stagnation=0)
        jobs = build_jobs(repo, root, config, plan, checkpoint_spec(repo / base["parent"]), corpus, 1, 180)
        # Mature parent + actual admitted expert corpus, not only random fixtures.
        for index in (0, 4, 6):
            name, trial = jobs[index]
            trial = {**trial, "batch_episodes": 32, "workers": 2, "epochs": 2,
                     "trajectory_directory": str(out / name / "trajectories")}
            trainer = Trainer(trial, "cuda")
            try:
                trainer.import_sampling_continuation(trial["continuation"])
                before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
                evaluation = trainer.evaluate(trial["curve_seeds"][:4])
                assert not any(g["curriculum_reset"] for g in evaluation["games"])
                assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
                row = trainer.batch()
                assert row["initial_ratio_max_error"] < 1e-4 and row["actor_change_l2"] > 0
                path = out / name / "latest.pt"
                trainer.save(path)
                expected = trainer.batch()
                weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            finally:
                trainer.close()
            other = Trainer(trial, "cuda")
            try:
                other.load(path)
                actual = other.batch()
                assert expected["games"] == actual["games"]
                for k, value in other.model.state_dict().items():
                    torch.testing.assert_close(value, weights[k], rtol=1e-10, atol=1e-10)
            finally:
                other.close()
            receipt["checks"].append({"arm": name, "initial_ratio_max_error": row["initial_ratio_max_error"],
                                     "actor_change_l2": row["actor_change_l2"], "reset_count": row["reset_count"],
                                     "actions_per_second": row["actions"] / row["batch_seconds"],
                                     "evaluation_score": evaluation["mean_score"], "resume": "passed"})
            print(receipt["checks"][-1], flush=True)
        receipt["passed"] = True
    finally:
        elapsed = time.monotonic() - started
        receipt.update(cpu_core_seconds=4 * elapsed, gpu_seconds=elapsed, elapsed_seconds=elapsed,
                       ended_unix=time.time(), accounting="four reserved CPUs; one physical GPU; failed gates included")
        write_json(out / "receipt.json", receipt)
        attempts = [read_json(p) for p in root.glob("preflight_attempt_*/receipt.json")]
        total = {**receipt, "cpu_core_seconds": sum(r["cpu_core_seconds"] for r in attempts),
                 "gpu_seconds": sum(r["gpu_seconds"] for r in attempts), "attempts": len(attempts)}
        write_json(root / "preflight_cost.json", total)
        lock.close()


if __name__ == "__main__":
    main()
