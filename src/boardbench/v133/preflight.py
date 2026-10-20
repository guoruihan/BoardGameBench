"""Bounded actual GPU, initial-policy parity, full-update and exact-resume gates."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import read_json, source_id, write_json
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from .training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
    config = read_json(root / "config.json")
    if os.uname().nodename != config["host"]:
        raise ValueError("authorized a226 required")
    cpus = physical_cpus()
    if len(cpus) < 8:
        raise ValueError("four preflight CPUs must fit half-machine cap")
    os.sched_setaffinity(0, set(cpus[:4]))
    lock = reserve(repo, config)
    out = root / f"preflight_attempt_{len(list(root.glob('preflight_attempt_*'))) + 1:02d}"
    out.mkdir()
    started, extra_gpu = time.monotonic(), 0.
    receipt = {"source_id": source_id(), "checks": [], "passed": False}
    try:
        # All selected cards execute a real FP64 CUDA matmul; nvidia-smi alone is insufficient.
        for index, gpu in enumerate(config["gpu_uuids"]):
            tick = time.monotonic()
            result = subprocess.run([sys.executable, "-c",
                "import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device='cuda',dtype=torch.float64); "
                "assert (x@x).sum().item()==32768; print(torch.cuda.get_device_name())"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, capture_output=True, text=True, timeout=45)
            if index:
                extra_gpu += time.monotonic() - tick
            receipt["checks"].append({"gpu": gpu, "returncode": result.returncode, "output": result.stdout,
                                      "error": result.stderr})
            if result.returncode:
                raise RuntimeError("actual CUDA probe failed")
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuids"][0]
        env = {**os.environ, "BOARDBENCH_V133_CUDA": "1"}
        with (out / "pytest.log").open("w") as stream:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                str(repo / "tests/test_v133.py") + "::test_cuda_exact_resume",
                "--basetemp", str(out / "pytest")], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError("CUDA fixture gate failed; inspect pytest.log")
        reference = None
        for name, original in read_json(root / "jobs.json")[::2]:
            trial = {**original, "batch_episodes": 32, "workers": 2, "epochs": 2,
                     "replay_directory": str(out / name / "replay"),
                     "trajectory_directory": str(out / name / "trajectories")}
            trainer = Trainer(trial, "cuda")
            try:
                trainer.import_sampling_continuation(trial["continuation"])
                initial = trainer.evaluate(trial["curve_seeds"][:8])
                if reference is None:
                    reference = initial["games"]
                elif initial["games"] != reference:
                    raise RuntimeError("mature initial policies differ across arms")
                row = trainer.batch()
                if row["initial_ratio_max_error"] > 1e-4 or row["actor_change_l2"] <= 0:
                    raise RuntimeError("actual mature-policy update failed")
                checkpoint = out / name / "latest.pt"
                trainer.save(checkpoint)
                expected = trainer.batch()
                weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            finally:
                trainer.close()
            other = Trainer(trial, "cuda")
            try:
                other.load(checkpoint)
                actual = other.batch()
                if expected["games"] != actual["games"]:
                    raise RuntimeError("actual mature-policy rollout resume differs")
                for key, value in weights.items():
                    torch.testing.assert_close(value, other.model.state_dict()[key], rtol=1e-10, atol=1e-10)
            finally:
                other.close()
            check = {"trial": name, "initial_score8": initial["mean_score"],
                     "initial_ratio_max_error": row["initial_ratio_max_error"],
                     "actor_change_l2": row["actor_change_l2"],
                     "mc_explained_variance": row["mc_explained_variance"],
                     "gae_explained_variance": row["explained_variance"],
                     "batch_seconds": row["batch_seconds"], "resume": "passed"}
            receipt["checks"].append(check)
            print(check, flush=True)
        receipt["passed"] = True
    finally:
        elapsed = time.monotonic() - started
        receipt.update(elapsed_seconds=elapsed, cpu_core_seconds=4 * elapsed,
                       gpu_seconds=elapsed + extra_gpu, ended_unix=time.time())
        write_json(out / "receipt.json", receipt)
        attempts = [read_json(p) for p in root.glob("preflight_attempt_*/receipt.json")]
        write_json(root / "preflight_cost.json", {
            **receipt, "cpu_core_seconds": sum(r["cpu_core_seconds"] for r in attempts),
            "gpu_seconds": sum(r["gpu_seconds"] for r in attempts), "attempts": len(attempts),
            "basis": "four reserved cores; GPU0 reserved for gate plus actual other-device probe times; all attempts"})
        lock.close()


if __name__ == "__main__":
    main()
