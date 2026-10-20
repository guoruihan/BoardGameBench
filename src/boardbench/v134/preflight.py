"""Actual CUDA, mature initial-policy parity, throughput, and exact next-update resume."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import read_json, source_id, write_json
from boardbench.v122.campaign import physical_cpus
from boardbench.v133.campaign import wait_resources
from .training import Trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    repo, root = Path(a.repo).resolve(), Path(a.root).resolve()
    config = read_json(root / "config.json")
    if os.uname().nodename != config["host"]:
        raise ValueError("authorized a226 only")
    cpus = physical_cpus()
    if len(cpus) < 16:
        raise ValueError("eight preflight CPUs must fit half-machine cap")
    os.sched_setaffinity(0, set(cpus[:8]))
    waiting = time.monotonic()
    lock = wait_resources(root, repo, config, time.time() + config["resource_wait_seconds"], "preflight")
    waited = time.monotonic() - waiting
    out = root / f"preflight_attempt_{len(list(root.glob('preflight_attempt_*'))) + 1:02d}"
    out.mkdir()
    tick, extra_gpu = time.monotonic(), 0.
    receipt = {"source_id": source_id(), "passed": False, "checks": [], "waiting_seconds": waited}
    try:
        for i, gpu in enumerate(config["gpu_uuids"]):
            start = time.monotonic()
            result = subprocess.run([sys.executable, "-c",
                "import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device='cuda',dtype=torch.float64); "
                "assert (x@x).sum().item()==32768; print(torch.cuda.get_device_name())"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, text=True, capture_output=True, timeout=45)
            if i:
                extra_gpu += time.monotonic() - start
            receipt["checks"].append({"gpu": gpu, "returncode": result.returncode, "output": result.stdout, "error": result.stderr})
            if result.returncode:
                raise RuntimeError("actual CUDA failed")
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuids"][0]
        with (out / "pytest.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                     str(repo / "tests/test_v134.py") + "::test_cuda_resume",
                                     "--basetemp", str(out / "pytest")],
                                    env={**os.environ, "BOARDBENCH_V134_CUDA": "1"},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError("CUDA fixture checks failed")
        reference = None
        for name, original in read_json(root / "stage1/jobs.json")[::2]:
            c = {**original, "batch_episodes": 128, "workers": 6,
                 "trajectory_directory": str(out / name / "trajectories")}
            trainer = Trainer(c, "cuda")
            try:
                trainer.import_sampling_continuation(c["continuation"])
                initial = trainer.evaluate(c["curve_seeds"][:8])
                if reference is None:
                    reference = initial["games"]
                if initial["games"] != reference:
                    raise RuntimeError("mature initial policies differ")
                row = trainer.batch()
                if row["initial_ratio_max_error"] > 1e-4 or row["plan_adapter_change"] <= 0:
                    raise RuntimeError("mature worker update failed")
                if c["arm"] == "manager" and row.get("manager_change", 0) <= 0:
                    raise RuntimeError("manager did not learn")
                checkpoint = out / name / "latest.pt"
                trainer.save(checkpoint)
                expected = trainer.batch()
                weights = {key: {k: v.clone() for k, v in getattr(trainer, key).state_dict().items()}
                           for key in ("model", "manager")}
            finally:
                trainer.close()
            restored = Trainer(c, "cuda")
            try:
                restored.load(checkpoint)
                actual = restored.batch()
                if actual["games"] != expected["games"]:
                    raise RuntimeError("mature next trajectory differs on resume")
                for key, state in weights.items():
                    for k, v in state.items():
                        torch.testing.assert_close(v, getattr(restored, key).state_dict()[k], atol=1e-10, rtol=1e-10)
            finally:
                restored.close()
            speed = expected["actions"] / expected["batch_seconds"]
            check = {"trial": name, "initial8": initial["mean_score"],
                     "ratio_error": row["initial_ratio_max_error"], "adapter_change": row["plan_adapter_change"],
                     "manager_change": row.get("manager_change"), "batch_seconds": expected["batch_seconds"],
                     "actions_per_second": speed, "environment_seconds": expected["environment_seconds"],
                     "update_seconds": expected["update_seconds"], "exact_resume": True}
            receipt["checks"].append(check)
            print(check, flush=True)
        if source_id() != receipt["source_id"]:
            raise RuntimeError("source changed during gate")
        receipt["passed"] = True
    finally:
        elapsed = time.monotonic() - tick
        receipt.update(cpu_core_seconds=8 * elapsed + waited, gpu_seconds=elapsed + extra_gpu,
                       elapsed_seconds=elapsed, ended_unix=time.time())
        write_json(out / "receipt.json", receipt)
        receipts = [read_json(path) for path in root.glob("preflight_attempt_*/receipt.json")]
        write_json(root / "preflight_cost.json", {
            **receipt, "cpu_core_seconds": sum(r["cpu_core_seconds"] for r in receipts),
            "gpu_seconds": sum(r["gpu_seconds"] for r in receipts), "attempts": len(receipts),
            "basis": "eight reserved CPU cores, GPU0 gate interval plus other-device probe times; failed attempts included"})
        lock.close()


if __name__ == "__main__":
    main()
