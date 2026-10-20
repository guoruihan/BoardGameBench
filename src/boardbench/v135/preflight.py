"""Actual four-device CUDA, mature probability/gradient/resume and advice diagnostics."""
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
from boardbench.v134.training import Trainer as PlanTrainer
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
    started = time.monotonic()
    lock = wait_resources(root, repo, config, time.time() + config["resource_wait_seconds"], "preflight")
    waited = time.monotonic() - started
    out = root / f"preflight_attempt_{len(list(root.glob('preflight_attempt_*'))) + 1:02d}"
    out.mkdir()
    tick, extra_gpu = time.monotonic(), 0.
    receipt = {"source_id": source_id(), "passed": False, "checks": [], "waiting_seconds": waited}
    try:
        for index, gpu in enumerate(config["gpu_uuids"]):
            t0 = time.monotonic()
            result = subprocess.run([sys.executable, "-c",
                "import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device='cuda',dtype=torch.float64); "
                "assert (x@x).sum().item()==32768; print(torch.cuda.get_device_name())"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, text=True, capture_output=True, timeout=45)
            if index:
                extra_gpu += time.monotonic() - t0
            receipt["checks"].append({"gpu": gpu, "returncode": result.returncode,
                                      "output": result.stdout, "error": result.stderr})
            if result.returncode:
                raise RuntimeError("actual CUDA failed")
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_uuids"][0]
        with (out / "pytest.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                     str(repo / "tests/test_v135.py") + "::test_cuda_resume",
                                     "--basetemp", str(out / "pytest")],
                                    env={**os.environ, "BOARDBENCH_V135_CUDA": "1"},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError("CUDA fixture checks failed")
        for name, original in read_json(root / "stage1/jobs.json")[::2]:
            c = {**original, "batch_episodes": 128, "workers": 6,
                 "trajectory_directory": str(out / name / "trajectories")}
            t = Trainer(c, "cuda")
            try:
                t.import_sampling_continuation(c["continuation"])
                # This is a training-pool diagnostic, never an independent result.
                probe_seeds = t.train_seeds[:64]
                initial = t.evaluate(probe_seeds)
                if c["method"] == "control":
                    saved = torch.load(c["continuation"]["checkpoint"], map_location="cuda", weights_only=False)
                    parent = PlanTrainer({**saved["config"], "workers": 6}, "cuda")
                    try:
                        parent.model.load_state_dict(saved["model"])
                        if initial["games"] != parent.evaluate(probe_seeds)["games"]:
                            raise RuntimeError("mature control no longer matches its parent")
                    finally:
                        parent.close()
                else:
                    t.model.advice_intervention = "zero"
                    zero = t.evaluate(probe_seeds)
                    t.model.advice_intervention = "normal"
                    write_json(out / f"{name}_initial_diagnostic.json",
                               {"normal": initial, "zero_action_advice": zero,
                                "scope": "fixed first64 training seeds; diagnostic only"})
                row = t.batch()
                if row["initial_ratio_max_error"] > 1e-4 or row["actor_change_l2"] <= 0:
                    raise RuntimeError("mature PPO update failed")
                if c["method"] in ("gated", "sticky") and row["advice_gate_change"] <= 0:
                    raise RuntimeError("trust gate has no gradient")
                checkpoint = out / name / "latest.pt"
                t.save(checkpoint)
                expected = t.batch()
                weights = {k: v.clone() for k, v in t.model.state_dict().items()}
            finally:
                t.close()
            restored = Trainer(c, "cuda")
            try:
                restored.load(checkpoint)
                actual = restored.batch()
                if actual["games"] != expected["games"]:
                    raise RuntimeError("mature next trajectory differs on resume")
                for k, v in weights.items():
                    torch.testing.assert_close(v, restored.model.state_dict()[k], atol=1e-10, rtol=1e-10)
            finally:
                restored.close()
            check = {"trial": name, "initial_training64_score": initial["mean_score"],
                     "initial_training64_animal": initial["animal_mean"],
                     "initial_primary_achieved": initial["primary_achieved"],
                     "ratio_error": row["initial_ratio_max_error"], "gate_change": row["advice_gate_change"],
                     "batch_seconds": expected["batch_seconds"],
                     "actions_per_second": expected["actions"] / expected["batch_seconds"],
                     "exact_resume": True}
            receipt["checks"].append(check)
            write_json(out / "progress.json", receipt)
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
            "basis": "eight reserved CPU cores, GPU0 gate interval plus other-device probes; failures included"})
        lock.close()


if __name__ == "__main__":
    main()
