"""Local policy manifests, verified artifacts and resident inference cache."""
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from boardbench.artifacts.store import digest, read_json, restore_solver_artifacts, validate_checkpoint
from boardbench.solvers import make_solver


class PolicyRegistry:
    def __init__(self, task, directory=None):
        self.task, self.entries, self.cache, self.replicas = task, {}, {}, []
        if directory and Path(directory).is_dir():
            for path in sorted(Path(directory).glob("*/manifest.json")):
                entry = read_json(path)
                if entry["environment_id"] != task["id"] or entry["rules_version"] != task["version"]:
                    raise ValueError("policy registry task/rules mismatch")
                artifact = self._relative(path.parent, entry["artifact_path"])
                if digest(artifact) != entry["artifact_sha256"]:
                    raise ValueError("policy artifact hash mismatch")
                if digest(self._relative(path.parent, entry["config_path"])) != entry["config_sha256"]:
                    raise ValueError("policy config hash mismatch")
                if entry["policy_id"] in self.entries:
                    raise ValueError("duplicate policy ID")
                self.entries[entry["policy_id"]] = (entry, path.parent)
        if not self.entries:
            for method in ("random", "heuristic", "search"):
                self.entries[method] = ({"policy_id": method, "environment_id": task["id"],
                    "rules_version": task["version"], "method": method,
                    "label": method+"（内置配置，未评测）", "validation_mean": None,
                    "solver": {"id": "board", "params": {"method": method, "width": 3,
                               "rollouts": 2, "depth": 2, "max_simulation_steps": 12}}}, None)

    @staticmethod
    def _relative(root, relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("policy paths must remain inside the artifact directory")
        return path

    def list(self):
        return [{k: entry.get(k) for k in ("policy_id", "method", "label", "validation_mean",
                                          "training_steps", "inference_device")}
                for entry, _ in self.entries.values()]

    def get(self, policy_id, *, fresh=False):
        if policy_id not in self.entries:
            raise ValueError("unknown policy")
        if not fresh and policy_id in self.cache:
            return self.cache[policy_id], 0.
        entry, root = self.entries[policy_id]
        started = time.perf_counter()
        if root is None:
            solver = make_solver(entry["solver"], 90210)
        else:
            checkpoint = self._relative(root, entry["checkpoint_path"])
            manifest = validate_checkpoint(checkpoint)
            if manifest["config"]["solver"]["id"] in ("ppo", "compact"):
                import torch
                torch.set_num_threads(2)
            solver = make_solver(manifest["config"]["solver"], 90210)
            replica = TemporaryDirectory(prefix="boardbench-policy-")
            self.replicas.append(replica)
            solver_dir = restore_solver_artifacts(checkpoint, Path(replica.name)/"workspace",
                                                  Path(replica.name)/"solver")
            solver.load(solver_dir)
        if not fresh:
            self.cache[policy_id] = solver
        return solver, time.perf_counter()-started

    def close(self):
        self.cache.clear()
        for replica in self.replicas:
            replica.cleanup()
        self.replicas.clear()
