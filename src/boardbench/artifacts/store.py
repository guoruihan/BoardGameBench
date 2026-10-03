from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hashes(directory):
    root = Path(directory)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def source_root():
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file():
        raise ValueError("checkpoint requires editable source installation (pip install -e SOURCE)")
    return root


def source_files(root=None):
    root = Path(root) if root is not None else source_root()
    paths = [root / "pyproject.toml"]
    paths += [p for p in (root / "src" / "boardbench").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def source_id(root=None):
    return hashlib.sha256(json.dumps(source_files(root), sort_keys=True).encode()).hexdigest()


class ArtifactStore:
    def __init__(self, output, config):
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ("workspace", "jobs", "checkpoints"):
            (self.root / name).mkdir()
        write_json(self.root / "config.json", config)
        self._events = (self.root / "events.jsonl").open("x", encoding="utf-8")
        self._episodes = (self.root / "episodes.jsonl").open("x", encoding="utf-8")
        self.sequence = 0

    def event(self, kind, *, episode=None, phase=None, decision_id=None, **data):
        self.sequence += 1
        record = {"seq": self.sequence, "run": self.root.name,
                  "kind": kind, "episode": episode, "phase": phase,
                  "decision_id": decision_id, **data}
        self._events.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._events.flush()
        return self.sequence

    def episode(self, result):
        self._episodes.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        self._episodes.flush()

    def close(self):
        self._events.close()
        self._episodes.close()


def save_checkpoint(store, solver, config, progress, resources, workspace, prior_resources=None):
    target = store.root / "checkpoints" / f"ep_{progress['completed_episodes']:06d}"
    temporary = target.with_name(target.name + ".partial")
    temporary.mkdir()
    (temporary / "solver").mkdir()
    solver.save(temporary / "solver")
    # A source snapshot includes any generated code/prompts in the solver workspace.
    shutil.copytree(workspace, temporary / "workspace",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    root = source_root()
    for relative in source_files(root):
        destination = temporary / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    cumulative = dict(resources)
    if prior_resources:
        for key in cumulative:
            previous = prior_resources.get(key, 0)
            cumulative[key] = (cumulative[key] + previous
                               if cumulative[key] is not None and previous is not None else None)
    manifest = {"format_version": 1, "task": config["task"],
                "solver": config["solver"], "loader": "boardbench.solvers.make_solver",
                "config": config, "progress": progress, "resources": resources,
                "cumulative_adaptation_resources": cumulative,
                "resource_capture_point": "before this checkpoint save",
                "source_id": source_id(temporary / "source"),
                "solver_version": solver.version, "python": sys.version,
                "files": file_hashes(temporary)}
    write_json(temporary / "manifest.json", manifest)
    temporary.rename(target)
    return target


def validate_checkpoint(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    actual = file_hashes(path)
    actual.pop("manifest.json", None)
    if actual != manifest["files"]:
        raise ValueError("checkpoint contents do not match manifest hashes")
    if source_id(path / "source") != manifest["source_id"] or source_id() != manifest["source_id"]:
        raise ValueError(f"source version mismatch; copy {path / 'source'} to a separate directory, "
                         "then install that copy with: python -m pip install -e COPY")
    if manifest["loader"] != "boardbench.solvers.make_solver":
        raise ValueError("unsupported solver loader")
    return manifest


def restore_solver_artifacts(checkpoint, workspace, solver_directory):
    """Copy every mutable solver artifact; the caller owns the replica's lifetime."""
    checkpoint = Path(checkpoint)
    solver_directory = Path(solver_directory)
    shutil.copytree(checkpoint / "solver", solver_directory)
    shutil.copytree(checkpoint / "workspace", workspace, dirs_exist_ok=True)
    return solver_directory
