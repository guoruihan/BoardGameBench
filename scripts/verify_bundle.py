"""Restore a review ZIP using only its checkpoint source in a new isolated venv."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

from boardbench.artifacts.store import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    with tempfile.TemporaryDirectory(prefix="boardbench-verify-") as directory:
        base = Path(directory)
        with zipfile.ZipFile(args.bundle) as archive:
            for name in archive.namelist():
                if not (base / name).resolve().is_relative_to(base):
                    raise ValueError("unsafe archive path")
            archive.extractall(base)
        project = base / "boardbench-v0"
        audits = list((project / "runs").glob("*/audit.json"))
        if len(audits) != 1:
            raise ValueError("review ZIP must contain exactly one acceptance evidence directory")
        evidence = audits[0].parent
        checkpoint = evidence / "collaboration_demo/checkpoints/ep_000006"
        # Editable installers write egg-info; keep the original checkpoint immutable.
        install_source = base / "installed_snapshot"
        shutil.copytree(checkpoint / "source", install_source)
        venv.EnvBuilder(with_pip=True).create(base / "venv")
        python = str(base / "venv/bin/python")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(PIP_CONFIG_FILE="/dev/null", PIP_EXTRA_INDEX_URL="", PIP_DISABLE_PIP_VERSION_CHECK="1")
        commands_to_run = [
            ("install_snapshot", [python, "-m", "pip", "install", "-e", str(install_source),
                                  "--index-url", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"]),
            ("restore", [python, "-I", "-m", "boardbench", "evaluate", "--checkpoint", str(checkpoint),
                         "--config", str(project / "configs/risk_eval.json"), "--out", str(base / "restored")]),
        ]
        for name, command in commands_to_run:
            p = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True, timeout=120)
            (out / f"{name}.log").write_text(p.stdout + p.stderr)
            commands.append({"name": name, "argv": command, "returncode": p.returncode})
            write_json(out / "commands.json", commands)
            if p.returncode:
                raise RuntimeError(f"{name} failed: {p.stdout}{p.stderr}")
        result = read_json(base / "restored/summary.json")
        expected = read_json(evidence / "eval_demo/summary.json")
        assert result["status"] == "completed"
        assert result["requested_episode_score_sum"] == expected["requested_episode_score_sum"]
        import json
        def transitions(path):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            return [(e["episode"], e["action"], e["result"]) for e in events if e["kind"] == "transition"]
        assert transitions(base / "restored/events.jsonl") == transitions(evidence / "eval_demo/events.jsonl")
        write_json(out / "verification.json", {"status": "passed", "fresh_venv": True,
                    "installed_only_checkpoint_source": True, "isolated_python": True,
                    "matching_actions_and_scores": True, "evaluation_summary": result})
    print(f"Unpacked checkpoint restored in clean environment: {out}")


if __name__ == "__main__":
    main()
