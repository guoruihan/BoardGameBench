"""Train-only coherent expert episodes; no private snapshots or teacher-Q targets."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.artifacts.trajectories import read_train_episode
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode, index
from boardbench.v121.registry import validate_manifest

FORMAT = "public_expert_suffix_v132"
PUBLIC_KEYS = set(Harmonies().reset(0)[0])


def build(archive, out, manifest_path, excluded_seeds, *, minimum_seconds=300.):
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise ValueError("new corpus path required")
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    excluded = set(excluded_seeds)
    allowed = set(manifest["sets"]["train"]) - excluded
    episodes, seen, skipped = [], set(), []
    for line in (archive / "index.jsonl").open():
        entry = json.loads(line)
        if (not entry["train_candidate"] or entry["behavior"] not in {"search", "learned_search"}
                or entry["job"]["environment_seed"] not in allowed
                or entry["job"].get("params", {}).get("seconds", 0) < minimum_seconds):
            continue
        signature = (entry["source_path"], entry["sha256"])
        if signature in seen:
            continue
        rows = read_train_episode(archive, entry["id"], excluded_seeds=excluded)
        episode = {
            "id": entry["id"], "environment_seed": entry["job"]["environment_seed"],
            "behavior": entry["behavior"], "job": entry["job"], "source_id": entry["source_id"],
            "terminal_score": entry["result"]["score"],
            "animal_score": rows[-1]["next_observation"]["score_breakdown"]["animals"],
            "observations": [r["observation"] for r in rows],
            "actions": [r["action"] for r in rows],
            "reward_deltas": [r["raw_score_delta"] for r in rows],
        }
        episodes.append(episode)
        seen.add(signature)
    if not episodes:
        raise ValueError("no admitted complete expert training episodes")
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = {"format": FORMAT, "rules_version": Harmonies.rules_version,
              "split_sha256": manifest["sha256"], "excluded_seeds": sorted(excluded),
              "minimum_seconds": minimum_seconds, "episodes": episodes}
    raw = json.dumps(corpus, separators=(",", ":"), allow_nan=False).encode()
    temporary = out.with_suffix(".partial")
    temporary.write_bytes(gzip.compress(raw, mtime=0))
    temporary.replace(out)
    metadata = {"sha256": digest(out), "format": FORMAT, "episodes": len(episodes),
                "environments": len({e["environment_seed"] for e in episodes}),
                "states": sum(len(e["actions"]) for e in episodes),
                "split_sha256": manifest["sha256"], "excluded_seeds": sorted(excluded),
                "mean_teacher_score": float(np.mean([e["terminal_score"] for e in episodes])),
                "archive": str(archive.resolve()), "skipped": skipped,
                "information": "public states only; fresh conditional future on each reset",
                "selection": "verified train expert behavior and minimum search budget; no score filter"}
    write_json(out.with_suffix(".json"), metadata)
    return metadata


def load(specification, allowed_seeds):
    path = Path(specification["path"])
    if digest(path) != specification["sha256"]:
        raise ValueError("curriculum corpus checksum mismatch")
    corpus = json.loads(gzip.decompress(path.read_bytes()))
    if corpus["format"] != FORMAT or corpus["rules_version"] != Harmonies.rules_version:
        raise ValueError("curriculum rules/format mismatch")
    episodes = corpus["episodes"]
    if not episodes or any(e["environment_seed"] not in allowed_seeds for e in episodes):
        raise ValueError("curriculum environment split leakage")
    for episode in episodes:
        if (not episode["actions"] or len(episode["actions"]) != len(episode["observations"])
                or len(episode["reward_deltas"]) != len(episode["actions"])
                or sum(episode["reward_deltas"]) != episode["terminal_score"]):
            raise ValueError("incomplete expert curriculum episode")
        for obs, action in zip(episode["observations"], episode["actions"]):
            if (set(obs) != PUBLIC_KEYS or obs["terminated"] or action not in obs["legal_actions"]
                    or obs["rules_version"] != Harmonies.rules_version):
                raise ValueError("invalid curriculum observation or action")
    return corpus


def encoded_sequences(corpus):
    """Preserve episode boundaries for contiguous teacher sequence supervision."""
    result = []
    for episode in corpus["episodes"]:
        enc = [encode(obs, allow_stop=False) for obs in episode["observations"]]
        result.append({"x": np.asarray([v["features"] for v in enc], np.float32),
                       "mask": np.asarray([v["action_mask"] for v in enc], bool),
                       "y": np.asarray([index(o, a, "slots240_v1", False)
                                        for o, a in zip(episode["observations"], episode["actions"])], np.int64)})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--excluded-json", required=True)
    p.add_argument("--minimum-seconds", type=float, default=300.)
    args = p.parse_args()
    print(json.dumps(build(args.archive, args.out, args.splits, read_json(args.excluded_json),
                           minimum_seconds=args.minimum_seconds)))
