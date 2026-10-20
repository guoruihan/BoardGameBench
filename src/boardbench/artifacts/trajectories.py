"""Lossless Harmonies trajectory archive; training access is separate and strict.

Archive all scores and failed attempts. Never replace behavior actions with
teacher labels, infer behavior probabilities from search scores, or silently
admit development/test experience into training.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from .store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v121.registry import validate_manifest

FORMAT = "harmonies_trajectory_archive_v1"
CAMPAIGN_FILES = ("plan.json", "splits.json", "allocation.json", "status.json")
EXPERT_BEHAVIORS = {"search", "learned_search", "heuristic"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode())


def discover(root, destination):
    """Skip duplicated source/checkpoint trees, not low-performing experiments."""
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in
                         {"source", "__pycache__", ".git", ".venv", ".venv-train", "checkpoints"}
                         and (Path(directory) / d).resolve() != destination)
        path = Path(directory)
        if "decisions.jsonl" in files and path.parent.name == "episodes":
            yield path / "decisions.jsonl", "episode"
        if "semantic_trajectories.jsonl" in files:
            yield path / "semantic_trajectories.jsonl", "legacy_raw_only"


def _read_optional(path):
    return read_json(path) if path.is_file() else {}


def _split_context(campaign):
    plan, manifest = campaign.get("plan.json", {}), campaign.get("splits.json", {})
    declared = plan.get("split", "unknown")
    split = "train" if declared == "train_calibration" else declared
    try:
        validate_manifest(manifest)
        if plan.get("split_sha256") != manifest["sha256"]:
            raise ValueError("plan/split hash mismatch")
    except (ValueError, KeyError, TypeError) as exc:
        return split, frozenset(), str(exc)
    return split, frozenset(manifest["sets"].get(split, [])), None


def _split(campaign, job):
    split, seeds, error = _split_context(campaign)
    if error is None and job.get("environment_seed") not in seeds:
        error = "seed outside declared split"
    return split, error is None, error


def _behavior(job):
    variant = job.get("params", {}).get("continuation")
    if variant == "dagger":
        return "student_with_expert_labels"
    if variant == "student":
        return "student"
    if variant in ("student_roots", "student_rollout"):
        return "learned_search"
    if job.get("method") == "heuristic":
        return "heuristic"
    if job.get("method") in ("terminal", "short"):
        return "search"
    return "unknown"


def _store_blob(root, data):
    digest = sha(data)
    path = root / "blobs" / f"{digest}.jsonl.gz"
    if path.exists():
        if gzip.decompress(path.read_bytes()) != data:
            raise ValueError(f"corrupt existing archive blob: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(gzip.compress(data, compresslevel=6, mtime=0))
        temporary.replace(path)
    return digest, path.stat().st_size


def archive(source, destination, *, progress=None, workers=4):
    """Add immutable entries; rerunning after interruption resumes/deduplicates."""
    source, root = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or source == root or not 1 <= workers <= 8:
        raise ValueError("distinct existing source directory required")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (root / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / "archive.json"
        if marker.exists() and read_json(marker).get("format") != FORMAT:
            raise ValueError("incompatible archive")
        if not marker.exists():
            write_json(marker, {"format": FORMAT, "created_unix": time.time(),
                               "source_root": str(source),
                               "scope": "actual episode logs plus raw-only legacy semantic logs",
                               "backup_scope": "local copy; not an offsite backup"})
        campaigns, tasks = {}, []
        skipped = []
        blob_locks, lock_guard = {}, threading.Lock()
        for path, kind in discover(source, root):
            try:
                owner = path.parents[2] if kind == "episode" else path.parent
                if owner not in campaigns:
                    campaign = {name: _read_optional(owner / name) for name in CAMPAIGN_FILES}
                    campaign_id = identity({"source": str(owner), "metadata": campaign})
                    campaigns[owner] = campaign, campaign_id, _split_context(campaign)
                    campaign_path = root / "campaigns" / f"{campaign_id}.json"
                    if not campaign_path.exists():
                        write_json(campaign_path, {"source_run": str(owner), "metadata": campaign})
                campaign, campaign_id, split_context = campaigns[owner]
                status = campaign["status.json"]
                if status.get("active_count", 0) or status.get("phase") == "running":
                    skipped.append({"path": str(path), "reason": "campaign may still be writing"})
                    continue
                tasks.append((path, kind, owner, campaign, campaign_id, split_context))
            except (OSError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})

        def save_one(task):
            path, kind, owner, campaign, campaign_id, split_context = task
            try:
                before = path.stat()
                data = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    return 0, {"path": str(path), "reason": "source changed during read"}
                # No need to open or analyse held-out outcomes during archiving.
                # Metadata and raw bytes are retained; only train readers replay.
                job = _read_optional(path.parent / "job.json") if kind == "episode" else {}
                result = _read_optional(path.parent / "result.json") if kind == "episode" else {}
                split, seeds, reason = split_context
                if reason is None and job.get("environment_seed") not in seeds:
                    reason = "seed outside declared split"
                verified = reason is None
                # Different blobs write concurrently; identical (including empty)
                # logs share a lock so .partial replacement cannot race.
                with lock_guard:
                    blob_lock = blob_locks.setdefault(sha(data), threading.Lock())
                with blob_lock:
                    digest, compressed_bytes = _store_blob(root, data)
                terminal = result.get("status") == "completed" and result.get("natural_terminal") is True
                entry = {
                    "format": FORMAT, "kind": kind, "source_path": str(path),
                    "campaign": campaign_id, "source_run": str(owner),
                    "source_id": campaign["allocation.json"].get("source_id"),
                    "sha256": digest, "raw_bytes": len(data), "compressed_bytes": compressed_bytes,
                    "job": job, "result": result, "split": split,
                    "split_verified": verified, "split_error": reason,
                    "behavior": _behavior(job), "complete_claimed": terminal,
                    "train_candidate": kind == "episode" and split == "train" and verified and terminal,
                    "replay_verified": False,
                    "behavior_probability_semantics": "unknown unless explicitly recorded; never inferred from candidate scores",
                    "legacy_warning": ("May contain private endgame snapshots; raw-only, not learner input."
                                       if kind != "episode" else None),
                }
                entry_id = identity(entry)
                target = root / "entries" / f"{entry_id}.json"
                if not target.exists():
                    write_json(target, entry)
                    return 1, None
                elif read_json(target) != entry:
                    raise ValueError("archive entry identity mismatch")
                return 0, None
            except (OSError, json.JSONDecodeError) as exc:
                # Keep previous successful entries; make omissions visible.
                return 0, {"path": str(path), "reason": str(exc)}

        scanned = added = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for new, error in pool.map(save_one, tasks):
                if error:
                    skipped.append(error)
                else:
                    scanned += 1
                    added += new
                if progress and scanned and scanned % 100 == 0:
                    progress({"scanned": scanned, "new_entries": added})
        counts, behaviors = Counter(), Counter()
        raw_bytes = compressed_bytes = total = candidates = 0
        seen_blobs = set()
        index = root / "index.jsonl"
        temporary = root / "index.jsonl.partial"
        with temporary.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            paths = sorted((root / "entries").glob("*.json"))
            for path, entry in zip(paths, pool.map(read_json, paths)):
                if identity(entry) != path.stem:
                    raise ValueError("corrupt archive entry")
                stream.write(json.dumps({"id": path.stem, **entry}, allow_nan=False) + "\n")
                total += 1
                counts[entry["split"]] += 1
                behaviors[entry["behavior"]] += 1
                candidates += entry["train_candidate"]
                if entry["sha256"] not in seen_blobs:
                    seen_blobs.add(entry["sha256"])
                    raw_bytes += entry["raw_bytes"]
                    compressed_bytes += entry["compressed_bytes"]
        temporary.replace(index)
        summary = {"format": FORMAT, "entries": total, "unique_blobs": len(seen_blobs),
                   "new_entries": added, "scanned": scanned,
                   "by_split": dict(counts), "by_behavior": dict(behaviors),
                   "train_candidates_pending_replay": candidates,
                   "unique_raw_bytes": raw_bytes, "compressed_bytes": compressed_bytes,
                   "elapsed_seconds": time.monotonic() - started, "workers": workers, "skipped": skipped,
                   "note": "Counts include weak, failed, student and calibration attempts; not all expert games."}
        write_json(root / "summary.json", summary)
        return summary


def read_train_episode(root, entry_id, *, excluded_seeds=(), expert_only=True):
    """Return transitions ONLY AFTER the entire actual episode replays exactly.

    ``excluded_seeds`` includes all caller's train-internal validation seeds.
    This reader is independent of the archive's advisory train_candidate flag.
    Environment/search seeds are provenance, not observation features.
    """
    root = Path(root)
    if len(entry_id) != 64 or any(c not in "0123456789abcdef" for c in entry_id):
        raise ValueError("invalid archive entry id")
    entry = read_json(root / "entries" / f"{entry_id}.json")
    if identity(entry) != entry_id or entry["format"] != FORMAT or entry["kind"] != "episode":
        raise ValueError("invalid episode entry")
    campaign = read_json(root / "campaigns" / f"{entry['campaign']}.json")
    if identity({"source": campaign["source_run"], "metadata": campaign["metadata"]}) != entry["campaign"]:
        raise ValueError("campaign metadata hash mismatch")
    split, verified, _ = _split(campaign["metadata"], entry["job"])
    if split != "train" or not verified:
        raise ValueError("only verified train-split trajectories may train")
    job, result = entry["job"], entry["result"]
    if job["environment_seed"] in set(excluded_seeds):
        raise ValueError("train-internal validation environment is excluded")
    if expert_only and _behavior(job) not in EXPERT_BEHAVIORS:
        raise ValueError("student or unknown behavior is not an expert trajectory")
    if result.get("status") != "completed" or result.get("natural_terminal") is not True:
        raise ValueError("incomplete trajectory is audit-only")
    if any(job.get(k) != result.get(k) for k in
           ("id", "environment_seed", "search_seed", "method", "multiplier", "params")):
        raise ValueError("job/result provenance mismatch")
    data = gzip.decompress((root / "blobs" / f"{entry['sha256']}.jsonl.gz").read_bytes())
    if sha(data) != entry["sha256"]:
        raise ValueError("trajectory hash mismatch")
    engine = Harmonies()
    obs, _ = engine.reset(job["environment_seed"])
    transitions = []
    for number, line in enumerate(data.splitlines(), 1):
        row = json.loads(line)
        if row["decision"] != number or obs["terminated"]:
            raise ValueError("noncontiguous or post-terminal trajectory")
        if row.get("observation_before", obs) != obs:
            raise ValueError("pre-observation replay mismatch")
        action = row["action"]  # NOT search.expert_action!
        if action not in obs["legal_actions"]:
            raise ValueError("illegal behavior action")
        search = row.get("search", {})
        teacher = search.get("expert_action")
        if teacher is not None and teacher not in obs["legal_actions"]:
            raise ValueError("illegal counterfactual expert label")
        if expert_only and (teacher is not None or search.get("labels_only")):
            raise ValueError("teacher labels do not make student roll-ins expert trajectories")
        step = engine.step(action)
        after = step.observation
        delta = after["score_breakdown"]["total"] - obs["score_breakdown"]["total"]
        if after != row["observation"] or row.get("raw_score_delta", delta) != delta:
            raise ValueError("successor/reward replay mismatch")
        for key, expected in (("reward", step.reward), ("terminated", step.terminated),
                              ("truncated", step.truncated)):
            if key in row and row[key] != expected:
                raise ValueError(f"{key} replay mismatch")
        transitions.append({
            "episode_id": entry_id, "decision": number,
            "observation": obs, "action": action, "next_observation": after,
            "environment_reward": step.reward, "raw_score_delta": delta,
            "terminated": step.terminated, "truncated": step.truncated,
            "score_delta_return_to_go": result["score"] - obs["score_breakdown"]["total"],
            "expert_action": teacher, "search": search,
            "behavior_log_probability": row.get("behavior_log_probability"),
            "candidate_value_semantics": "continuation-specific search estimate, not observed reward or optimal Q",
        })
        obs = after
    if (not transitions or not obs["terminated"] or obs["score"] != result["score"]
            or len(transitions) != result.get("real_steps")):
        raise ValueError("incomplete or inconsistent terminal trajectory")
    return transitions


def verify_archive(root):
    """Verify all saved bytes and provenance; no evaluation outcomes are analysed."""
    root = Path(root)
    blobs, campaigns = set(), set()
    entries = raw_bytes = 0
    if read_json(root / "archive.json").get("format") != FORMAT:
        raise ValueError("incompatible archive")

    def verify_blob(pair):
        digest, length = pair
        data = gzip.decompress((root / "blobs" / f"{digest}.jsonl.gz").read_bytes())
        if sha(data) != digest or len(data) != length:
            raise ValueError("trajectory hash/length mismatch")
        return len(data)

    blob_lengths = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = sorted((root / "entries").glob("*.json"))
        for path, entry in zip(paths, pool.map(read_json, paths)):
            if entry.get("format") != FORMAT or identity(entry) != path.stem:
                raise ValueError("corrupt archive entry")
            if entry["campaign"] not in campaigns:
                campaign = read_json(root / "campaigns" / f"{entry['campaign']}.json")
                if identity({"source": campaign["source_run"], "metadata": campaign["metadata"]}) != entry["campaign"]:
                    raise ValueError("campaign metadata hash mismatch")
                campaigns.add(entry["campaign"])
            if entry["sha256"] in blob_lengths and blob_lengths[entry["sha256"]] != entry["raw_bytes"]:
                raise ValueError("inconsistent blob length metadata")
            blob_lengths[entry["sha256"]] = entry["raw_bytes"]
            entries += 1
        raw_bytes = sum(pool.map(verify_blob, blob_lengths.items()))
        blobs = set(blob_lengths)
    return {"verified_entries": entries, "verified_blobs": len(blobs),
            "verified_raw_bytes": raw_bytes, "game_replay": "not performed by byte verification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="1-8 local I/O threads")
    args = parser.parse_args()
    if args.verify:
        summary = verify_archive(args.out)
    elif args.source:
        summary = archive(args.source, args.out, workers=args.workers,
                          progress=lambda value: print(json.dumps(value), flush=True))
    else:
        parser.error("--source is required unless --verify is specified")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
