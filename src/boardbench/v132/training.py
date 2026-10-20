"""Public expert-state curricula, coherent replay auxiliary, unchanged true-score PPO."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from boardbench.artifacts.store import digest, read_json
from boardbench.v12.training import policy_distribution, returns
from boardbench.v121.collector import REWARD, SCORE, ANIMAL, COMPLETED, NATURAL, STOPPED
from boardbench.v127.training import Trainer as BaseTrainer, advantages, run as base_run
from .collector import EnvironmentPool
from .data import load, encoded_sequences


def remaining_window(config, additional_actions):
    """Backward curriculum gradually exposes more student-controlled decisions."""
    if config["reset_mode"] == "random":
        return .05, .95
    progress = min(1., max(0., additional_actions / config["curriculum_actions"]))
    start = (1 - progress) * config.get("curriculum_start", .75) + progress * config.get("curriculum_end", .05)
    return start, min(.98, start + .15)


class Trainer(BaseTrainer):
    def __init__(self, config, device="cpu"):
        if config.get("curriculum_version") != "1.2.12":
            raise ValueError("explicit curriculum version required")
        super().__init__(config, device)
        if (config["reset_mode"] not in ("none", "random", "backward")
                or not 0 <= config["reset_fraction"] <= .9
                or not 0 <= config.get("sequence_bc_weight", 0) <= .05
                or config["curriculum_actions"] <= 0
                or config.get("environment_backend") != "native_v129"):
            raise ValueError("invalid bounded curriculum configuration")
        self.corpus = load(config["corpus"], set(self.train_seeds))
        if self.corpus["split_sha256"] != self.manifest["sha256"]:
            raise ValueError("curriculum split manifest mismatch")
        self.sequences = encoded_sequences(self.corpus) if config.get("sequence_bc_weight", 0) else None
        self.last_trajectories = []

    def make_pool(self, count, workers):
        return EnvironmentPool(count, workers, native_artifact=self.config["native_artifact"])

    def specifications(self, seeds):
        specifications = list(seeds)
        if self.config["reset_mode"] == "none" or not self.config["reset_fraction"]:
            return specifications
        episodes = self.corpus["episodes"]
        additional = self.ppo_actions - self.lineage.get("starting_ppo_actions", 0)
        low, high = remaining_window(self.config, additional)
        count = round(len(seeds) * self.config["reset_fraction"])
        for slot in self.rng.permutation(len(seeds))[:count]:
            episode = episodes[int(self.rng.integers(len(episodes)))]
            observations = episode["observations"]
            first, last = int(len(observations) * low), max(1, int(len(observations) * high))
            choices = [i for i in range(first, min(len(observations), last + 1))
                       if len(observations[i]["legal_actions"]) > 1]
            if not choices:
                choices = [i for i, obs in enumerate(observations) if len(obs["legal_actions"]) > 1]
            position = choices[int(self.rng.integers(len(choices)))]
            specifications[slot] = {
                "observation": observations[position],
                "future_seed": int(self.rng.integers(1, 2**62)),
                "environment_seed": episode["environment_seed"],
                "source": {"episode": episode["id"], "decision": position + 1},
            }
        return specifications

    def collect(self, pool, seeds, greedy=False, keep_rows=True):
        specifications = self.specifications(seeds) if keep_rows and not greedy else list(seeds)
        arrays = pool.reset(specifications)
        initial = arrays["stats"][:, SCORE].copy()
        reset = np.asarray([isinstance(s, dict) for s in specifications])
        trajectories = [[] for _ in seeds]
        calls = decisions = 0
        env_seconds = model_seconds = 0.
        while not arrays["done"].all():
            slots = np.flatnonzero(~arrays["done"])
            x, mask = arrays["features"][slots].copy(), arrays["masks"][slots, :240].copy()
            begin = time.monotonic()
            with torch.no_grad():
                logits, value = self.model(torch.as_tensor(x, device=self.device, dtype=torch.float64),
                                          torch.as_tensor(mask, device=self.device))
                distribution = policy_distribution(logits)
                chosen = logits.argmax(-1) if greedy else distribution.sample()
                lp = distribution.log_prob(chosen).cpu().numpy()
                val, action = value.cpu().numpy(), chosen.cpu().numpy()
            model_seconds += time.monotonic() - begin
            begin = time.monotonic()
            pool.step(slots, action)
            env_seconds += time.monotonic() - begin
            if keep_rows:
                for j, slot in enumerate(slots):
                    trajectories[slot].append((x[j], mask[j], int(action[j]), float(lp[j]),
                                               float(arrays["stats"][slot, REWARD]), float(val[j])))
            decisions += len(slots)
            calls += 1
        stats = arrays["stats"].copy()
        if stats[:, STOPPED].any() or not stats[:, NATURAL].all():
            raise RuntimeError("non-natural curriculum terminal")
        rows = []
        for slot, trajectory in enumerate(trajectories):
            rewards, values = [r[4] for r in trajectory], [r[5] for r in trajectory]
            adv, target = advantages(rewards, values, self.config["gae_lambda"])
            mc = returns(rewards)
            for i, row in enumerate(trajectory):
                rows.append((*row[:4], float(target[i]), float(adv[i]), float(mc[i])))
            if keep_rows and abs(sum(rewards) * 150. - (stats[slot, SCORE] - initial[slot])) > 1e-6:
                raise RuntimeError("suffix rewards must equal final-minus-prefix, not full score")
        games = [{
            "environment_seed": int(specifications[i]["environment_seed"] if reset[i] else seeds[i]),
            "score": float(stat[SCORE]), "animal_score": float(stat[ANIMAL]),
            "completed_cards": float(stat[COMPLETED]), "natural_terminal": True,
            "curriculum_reset": bool(reset[i]), "initial_score": float(initial[i]),
            "earned_score": float(stat[SCORE] - initial[i]),
        } for i, stat in enumerate(stats)]
        if keep_rows:
            keep = min(self.config.get("record_games_per_batch", 4), len(games))
            self.last_trajectories = [{
                "reset": specifications[i], "game": games[i], "actions": [r[2] for r in trajectories[i]],
                "behavior_log_probabilities": [r[3] for r in trajectories[i]],
                "scaled_rewards": [r[4] for r in trajectories[i]],
                "encoding": "cards_v1/slots240_v1/no_STOP", "reward_scale": 150.,
            } for i in range(keep)]
        return rows, {
            "games": games, "actions": decisions, "mean_score": float(stats[:, SCORE].mean()),
            "animal_mean": float(stats[:, ANIMAL].mean()), "completed_cards": float(stats[:, COMPLETED].mean()),
            "normal_start_mean": float(stats[~reset, SCORE].mean()) if (~reset).any() else None,
            "reset_earned_score": float((stats[reset, SCORE] - initial[reset]).mean()) if reset.any() else None,
            "reset_count": int(reset.sum()), "environment_seconds": env_seconds,
            "model_seconds": model_seconds, "mean_inference_batch": decisions / calls,
            "training_score_semantics": "mixed-prefix total is not normal-start evaluation; use earned_score for reset episodes",
        }

    def imitation_weight(self):
        return self.config.get("sequence_bc_weight", 0.)

    def demonstration_loss(self):
        # One sequence has coherent downstream labels; never substitute labels
        # into on-policy PPO transitions or use teacher rollout means as returns.
        examples = []
        for _ in range(16):
            episode = self.sequences[int(self.rng.integers(len(self.sequences)))]
            start = int(self.rng.integers(len(episode["y"])))
            examples.append({k: v[start:start + 16] for k, v in episode.items()})
        x, mask, y = [np.concatenate([e[k] for e in examples]) for k in ("x", "mask", "y")]
        logits, _ = self.model(torch.as_tensor(x, device=self.device, dtype=torch.float64),
                              torch.as_tensor(mask, device=self.device))
        return F.cross_entropy(logits, torch.as_tensor(y, device=self.device))

    def batch(self):
        # Preserve sufficient evidence to replay a bounded student sample without
        # storing gigabytes of encoded state per PPO batch. Expert traces are all retained.
        policy_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                           for v in self.model.state_dict().values())).hexdigest()
        before = self.ppo_actions
        row = super().batch()
        out = self.config.get("trajectory_directory")
        if out and self.last_trajectories:
            path = Path(out) / f"actions_{before:012d}_batch_{self.batches:07d}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"format": "student_public_reset_replay_v132", "source_id": self.source,
                       "policy_sha256": policy_hash, "policy_actions": before,
                       "episodes": self.last_trajectories, "sampling": "first fixed slots, not reward-selected"}
            content = gzip.compress(json.dumps(payload, allow_nan=False).encode(), mtime=0)
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
        return row

    def import_sampling_continuation(self, specification):
        if self.actions or self.updates or self.optimizer.state:
            raise ValueError("fresh trainer required")
        path = Path(specification["checkpoint"])
        metadata = read_json(path.with_suffix(".json"))
        if (digest(path) != specification["sha256"] or metadata["sha256"] != specification["sha256"]
                or metadata["source_id"] != specification["source_id"]):
            raise ValueError("parent checksum/source mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["version"] != 127 or saved["source_id"] != specification["source_id"]:
            raise ValueError("unsupported parent checkpoint")
        for key in ("hidden", "reward_scale", "dataset_sha256", "split_sha256"):
            if saved["config"][key] != self.config[key]:
                raise ValueError("protected parent field: " + key)
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("finite FP64 parent required")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["learning_rate"]
        for key in ("episodes", "actions", "ppo_actions", "batches", "warm_batches", "updates"):
            setattr(self, key, saved["counters"][key])
        self.lineage = {
            "kind": "expert_suffix_curriculum", "parent_checkpoint": str(path),
            "sha256": specification["sha256"], "parent_source_id": specification["source_id"],
            "starting_ppo_actions": self.ppo_actions,
            "preserved": "actor, critic, both Adam states and lifetime counters",
            "rng": "fresh paired experiment seed; same-run resume preserves exact RNG",
            "changed": "declared curriculum/credit/auxiliary settings; original real-score objective",
        }


def run(config, out, device="cuda"):
    return base_run(config, out, device, trainer_class=Trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(read_json(args.config), args.out, args.device)
