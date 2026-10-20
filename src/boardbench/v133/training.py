"""Continuous PPO, independent critic, and bounded own-experience self-imitation."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v12.training import policy_distribution
from boardbench.v127.training import Trainer as BaseTrainer, run as base_run
from boardbench.v132.training import Trainer as RecordingTrainer

ARMS = ("ppo", "separate_critic", "self_imitation")
COUNTERS = ("episodes", "actions", "ppo_actions", "batches", "warm_batches", "updates",
            "wall_seconds", "best_score", "evaluated", "initial_scores")


class SeparateCritic(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Preserve original parameter ordering; new parameters come last.
        self.body = deepcopy(original.body)
        self.actor = deepcopy(original.actor)
        self.critic = deepcopy(original.critic)
        self.critic_body = deepcopy(original.body)

    def forward(self, x, mask):
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("terminal/all-zero mask")
        return (self.actor(self.body(x)).masked_fill(~mask, -1e9),
                self.critic(self.critic_body(x)).squeeze(-1))


class Replay:
    """Uniform bounded FIFO of complete batches; immutable compressed disk chunks."""
    def __init__(self, directory, capacity, seed):
        self.directory = Path(directory)
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.blocks, self.receipts = [], []
        if capacity < 1:
            raise ValueError("positive replay capacity required")

    def add(self, rows, identifier, seeds):
        block = {
            "x": np.stack([r[0] for r in rows]).astype(np.float32),
            "mask": np.stack([r[1] for r in rows]).astype(bool),
            "action": np.array([r[2] for r in rows], np.int64),
            "behavior_logp": np.array([r[3] for r in rows], np.float64),
            "mc": np.array([r[6] for r in rows], np.float64),
        }
        if not block["mask"][np.arange(len(rows)), block["action"]].all():
            raise ValueError("illegal own-replay actions")
        path = self.directory / f"actions_{identifier:012d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if any(not np.array_equal(block[k], old[k]) for k in block):
                    raise ValueError("immutable replay chunk changed")
        else:
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **block)
            temporary.replace(path)
        receipt = {"path": str(path), "sha256": digest(path), "rows": len(rows),
                   "environment_seeds": list(map(int, seeds)), "selection": "all rows, no score filtering"}
        write_json(path.with_suffix(".json"), receipt)
        self.blocks.append(block)
        self.receipts.append(receipt)
        self.blocks, self.receipts = self.blocks[-self.capacity:], self.receipts[-self.capacity:]

    def sample(self, count):
        sizes = np.array([len(b["action"]) for b in self.blocks])
        if not len(sizes):
            raise ValueError("empty replay")
        indices = self.rng.integers(int(sizes.sum()), size=count)
        offsets = np.r_[0, sizes.cumsum()]
        owners = np.searchsorted(offsets[1:], indices, side="right")
        samples = {}
        for key in self.blocks[0]:
            samples[key] = np.stack([self.blocks[int(owner)][key][index - offsets[owner]]
                                     for owner, index in zip(owners, indices)])
        return samples

    def state(self):
        return {"receipts": deepcopy(self.receipts), "rng": deepcopy(self.rng.bit_generator.state)}

    def restore(self, state, allowed_seeds):
        self.blocks, self.receipts = [], deepcopy(state["receipts"])
        if len(self.receipts) > self.capacity:
            raise ValueError("replay capacity changed")
        for receipt in self.receipts:
            if not set(receipt["environment_seeds"]) <= allowed_seeds:
                raise ValueError("replay split leakage")
            if digest(receipt["path"]) != receipt["sha256"]:
                raise ValueError("replay checksum mismatch")
            with np.load(receipt["path"], allow_pickle=False) as archive:
                self.blocks.append({k: archive[k].copy() for k in archive.files})
        self.rng.bit_generator.state = deepcopy(state["rng"])


class Trainer(BaseTrainer):
    def __init__(self, config, device="cpu"):
        if (config.get("continuous_version") != "1.3.3" or config["arm"] not in ARMS
                or config.get("bc_weight", 0) or config.get("sequence_bc_weight", 0)
                or config.get("reset_fraction", 0) or config.get("critic_batches", 0)
                or config["environment_backend"] != "native_v129"):
            raise ValueError("explicit model-only continuous configuration required")
        super().__init__(config, device)
        if config["arm"] == "separate_critic":
            self.model = SeparateCritic(self.model)
            self.solver.model = self.model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
            self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=.001)
        self.replay = (Replay(config["replay_directory"], config["replay_batches"], config["seed"] + 133000)
                       if config["arm"] == "self_imitation" else None)
        self.last_trajectories = []

    def specifications(self, seeds):
        return list(seeds)

    def collect(self, pool, seeds, greedy=False, keep_rows=True):
        rows, metrics = RecordingTrainer.collect(self, pool, seeds, greedy, keep_rows)
        if keep_rows:
            if not set(seeds) <= set(self.train_seeds):
                raise ValueError("only training episodes may enter learning/replay")
            if self.replay is not None:
                self.replay.add(rows, self.ppo_actions, seeds)
        return rows, metrics

    def imitation_weight(self):
        return self.config["self_imitation_weight"] if self.replay is not None else 0.

    def demonstration_loss(self):
        # This is a separate off-policy SIL auxiliary, never a PPO likelihood ratio.
        sample = self.replay.sample(self.config["replay_minibatch"])
        x = torch.as_tensor(sample["x"], device=self.device, dtype=torch.float64)
        mask = torch.as_tensor(sample["mask"], device=self.device)
        action = torch.as_tensor(sample["action"], device=self.device)
        target = torch.as_tensor(sample["mc"], device=self.device)
        logits, value = self.model(x, mask)
        positive = (target - value).clamp_min(0)
        weight = positive.detach().clamp_max(self.config["replay_advantage_cap"])
        return (-policy_distribution(logits).log_prob(action) * weight).mean() + .5 * positive.square().mean()

    def update(self, rows):
        with torch.no_grad():
            x, mask, _, _, _, _, mc = self.tensors(rows)
            _, values = self.model(x, mask)
            residual = mc - values
            diagnostics = {
                "mc_value_mse": float(residual.square().mean()),
                "mc_explained_variance": 1 - float(residual.var(unbiased=False)) /
                                         max(float(mc.var(unbiased=False)), 1e-12),
                "replay_rows": sum(len(b["action"]) for b in self.replay.blocks) if self.replay else 0,
            }
        update = super().update(rows)
        return {**update, **diagnostics, "expert_bc_weight": 0.,
                "auxiliary_kind": "own_replay_SIL" if self.replay is not None else "none",
                "self_imitation_loss": update["bc_loss"] if self.replay is not None else 0.}

    def batch(self):
        before = self.ppo_actions
        row = super().batch()
        out = self.config.get("trajectory_directory")
        if out:
            path = Path(out) / f"actions_{before:012d}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"format": "student_public_reset_replay_v133", "source_id": self.source,
                       "policy_actions": before, "episodes": self.last_trajectories,
                       "sampling": "fixed first slots, not score-selected"}
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(gzip.compress(json.dumps(payload, allow_nan=False).encode(), mtime=0))
            temporary.replace(path)
        return row

    def evaluate(self, seeds):
        # Fixed128batch shape, even when the registered development panel has512 games.
        batches = [super(Trainer, self).evaluate(seeds[i:i + 128]) for i in range(0, len(seeds), 128)]
        games = [g for batch in batches for g in batch["games"]]
        return {"games": games, "mean_score": float(np.mean([g["score"] for g in games])),
                "animal_mean": float(np.mean([g["animal_score"] for g in games])),
                "completed_cards": float(np.mean([g["completed_cards"] for g in games])),
                **{k: sum(b[k] for b in batches) for k in ("actions", "environment_seconds", "model_seconds")}}

    def import_sampling_continuation(self, spec):
        if self.actions or self.updates or self.optimizer.state:
            raise ValueError("fresh trainer required")
        path = Path(spec["checkpoint"])
        metadata = read_json(path.with_suffix(".json"))
        if (digest(path) != spec["sha256"] or metadata["sha256"] != spec["sha256"]
                or metadata["source_id"] != spec["source_id"]):
            raise ValueError("parent checksum/source mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["version"] != 127 or saved["source_id"] != spec["source_id"]:
            raise ValueError("V132 mature parent required")
        for key in ("hidden", "reward_scale", "dataset_sha256"):
            if saved["config"][key] != self.config[key]:
                raise ValueError("protected parent field: " + key)
        old_path = saved["config"]["split_manifest"]
        if (digest(old_path) != saved["config"]["split_sha256"]
                or saved["config"]["split_sha256"] != self.manifest["parent_split_sha256"]
                or read_json(old_path)["sets"]["train"] != self.manifest["sets"]["train"]
                or saved["config"].get("excluded_train_seeds", []) != self.config.get("excluded_train_seeds", [])):
            raise ValueError("only evaluation panels may change; training pool must match parent")
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("finite FP64 parent required")
        weights = dict(saved["model"])
        optimizer = deepcopy(saved["optimizer"])
        if self.config["arm"] == "separate_critic":
            weights.update({"critic_body." + k[5:]: v.clone()
                            for k, v in saved["model"].items() if k.startswith("body.")})
            # Preserve all existing Adam states; only added critic-trunk moments start fresh.
            expected = list(self.model.state_dict())[:len(saved["model"])]
            if expected != list(saved["model"]):
                raise ValueError("optimizer parameter mapping changed")
            optimizer["param_groups"][0]["params"] = self.optimizer.state_dict()["param_groups"][0]["params"]
        self.model.load_state_dict(weights, strict=True)
        self.optimizer.load_state_dict(optimizer)
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["learning_rate"]
        for key in COUNTERS[:6]:
            setattr(self, key, saved["counters"][key])
        self.lineage = {"kind": "continuous_model_only", **spec, "arm": self.config["arm"],
                        "starting_ppo_actions": self.ppo_actions,
                        "preserved": "actor, critic outputs and existing Adam states",
                        "changed": "declared arm; fresh paired RNG; new evaluation panels only",
                        "new_critic_trunk_moments": "fresh" if self.config["arm"] == "separate_critic" else None}

    def save(self, path):
        if not self.boundary:
            raise RuntimeError("checkpoint requires complete batch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {"version": 133, "source_id": self.source, "config": self.config, "device": self.device,
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "critic_optimizer": self.critic_optimizer.state_dict(), "lineage": self.lineage,
                 "rng_numpy": self.rng.bit_generator.state, "rng_torch": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else [],
                 "counters": {k: getattr(self, k) for k in COUNTERS},
                 "replay": self.replay.state() if self.replay is not None else None}
        temporary = path.with_suffix(".partial")
        torch.save(saved, temporary)
        temporary.replace(path)
        write_json(path.with_suffix(".json"), {"sha256": digest(path), "source_id": self.source,
                   "ppo_actions": self.ppo_actions, "episodes": self.episodes, "lineage": self.lineage})

    def load(self, path):
        if digest(path) != read_json(Path(path).with_suffix(".json"))["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if (saved["version"] != 133 or saved["source_id"] != self.source
                or saved["config"] != self.config or saved["device"] != self.device):
            raise ValueError("exact resume source/config/device mismatch")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        self.lineage = saved["lineage"]
        if self.replay is not None:
            self.replay.restore(saved["replay"], set(self.train_seeds))
        elif saved["replay"] is not None:
            raise ValueError("unexpected replay in baseline")
        self.rng.bit_generator.state = saved["rng_numpy"]
        torch.set_rng_state(saved["rng_torch"].cpu())
        if saved["rng_cuda"]:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["rng_cuda"]])
        for key, value in saved["counters"].items():
            setattr(self, key, value)


def run(config, out, device="cuda"):
    return base_run(config, out, device, trainer_class=Trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(read_json(args.config), args.out, args.device)
