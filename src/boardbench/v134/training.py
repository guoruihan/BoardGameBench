"""Plan-conditioned worker PPO and optional separate semi-Markov manager PPO."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v12.training import policy_distribution, returns
from boardbench.v121.collector import REWARD, SCORE, ANIMAL, COMPLETED, NATURAL, STOPPED
from boardbench.v127.training import Trainer as BaseTrainer, advantages
from boardbench.v133.training import COUNTERS
from .collector import EnvironmentPool
from .plans import ARMS, GOAL_DIM, PLAN_DIM


class PlanPolicy(nn.Module):
    """Identical-capacity residual adapters, initialized to exact parent policy."""
    def __init__(self, original):
        super().__init__()
        self.body, self.actor, self.critic = deepcopy(original.body), deepcopy(original.actor), deepcopy(original.critic)
        self.adapter = nn.Sequential(nn.Linear(original.actor.in_features + PLAN_DIM, 64), nn.Tanh(), nn.Linear(64, 241))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.intervention = "normal"

    def forward(self, x, mask):
        base, goals = x[:, :1346], x[:, 1346:]
        if goals.shape[-1] != PLAN_DIM or not mask.any(-1).all():
            raise ValueError("plan feature width or legal mask mismatch")
        if self.intervention == "zero":
            goals = torch.zeros_like(goals)
        elif self.intervention == "shuffle":
            # Fixed deterministic batch permutation; no evaluation RNG consumed.
            goals = goals.roll(1, 0)
        z = self.body(base)
        delta = self.adapter(torch.cat((z, goals), -1))
        return ((self.actor(z) + delta[:, :240]).masked_fill(~mask, -1e9),
                self.critic(z).squeeze(-1) + delta[:, 240])


class Manager(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.body, self.critic = deepcopy(original.body), deepcopy(original.critic)
        self.score = nn.Sequential(nn.Linear(original.actor.in_features + PLAN_DIM, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x, plans, mask):
        if not mask.any(-1).all():
            raise ValueError("manager requires free-play or feasible plan")
        z = self.body(x)
        expanded = z[:, None, :].expand(-1, plans.shape[1], -1)
        # Public heuristic is only an initial logit prior, never a game reward.
        prior = (10 * (plans[:, :, GOAL_DIM - 2] + .3 * plans[:, :, 2 * GOAL_DIM - 2])).clamp(0, 4)
        logits = self.score(torch.cat((expanded, plans), -1)).squeeze(-1) + prior
        return logits.masked_fill(~mask, -1e9), self.critic(z).squeeze(-1)


def manager_targets(trajectory, decisions, lam):
    """Variable-duration option rewards telescope; gamma=1, lambda per option."""
    rewards = [row[4] for row in trajectory]
    boundaries = [d["start"] for d in decisions] + [len(rewards)]
    segments = [sum(rewards[a:b]) for a, b in zip(boundaries, boundaries[1:])]
    values = [d["value"] for d in decisions]
    adv, target = advantages(segments, values, lam)
    mc = returns(segments)
    return [{**d, "target": float(target[i]), "advantage": float(adv[i]),
             "mc": float(mc[i]), "duration": boundaries[i + 1] - boundaries[i]}
            for i, d in enumerate(decisions)]


class Trainer(BaseTrainer):
    checkpoint_version = 134

    def __init__(self, config, device="cpu"):
        if (config.get("planning_version") != "1.3.4" or config["arm"] not in ARMS
                or config.get("bc_weight", 0) or config.get("critic_batches", 0)):
            raise ValueError("explicit V134 model-only configuration required")
        super().__init__(config, device)
        original = self.model
        self.model = PlanPolicy(original).to(device=device, dtype=torch.float64)
        self.solver.model = self.model
        self.manager = Manager(original).to(device=device, dtype=torch.float64)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=.001)
        self.manager_optimizer = torch.optim.Adam(self.manager.parameters(), lr=config["manager_lr"])
        self.manager_rng = torch.Generator(device=device).manual_seed(config["seed"] + 134000)
        self.manager_rows, self.last_trajectories = [], []

    def make_pool(self, count, workers):
        return EnvironmentPool(count, workers, native_artifact=self.config["native_artifact"],
                               arm=self.config["arm"], horizon=self.config["plan_horizon"])

    def collect(self, pool, seeds, greedy=False, keep_rows=True):
        if len(seeds) != pool.count:
            raise ValueError("one seed per slot")
        started = time.monotonic()
        arrays = pool.reset(seeds)
        env_seconds, model_seconds = time.monotonic() - started, 0.
        trajectories, high = [[] for _ in seeds], [[] for _ in seeds]
        plans_used, successes, decisions, requests = 0, 0, 0, 0
        while not arrays["done"].all():
            slots = np.flatnonzero(~arrays["done"])
            extra = pool.extra
            selected = extra["plan"][slots].copy()
            request_indices = np.flatnonzero(extra["request"][slots])
            requesting = slots[request_indices]
            tick = time.monotonic()
            if len(requesting):
                bank = extra["plans"][requesting].copy()
                bank_mask = extra["plan_mask"][requesting].copy()
                hx = arrays["features"][requesting].copy()
                if self.config["arm"] == "manager":
                    with torch.no_grad():
                        hlogits, hv = self.manager(torch.as_tensor(hx, device=self.device, dtype=torch.float64),
                                                  torch.as_tensor(bank, device=self.device, dtype=torch.float64),
                                                  torch.as_tensor(bank_mask, device=self.device))
                        hd = policy_distribution(hlogits)
                        hc = hlogits.argmax(-1) if greedy else torch.multinomial(
                            hd.probs, 1, generator=self.manager_rng).squeeze(-1)
                        choices = hc.cpu().numpy()
                        hp, hvalues = hd.log_prob(hc).cpu().numpy(), hv.cpu().numpy()
                    if keep_rows:
                        for j, slot in enumerate(requesting):
                            high[slot].append({"x": hx[j], "plans": bank[j], "mask": bank_mask[j],
                                              "action": int(choices[j]), "logp": float(hp[j]),
                                              "value": float(hvalues[j]), "start": len(trajectories[slot])})
                else:
                    # Top public candidate, or no goal when no feasible blueprint.
                    choices = np.where(bank_mask[:, 1], 1, 0)
                extra["choice"][requesting] = choices
                selected[request_indices] = bank[np.arange(len(requesting)), choices]
                requests += len(requesting)
                plans_used += int((choices != 0).sum())
            if self.config["arm"] == "capacity":
                selected.fill(0)
            x = np.concatenate((arrays["features"][slots], selected), axis=-1)
            mask = arrays["masks"][slots, :240].copy()
            with torch.no_grad():
                logits, values = self.model(torch.as_tensor(x, device=self.device, dtype=torch.float64),
                                            torch.as_tensor(mask, device=self.device))
                dist = policy_distribution(logits)
                chosen = logits.argmax(-1) if greedy else dist.sample()
                action = chosen.cpu().numpy()
                logp, value = dist.log_prob(chosen).cpu().numpy(), values.cpu().numpy()
            model_seconds += time.monotonic() - tick
            tick = time.monotonic()
            pool.step(slots, action)
            env_seconds += time.monotonic() - tick
            successes += int(extra["success"][slots].sum())
            for j, slot in enumerate(slots):
                if keep_rows:
                    trajectories[slot].append((x[j], mask[j], int(action[j]), float(logp[j]),
                                               float(arrays["stats"][slot, REWARD]), float(value[j])))
            decisions += len(slots)
        stats, rows, high_rows = arrays["stats"].copy(), [], []
        if stats[:, STOPPED].any() or not stats[:, NATURAL].all():
            raise RuntimeError("only natural games allowed")
        for trajectory, hs, stat in zip(trajectories, high, stats):
            rewards, vals = [r[4] for r in trajectory], [r[5] for r in trajectory]
            adv, target = advantages(rewards, vals, self.config["gae_lambda"])
            mc = returns(rewards)
            rows.extend((*r[:4], float(target[i]), float(adv[i]), float(mc[i])) for i, r in enumerate(trajectory))
            if keep_rows:
                if abs(sum(rewards) * 150 - stat[SCORE]) > 1e-6:
                    raise RuntimeError("changed true-score objective")
                if hs:
                    high_rows.extend(manager_targets(trajectory, hs, self.config["gae_lambda"]))
        if keep_rows:
            self.manager_rows = high_rows
            self.last_trajectories = [
                {"environment_seed": int(seeds[i]), "actions": [r[2] for r in t],
                 "rewards": [r[4] for r in t], "behavior_logp": [r[3] for r in t],
                 "worker_plan_vectors": [r[0][1346:].tolist() for r in t],
                 "plans": [{"start": d["start"], "choice": d["action"],
                            "vector": d["plans"][d["action"]].tolist()} for d in high[i]]}
                for i, t in enumerate(trajectories[:self.config.get("record_games_per_batch", 4)])]
        games = [{"environment_seed": int(s), "score": float(v[SCORE]), "animal_score": float(v[ANIMAL]),
                  "completed_cards": float(v[COMPLETED]), "natural_terminal": True} for s, v in zip(seeds, stats)]
        return rows, {"games": games, "actions": decisions, "mean_score": float(stats[:, SCORE].mean()),
                      "animal_mean": float(stats[:, ANIMAL].mean()),
                      "completed_cards": float(stats[:, COMPLETED].mean()),
                      "environment_seconds": env_seconds, "model_seconds": model_seconds,
                      "plan_requests": requests, "nonempty_plans": plans_used, "primary_achieved": successes}

    def update_manager(self):
        rows = self.manager_rows
        if not rows:
            return {}
        tensor = lambda key, dtype=torch.float64: torch.as_tensor(np.stack([r[key] for r in rows]),
                                                                 device=self.device, dtype=dtype)
        x, plans, mask = tensor("x"), tensor("plans"), tensor("mask", torch.bool)
        action, old = tensor("action", torch.int64), tensor("logp")
        target, adv = tensor("target"), tensor("advantage")
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        with torch.no_grad():
            logits, _ = self.manager(x, plans, mask)
            error = float(((policy_distribution(logits).log_prob(action) - old).exp() - 1).abs().max())
        if error > 1e-4:
            raise RuntimeError("manager behavior probability mismatch")
        updates, losses = 0, []
        before = self.manager.score[-1].weight.detach().clone()
        for _ in range(2):
            for ids in torch.randperm(len(rows), device=self.device).split(self.config["minibatch"]):
                logits, value = self.manager(x[ids], plans[ids], mask[ids])
                dist = policy_distribution(logits)
                logratio = dist.log_prob(action[ids]) - old[ids]
                ratio = logratio.exp()
                if float(((ratio - 1) - logratio).mean().detach()) > self.config["target_kl"]:
                    break
                loss = (-torch.minimum(ratio * adv[ids], ratio.clamp(.8, 1.2) * adv[ids]).mean()
                        + .5 * (value - target[ids]).square().mean() - .01 * dist.entropy().mean())
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite manager loss")
                self.manager_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.manager.parameters(), .5, error_if_nonfinite=True)
                self.manager_optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
        return {"manager_updates": updates, "manager_ratio_error": error,
                "manager_change": float((self.manager.score[-1].weight.detach() - before).norm()),
                "manager_loss": float(np.mean(losses)) if losses else 0.,
                "manager_decisions": len(rows), "mean_option_actions": float(np.mean([r["duration"] for r in rows]))}

    def update(self, rows):
        with torch.no_grad():
            x, mask, _, _, _, _, mc = self.tensors(rows)
            _, value = self.model(x, mask)
            mc_mse = float((mc - value).square().mean())
            adapter_before = self.model.adapter[-1].weight.detach().clone()
        result = super().update(rows)
        if self.config["arm"] == "manager":
            result.update(self.update_manager())
        result.update(plan_adapter_change=float((self.model.adapter[-1].weight.detach() - adapter_before).norm()),
                      mc_value_mse=mc_mse, expert_bc_weight=0.)
        return result

    def batch(self):
        before = self.ppo_actions
        row = super().batch()
        if self.config.get("trajectory_directory"):
            path = Path(self.config["trajectory_directory"]) / f"actions_{before:012d}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"format": "v134_public_plan_trajectory", "source_id": self.source,
                    "episodes": self.last_trajectories, "selection": "fixed first slots"}
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))
            temporary.replace(path)
        return row

    def evaluate(self, seeds):
        batches = []
        count = min(128, len(seeds))
        workers = min(self.config["workers"], count)
        # A reusable cohort amortizes process startup. Never borrow extra cores:
        # workers use the same trial affinity as training.
        with self.make_pool(count, workers) as pool:
            for i in range(0, len(seeds), count):
                chunk = seeds[i:i + count]
                if len(chunk) == count:
                    _, metric = self.collect(pool, chunk, greedy=True, keep_rows=False)
                else:
                    with self.make_pool(len(chunk), 0) as remainder:
                        _, metric = self.collect(remainder, chunk, greedy=True, keep_rows=False)
                batches.append(metric)
        games = [g for b in batches for g in b["games"]]
        return {"games": games, "mean_score": float(np.mean([g["score"] for g in games])),
                "animal_mean": float(np.mean([g["animal_score"] for g in games])),
                "completed_cards": float(np.mean([g["completed_cards"] for g in games])),
                **{k: sum(b[k] for b in batches) for k in
                   ("actions", "environment_seconds", "model_seconds", "plan_requests", "nonempty_plans", "primary_achieved")}}

    def import_sampling_continuation(self, spec):
        path = Path(spec["checkpoint"])
        meta = read_json(path.with_suffix(".json"))
        if digest(path) != spec["sha256"] or meta["sha256"] != spec["sha256"] or meta["source_id"] != spec["source_id"]:
            raise ValueError("parent hash/source mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["version"] == self.checkpoint_version:
            mutable = {"continuation", "target_actions", "milestones", "max_seconds", "trajectory_directory", "workers"}
            if (saved["source_id"] != self.source or saved["device"] != self.device or
                    {k: v for k, v in saved["config"].items() if k not in mutable} !=
                    {k: v for k, v in self.config.items() if k not in mutable}):
                raise ValueError("stage continuation must preserve algorithm/data/source")
            for key in ("model", "manager", "optimizer", "critic_optimizer", "manager_optimizer"):
                getattr(self, key).load_state_dict(saved[key])
            self.rng.bit_generator.state = saved["rng_numpy"]
            torch.set_rng_state(saved["rng_torch"].cpu())
            self.manager_rng.set_state(saved["rng_manager"].cpu())
            if saved["rng_cuda"]:
                torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng_cuda"]])
            for key in COUNTERS[:6]:
                setattr(self, key, saved["counters"][key])
            self.lineage = {"kind": "continuous_plan_stage", **spec, "starting_ppo_actions": self.ppo_actions,
                            "preserved": "worker/manager, all optimizer states and RNG, training counters"}
            return
        if saved["version"] not in (127, 133) or saved["source_id"] != spec["source_id"]:
            raise ValueError("flat mature parent required")
        for key in ("hidden", "reward_scale", "dataset_sha256", "excluded_train_seeds"):
            if saved["config"].get(key) != self.config.get(key):
                raise ValueError("protected parent field: " + key)
        old_split = read_json(saved["config"]["split_manifest"])
        if digest(saved["config"]["split_manifest"]) != saved["config"]["split_sha256"] or old_split["sets"]["train"] != self.manifest["sets"]["train"]:
            raise ValueError("parent training pool changed")
        if set(saved["model"]) != {"body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias",
                                   "actor.weight", "actor.bias", "critic.weight", "critic.bias"}:
            raise ValueError("parent architecture must be the flat policy")
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("finite FP64 parent required")
        weights = self.model.state_dict()
        weights.update(saved["model"])
        self.model.load_state_dict(weights)
        high_weights = self.manager.state_dict()
        high_weights.update({k: v for k, v in saved["model"].items() if k.startswith(("body.", "critic."))})
        self.manager.load_state_dict(high_weights)
        optimizer = deepcopy(saved["optimizer"])
        optimizer["param_groups"][0]["params"] = self.optimizer.state_dict()["param_groups"][0]["params"]
        self.optimizer.load_state_dict(optimizer)
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["learning_rate"]
        for key in COUNTERS[:6]:
            setattr(self, key, saved["counters"][key])
        self.lineage = {"kind": "plan_conditioned_mature_import", **spec,
                        "starting_ppo_actions": self.ppo_actions,
                        "preserved": "flat actor/critic and existing worker Adam moments",
                        "new": "zero-output plan residual, separate manager, paired fresh RNG"}

    def save(self, path):
        if not self.boundary:
            raise RuntimeError("complete batch checkpoint required")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {"version": self.checkpoint_version, "source_id": self.source, "config": self.config, "device": self.device,
                 "model": self.model.state_dict(), "manager": self.manager.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                 "manager_optimizer": self.manager_optimizer.state_dict(), "lineage": self.lineage,
                 "rng_numpy": self.rng.bit_generator.state, "rng_torch": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else [],
                 "rng_manager": self.manager_rng.get_state(),
                 "counters": {k: getattr(self, k) for k in COUNTERS}}
        temporary = path.with_suffix(".partial")
        torch.save(saved, temporary)
        temporary.replace(path)
        write_json(path.with_suffix(".json"), {"sha256": digest(path), "source_id": self.source,
                   "ppo_actions": self.ppo_actions, "episodes": self.episodes, "lineage": self.lineage})

    def load(self, path):
        if digest(path) != read_json(Path(path).with_suffix(".json"))["sha256"]:
            raise ValueError("resume checksum mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["version"] != self.checkpoint_version or saved["source_id"] != self.source or saved["config"] != self.config or saved["device"] != self.device:
            raise ValueError("resume source/config/device mismatch")
        for key in ("model", "manager", "optimizer", "critic_optimizer", "manager_optimizer"):
            getattr(self, key).load_state_dict(saved[key])
        self.lineage = saved["lineage"]
        self.rng.bit_generator.state = saved["rng_numpy"]
        torch.set_rng_state(saved["rng_torch"].cpu())
        self.manager_rng.set_state(saved["rng_manager"].cpu())
        if saved["rng_cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng_cuda"]])
        for key, value in saved["counters"].items():
            setattr(self, key, value)


def run(config, out, device="cuda", *, trainer_class=Trainer):
    """Save the worker AND manager at every boundary; never swap only half a policy."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and read_json(out / "config.json") != config:
        raise ValueError("immutable trial config")
    write_json(out / "config.json", config)
    trainer = trainer_class(config, device)
    started, prior_wall, stopping = time.monotonic(), 0., [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    def elapsed():
        return prior_wall + time.monotonic() - started
    def publish(phase, **other):
        write_json(out / "progress.json", {"phase": phase, "pid": os.getpid(), "updated_unix": time.time(),
                   "ppo_actions": trainer.ppo_actions, "wall_seconds": elapsed(), **other})
    def save(name):
        trainer.wall_seconds = elapsed()
        trainer.save(out / name)
    def evaluate(label, seeds):
        publish("evaluating", label=label)
        m = trainer.evaluate(seeds)
        m.update(label=label, ppo_actions=trainer.ppo_actions, wall_seconds=elapsed(),
                 policy="model_with_public_blueprints_no_search")
        write_json(out / "evaluations" / (label + ".json"), m)
        return m
    try:
        if (out / "latest.pt").exists():
            trainer.load(out / "latest.pt")
            prior_wall = trainer.wall_seconds
        else:
            trainer.import_sampling_continuation(config["continuation"])
        if trainer.initial_scores is None:
            save("initial.pt")
            m = evaluate("initial", config["curve_seeds"])
            trainer.initial_scores, trainer.best_score = m["games"], m["mean_score"]
            save("best.pt")
            save("latest.pt")
        with (out / "training.jsonl").open("a") as log:
            while trainer.ppo_actions < config["target_actions"] and elapsed() < config["max_seconds"] and not stopping[0]:
                publish("collecting_and_updating")
                row = trainer.batch()
                row.update(wall_seconds=elapsed(), timestamp=time.time())
                save("latest.pt")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                publish("ppo", metrics={k: v for k, v in row.items() if k != "games"})
                print(json.dumps({k: v for k, v in row.items() if k != "games"}), flush=True)
                due = [n for n in config["milestones"] if n <= trainer.ppo_actions and n not in trainer.evaluated]
                if due and not stopping[0] and elapsed() < config["max_seconds"]:
                    m = evaluate(f"actions_{trainer.ppo_actions}", config["curve_seeds"])
                    if m["mean_score"] > trainer.best_score:
                        trainer.best_score = m["mean_score"]
                        save("best.pt")
                    trainer.evaluated.extend(due)
                    save(f"milestone_{max(due)}.pt")
                    save("latest.pt")
        status = "interrupted" if stopping[0] else "completed" if trainer.ppo_actions >= config["target_actions"] else "time_limit"
        if status == "completed" and elapsed() < config["max_seconds"]:
            evaluate("final", config["final_seeds"])
        save("final.pt")
        save("latest.pt")
        summary = {"status": status, "source_id": trainer.source, "ppo_actions": trainer.ppo_actions,
                   "wall_seconds": elapsed(), "lineage": trainer.lineage, "best_curve_score": trainer.best_score}
        write_json(out / "summary.json", summary)
        publish(status)
        return summary
    except BaseException as exc:
        publish("failed", error=repr(exc))
        raise
    finally:
        trainer.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(read_json(a.config), a.out, a.device)
