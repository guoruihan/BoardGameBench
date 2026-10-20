"""Mature PPO plus bounded strategic logit advice; no synthetic rewards or search."""
import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn

from boardbench.artifacts.store import digest, read_json
from boardbench.v133.training import COUNTERS
from boardbench.v134.plans import PLAN_DIM
from boardbench.v134.training import PlanPolicy, Trainer as PlanTrainer, run as plan_run
from .advice import ActionAdvice, METHODS


class AdvicePolicy(PlanPolicy):
    def __init__(self, original, method, strength):
        super().__init__(original)
        self.adapter = deepcopy(original.adapter)
        self.method, self.strength = method, float(strength)
        self.advice = ActionAdvice()
        dtype = original.actor.weight.dtype
        self.gate = nn.Sequential(nn.Linear(original.actor.in_features + PLAN_DIM, 32, dtype=dtype),
                                  nn.Tanh(), nn.Linear(32, 1, dtype=dtype))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(strength / (2 - strength)))
        self.advice_intervention = "normal"

    def forward(self, x, mask):
        logits, value = super().forward(x, mask)
        if self.method == "control" or self.advice_intervention == "zero":
            return logits, value
        base, goals = x[:, :1346], x[:, 1346:]
        if self.advice_intervention == "shuffle":
            goals = goals.roll(1, 0)
        advice = self.advice(base, goals)
        # Center only over legal actions. Illegal actions stay strictly masked.
        advice = advice - (advice * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        strength = self.strength
        if self.method in ("gated", "sticky"):
            strength = 2 * self.gate(torch.cat((self.body(base), goals), -1)).sigmoid()
        return (logits + strength * advice).masked_fill(~mask, -1e9), value


class Trainer(PlanTrainer):
    checkpoint_version = 135

    def __init__(self, config, device="cpu"):
        if (config.get("advice_version") != "1.3.5" or config.get("method") not in METHODS
                or not 0 < config.get("advice_strength", 0) < 2
                or config["arm"] != ("persistent" if config["method"] == "sticky" else "reactive")):
            raise ValueError("explicit V135 advice method and consistent public planner required")
        super().__init__(config, device)
        self.model = AdvicePolicy(self.model, config["method"], config["advice_strength"]).to(
            device=device, dtype=torch.float64)
        self.solver.model = self.model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=.001)

    def import_sampling_continuation(self, spec):
        path = Path(spec["checkpoint"])
        meta = read_json(path.with_suffix(".json"))
        if digest(path) != spec["sha256"] or meta["sha256"] != spec["sha256"] or meta["source_id"] != spec["source_id"]:
            raise ValueError("parent hash/source mismatch")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["version"] == 135:
            return super().import_sampling_continuation(spec)
        if saved["version"] != 134 or saved["source_id"] != spec["source_id"]:
            raise ValueError("mature V134 parent required")
        for key in ("hidden", "reward_scale", "dataset_sha256", "excluded_train_seeds"):
            if saved["config"].get(key) != self.config.get(key):
                raise ValueError("protected parent field: " + key)
        old = read_json(saved["config"]["split_manifest"])
        if digest(saved["config"]["split_manifest"]) != saved["config"]["split_sha256"] or old["sets"]["train"] != self.manifest["sets"]["train"]:
            raise ValueError("parent training pool changed")
        names = list(dict(self.model.named_parameters()))
        if names[:len(saved["model"])] != list(saved["model"]):
            raise ValueError("parent optimizer parameter order changed")
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("finite FP64 parent required")
        state = self.model.state_dict()
        state.update(saved["model"])
        self.model.load_state_dict(state)
        self.manager.load_state_dict(saved["manager"])
        optimizer = deepcopy(saved["optimizer"])
        optimizer["param_groups"][0]["params"] = self.optimizer.state_dict()["param_groups"][0]["params"]
        self.optimizer.load_state_dict(optimizer)
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["learning_rate"]
        for key in COUNTERS[:6]:
            setattr(self, key, saved["counters"][key])
        self.lineage = {"kind": "agent_action_advice_import", **spec, "starting_ppo_actions": self.ppo_actions,
                        "preserved": "V134 actor/critic/adapter and their Adam moments",
                        "new": "public action advice, bounded trust gate, paired fresh RNG"}

    def update(self, rows):
        before = self.model.gate[-1].weight.detach().clone()
        result = super().update(rows)
        result["advice_gate_change"] = float((self.model.gate[-1].weight.detach() - before).norm())
        with torch.no_grad():
            x, mask, *_ = self.tensors(rows)
            advice = self.model.advice(x[:, :1346], x[:, 1346:])
            strength = 2 * self.model.gate(torch.cat((self.model.body(x[:, :1346]), x[:, 1346:]), -1)).sigmoid()
            result.update(advice_legal_nonzero=float(((advice.abs() > 0) & mask).sum() / mask.sum()),
                          advice_strength_mean=float(strength.mean()) if self.config["method"] in ("gated", "sticky")
                          else self.config["advice_strength"] if self.config["method"] == "advice" else 0.)
        return result


def run(config, out, device="cuda"):
    return plan_run(config, out, device, trainer_class=Trainer)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(read_json(a.config), a.out, a.device)
