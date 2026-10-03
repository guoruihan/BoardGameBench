from copy import deepcopy
import math

from boardbench.environments import get_task

DEFAULTS = {
    "task": {"id": "risk_collect", "version": "1"},
    "solver": {"id": "random", "params": {}},
    "mode": "adaptation", "allow_learning": True, "episodes": 6, "seed": 123,
    "capabilities": {"reference_simulator": False},
    "limits": {"max_action_requests": 100, "max_actions_per_episode": 16,
               "max_wall_seconds": 60.0},
    "model": {"provider": "mock", "max_calls": 1, "timeout_seconds": 10.0},
    "checkpoint": {"every_episodes": 2},
}


def merge(base, changes):
    out = deepcopy(base)
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"unknown config option: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            out[key] = deepcopy(value) if key == "params" else merge(base[key], value)
        else:
            out[key] = value
    return out


def number(value, name, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")


def normalize(config, *, evaluation=False):
    out = merge(DEFAULTS, config)
    supported = ("evaluation", False) if evaluation else ("adaptation", True)
    if type(out["allow_learning"]) is not bool or (out["mode"], out["allow_learning"]) != supported:
        raise ValueError("V0 supports adaptation+learning via run and evaluation+no learning via evaluate")
    get_task(out["task"])
    if type(out["capabilities"]["reference_simulator"]) is not bool:
        raise ValueError("reference_simulator must be a boolean")
    if out["model"]["provider"] != "mock":
        raise ValueError("V0 provider is mock; real API integration is unverified")
    number(out["episodes"], "episodes", integer=True, positive=True)
    number(out["seed"], "seed", integer=True)
    for name, val in out["limits"].items():
        number(val, name, integer=name != "max_wall_seconds")
    number(out["model"]["max_calls"], "max_calls", integer=True)
    number(out["model"]["timeout_seconds"], "timeout_seconds", positive=True)
    number(out["checkpoint"]["every_episodes"], "every_episodes", integer=True)
    return out


def evaluation_config(raw, manifest):
    allowed = {"mode", "allow_learning", "episode_seeds", "per_episode_limits"}
    if set(raw) - allowed:
        raise ValueError(f"evaluation cannot override: {sorted(set(raw) - allowed)}")
    if raw.get("mode", "evaluation") != "evaluation" or raw.get("allow_learning", False) is not False:
        raise ValueError("V0 evaluation requires allow_learning=false")
    seeds = raw.get("episode_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("evaluation requires nonempty episode_seeds")
    for seed in seeds:
        number(seed, "episode seed", integer=True)
    limits = merge({"max_action_requests": 16, "max_wall_seconds": 30.0,
                    "max_model_calls": 0}, raw.get("per_episode_limits", {}))
    config = deepcopy(manifest["config"])
    config.update(mode="evaluation", allow_learning=False, episodes=len(seeds))
    config["limits"] = {"max_action_requests": limits["max_action_requests"],
                        "max_actions_per_episode": limits["max_action_requests"],
                        "max_wall_seconds": limits["max_wall_seconds"]}
    config["model"]["max_calls"] = limits["max_model_calls"]
    config["checkpoint"]["every_episodes"] = 0
    return normalize(config, evaluation=True), seeds
