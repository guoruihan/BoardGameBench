"""Finite-budget on-policy masked PPO; direct engines, no Runner log per sample."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, source_files, source_id, source_root, write_json
from boardbench.environments import TASKS
from boardbench.environments.numerical import encode, decode
from boardbench.solvers.neural import NeuralSolver


def parameter_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(config, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    for relative in source_files():
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root() / relative, target)
    code_version = source_id(out / "source")
    task, seed, device = config["task_id"], config["training_seed"], config.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    threads = config.get("threads", 2)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    solver = NeuralSolver(seed, task, hidden=config.get("hidden", 128), device=device)
    optimizer = torch.optim.Adam(solver.model.parameters(), lr=config.get("learning_rate", .0003))
    episodes, batch_episodes = config["episodes"], config.get("batch_episodes", 32)
    if episodes < 2*batch_episodes or episodes % batch_episodes:
        raise ValueError("episodes must be a batch multiple and allow two trained checkpoints")
    env_seed_start = config.get("train_env_seed_start", 100000)
    validation = config.get("validation_seeds", list(range(200000, 200032)))
    test = config.get("test_seeds", list(range(300000, 300128)))
    training_seeds = list(range(env_seed_start, env_seed_start+episodes))
    if set(training_seeds) & set(validation) or set(training_seeds) & set(test) or set(validation) & set(test):
        raise ValueError("train/validation/test environment seed overlap")
    write_json(out / "splits.json", {"train": training_seeds, "validation": validation, "test": test})
    if config.get("gamma", 1.) != 1. or config.get("shaping", "score_delta") not in ("score_delta", "terminal"):
        raise ValueError("this finite-horizon PPO implements gamma=1 and score_delta/terminal rewards")
    scale = float(config.get("reward_scale", {"micro_tiles": 30, "take_it_easy": 200, "harmonies": 150}[task]))
    start, steps, gradients, completed = time.monotonic(), 0, 0, 0
    initial_hash = parameter_hash(solver.model)
    saved = []
    def save(label):
        metadata = {"training_seed": seed, "episodes": completed, "environment_steps": steps,
                    "gradient_steps": gradients, "wall_seconds": time.monotonic()-start,
                    "device": str(device), "device_name": torch.cuda.get_device_name() if str(device).startswith("cuda") else "CPU",
                    "threads": threads, "parallel_envs": 1, "gamma": 1., "discount_unit": "atomic_action",
                    "shaping": config.get("shaping", "score_delta"), "reward_scale": scale,
                    "parameter_count": sum(p.numel() for p in solver.model.parameters()),
                    "parameter_sha256": parameter_hash(solver.model), "initial_parameter_sha256": initial_hash,
                    "code_version": code_version, "torch_version": str(torch.__version__),
                    "numpy_version": np.__version__}
        solver.training = metadata
        path = out / (label+".pt")
        solver.save_weights(path)
        saved.append({"path": path.name, "sha256": digest(path), **metadata})
        write_json(out / "checkpoints.json", saved)
    save("untrained")
    midpoint = max(batch_episodes, (episodes//2//batch_episodes)*batch_episodes)
    with (out / "training.jsonl").open("x") as log:
        while completed < episodes and time.monotonic()-start < config.get("max_wall_seconds", 1800):
            rows, scores = [], []
            solver.model.eval()
            for _ in range(batch_episodes):
                env = TASKS[task].factory()
                obs, _ = env.reset(env_seed_start+completed)
                trajectory = []
                while not obs["terminated"]:
                    data = encode(obs)
                    x = np.asarray(data["features"], dtype=np.float32)
                    mask = np.asarray(data["action_mask"], dtype=bool)
                    with torch.no_grad():
                        logits, value = solver.model(torch.as_tensor(x, device=device)[None],
                                                     torch.as_tensor(mask, device=device)[None])
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        logp = dist.log_prob(action).item()
                    result = env.step(decode(obs, int(action.item())))
                    reward = (result.observation["score_breakdown"]["total"]-obs["score_breakdown"]["total"]
                              if config.get("shaping", "score_delta") == "score_delta" else result.reward)
                    trajectory.append((x, mask, int(action.item()), logp, value.item(), reward/scale))
                    steps += 1
                    obs = result.observation
                    if len(trajectory) > 256:
                        raise RuntimeError("engine exceeded finite action bound")
                target = 0.
                for x, mask, action, logp, value, reward in reversed(trajectory):
                    target += reward  # Monte Carlo advantages, gamma=lambda=1.
                    rows.append((x, mask, action, logp, target, target-value))
                completed += 1
                scores.append(obs["score"])
            xs = torch.as_tensor(np.stack([r[0] for r in rows]), device=device)
            masks = torch.as_tensor(np.stack([r[1] for r in rows]), device=device)
            actions = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
            old_logps = torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device)
            targets = torch.tensor([r[4] for r in rows], dtype=torch.float32, device=device)
            advantages = torch.tensor([r[5] for r in rows], dtype=torch.float32, device=device)
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
            solver.model.train()
            losses = []
            for _ in range(config.get("epochs", 4)):
                order = np_rng.permutation(len(rows))
                for offset in range(0, len(rows), config.get("minibatch_size", 256)):
                    ids = torch.as_tensor(order[offset:offset+config.get("minibatch_size", 256)], device=device)
                    logits, values = solver.model(xs[ids], masks[ids])
                    dist = torch.distributions.Categorical(logits=logits)
                    ratios = (dist.log_prob(actions[ids])-old_logps[ids]).exp()
                    clip = config.get("clip_ratio", .2)
                    policy_loss = -torch.minimum(ratios*advantages[ids], ratios.clamp(1-clip, 1+clip)*advantages[ids]).mean()
                    value_loss = (values-targets[ids]).square().mean()
                    loss = policy_loss+.5*value_loss-config.get("entropy_coef", .01)*dist.entropy().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(solver.model.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    gradients += 1
                    losses.append(loss.item())
            record = {"episodes": completed, "environment_steps": steps, "gradient_steps": gradients,
                      "mean_training_score": float(np.mean(scores)), "loss": float(np.mean(losses)),
                      "gradient_norm": float(grad_norm), "wall_seconds": time.monotonic()-start}
            log.write(json.dumps(record)+"\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if completed == midpoint or completed == episodes:
                save(f"trained_{completed:06d}")
        if saved[-1]["episodes"] != completed:
            save(f"trained_{completed:06d}")
    summary = {"status": "completed" if completed == episodes else "wall_budget", "checkpoints": saved,
               "parameter_changed": parameter_hash(solver.model) != initial_hash,
               "training_wall_seconds": time.monotonic()-start, "test_used": False}
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(read_json(args.config), args.out)


if __name__ == "__main__":
    main()
