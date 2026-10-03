"""Fixed-set PPO learning probe; never evaluates the declared final test set."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments import TASKS
from boardbench.environments.numerical import encode, decode
from boardbench.solvers.neural import NeuralSolver
from boardbench.training import train


def assess(solver, seeds, sample=False):
    torch.manual_seed(913)
    scores, actions, entropies = [], Counter(), []
    for seed in seeds:
        env = TASKS[solver.task_id].factory()
        obs, _ = env.reset(seed)
        while not obs['terminated']:
            data = encode(obs)
            with torch.no_grad():
                logits, _ = solver.model(torch.tensor(data['features'], device=solver.device)[None],
                                          torch.tensor(data['action_mask'], device=solver.device)[None])
                dist = torch.distributions.Categorical(logits=logits)
                index = int((dist.sample() if sample else logits.argmax(-1)).item())
                entropies.append(dist.entropy().item())
            action = decode(obs, index)
            actions[index] += 1
            obs = env.step(action).observation
        scores.append(obs['score'])
    return {'mean': statistics.mean(scores), 'scores': scores,
            'mean_entropy': statistics.mean(entropies), 'action_histogram': dict(actions)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(task_id='micro_tiles', training_seed=811, device=args.device,
                  hidden=128, episodes=8192, batch_episodes=32, train_seed_count=8,
                  checkpoint_every=1024, epochs=4, threads=2, learning_rate=.0003,
                  train_env_seed_start=4100000, validation_seeds=list(range(5000000,5000064)),
                  test_seeds=list(range(6000000,6000128)), max_wall_seconds=600)
    with (out/'training.log').open('x') as stream, redirect_stdout(stream):
        train(config, out/'train')
    splits = read_json(out/'train/splits.json')
    report = {'config':config, 'test_used':False, 'checkpoints':[]}
    torch.set_num_threads(2)
    for cp in read_json(out/'train/checkpoints.json'):
        solver = NeuralSolver(0, 'micro_tiles', hidden=128, device='cpu')
        solver.load_weights(out/'train'/cp['path'])
        item = {'checkpoint':cp['path'], 'episodes':cp['episodes'], 'train_seconds':cp['wall_seconds']}
        for name in ('train','validation'):
            for mode in ('argmax','sample'):
                item[name+'_'+mode] = assess(solver, splits[name], sample=mode=='sample')
        report['checkpoints'].append(item)
        write_json(out/'report.json', report)
        print(json.dumps({k:(v['mean'] if isinstance(v,dict) else v) for k,v in item.items()}), flush=True)


if __name__ == '__main__':
    main()
