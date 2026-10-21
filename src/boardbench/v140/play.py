"""Run a real two-seat match from portable policies, or transfer the solo parent."""
import argparse
import json
import random
import time
import torch

from boardbench.artifacts.store import write_json
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v138.baseline import load_model
from .model import DuoPolicy
from .policy import DuoSolver, load_policy


def match(policies, seed, first_player=0, max_seconds_per_seat=900.):
    env = HarmoniesDuo(); obs, _ = env.reset(seed, first_player)
    for p in policies:
        if p is not None: p.start_episode(obs)
    used = [0., 0.]; trace = []; rng = random.Random(seed+140)
    while not obs['terminated']:
        seat = obs['active_player']; tick = time.monotonic()
        action = policies[seat].decide(obs) if policies[seat] is not None else rng.choice(obs['legal_actions'])
        used[seat] += time.monotonic()-tick
        if used[seat] > max_seconds_per_seat:
            return dict(status='incomplete', reason='decision_budget', seed=seed, scores=None, winners=None,
                        decision_seconds=used, trajectory=trace)
        trace.append(dict(seat=seat, action=action, observation=obs))
        obs = env.step(action).observation
    return dict(status='completed', seed=seed, first_player=first_player, scores=obs['scores'],
                winners=obs['winners'], breakdown=obs['score_breakdown'],
                decision_seconds=used, trajectory=trace, final_observation=obs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights'); p.add_argument('--solo-parent')
    p.add_argument('--opponent', default='mirror', choices=('mirror', 'random'))
    p.add_argument('--seed', type=int, default=140); p.add_argument('--first-player', type=int, default=0)
    p.add_argument('--out', required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    if bool(args.weights) == bool(args.solo_parent): p.error('choose --weights or --solo-parent')
    model = load_policy(args.weights) if args.weights else DuoPolicy(load_model(args.solo_parent)[0])
    policies = [DuoSolver(model), DuoSolver(model) if args.opponent == 'mirror' else None]
    result = match(policies, args.seed, args.first_player)
    write_json(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trajectory', 'final_observation')}))
