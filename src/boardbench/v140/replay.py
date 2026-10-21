"""Validate/reconstruct all archived real actions as public offline transitions."""
import argparse
import gzip
import json

from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from .encoding import action_map, LAYOUT


def replay(game):
    env = FastHarmoniesDuo(); obs, _ = env.reset(game['seed'])
    for row in game['decisions']:
        if obs['terminated'] or row['seat'] != obs['active_player']:
            raise ValueError('trajectory seat/terminal mismatch')
        mapping = action_map(obs)
        if row['action'] not in mapping: raise ValueError('illegal recorded action')
        action = mapping[row['action']]; result = env.step(action)
        yield dict(observation=obs, action=action, next_observation=result.observation,
                   acting_player=row['seat'], behavior_logp=row['logp'],
                   engine_reward=result.reward, terminated=result.terminated,
                   reward_to_next_own_decision=row.get('reward_to_next_own_decision'))
        obs = result.observation
    if (not obs['terminated'] or obs['scores'][game['seat']] != game['score']
            or obs['scores'][1-game['seat']] != game['opponent_score']):
        raise ValueError('recorded outcome does not replay')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('archive')
    args = p.parse_args()
    with gzip.open(args.archive, 'rt') as f: saved = json.load(f)
    if saved['layout'] != LAYOUT: raise ValueError('unknown archive layout')
    counts = [sum(1 for _ in replay(g)) for g in saved['games']]
    print(dict(games=len(counts), verified_actions=sum(counts), source_id=saved['source_id']))
