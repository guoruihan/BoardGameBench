"""Search demonstrations: public observations, semantic actions and full costs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers.board import BoardSolver
from .encoding import encode, index


def generate(seed):
    begin = time.monotonic()
    env = Harmonies(); obs, _ = env.reset(seed)
    solver = BoardSolver(seed, method='search', width=6, rollouts=4, depth=2,
                         max_simulation_steps=384, depth_unit='turn', leaf='potential')
    steps = [0]
    def count(): steps[0] += 1
    context = SimpleNamespace(reference_simulator=SearchAccess(env.fork, count))
    examples, snapshots, actions = [], [], []
    while not obs['terminated']:
        action = solver.decide(obs, context)
        # No random refill can occur once <=2 empty cells; end_turn is natural.
        if (sum(c['stack_id'] == 0 for c in obs['board']) <= 2 and len(snapshots) < 3
                and any(a.get('type') == 'place_animal' for a in obs['legal_actions'])):
            snapshots.append(env.save_state())
        examples.append({'observation': obs, 'action': action})
        actions.append(action)
        obs = env.step(action).observation
    return {'seed': seed, 'score': obs['score'], 'seconds': time.monotonic() - begin,
            'simulation_steps': steps[0], 'examples': examples, 'endgames': snapshots}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    args = p.parse_args()
    config = read_json(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    dataset, summaries, endgames = [], [], []
    with ProcessPoolExecutor(max_workers=config.get('workers', 4), mp_context=mp.get_context('spawn')) as pool:
        with (out / 'semantic_trajectories.jsonl').open('x') as stream:
            for game in pool.map(generate, config['seeds']):
                stream.write(json.dumps(game) + '\n'); stream.flush()
                summaries.append({k: v for k, v in game.items() if k not in ('examples', 'endgames')})
                endgames.extend(game['endgames'])
                for example in game['examples']:
                    obs = example['observation']
                    data = encode(obs)
                    dataset.append({'features': data['features'], 'mask': data['action_mask'],
                                    'action': index(obs, example['action'], 'slots240_v1'),
                                    'seed': game['seed']})
    write_json(out / 'dataset.json', dataset)
    write_json(out / 'endgames.json', endgames)
    write_json(out / 'summary.json', {'games': summaries, 'examples': len(dataset),
                'sum_teacher_seconds': sum(g['seconds'] for g in summaries),
                'total_simulation_steps': sum(g['simulation_steps'] for g in summaries),
                'training_only': True, 'inference_teacher_calls': 0})


if __name__ == '__main__':
    main()
