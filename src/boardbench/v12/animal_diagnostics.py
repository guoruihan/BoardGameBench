"""True on-policy PPO on legal, no-chance animal curricula; grouped hold-outs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from .diagnostics import exact_values
from .encoding import STOP, index
from .training import Trainer

LEVELS = ('placement', 'construction', 'card_choice')


def case_split(cases):
    """All layouts from a source game stay together, including across levels."""
    train = [c for c in cases if c['source_seed'] % 4 != 3]
    heldout = [c for c in cases if c['source_seed'] % 4 == 3]
    if {c['source_seed'] for c in train} & {c['source_seed'] for c in heldout}:
        raise ValueError('source-game leakage')
    if {c['layout_id'] for c in train} & {c['layout_id'] for c in heldout}:
        raise ValueError('layout leakage')
    return train, heldout


def prepare(trajectories, allowed_seeds, *, per_level=6, max_nodes=3000, max_seconds=120):
    """Re-execute recorded legal prefixes, never manufacture arbitrary boards.

    Teacher windows identify animal-relevant tasks; every legal action plus STOP
    is considered by the exact oracle. Missing levels are reported, never invented.
    """
    deadline = time.monotonic() + max_seconds
    allowed = set(allowed_seeds)
    cases, skipped = [], {'node_cap': 0, 'no_score_choice': 0}
    layouts = {}
    counts = {(level, split): 0 for level in LEVELS for split in ('train', 'heldout')}
    with Path(trajectories).open() as stream:
        for line in stream:
            game = json.loads(line)
            if game['seed'] not in allowed:
                raise ValueError('teacher source game is outside the training split')
            env = Harmonies(); obs, _ = env.reset(game['seed'])
            examples = game['examples']
            split = 'heldout' if game['seed'] % 4 == 3 else 'train'
            seen_levels = set()
            for i, example in enumerate(examples):
                if time.monotonic() >= deadline:
                    break
                if obs != example['observation']:
                    raise ValueError('teacher prefix does not reproduce its public observation')
                action = example['action']
                kind = action['type']
                window = examples[i:i+3]
                animal_at = next((j for j, e in enumerate(window) if e['action']['type'] == 'place_animal'), None)
                level = ('placement' if kind == 'place_animal' else
                         'construction' if kind == 'place_token' and animal_at is not None else
                         'card_choice' if kind == 'take_card' and animal_at is not None else None)
                if (level and level not in seen_levels and counts[level, split] < per_level
                        and sum(c['stack_id'] == 0 for c in obs['board']) <= 2):
                    horizon = 1 if level == 'placement' else animal_at + 1
                    layout = json.dumps({k: obs[k] for k in ('board', 'active_cards', 'animal_market', 'pending_tokens')}, sort_keys=True)
                    layout_id = hashlib.sha256(layout.encode()).hexdigest()
                    if layout_id not in layouts:
                        snapshot = env.save_state()
                        try:
                            _, values = exact_values(snapshot, horizon, max_nodes)
                        except ValueError as exc:
                            if 'node cap' not in str(exc):
                                raise
                            skipped['node_cap'] += 1
                        else:
                            if max(values.values()) == min(values.values()):
                                skipped['no_score_choice'] += 1
                            else:
                                cases.append({'source_seed': game['seed'], 'decision_index': i,
                                              'level': level, 'horizon': horizon, 'snapshot': snapshot,
                                              'layout_id': layout_id, 'values': values})
                                layouts[layout_id] = split
                                seen_levels.add(level); counts[level, split] += 1
                obs = env.step(action).observation
            if time.monotonic() >= deadline or all(v >= per_level for v in counts.values()):
                break
    train, heldout = case_split(cases)
    return {'cases': cases, 'teacher_sha256': digest(trajectories), 'skipped': skipped,
            'counts': {level: {s: counts[level, s] for s in ('train', 'heldout')} for level in LEVELS},
            'train_source_games': sorted({c['source_seed'] for c in train}),
            'heldout_source_games': sorted({c['source_seed'] for c in heldout}),
            'boundary': 'auxiliary final-turn finite-action tasks, not full-game planning proof'}


def measure(solver, cases):
    rows = []
    for case in cases:
        env = Harmonies(); obs = env.load_state(case['snapshot'])
        values = {int(k): v for k, v in case['values'].items()}
        optimum = max(values.values())
        first_gap = None
        initial = obs['score_breakdown']['total']
        for _ in range(case['horizon']):
            action = solver.decide(obs, None)
            if first_gap is None:
                chosen = index(obs, action, 'slots240_v1', True)
                first_gap = optimum - values[chosen]
            if action == STOP:
                break
            obs = env.step(action).observation
            if obs['terminated']:
                break
        rows.append({'source_seed': case['source_seed'], 'level': case['level'],
                     'first_action_gap': first_gap, 'rollout_gap': optimum - obs['score_breakdown']['total'],
                     'score_gain': obs['score_breakdown']['total'] - initial})
    return rows


def diagnose_cases(prepared, config, device='cpu', *, seconds_per_method=45, max_batches=512):
    if config['network'].get('action_encoding') != 'slots240_v1':
        raise ValueError('animal diagnostics require the shared compact action encoding')
    train, heldout = case_split(prepared['cases'])
    allowed = set(config['splits']['train'])
    if any(c['source_seed'] not in allowed for c in train + heldout):
        raise ValueError('diagnostic source games must be training-only, not benchmark holdouts')
    results = {}
    for level in LEVELS:
        fitted = [c for c in train if c['level'] == level]
        unseen = [c for c in heldout if c['level'] == level]
        if len(fitted) < 2 or len(unseen) < 2:
            results[level] = {'status': 'insufficient_cases', 'train': len(fitted), 'heldout': len(unseen)}
            continue
        methods = {}
        for method in ('bc', 'ppo'):
            cfg = {**deepcopy(config), 'algorithm': method, 'batch_episodes': 16,
                   'epochs': 4, 'target_kl': .02}
            trainer = Trainer(cfg, device)
            dataset = []
            for case in fitted:
                env = Harmonies(); obs = env.load_state(case['snapshot'])
                data = trainer.solver.encode(obs)
                values = {int(k): v for k, v in case['values'].items()}
                for action, value in values.items():
                    if value == max(values.values()):
                        dataset.append({'features': data['features'], 'mask': data['action_mask'],
                                        'action': action, 'seed': case['source_seed']})
            initial = {'train': measure(trainer.solver, fitted), 'heldout': measure(trainer.solver, unseen)}
            deadline = time.monotonic() + seconds_per_method
            curve = []
            for batch in range(max_batches):
                if time.monotonic() >= deadline:
                    break
                metrics = trainer.bc_batch(dataset) if method == 'bc' else trainer.ppo_batch(fitted)
                if batch % 16 == 0:
                    gaps = measure(trainer.solver, fitted)
                    curve.append({'batch': batch + 1, 'loss': metrics['loss'], 'gaps': gaps})
                    # BC fits one-step Q labels; PPO must solve the complete small task.
                    field = 'first_action_gap' if method == 'bc' else 'rollout_gap'
                    if max(r[field] for r in gaps) == 0:
                        break
            final = {'train': measure(trainer.solver, fitted), 'heldout': measure(trainer.solver, unseen)}
            field = 'first_action_gap' if method == 'bc' else 'rollout_gap'
            methods[method] = {'status': 'fit_passed' if max(r[field] for r in final['train']) == 0 else 'fit_incomplete',
                               'initial': initial, 'final': final, 'curve': curve,
                               'optimizer_steps': trainer.updates, 'training_seconds': trainer.elapsed,
                               'learning_evidence': any(r[field] > 0 for r in initial['train']) and
                                                    max(r[field] for r in final['train']) == 0,
                               'oracle_used_for_training': method == 'bc'}
        results[level] = methods
    # A curriculum failure is a research finding, not an exception or a BC-as-RL pass.
    return {'levels': results, 'counts': prepared['counts'], 'source_game_holdout': True,
            'capability_verified': False, 'objective': 'true unweighted total-score difference',
            'boundary': 'BC first-action fit and PPO full short-rollout fit are distinct; full games still required'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectories', required=True); p.add_argument('--config', required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--prepare-seconds', type=float, default=120)
    p.add_argument('--seconds-per-method', type=float, default=45)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=False, parents=True)
    config = read_json(args.config)
    prepared = prepare(args.trajectories, config['splits']['train'], max_seconds=args.prepare_seconds)
    write_json(out / 'cases.json', prepared)
    report = diagnose_cases(prepared, config, args.device, seconds_per_method=args.seconds_per_method)
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    main()
