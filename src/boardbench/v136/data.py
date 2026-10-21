"""Hindsight blueprint labels, not expert intent or privileged future inputs."""
import hashlib
from pathlib import Path

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import PATTERNS
from boardbench.v12.encoding import encode, index
from boardbench.v132.data import load as load_corpus
from boardbench.v134.plans import candidates, encode_plan, refresh_goal


def identity(goal):
    return goal['card_id'], goal['anchor'], tuple(sorted(goal['targets'].items()))


def segments(corpus, horizon=2, max_actions=16):
    """Stop backward at invisibility/infeasibility; never label a future card."""
    result = []
    for episode in corpus['episodes']:
        observations, actions = episode['observations'], episode['actions']
        for end, action in enumerate(actions):
            if action['type'] != 'place_animal':
                continue
            final = observations[end]
            card = next(c for c in final['active_cards'] if c['instance_id'] == action['instance_id'])
            anchor, orientation = action['anchor_cell_id'], action['orientation']
            goal = {'card_id': card['card_id'], 'placed': card['placed_count'], 'anchor': anchor,
                    'orientation': orientation, 'targets': {cell: final['board'][cell]['stack_id']
                        for cell, _ in PATTERNS[card['card_id']][anchor][orientation]}}
            start = end
            for j in range(end - 1, max(-1, end - max_actions), -1):
                obs = observations[j]
                if final['turn_index'] - obs['turn_index'] >= horizon or refresh_goal(obs, goal) is None:
                    break
                start = j
            if end - start < 2 or refresh_goal(observations[start], goal)['missing'] < 1:
                continue  # Not a layout skill if all we teach is the last placement.
            result.append({'episode_id': episode['id'], 'environment_seed': episode['environment_seed'],
                           'start': start, 'end': end, 'goal': goal, 'observations': observations[start:end+1],
                           'actions': actions[start:end+1]})
    return result


def build(corpus_spec, allowed_seeds, out):
    out = Path(out)
    if out.exists():
        raise ValueError('new skill dataset required')
    corpus = load_corpus(corpus_spec, set(allowed_seeds))
    rows = segments(corpus)
    if not rows:
        raise ValueError('no feasible multi-action expert segments')
    seeds = sorted({r['environment_seed'] for r in rows},
                   key=lambda s: hashlib.sha256(f'v136-skill-holdout-{s}'.encode()).hexdigest())
    if len(seeds) < 2:
        raise ValueError('at least two training environments for skill fit/probe separation')
    held = set(seeds[:max(1, len(seeds)//8)])
    x, goals, ordinary, masks, y, groups, envs, validation, coverage, probes = [], [], [], [], [], [], [], [], [], []
    for group, row in enumerate(rows):
        start_obs, goal = row['observations'][0], row['goal']
        bank = candidates(start_obs)
        coverage.append({'environment_seed': row['environment_seed'], 'segment': group,
                         'exact_primary_in_bank': any(p and identity(p[0]) == identity(goal) for p in bank),
                         'card_anchor_in_bank': any(p and (p[0]['card_id'], p[0]['anchor']) == (goal['card_id'], goal['anchor']) for p in bank),
                         'missing': refresh_goal(start_obs, goal)['missing']})
        if row['environment_seed'] in held:
            probes.append({k: row[k] for k in ('episode_id', 'environment_seed', 'start', 'end', 'goal')}
                            | {'observation': start_obs})
        for obs, action in zip(row['observations'], row['actions']):
            encoded = encode(obs, allow_stop=False)
            current = candidates(obs)
            x.append(encoded['features']); masks.append(encoded['action_mask'])
            goals.append(encode_plan(obs, [goal], obs['turn_index'] - start_obs['turn_index']))
            ordinary.append(encode_plan(obs, current[1] if len(current) > 1 else []))
            y.append(index(obs, action, 'slots240_v1', False))
            groups.append(group); envs.append(row['environment_seed']); validation.append(row['environment_seed'] in held)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, x=np.asarray(x, np.float32), goal=np.asarray(goals, np.float32),
                            ordinary=np.asarray(ordinary, np.float32), mask=np.asarray(masks, bool),
                            y=np.asarray(y, np.int64), group=np.asarray(groups), seed=np.asarray(envs),
                            validation=np.asarray(validation))
    temporary.replace(out)
    metadata = {'format': 'v136_public_goal_segments', 'sha256': digest(out), 'corpus': corpus_spec,
                'states': len(y), 'segments': len(rows), 'fit_environments': len(seeds)-len(held),
                'probe_environments': sorted(held), 'coverage': coverage,
                'exact_primary_coverage': float(np.mean([r['exact_primary_in_bank'] for r in coverage])),
                'card_anchor_coverage': float(np.mean([r['card_anchor_in_bank'] for r in coverage])),
                'labels': 'retrospective achieved public blueprint; NOT expert intention or known future',
                'probe_scope': 'held out from skill BC, NOT from prior or online PPO training',
                'group_sampling': 'uniform segment, then uniform transition; no independent-row inference'}
    write_json(out.with_suffix('.json'), metadata)
    write_json(out.with_suffix('.probes.json'), probes)
    return metadata


def load(spec, allowed_seeds):
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('skill dataset hash mismatch')
    with np.load(spec['path'], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if not set(data['seed'].tolist()) <= set(allowed_seeds):
        raise ValueError('skill split leakage')
    fit, hold = ~data['validation'], data['validation']
    if not fit.any() or not hold.any() or set(data['seed'][fit]) & set(data['seed'][hold]):
        raise ValueError('skill fit/probe overlap')
    if (data['x'].shape != (len(data['y']), 1346) or data['goal'].shape != (len(data['y']), 819)
            or data['ordinary'].shape != data['goal'].shape or data['mask'].shape != (len(data['y']), 240)
            or not data['mask'][np.arange(len(data['y'])), data['y']].all()
            or any(not np.isfinite(data[k]).all() for k in ('x', 'goal', 'ordinary'))):
        raise ValueError('skill encoding/mask mismatch')
    return data
