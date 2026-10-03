"""Fail-closed training-source admission before any validation rollout."""
import hashlib
import json
from pathlib import Path

from boardbench.artifacts.store import digest, read_json
from boardbench.environments import TASKS
from boardbench.environments.numerical import ENCODING_VERSION
from boardbench.solvers.action_features import ACTION_FEATURE_VERSION


def validate_splits(splits):
    sets = {}
    for name in ('train', 'validation', 'test'):
        values = splits[name]
        if not values or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f'invalid {name} seeds')
        if len(set(values)) != len(values):
            raise ValueError(f'duplicate {name} seeds')
        sets[name] = set(values)
    if any(sets[a] & sets[b] for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('training/validation/test seed overlap')
    return sets


def training_candidates(directory):
    import torch
    root = Path(directory).resolve()
    config, splits = read_json(root/'config.json'), read_json(root/'splits.json')
    validate_splits(splits)
    task = config['task_id']
    if config.get('algorithm','ppo') not in ('ppo','imitation'):
        raise ValueError('unsupported training algorithm provenance')
    checkpoints = read_json(root/'checkpoints.json')
    identity = {'config': config, 'splits': splits, 'checkpoints': checkpoints}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    origin = {'run_id': run_id, 'directory_hint': str(root), 'config': config,
              'splits': splits, 'task_id': task, 'rules_version': TASKS[task].spec['version']}
    items = []
    for checkpoint in checkpoints:
        weights = (root/checkpoint['path']).resolve()
        if not weights.is_relative_to(root) or digest(weights) != checkpoint['sha256']:
            raise ValueError('training weight path/hash mismatch')
        metadata = torch.load(weights, map_location='cpu', weights_only=True)
        network = metadata.get('network', 'flat_mlp')
        if network != config.get('network', 'flat_mlp') or network not in ('flat_mlp','action_mlp'):
            raise ValueError('training configuration/artifact network mismatch')
        for key, expected in (('task_id', task), ('rules_version', origin['rules_version']),
                              ('encoding_version', ENCODING_VERSION if network=='flat_mlp' else ACTION_FEATURE_VERSION), ('format_version', 1)):
            if metadata[key] != expected:
                raise ValueError(f'training artifact mismatch: {key}')
        hidden = metadata['hidden']
        if type(hidden) is not int or hidden <= 0 or config.get('hidden', 128) != hidden:
            raise ValueError('training configuration/artifact hidden mismatch')
        method = 'imitation' if config.get('algorithm')=='imitation' else 'rl'
        prefix = 'bc' if method=='imitation' else 'rl'
        items.append({'id': f"{prefix}_{weights.stem}__{run_id[:16]}", 'method': method, 'params': {},
                      'hidden': hidden, 'network': network, 'untrained': checkpoint['episodes'] == 0,
                      'weights': str(weights), 'weights_sha256': checkpoint['sha256'],
                      'training': checkpoint, 'origin': origin})
    if not items or not any(c['untrained'] for c in items):
        raise ValueError('training run requires checkpoints and an untrained control')
    return items


def admit_sources(task, definitions, main_splits):
    holdout = validate_splits(main_splits)
    seen = {}
    for candidate in definitions:
        if candidate['method'] not in ('rl','imitation'):
            continue
        origin = candidate['origin']
        if origin['task_id'] != task or origin['rules_version'] != TASKS[task].spec['version']:
            raise ValueError('candidate source task/rules mismatch')
        splits = validate_splits(origin['splits'])
        if splits['train'] & (holdout['validation'] | holdout['test']):
            raise ValueError(f"source training/primary holdout overlap: {origin['run_id']}")
        prior = seen.setdefault(candidate['id'], candidate['weights_sha256'])
        if prior != candidate['weights_sha256']:
            raise ValueError('candidate identity collision')
