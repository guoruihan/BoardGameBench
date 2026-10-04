"""Audited new-root numeric recovery, selecting by progress, never by held-out score."""
from copy import deepcopy
from pathlib import Path

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.training import config_id
from .registry import NUMERIC, definitions, identity


def numeric_parent_config(config):
    parent = NUMERIC.get(config['experiment_id'])
    if parent is None or config['network'].get('numeric_dtype') != 'float64':
        raise ValueError('numeric import requires registered FP64 variant')
    result = deepcopy(config)
    result['experiment_id'] = parent
    result['experiment_sha256'] = identity(definitions()[parent])
    result['network']['experiment_id'] = parent
    del result['network']['numeric_dtype']
    return result


def prepare_recovery(parent_root, root, experiments, seeds):
    parent_root, root = Path(parent_root).resolve(), Path(root).resolve()
    allocation = read_json(parent_root / 'allocation.json')
    status = read_json(parent_root / 'status.json')
    read_json(parent_root / 'watchdog_result.json')  # A stopped controller alone may still have children.
    if status['phase'] not in ('failed', 'stopped_by_request'):
        raise ValueError('numeric recovery requires a stopped parent campaign')
    parent_source = allocation['source_id']
    if source_id(parent_root / 'source') != parent_source or status['source_id'] != parent_source:
        raise ValueError('parent frozen source mismatch')
    if read_json(parent_root / 'plan/splits.json') != read_json(root / 'plan/splits.json'):
        raise ValueError('numeric recovery split mismatch')
    if {NUMERIC[x] for x in experiments} != set(allocation['experiments']) or list(seeds) != allocation['training_seeds']:
        raise ValueError('numeric recovery requires all matched experiments and seeds')
    entries = {}
    for experiment in experiments:
        for seed in seeds:
            config = read_json(root / 'plan/configs' / f'{experiment}_{seed}.json')
            parent_config = numeric_parent_config(config)
            parent_key = f'{NUMERIC[experiment]}_{seed}'
            if read_json(parent_root / 'plan/configs' / f'{parent_key}.json') != parent_config:
                raise ValueError('numeric recovery may only change the registered dtype variant')
            candidates = []
            for phase in (parent_root / 'training').glob(parent_key + '_a*_chunk*'):
                for name in ('initial', 'latest', 'final'):
                    path = phase / f'{name}.resume.pt'
                    if not path.with_suffix('.json').exists():
                        continue  # Incomplete atomic saves without metadata are never imported.
                    meta = read_json(path.with_suffix('.json'))
                    if meta['source_id'] != parent_source or meta['config_id'] != config_id(parent_config):
                        raise ValueError('parent checkpoint metadata config/source mismatch')
                    candidates.append((meta['real_atomic_actions'], meta['updates'], meta['batches'], name == 'final', str(path), meta))
            if not candidates:
                raise ValueError(f'missing recovery checkpoint for {parent_key}')
            *_, saved, meta = max(candidates, key=lambda x: x[:-1])
            path = Path(saved)
            if digest(path) != meta['resume_sha256'] or digest(path.with_suffix('.weights.pt')) != meta['weights_sha256']:
                raise ValueError('selected recovery checkpoint hash mismatch; inspect, do not silently roll back')
            entries[f'{experiment}_{seed}'] = {'resume': saved, 'actions': meta['real_atomic_actions'],
                'numeric_import': True, 'parent_source': parent_source, 'parent_sha256': meta['resume_sha256'],
                'parent_updates': meta['updates'], 'parent_batches': meta['batches']}
    manifest = {'mode': 'explicit_fp32_to_fp64_numeric_continuation', 'parent_root': str(parent_root),
        'parent_source_id': parent_source, 'entries': entries,
        'selection': 'greatest complete checkpoint (actions, updates, batches); no score selection',
        'historical_curve': str(parent_root / 'learning_curve.json'),
        'historical_curve_sha256': digest(parent_root / 'learning_curve.json'),
        'prior_gpu_seconds': status['gpu_seconds'], 'prior_cpu_core_seconds': status['cpu_core_seconds'],
        'prior_elapsed_seconds': status['elapsed_seconds'], 'parent_target_actions': status['current_target_actions'],
        'accounting': 'parent reservation costs retained including discarded post-checkpoint work; new costs additional'}
    write_json(root / 'recovery.json', manifest)
    return manifest
