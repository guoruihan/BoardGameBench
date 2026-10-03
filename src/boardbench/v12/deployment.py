"""Explicit legacy-weight imports and atomic, verified release pointers."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

from boardbench.artifacts.store import (ArtifactStore, digest, read_json, save_checkpoint,
                                       source_id, validate_checkpoint, write_json)
from boardbench.environments import TASKS
from boardbench.provenance import validate_splits
from boardbench.runner.config import normalize
from boardbench.runner.context import empty_resources
from boardbench.ui.policies import PolicyRegistry
from .timed import build_policy


def legacy_policies(root, holdouts):
    result = {}
    for task in ('micro_tiles', 'take_it_easy', 'harmonies'):
        policies = []
        for path in sorted((Path(root) / task / 'policies').glob('*/manifest.json')):
            entry = read_json(path)
            if entry['environment_id'] != task or entry['rules_version'] != TASKS[task].spec['version']:
                raise ValueError('legacy task/rules mismatch')
            config_path = PolicyRegistry._relative(path.parent, entry['config_path'])
            artifact = PolicyRegistry._relative(path.parent, entry['artifact_path'])
            if digest(config_path) != entry['config_sha256'] or digest(artifact) != entry['artifact_sha256']:
                raise ValueError('legacy artifact changed')
            origin = entry.get('origin', {})
            if origin:
                splits = validate_splits(origin['splits'])
                if set.union(*splits.values()) & set(holdouts):
                    raise ValueError('new holdout overlaps any historical split')
            spec = {'id': entry['policy_id'], 'method': entry['method'], 'label': entry['label'],
                    'solver': read_json(config_path)['solver'], 'legacy': {
                        'manifest': str(path.resolve()), 'manifest_sha256': digest(path),
                        'artifact_sha256': digest(artifact), 'source_id': entry['code_version'],
                        'import_kind': 'weights/config re-export; NOT original checkpoint restore'},
                    'validation_mean': entry.get('validation_mean'),
                    'training_seed': entry.get('training_seed')}
            if spec['solver']['id'] == 'ppo':
                spec.update(weights=str(artifact.resolve()), weights_sha256=digest(artifact))
            policies.append(spec)
        if not policies:
            raise ValueError(f'no legacy policies for {task}')
        result[task] = policies
    return result


def export(task, spec, root, summary=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    solver = build_policy(spec)
    config = normalize({'task': {'id': task, 'version': TASKS[task].spec['version']},
                        'solver': spec['solver'], 'episodes': 1, 'seed': 90210,
                        'model': {'max_calls': 0}, 'capabilities': {'reference_simulator': spec['method'] == 'search'}})
    store = ArtifactStore(root / 'export', config)
    try:
        checkpoint = save_checkpoint(store, solver, config, {'completed_episodes': 0, 'started_episodes': 0},
                                     empty_resources(), store.root / 'workspace')
    finally:
        store.close()
    artifact = checkpoint / 'solver' / ('weights.pt' if spec['solver']['id'] in ('ppo', 'compact') else 'board_solver.json')
    write_json(root / 'config.json', config)
    write_json(root / 'provenance.json', spec)
    write_json(root / 'validation.json', summary or {})
    write_json(root / 'manifest.json', {
        'policy_id': spec['id'], 'environment_id': task, 'rules_version': TASKS[task].spec['version'],
        'method': spec['method'], 'label': spec['label'], 'code_version': source_id(),
        'config_path': 'config.json', 'config_sha256': digest(root / 'config.json'),
        'artifact_path': str(artifact.relative_to(root)), 'artifact_sha256': digest(artifact),
        'checkpoint_path': str(checkpoint.relative_to(root)), 'inference_device': 'cpu',
        'validation_mean': (summary or {}).get('score_mean'),
        'training_seed': spec.get('training_seed'), 'legacy_import': spec.get('legacy')})
    validate_checkpoint(checkpoint)


def publish(release, pointer, defaults, *, kind, gate=None):
    release, pointer = Path(release).resolve(), Path(pointer).resolve()
    if not release.is_relative_to(pointer.parent):
        raise ValueError('release must remain inside the deployment root')
    if kind == 'promotion' and not (gate and gate.get('passed')):
        raise ValueError('promotion requires an explicit passing validation gate')
    if kind not in ('promotion', 'legacy_import'):
        raise ValueError('unknown publication type')
    manifests = {}
    for task, policy_id in defaults.items():
        registry = PolicyRegistry({'id': task, 'version': TASKS[task].spec['version']}, release / task / 'policies')
        try:
            for name in registry.entries:
                registry.get(name, fresh=True)
            if policy_id not in registry.entries:
                raise ValueError('default policy is absent from verified release')
        finally:
            registry.close()
        for path in (release / task / 'policies').glob('*/manifest.json'):
            manifests[str(path.relative_to(release))] = digest(path)
    previous = read_json(pointer) if pointer.exists() else None
    write_json(release / 'publication.json', {'kind': kind, 'gate': gate, 'previous': previous})
    value = {'format_version': 1, 'release': str(release), 'defaults': defaults,
             'source_id': source_id(), 'manifests': manifests, 'published_unix': time.time(), 'kind': kind}
    # Previous pointer is preserved in the immutable release before atomic replace.
    write_json(pointer, value)
    return value


def resolve_release(pointer, task):
    pointer = Path(pointer).resolve()
    if not pointer.is_file():
        return None, None
    value = read_json(pointer)
    release = Path(value['release']).resolve()
    if not release.is_relative_to(pointer.parent) or value['source_id'] != source_id():
        raise ValueError('deployment path/source mismatch; use the matching frozen source')
    for relative, expected in value['manifests'].items():
        path = PolicyRegistry._relative(release, relative)
        if digest(path) != expected:
            raise ValueError('published policy manifest changed')
    return release / task / 'policies', value['defaults'].get(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    value = read_json(args.config)
    for task, specs in value['policies'].items():
        for spec in specs:
            export(task, spec, Path(value['release']) / task / 'policies' / spec['id'], spec.get('summary'))
    publish(value['release'], value['pointer'], value['defaults'], kind=value['kind'], gate=value.get('gate'))


if __name__ == '__main__':
    main()
