"""Freeze parent and replay-verified panels before independent future sampling."""
import argparse
from collections import defaultdict
from pathlib import Path

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v130.training import export_checkpoint
from .counterfactual import select_states


def prepare(config, root, repo):
    root, repo = Path(root), Path(repo)
    if root.exists(): raise ValueError('refusing prepared campaign overwrite')
    manifest = repo/config['split_manifest']
    if digest(manifest) != config['split_sha256']:
        raise ValueError('split hash mismatch')
    root.mkdir(parents=True)
    meta = export_checkpoint(repo/config['parent'], root/'student.npz')
    rows, sources = select_states([repo/p for p in config['source_runs']], root/'student.npz',
                                  meta['sha256'], read_json(manifest), config['states_per_environment'])
    validation = sorted({seed for p in config['source_runs']
                         for seed in read_json((repo/p).parent/'corrections.json')['validation_seeds']})
    by_env = defaultdict(list)
    for r in rows: by_env[r['environment_seed']].append(r)
    if len(by_env) < 16 or any(len(v) < 4 for v in by_env.values()):
        raise ValueError('insufficient representative panel')
    diagnostic = [row for seed in sorted(by_env)[:16] for row in by_env[seed][:4]]
    for name, selected in [('diagnostic', diagnostic), ('labels', rows)]:
        path = root/(name+'_states.json')
        write_json(path, {'states': selected, 'sources': sources, 'validation_seeds': validation,
                         'selection': 'stage/action-stratified disagreements, no Q-gap selection',
                         'parent': meta, 'train_only': True})
        write_json(root/(name+'_plan.json'), {
            'split': 'train_calibration', 'split_manifest': str(manifest),
            'states': str(path), 'states_sha256': digest(path),
            'checkpoint': str(root/'student.npz'), 'checkpoint_sha256': meta['sha256'],
            'repeats': config[name+'_repeats'], 'sample_seed': 13120261020+(name == 'labels'),
            'job_seconds': config['counterfactual_job_seconds'],
            'task_seconds': config[name+'_task_seconds'],
            'cpu_cores': config['counterfactual_cores'], 'resource_profile': 'b35_reserve16_v1'})
    write_json(root/'config.json', config)
    write_json(root/'prepared.json', {'states': len(rows), 'environments': len(by_env),
        'diagnostic_states': len(diagnostic), 'validation_seeds': validation, 'parent': meta})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True); p.add_argument('--root', required=True); p.add_argument('--config', required=True)
    a = p.parse_args(); prepare(read_json(a.config), Path(a.root).resolve(), Path(a.repo).resolve())
