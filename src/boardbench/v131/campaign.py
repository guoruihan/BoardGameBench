"""Bounded CPU-only paired diagnostics using the existing frozen-source watchdog."""
import argparse
from pathlib import Path
import sys
import time

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v121.registry import validate_manifest
from boardbench.v122 import campaign as base
from boardbench.v125.campaign import cpu_core_limit
from .counterfactual import fingerprint, summarize

output_version = 'v131'
episode_module = 'boardbench.v131.counterfactual'
controller_module = 'boardbench.v131.campaign'


def prepare_plan(raw, manifest):
    validate_manifest(manifest)
    if raw['split'] != 'train_calibration' or digest(raw['states']) != raw['states_sha256']:
        raise ValueError('train-only checksum-verified state manifest required')
    if digest(raw['checkpoint']) != raw['checkpoint_sha256']:
        raise ValueError('student checkpoint mismatch')
    if type(raw['repeats']) is not int or not 2 <= raw['repeats'] <= 128 or not 10 < raw['job_seconds'] <= 1800:
        raise ValueError('bounded positive paired sample count and timeout required')
    rows = read_json(raw['states'])['states']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique nonempty state IDs required')
    for r in rows:
        if r['environment_seed'] not in set(manifest['sets']['train']):
            raise ValueError('state split leakage')
        if r['student_checkpoint_sha256'] != raw['checkpoint_sha256']:
            raise ValueError('state selected for different student')
    return {**raw, 'protocol': 'public_paired_counterfactual_v131',
            'rules_version': 'harmonies_solo_a_v1', 'split_sha256': manifest['sha256']}


def make_jobs(plan):
    if digest(plan['states']) != plan['states_sha256']:
        raise ValueError('state manifest changed')
    return [{**r, 'observation_sha256': fingerprint(r['observation']),
             'checkpoint': plan['checkpoint'], 'checkpoint_sha256': plan['checkpoint_sha256'],
             'repeats': plan['repeats'], 'timeout_seconds': plan['job_seconds'],
             'future_seed': int(fingerprint(['v131-fresh-futures', plan['sample_seed'], r['id']])[:16], 16)}
            for r in read_json(plan['states'])['states']]


def publish(root, plan, jobs, results, status):
    summary = {**summarize(results), 'phase': status['phase'], 'planned': len(jobs),
               'updated_unix': status['updated_unix'], 'source_id': status.get('source_id')}
    write_json(Path(root)/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('launch','control','report'))
    p.add_argument('--root', required=True); p.add_argument('--repo'); p.add_argument('--plan')
    a = p.parse_args(); backend = sys.modules[__name__]
    if a.mode == 'control': base.supervise(a.root, backend=backend)
    elif a.mode == 'report':
        root = Path(a.root)
        publish(root, read_json(root/'plan.json'), read_json(root/'jobs.json'),
                base.collect_results(root), read_json(root/'status.json'))
    else:
        raw = read_json(a.plan); raw['deadline_unix'] = time.time()+raw.pop('task_seconds')
        path = Path(a.root).with_suffix('.plan.json')
        if path.exists(): raise ValueError('refusing attempt overwrite')
        write_json(path, raw); base.launch(path, a.root, a.repo, backend=backend)


if __name__ == '__main__': main()
