"""b35 CPU-only bounded correction campaigns using the existing watchdog."""
import argparse
import math
from pathlib import Path
import sys
import time

from boardbench.artifacts.store import digest, read_json, write_json, source_root
from boardbench.v122 import campaign as base
from boardbench.v125.campaign import cpu_core_limit, initial_results, resume_parent

output_version = 'v130'
episode_module = 'boardbench.v130.teacher'
controller_module = 'boardbench.v130.campaign'
make_jobs = base.make_jobs


def snapshot_repository(repo, plan):
    return resume_parent(plan)/'source' if plan.get('resume_from') else source_root()


def prepare_plan(raw, manifest):
    plan = base.prepare_plan(raw, manifest)
    if plan['split'] != 'train_calibration':
        raise ValueError('correction labels only on training split')
    indices = raw['environment_indices']; pool = manifest['sets']['train']
    if len(indices) != plan['panels'][-1] or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(pool) for i in indices):
        raise ValueError('invalid unique train indices')
    plan.update(environment_seeds=[pool[i] for i in indices], protocol='student_correction_v130')
    for arm in plan['arms']:
        p = arm['params']
        if (set(p) != {'mode','continuation','seconds','checkpoint','checkpoint_sha256','stochastic'} or
            p['mode'] != 'timed' or p['continuation'] != 'dagger' or type(p['stochastic']) is not bool or
            arm['method'] != 'terminal' or arm['multiplier'] != 1 or
            not math.isfinite(p['seconds']) or not 0 < p['seconds'] <= 900 or arm['timeout_seconds'] <= p['seconds'] or
            digest(p['checkpoint']) != p['checkpoint_sha256']):
            raise ValueError('invalid correction policy/budget/checkpoint')
    return plan


def publish(root, plan, jobs, results, status):
    # These scores belong to student roll-in, NOT teacher-controlled games.
    completed = [r for r in results if r['status'] == 'completed' and r['natural_terminal']]
    summary = {'phase':status['phase'], 'completed':len(completed), 'planned':len(jobs),
               'score_semantics':'student roll-in trajectory; search labels only',
               'rollin_mean':sum(r['score'] for r in completed)/len(completed) if completed else None,
               'source_id':status.get('source_id'), 'watchdog':status.get('watchdog')}
    write_json(Path(root)/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('launch','control'))
    p.add_argument('--root', required=True); p.add_argument('--plan'); p.add_argument('--repo')
    a = p.parse_args(); backend = sys.modules[__name__]
    if a.mode == 'control':
        base.supervise(a.root, backend=backend)
    else:
        raw = read_json(a.plan)
        raw['deadline_unix'] = time.time()+raw.pop('task_seconds')
        prepared = Path(a.root).with_suffix('.plan.json')
        if prepared.exists():
            raise ValueError('attempt plan exists; do not overwrite')
        write_json(prepared, raw)
        base.launch(prepared, a.root, a.repo, backend=backend)


if __name__ == '__main__':
    main()
