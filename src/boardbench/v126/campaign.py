"""V126 adapter for existing frozen-source campaign and deadline watchdog."""
import argparse
import math
from pathlib import Path
import sys
import time

from boardbench.artifacts.store import read_json, write_json, digest, source_root
from boardbench.v122 import campaign as base
from boardbench.v125.campaign import cpu_core_limit, initial_results, resume_parent
from boardbench.v125.report import publish as prior_publish

output_version='v126'
episode_module='boardbench.v126.episode'
controller_module='boardbench.v126.campaign'
make_jobs=base.make_jobs


def snapshot_repository(repo, plan):
    # The outer pipeline may wait/run for hours while the worktree changes.
    # Always inherit the executing frozen package, not the mutable repo argument.
    return resume_parent(plan)/'source' if plan.get('resume_from') else source_root()


def prepare_plan(raw, manifest):
    plan=base.prepare_plan(raw,manifest)
    pool=manifest['sets']['train' if plan['split']=='train_calibration' else 'development']
    ids=raw['environment_indices']
    if len(ids)!=plan['panels'][-1] or len(set(ids))!=len(ids) or any(type(i)is not int or not 0<=i<len(pool) for i in ids):
        raise ValueError('invalid unique in-split environment indices')
    plan.update(environment_seeds=[pool[i] for i in ids],protocol='search_guided_learning_v126')
    learned={'student','student_roots','student_rollout','dagger'}
    for arm in plan['arms']:
        p=arm['params']; variant=p['continuation']
        keys={'mode','continuation','seconds'} | ({'checkpoint','checkpoint_sha256'} if variant in learned else set())
        if (set(p)!=keys or variant not in learned|{'mixed_roots','habitat','original'} or
            arm['method']!='terminal' or arm['multiplier']!=1 or
            not math.isfinite(p['seconds']) or not 0<p['seconds']<=900 or arm['timeout_seconds']<=p['seconds'] or
            p['mode']!=('student' if variant=='student' else 'timed')):
            raise ValueError('invalid V126 policy or budget')
        if variant=='dagger' and plan['split']!='train_calibration':
            raise ValueError('DAgger labels require train split')
        if variant in learned and digest(p['checkpoint'])!=p['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
    return plan


def publish(root, plan, jobs, results, status):
    summary=prior_publish(root,plan,jobs,results,status)
    p=Path(root)/'REPORT.md'
    text=p.read_text().replace('Harmonies V1.2.5 小批量策略迭代','Harmonies V1.2.6 搜索指导学习')
    text+='\n\nV126: student组是纯模型自然终局，非等耗时搜索。训练及示范成本另见上级REPORT。\n'
    p.write_text(text)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('launch','control','report'))
    p.add_argument('--root',required=True);p.add_argument('--plan');p.add_argument('--repo')
    a=p.parse_args();backend=sys.modules[__name__]
    if a.mode=='launch':
        raw=read_json(a.plan)
        raw['deadline_unix']=time.time()+raw.pop('task_seconds')
        target=Path(a.repo)/'outputs/v126/plans'/(Path(a.root).name+'.json')
        if target.exists():
            raise ValueError('plan exists; choose new attempt identity')
        write_json(target,raw)
        base.launch(target,a.root,a.repo,backend=backend)
    elif a.mode=='control':
        base.supervise(a.root,backend=backend)
    else:
        root=Path(a.root)
        publish(root,read_json(root/'plan.json'),read_json(root/'jobs.json'),base.collect_results(root),read_json(root/'status.json'))


if __name__=='__main__':
    main()
