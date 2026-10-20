"""Admit independently sampled paired advantages, grouped by environment."""
from pathlib import Path
import random
import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v12.encoding import encode, index
from boardbench.v121.registry import validate_manifest
from .counterfactual import fingerprint, paired_stats


def build(run, destination):
    run, destination = Path(run), Path(destination)
    if destination.exists(): raise ValueError('refusing dataset overwrite')
    plan=read_json(run/'plan.json'); manifest=read_json(run/'splits.json');validate_manifest(manifest)
    status=read_json(run/'status.json')
    if plan['split']!='train_calibration' or status['phase']!='completed' or status['active_count']:
        raise ValueError('complete inactive train counterfactual campaign required')
    if digest(plan['states'])!=plan['states_sha256']:
        raise ValueError('state manifest hash mismatch')
    states=read_json(plan['states']); val=set(states['validation_seeds'])
    jobs={j['id']:j for j in read_json(run/'jobs.json')}
    from .campaign import make_jobs
    if list(jobs.values())!=make_jobs(plan):raise ValueError('jobs differ from frozen state manifest')
    resolved={p.stem:read_json(p) for p in (run/'resolved').glob('*.json')}
    if set(jobs)!=set(resolved): raise ValueError('missing counterfactual tasks')
    arrays={k:[] for k in ('x','mask','y','student_y','seed','validation','habitat_mean','habitat_se',
                          'student_mean','student_se','samples')}
    for name,job in jobs.items():
        r=resolved[name];obs=job['observation'];seed=job['environment_seed']
        if (r['status']!='completed' or not r['natural_terminal'] or seed not in set(manifest['sets']['train'])
                or r['environment_seed']!=seed or r['observation_sha256']!=fingerprint(obs)
                or r['checkpoint_sha256']!=plan['checkpoint_sha256'] or len(r['samples'])!=plan['repeats']
                or r['teacher_action']!=job['teacher_action'] or r['student_action']!=job['student_action']):
            raise ValueError('invalid counterfactual provenance or split')
        if len({s['future_seed'] for s in r['samples']})!=len(r['samples']):
            raise ValueError('duplicate future samples')
        rng=random.Random(job['future_seed'])
        if [s['future_seed'] for s in r['samples']]!=[rng.getrandbits(63) for _ in range(plan['repeats'])]:
            raise ValueError('future samples differ from frozen independent stream')
        encoded=encode(obs,allow_stop=False)
        y=index(obs,job['teacher_action'],'slots240_v1',allow_stop=False)
        sy=index(obs,job['student_action'],'slots240_v1',allow_stop=False)
        if y==sy or not encoded['action_mask'][y] or not encoded['action_mask'][sy]:
            raise ValueError('distinct legal paired actions required')
        stats={}
        for policy in ('habitat','student'):
            stats[policy]=paired_stats([s['values'][policy+'_teacher']['score']-
                                       s['values'][policy+'_student']['score'] for s in r['samples']])
            if stats[policy]!=r['statistics'][policy]: raise ValueError('paired statistic mismatch')
        values=(np.asarray(encoded['features'],np.float32), np.asarray(encoded['action_mask'],bool),
                y,sy,seed,seed in val,stats['habitat']['mean'],stats['habitat']['se'],
                stats['student']['mean'],stats['student']['se'],len(r['samples']))
        for k,v in zip(arrays,values):arrays[k].append(v)
    seeds=set(arrays['seed'])
    if not val or not val<=seeds or not seeds-val:raise ValueError('environment-grouped fit and validation required')
    temporary=destination.with_suffix('.partial');destination.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{k:np.asarray(v) for k,v in arrays.items()})
    temporary.replace(destination)
    info={'sha256':digest(destination),'records':len(arrays['y']),'environments':len(seeds),
          'validation_seeds':sorted(val),'source':str(run),'source_id':read_json(run/'allocation.json')['source_id'],
          'plan_sha256':digest(run/'plan.json'),'checkpoint_sha256':plan['checkpoint_sha256'],
          'semantics':'independent paired teacher-minus-student original terminal score under named continuation; not PPO returns'}
    write_json(destination.with_suffix('.json'),info);return info


def weights(mean,se,*,z=1.96,minimum_gain=1.,scale=5.):
    if z<0 or minimum_gain<0 or scale<=0:raise ValueError('invalid conservative advantage weighting')
    mean,se=np.asarray(mean),np.asarray(se)
    if mean.shape!=se.shape or not np.isfinite(mean).all() or not np.isfinite(se).all() or (se<0).any():
        raise ValueError('finite matched paired means and errors required')
    return np.clip((mean-z*se-minimum_gain)/scale,0.,1.)
