"""Train-only teacher data, grouped by environment and never by adjacent rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from boardbench.artifacts.store import read_json, write_json, digest
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode, index
from boardbench.v121.registry import validate_manifest


def targets(obs, action, candidates, temperature=5.):
    data=encode(obs,allow_stop=False)
    label=index(obs,action,'slots240_v1',allow_stop=False)
    if action not in obs['legal_actions'] or not data['action_mask'][label]:
        raise ValueError('illegal teacher label')
    q=np.zeros(240,np.float32); valid=np.zeros(240,bool)
    values=[];ids=[]
    for c in candidates:
        if c['action'] not in obs['legal_actions']:
            raise ValueError('illegal candidate')
        if c.get('mean') is None or c.get('samples',0)<2:
            continue
        value=float(c['mean'])
        if not np.isfinite(value):
            raise ValueError('nonfinite teacher target')
        code=index(obs,c['action'],'slots240_v1',allow_stop=False)
        if code in ids:
            raise ValueError('duplicate candidate target')
        ids.append(code);values.append(value)
    if len(ids)>1:
        probabilities=np.exp((np.asarray(values)-max(values))/temperature)
        q[ids]=probabilities/probabilities.sum();valid[ids]=True
    return np.asarray(data['features'],np.float32),np.asarray(data['action_mask'],bool),label,q,valid


def build(runs, destination, validation_seeds):
    destination=Path(destination)
    if destination.exists():
        raise ValueError('refusing dataset overwrite')
    arrays={k:[] for k in ('x','mask','y','q','qmask','seed','validation','return_to_go')}
    games=0;sources=[]; val=set(validation_seeds); seen=set()
    for run in map(Path,runs):
        plan=read_json(run/'plan.json');status=read_json(run/'status.json')
        manifest=read_json(run/'splits.json');validate_manifest(manifest)
        if plan['split']!='train_calibration' or status['phase']!='completed' or status['active_count']:
            raise ValueError('only complete train-split campaigns may train students')
        allowed=set(manifest['sets']['train'])
        if not val<=allowed:
            raise ValueError('validation must be train-internal')
        sources.append({'run':str(run),'plan_sha256':digest(run/'plan.json'),
                        'source_id':read_json(run/'allocation.json')['source_id']})
        jobs={j['id']:j for j in read_json(run/'jobs.json')}
        results=list((run/'resolved').glob('*.json'))
        if len(results)!=len(jobs):
            raise ValueError('missing teacher attempts')
        for path in sorted(results):
            r=read_json(path);job=jobs[r['id']]
            if (r['status']!='completed' or not r['natural_terminal'] or
                r['environment_seed'] not in allowed or r['params']!=job['params'] or
                r['params']['continuation'] not in ('mixed_roots','dagger') or
                r['environment_seed']!=job['environment_seed']):
                raise ValueError('invalid teacher episode or split')
            engine=Harmonies();obs,_=engine.reset(r['environment_seed'])
            origin=Path(r.get('inherited_from',str(run)))
            rows=[json.loads(line) for line in (origin/'episodes'/r['id']/'decisions.jsonl').read_text().splitlines()]
            for row in rows:
                if obs!=row['observation_before']:
                    raise ValueError('teacher replay state mismatch')
                teacher=row['search'].get('expert_action',row['action'])
                x,mask,y,q,qmask=targets(obs,teacher,row['search'].get('candidates',[]))
                values=(x,mask,y,q,qmask,r['environment_seed'],r['environment_seed'] in val,
                        (r['score']-obs['score_breakdown']['total'])/200.)
                for k,value in zip(arrays,values): arrays[k].append(value)
                obs=engine.step(row['action']).observation
                if obs!=row['observation']:
                    raise ValueError('teacher replay successor mismatch')
            if not obs['terminated'] or obs['score']!=r['score']:
                raise ValueError('teacher terminal mismatch')
            games+=1;seen.add(r['environment_seed'])
    if not val or not val<=seen or not seen-val:
        raise ValueError('both train and complete validation environments required')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{k:np.asarray(v) for k,v in arrays.items()})
    info={'records':len(arrays['y']),'games':games,'environments':len(seen),'validation_seeds':sorted(val),
          'sha256':digest(temporary),'sources':sources,'q_semantics':'terminal score under habitat continuation; not optimal Q',
          'value_semantics':'behavior trajectory score delta return, not candidate Q; diagnostic only',
          'training_permission':'train split only; environment seed is metadata, never a feature'}
    write_json(destination.with_suffix('.json'),info)
    temporary.rename(destination)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--validation-seeds',nargs='+',type=int,required=True)
    a=p.parse_args();print(json.dumps(build(a.runs,a.out,a.validation_seeds)))
